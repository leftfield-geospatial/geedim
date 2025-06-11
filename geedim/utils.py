"""
Copyright 2021 Dugal Harris - dugalh@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import asyncio
import atexit
import io
import json
import logging
import os
import threading
import warnings
from asyncio.runners import Runner
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generic, TypeVar

import aiohttp
import ee
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

T = TypeVar('T')


def Initialize(opt_url: str = 'https://earthengine-highvolume.googleapis.com', **kwargs) -> None:
    """
    Initialise Earth Engine.

    Credentials will be read from the `EE_SERVICE_ACC_PRIVATE_KEY` environment variable if it
    exists (useful for integrating with e.g. GitHub actions).

    :param opt_url:
        The Earth Engine endpoint to use.  Defaults to the high volume endpoint.  See `the Earth
        Engine docs <https://developers.google.com/earth-engine/guides/processing_environments
        #endpoints>`_ for more information.
    :param kwargs:
        Optional arguments to pass to ``ee.Initialize``.
    """
    # Adapted from https://gis.stackexchange.com/questions/380664/how-to-de-authenticate-from-earth-engine-api.
    env_key = 'EE_SERVICE_ACC_PRIVATE_KEY'

    if env_key in os.environ:
        # authenticate with service account
        key_dict = json.loads(os.environ[env_key])
        credentials = ee.ServiceAccountCredentials(
            key_dict['client_email'], key_data=key_dict['private_key']
        )
        ee.Initialize(credentials, opt_url=opt_url, project=key_dict['project_id'], **kwargs)
    else:
        ee.Initialize(opt_url=opt_url, **kwargs)


def singleton(cls: T, *args, **kwargs) -> T:
    """Class decorator to make it a singleton."""
    instances = {}

    def getinstance() -> cls:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return getinstance


def split_id(image_id: str) -> tuple[str | None, str | None]:
    """
    Split an Earth Engine image ID into collection and index components.

    :param image_id:
        Earth Engine ID.

    :return:
        A tuple of the image collection name and image index.
    """
    if not image_id:
        return None, None
    parts = image_id.strip('/').split('/')
    index = parts[-1]
    ee_coll_name = '/'.join(parts[:-1])
    return ee_coll_name, index


class Spinner(tqdm):
    _ascii = r'/-\|'

    def __init__(
        self,
        desc: str | None = None,
        leave: bool | str = '',
        file: io.TextIOBase | None = None,
        ascii: str = _ascii,
        disable: bool = False,
        position: int | None = None,
        interval: float = 0.2,
    ):
        """
        Spinner context manager that cooperates with tqdm.

        :param desc:
            Prefix for the spinner.
        :param leave:
            Whether to leave the spinner on termination (``True``), clear it (``False``),
            or replace the spinner character with the ``leave`` value when it is a string.
        :param file:
            File object to write to (defaults to ``sys.stderr``).
        :param ascii:
            Spinner characters to cycle through.
        :param disable:
            Whether to disable spinner display.
        :param position:
            Line offset to print the spinner.  Automatic if ``None``.  Useful for multiple
            spinners / tqdm bars.
        :param interval:
            Update interval (seconds).
        """
        super().__init__(
            desc=desc,
            leave=leave,
            file=file,
            disable=disable,
            position=position,
            miniters=0,
            smoothing=0,
        )
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.disable:
            return self

        def run():
            while not self._stop.wait(self._interval):
                self.update()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._thread:
            self._stop.set()
            self._thread.join()
        super().__exit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def format_meter(n: int, prefix: str = '', **kwargs):
        return prefix + Spinner._ascii[n % len(Spinner._ascii)]

    def close(self):
        if isinstance(self.leave, str):
            self.display(self.desc + self.leave + '\n')
            self.leave = False
        super().close()


def asset_id(filename: str, folder: str | None = None):
    """
    Convert a ``filename`` and ``folder`` into an Earth Engine asset ID.

    If ``folder`` is not supplied, ``filename`` is returned as is.  Otherwise ``folder`` is
    split to create an Earth Engine asset ID as:

        projects/<root folder>/assets/<sub folder(s)>/<filename>.
    """
    if not folder:
        return filename
    parts = folder.strip('/').split('/')
    asset_path = '/'.join(parts[1:] + [filename])
    return f'projects/{parts[0]}/assets/{asset_path}'


def register_accessor(name: str, cls: type) -> Callable[[type[T]], type[T]]:
    """
    Decorator function to register a custom accessor on a given class.

    Adapted from https://github.com/pydata/xarray/blob/main/xarray/core/extensions.py, Apache-2.0
    Licence.

    :param name:
        Name under which the accessor should be registered.
    :param cls:
        Class on which to register the accessor.
    """

    class CachedAccessor(Generic[T]):
        """Custom property-like object (descriptor) for caching accessors."""

        def __init__(self, name: str, accessor: type[T]):
            self._name = name
            self._accessor = accessor

        def __get__(self, obj, cls) -> type[T] | T:
            if obj is None:
                # return accessor type when the class attribute is accessed e.g. ee.Image.gd
                return self._accessor

            # retrieve the cache, creating if it does not yet exist
            try:
                cache = obj._accessor_cache
            except AttributeError:
                cache = obj._accessor_cache = {}

            # return accessor if it has been cached
            try:
                return cache[self._name]
            except KeyError:
                pass

            # otherwise create & cache the accessor
            try:
                accessor_obj = self._accessor(obj)
            except AttributeError as ex:
                # __getattr__ on data object will swallow any AttributeErrors raised when
                # initializing the accessor, so we need to raise as something else
                raise RuntimeError(f"Error initializing {self._name!r} accessor.") from ex

            cache[self._name] = accessor_obj
            return accessor_obj

    def decorator(accessor: type[T]) -> type[T]:
        if hasattr(cls, name):
            warnings.warn(
                f"Registration of accessor {accessor!r} under name '{name}' for type {cls!r} is"
                "overriding a preexisting attribute with the same name.",
                category=RuntimeWarning,
                stacklevel=2,
            )
        setattr(cls, name, CachedAccessor(name, accessor))
        return accessor

    return decorator


def get_tqdm_kwargs(desc: str | None = None, unit: str | None = None, **kwargs) -> dict[str, Any]:
    """Return a dictionary of kwargs for a tqdm progress bar."""
    tqdm_kwargs: dict[str, Any] = dict(dynamic_ncols=True, leave=None, smoothing=0)
    tqdm_kwargs.update(**kwargs)
    if desc:
        # clip the desc to max_width
        max_width = 40  # length of an S2 system:index
        desc_width = len(desc)
        desc = '...' + desc[-max_width + 3 :] if desc_width > max_width else desc
        tqdm_kwargs.update(desc=desc)

    if unit:
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]'
        tqdm_kwargs.update(bar_format=bar_format, unit=unit)
    else:
        bar_format = '{l_bar}{bar}| [{elapsed}<{remaining}]'
        tqdm_kwargs.update(bar_format=bar_format)
    return tqdm_kwargs


@singleton
class AsyncRunner:
    def __init__(self, **kwargs):
        """
        A singleton that manages the lifecycle of an :mod:`~python.asyncio` event loop and
        :mod:`aiohttp` client session.

        The :meth:`_close` method is executed on normal python exit.

        :param kwargs:
            Optional keywords arguments to :class:`~python.asyncio.runners.Runner`.
        """
        self._runner = Runner(**kwargs)
        self._executor = None
        self._session = None
        self._closed = False
        atexit.register(self._close)

    def __del__(self, _warnings: Any = warnings):
        # _warnings kwarg keeps the warnings module available until after AsyncRunner() is deleted
        if not self._closed:
            _warnings.warn(
                f'{type(self).__name__} was never closed: {self!r}.',
                category=ResourceWarning,
                stacklevel=2,
            )

    def _close(self):
        """Shutdown and close the client session and event loop."""

        async def close_session():
            logger.debug('Cancelling pending tasks...')
            tasks = asyncio.all_tasks(self.loop)
            tasks.remove(asyncio.current_task(self.loop))
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug('Closing session...')
            await self._session.close()
            # https://docs.aiohttp.org/en/latest/client_advanced.html#graceful-shutdown
            await asyncio.sleep(0.25)

        if self._closed:
            return
        if self._session:
            self.run(close_session())
        logger.debug('Closing runner...')
        self._runner.close()
        if self._executor:
            logger.debug('Shutting down executor...')
            self._executor.shutdown(cancel_futures=True)
        atexit.unregister(self._close)
        self._closed = True
        logger.debug('Close complete.')

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Embedded event loop."""
        return self._runner.get_loop()

    @property
    def session(self) -> aiohttp.ClientSession:
        """Embedded client session."""

        async def create_session():
            timeout = aiohttp.ClientTimeout(total=300, sock_connect=30, ceil_threshold=5)
            return aiohttp.ClientSession(raise_for_status=True, timeout=timeout)

        # a client session is bound to an event loop, so a persistent session requires a
        # persistent event loop
        self._session = self._session or self.run(create_session())
        return self._session

    def run(self, coro: Coroutine[Any, Any, T], **kwargs) -> T:
        """Run a coroutine in the embedded event loop.

        Uses a separate thread if an event loop is already running in the current thread (e.g. the
        main thread is a jupyter notebook).

        :param coro:
            Coroutine to run.
        :param kwargs:
            Optional keywords arguments to :meth:`~python.asyncio.runners.Runner.run`.

        :return:
            Coroutine result.
        """
        # Runner.run() cannot be called from a thread with an existing event loop, so test if
        # there is a loop running in this thread (see https://stackoverflow.com/a/75341431)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # run in a separate thread if there is an existing loop (e.g. we are in a jupyter
            # notebook)
            self._executor = self._executor or ThreadPoolExecutor(max_workers=1)
            return self._executor.submit(lambda: self._runner.run(coro, **kwargs)).result()
        else:
            # run in this thread if there is no existing loop
            return self._runner.run(coro, **kwargs)


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler that logs to ``tqdm.write()``.  Prevents logs interacting with any tqdm
    progress bars.

    Adapted from https://github.com/tqdm/tqdm/blob/master/tqdm/contrib/logging.py.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # noqa: E722
            self.handleError(record)


def auto_leave_tqdm(*args, leave: bool | None = None, **kwargs):
    """tqdm wrapper that sets leave attribute based on the bar's position."""
    bar = tqdm(*args, leave=leave, **kwargs)
    if leave is None:
        # work around leave is None logic not working in tqdm.notebook
        bar.leave = False if bar.pos > 0 else True
    return bar
