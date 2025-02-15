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
import itertools
import json
import logging
import os
import pathlib
import sys
import time
import warnings
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any, Callable, Generic, TypeVar

if sys.version_info >= (3, 11):
    from asyncio.runners import Runner
else:
    # TODO: remove when min supported python >= 3.11
    from geedim.runners import Runner

import aiohttp
import ee
import rasterio as rio
from rasterio.env import GDALVersion
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path.cwd()

_GDAL_AT_LEAST_35 = GDALVersion.runtime().at_least('3.5')
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
    # TODO: is the high vol endpoint still the right default value here?
    if not ee.data._credentials:
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


class Spinner(Thread):
    def __init__(self, label='', interval=0.2, leave=True, **kwargs):
        """
        Thread subclass to run a non-blocking spinner.

        :param label:
            Prepend spinner with this label.
        :param interval:
            Spinner update interval (s).
        :param leave:
            What to do with the spinner display on stop():
                ``False``: clear the label + spinner.
                ``True``:  leave the label + spinner as is.
                <string message>: print this message in place of the spinner
        :param kwargs:
            Additional kwargs to pass to ``Thread.__init__()``.
        """
        Thread.__init__(self, **kwargs)
        self._label = label
        self._interval = interval
        self._run = True
        self._leave = leave
        self._file = sys.stderr

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join()

    def run(self):
        """Run the spinner thread."""
        cursors_it = itertools.cycle(r'/-\|')

        while self._run:
            cursor = next(cursors_it)
            tqdm.write('\r' + self._label + cursor, file=self._file, end='')
            self._file.flush()
            time.sleep(self._interval)

        if self._leave is True:
            tqdm.write('', file=self._file, end='\n')
        elif self._leave is False:
            tqdm.write('\r', file=self._file, end='')
        elif isinstance(self._leave, str):
            tqdm.write('\r' + self._label + self._leave + ' ', file=self._file, end='\n')
        self._file.flush()

    def start(self):
        """Start the spinner thread."""
        self._run = True
        Thread.start(self)

    def stop(self):
        """Stop the spinner thread."""
        self._run = False


def rio_crs(crs: str | rio.CRS) -> str | rio.CRS:
    """Convert a GEE CRS string to a rasterio compatible CRS string."""
    if crs == 'SR-ORG:6974':
        # This is a workaround for https://issuetracker.google.com/issues/194561313,
        # that replaces the alleged GEE SR-ORG:6974 with actual WKT for SR-ORG:6842 taken from
        # https://github.com/OSGeo/spatialreference.org/blob/master/scripts/sr-org.json.
        crs = """PROJCS["Sinusoidal",
        GEOGCS["GCS_Undefined",
            DATUM["Undefined",
                SPHEROID["User_Defined_Spheroid",6371007.181,0.0]],
            PRIMEM["Greenwich",0.0],
            UNIT["Degree",0.0174532925199433]],
        PROJECTION["Sinusoidal"],
        PARAMETER["False_Easting",0.0],
        PARAMETER["False_Northing",0.0],
        PARAMETER["Central_Meridian",0.0],
        UNIT["Meter",1.0]]"""
    return crs


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
    tqdm_kwargs: dict[str, Any] = dict(dynamic_ncols=True, leave=True)
    tqdm_kwargs.update(**kwargs)
    if desc:
        # clip / pad the desc to max_width so that nested bars are aligned
        max_width = 40  # length of an S2 system:index
        desc_width = len(desc)
        desc = '...' + desc[-max_width + 3 :] if desc_width > max_width else desc.rjust(max_width)
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

        The :meth:`close` method is executed on normal python exit.  This class can also be used
        as a context manager.

        :param kwargs:
            Optional keywords arguments to :class:`~python.asyncio.runners.Runner`.
        """
        self._runner = Runner(**kwargs)
        self._executor = None
        self._session = None
        self._closed = False
        atexit.register(self.close)

    def __enter__(self) -> AsyncRunner:
        self._runner.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if not self._closed:
            warnings.warn(
                f'{type(self).__name__} was never closed: {self!r}.',
                category=ResourceWarning,
                stacklevel=2,
            )

    def close(self):
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

        # a client session is bound to an event loop, so a persistent session requires the
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
        # TODO:
        #  - test a sensible exception is raised if Runner is closed
        #  - test runner & loop are re-usable after async error
        #  - test session is re-usable after aiohttp error
        #  - what happens if the user has created their own loop on the main thread before
        #  AsyncRunner is called - will this use the executor?  also, think about if users can
        #  use this loop (either via AsyncRunner, or directly via asyncio) for their own async code

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
