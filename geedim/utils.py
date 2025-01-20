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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Thread
from typing import Any, Callable, Coroutine, Generic, Optional, Tuple, TypeVar

if sys.version_info >= (3, 11):
    from asyncio.runners import Runner
else:
    # TODO: remove when min supported python >= 3.11
    from geedim.runners import Runner

import aiohttp
import ee
import numpy as np
import rasterio as rio
import requests
from rasterio import warp
from rasterio.env import GDALVersion
from rasterio.windows import Window
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm

from geedim.enums import ResamplingMethod
from geedim.errors import GeedimError

logger = logging.getLogger(__name__)

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())

_GDAL_AT_LEAST_35 = GDALVersion.runtime().at_least("3.5")
T = TypeVar('T')


def Initialize(opt_url: Optional[str] = 'https://earthengine-highvolume.googleapis.com', **kwargs):
    """
    Initialise Earth Engine.

    Credentials will be read from the `EE_SERVICE_ACC_PRIVATE_KEY` environment variable if it exists
    (useful for integrating with e.g. GitHub actions).

    .. note::

        Earth Engine recommends using the `high volume endpoint` for applications like ``geedim``.
        See `the docs <https://developers.google.com/earth-engine/cloud/highvolume>`_ for more information.

    Parameters
    ----------
    opt_url: str
        The Earth Engine endpoint to use.  If ``None``, the default is used.
    kwargs
        Optional arguments to pass to `ee.Initialize`.
    """

    if not ee.data._credentials:
        # Adpated from https://gis.stackexchange.com/questions/380664/how-to-de-authenticate-from-earth-engine-api.
        env_key = 'EE_SERVICE_ACC_PRIVATE_KEY'

        if env_key in os.environ:
            # authenticate with service account
            key_dict = json.loads(os.environ[env_key])
            credentials = ee.ServiceAccountCredentials(
                key_dict['client_email'], key_data=key_dict['private_key']
            )
            ee.Initialize(credentials, opt_url=opt_url, **kwargs)
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
    Split Earth Engine image ID into collection and index components.

    Parameters
    ----------
    image_id: str
        Earth engine image ID.

    Returns
    -------
    tuple(str, str)
        A tuple of strings: (collection name, image index).
    """
    if not image_id:
        return None, None
    index = image_id.split("/")[-1]
    ee_coll_name = "/".join(image_id.split("/")[:-1])
    return ee_coll_name, index


@contextmanager
def suppress_rio_logs(level: int = logging.ERROR):
    """A context manager that sets the `rasterio` logging level, then returns it to its original value."""
    # TODO: this should not be necessary if logging level changes are limited to geedim.  if it has to be used,
    #  it should be made thread-safe.
    try:
        # GEE sets GeoTIFF `colorinterp` tags incorrectly. This suppresses `rasterio` warning relating to this:
        # 'Sum of Photometric type-related color channels and ExtraSamples doesn't match SamplesPerPixel'
        rio_level = logging.getLogger('rasterio').getEffectiveLevel()
        logging.getLogger('rasterio').setLevel(level)
        yield
    finally:
        logging.getLogger('rasterio').setLevel(rio_level)


def get_bounds(filename: pathlib.Path, expand: float = 5):
    """
    Get a geojson polygon representing the bounds of an image.

    Parameters
    ----------
    filename: str, pathlib.Path
        Path of the image file whose bounds to find.
    expand : int, optional
        Percentage (0-100) by which to expand the bounds (default: 5).

    Returns
    -------
    dict
        Geojson polygon.
    """
    with suppress_rio_logs(), rio.Env(GTIFF_FORCE_RGBA=False), rio.open(filename) as im:
        bbox = im.bounds
        if (im.crs.linear_units == "metre") and (expand > 0):  # expand the bounding box
            expand_x = (bbox.right - bbox.left) * expand / 100.0
            expand_y = (bbox.top - bbox.bottom) * expand / 100.0
            bbox_expand = rio.coords.BoundingBox(
                bbox.left - expand_x,
                bbox.bottom - expand_y,
                bbox.right + expand_x,
                bbox.top + expand_y,
            )
        else:
            bbox_expand = bbox

        coordinates = [
            [bbox_expand.right, bbox_expand.bottom],
            [bbox_expand.right, bbox_expand.top],
            [bbox_expand.left, bbox_expand.top],
            [bbox_expand.left, bbox_expand.bottom],
            [bbox_expand.right, bbox_expand.bottom],
        ]

        bbox_expand_dict = dict(type="Polygon", coordinates=[coordinates])
        src_bbox_wgs84 = warp.transform_geom(
            im.crs, "WGS84", bbox_expand_dict
        )  # convert to WGS84 geojson
    return src_bbox_wgs84


def get_projection(image: ee.Image, min_scale: bool = True) -> ee.Projection:
    """
    Get the min/max scale projection of image bands.  Server side - no calls to getInfo().
    Adapted from from https://github.com/gee-community/gee_tools, MIT license.

    Parameters
    ----------
    image : ee.Image
            Image whose min/max projection to retrieve.
    min_scale: bool, optional
         Retrieve the projection corresponding to the band with the minimum (True) or maximum (False) scale.
         (default: True)

    Returns
    -------
    ee.Projection
        Requested projection.
    """
    if not isinstance(image, ee.Image):
        raise TypeError('image is not an instance of ee.Image')

    bands = image.bandNames()
    scales = bands.map(lambda band: image.select(ee.String(band)).projection().nominalScale())
    projs = bands.map(lambda band: image.select(ee.String(band)).projection())
    projs = projs.sort(scales)

    return ee.Projection(projs.get(0) if min_scale else projs.get(-1))


class Spinner(Thread):
    def __init__(self, label='', interval=0.2, leave=True, **kwargs):
        """
        Thread sub-class to run a non-blocking spinner.

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
        cursors_it = itertools.cycle('/-\|')

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


def resample(ee_image: ee.Image, method: ResamplingMethod) -> ee.Image:
    """
    Resample an ee.Image. Extends ee.Image.resample by only resampling when the image has a fixed projection, and by
    providing an additional 'average' method for downsampling.  This logic is performed server-side.

    Note that for the :attr:`ResamplingMethod.average` ``method``, the returned image has a minimum scale default
    projection.

    See https://developers.google.com/earth-engine/guides/resample for more info.

    Parameters
    ----------
    ee_image: ee.Image
        Image to resample.
    method: ResamplingMethod
        Resampling method to use.

    Returns
    -------
    ee_image: ee.Image
        Resampled image.
    """
    # TODO : use STAC to only resample continuous qty type bands
    method = ResamplingMethod(method)
    if method == ResamplingMethod.near:
        return ee_image

    # resample the image, if it has a fixed projection
    proj = get_projection(ee_image, min_scale=True)
    has_fixed_proj = (
        proj.crs().compareTo('EPSG:4326').neq(0).Or(proj.nominalScale().toInt64().neq(111319))
    )

    def _resample(ee_image: ee.Image) -> ee.Image:
        """Resample the given image, allowing for additional 'average' method."""
        if method == ResamplingMethod.average:
            # set the default projection to the minimum scale projection (required for e.g. S2 images that have
            # non-fixed projection bands)
            ee_image = ee_image.setDefaultProjection(proj)
            return ee_image.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        else:
            return ee_image.resample(method.value)

    return ee.Image(ee.Algorithms.If(has_fixed_proj, _resample(ee_image), ee_image))


def retry_session(
    retries: int = 5,
    backoff_factor: float = 2.0,
    status_forcelist: Tuple = (429, 500, 502, 503, 504),
    session: requests.Session = None,
    **kwargs,
) -> requests.Session:
    """requests session configured for retries."""
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry, **kwargs)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def expand_window_to_grid(win: Window, expand_pixels: Tuple[int, int] = (0, 0)) -> Window:
    """
    Expand rasterio window extents to the nearest whole numbers i.e. for ``expand_pixels`` >= (0, 0), it will return a
    window that contains the original extents.

    Parameters
    ----------
    win: rasterio.windows.Window
        Window to expand.
    expand_pixels: tuple, optional
        Tuple specifying the number of (rows, columns) pixels to expand the window by.

    Returns
    -------
    rasterio.windows.Window
        Expanded window.
    """
    col_off, col_frac = np.divmod(win.col_off - expand_pixels[1], 1)
    row_off, row_frac = np.divmod(win.row_off - expand_pixels[0], 1)
    width = np.ceil(win.width + 2 * expand_pixels[1] + col_frac)
    height = np.ceil(win.height + 2 * expand_pixels[0] + row_frac)
    exp_win = Window(int(col_off), int(row_off), int(width), int(height))
    return exp_win


def rio_crs(crs: str | rio.CRS) -> str | rio.CRS:
    """Convert a GEE CRS string to a rasterio compatible CRS string."""
    if crs == 'SR-ORG:6974':
        # This is a workaround for https://issuetracker.google.com/issues/194561313, that replaces the alleged GEE
        # SR-ORG:6974 with actual WKT for SR-ORG:6842 taken from
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


def asset_id(image_id: str, folder: str = None):
    """
    Convert an EE image ID and EE asset project into an EE asset ID.

    If ``folder`` is not supplied, ``image_id`` is returned as is. Otherwise, ``image_id`` is
    converted to a name by changing forward slashes to dashes, ``folder`` is split into <root
    folder> and <sub folder> sections, and a string is returned with EE asset ID format:

        projects/<root folder>/assets/<sub folder(s)>/<image_name>.
    """
    if not folder:
        return image_id
    im_name = image_id.replace('/', '-')
    folder = pathlib.PurePosixPath(folder)
    cloud_folder = pathlib.PurePosixPath(folder.parts[0])
    asset_path = pathlib.PurePosixPath('/'.join(folder.parts[1:])).joinpath(im_name)
    return f'projects/{str(cloud_folder)}/assets/{str(asset_path)}'


def register_accessor(name: str, cls: type) -> Callable[[type[T]], type[T]]:
    """
    Decorator function to register a custom accessor on a given class.

    Adapted from https://github.com/pydata/xarray/blob/main/xarray/core/extensions.py, Apache-2.0
    License.

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
                raise GeedimError(f"Error initializing {self._name!r} accessor.") from ex

            cache[self._name] = accessor_obj
            return accessor_obj

    def decorator(accessor: type[T]) -> type[T]:
        if hasattr(cls, name):
            warnings.warn(
                f"Registration of accessor {accessor!r} under name '{name}' for type {cls!r} is"
                "overriding a preexisting attribute with the same name.",
                category=RuntimeWarning,
            )
        setattr(cls, name, CachedAccessor(name, accessor))
        return accessor

    return decorator


def get_tqdm_kwargs(desc: str = None, unit: str = None, **kwargs) -> dict[str, Any]:
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

        The :meth:`close` method is executed on normal python exit.  Can also be used as a
        context manager.

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
                f'{type(self).__name__} was never closed: {self!r}.', category=ResourceWarning
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
