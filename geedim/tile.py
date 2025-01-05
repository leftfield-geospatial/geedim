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
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from io import BytesIO
from itertools import product
from typing import Any, Coroutine, Generator, Sequence, TypeVar

import aiohttp
import numpy as np
import rasterio as rio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geedim import download, utils

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass(frozen=True)
class Tile:
    """Description of a 3D image tile.

    Defines the tile band, row & column extents, and provides properties to assist accessing it
    in an Earth Engine image, Rasterio dataset or NumPy array.
    """

    # band, row & col extents of the tile in the source image (included in repr)
    band_start: int
    row_start: int
    col_start: int
    band_stop: int
    row_stop: int
    col_stop: int
    # source image geo-referencing transform (excluded from repr)
    image_transform: Sequence[float] = field(repr=False)

    @dataclass(frozen=True)
    class Slices:
        """3D slices that make dimensions explicit."""

        band: slice
        row: slice
        col: slice

    @cached_property
    def shape(self) -> tuple[int, int]:
        """Tile (height, width) dimensions in pixels."""
        return (
            self.row_stop - self.row_start,
            self.col_stop - self.col_start,
        )

    @cached_property
    def count(self) -> int:
        """Number of tile bands."""
        return self.band_stop - self.band_start

    @cached_property
    def indexes(self) -> range:
        """Tile bands as a range of source image band indexes in one-based / Rasterio convention."""
        return range(self.band_start + 1, self.band_stop + 1)

    @cached_property
    def window(self) -> Window:
        """Tile row & column region in the source image as a Rasterio window."""
        return Window(self.col_start, self.row_start, *self.shape[::-1])

    @cached_property
    def tile_transform(self) -> list[float]:
        """Tile geo-referencing transform."""
        transform = rio.Affine(*self.image_transform) * rio.Affine.translation(
            self.col_start, self.row_start
        )
        return transform[:6]

    @cached_property
    def slices(self) -> Slices:
        """Slices defining the 3D tile extent in a source image array."""
        return self.Slices(
            slice(self.band_start, self.band_stop),
            slice(self.row_start, self.row_stop),
            slice(self.col_start, self.col_stop),
        )


def _asyncio_run(coro: Coroutine[Any, Any, T], executor: ThreadPoolExecutor, **kwargs) -> T:
    """Run a coroutine and return the result, using a separate thread for the event loop if one is
    already running.
    """
    # asyncio.run() cannot be called from a thread with an existing event loop, so test if there
    # is a loop running in this thread (see https://stackoverflow.com/a/75341431)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # run in a separate thread if there is an existing loop (e.g. we are in a jupyter notebook)
        res = executor.submit(lambda: asyncio.run(coro, **kwargs)).result()
    else:
        # run in this thread if there is no existing loop
        res = asyncio.run(coro, **kwargs)
    return res


class Tiler:
    # TODO: if there's little speed cost, default max_tile_size to << 32 to avoid memory limit (
    #  could actually speed up downloads with few tiles by increasing concurrency)
    _ee_max_tile_size = 32
    _ee_max_tile_dim = 10000
    _ee_max_tile_bands = 1024
    _max_requests = 32
    _retry_exceptions = (
        asyncio.TimeoutError,
        aiohttp.ClientError,
        ConnectionError,
        RasterioIOError,
    )
    try:
        # retry on requests errors from ee.Image.getDownloadURL()
        from requests import RequestException

        _retry_exceptions += (RequestException,)
    except ImportError:
        pass

    def __init__(
        self,
        image: download.BaseImageAccessor,
        max_tile_size: float = _ee_max_tile_size,
        max_tile_dim: int = _ee_max_tile_dim,
        max_tile_bands: int = _ee_max_tile_bands,
        max_requests: int = _max_requests,
        max_cpus: int = None,
    ):
        """
        Image tiler.

        Splits an images into tiles and downloads / decompresses them concurrently.

        Tile size can be controlled with ``max_tile_size``, ``max_tile_dim`` and
        ``max_tile_bands``, and download / decompress concurrency with ``max_requests`` and
        ``max_cpus``.

        :param image:
            Image to be tiled.  Should have a fixed projection.
        :param max_tile_size:
            Maximum tile size (MB).  Should be less than the `Earth Engine size limit
            <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`__ (32 MB).
        :param max_tile_dim:
            Maximum tile width / height (pixels).  Should be less than the `Earth Engine limit
            <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`__ (10000).
        :param max_tile_bands:
            Maximum number of tile bands.  Should be less than the Earth Engine limit (1024).
        :param max_requests:
            Maximum number of concurrent tile downloads.  Should be less than the `max concurrent
            requests quota <https://developers.google.com/earth-engine/guides/usage
            #adjustable_quota_limits>`__.
        :param max_cpus:
            Maximum number of tiles to decompress concurrently.  Defaults to two less than the
            number of CPUs, or one, whichever is greater.  Values larger than the default can
            stall the asynchronous event loop and are not recommended.
        """
        self._validate_image(image)
        self._im = image
        self._tile_shape = self._get_tile_shape(
            max_tile_size=max_tile_size, max_tile_dim=max_tile_dim, max_tile_bands=max_tile_bands
        )

        # default max_cpus to two less than the number of CPUs (leaves capacity for one cpu to
        # run the map_tiles() function, and another to run the event loop)
        max_cpus = max_cpus or max((os.cpu_count() or 0) - 2, 1)
        self._limit_requests = asyncio.Semaphore(max_requests)
        self._limit_cpus = asyncio.Semaphore(max_cpus)

        # create thread pool with capacity for all synchronous tasks (+2 is for the map_tiles()
        # map function, and async event loop)
        self._executor = ThreadPoolExecutor(max_workers=max_requests + max_cpus + 2)
        # use one thread per tile for reading GeoTIFF buffers so that CPU loading can be
        # controlled with max_cpus
        self._env = rio.Env(GDAL_NUM_THREAHDS=1)

    @staticmethod
    def _validate_image(self, image: download.BaseImageAccessor):
        """Raise an error if the image does not have a fixed projection."""
        if not image.shape:
            raise ValueError(
                "This image cannot be exported as it does not have a fixed projection.  "
                "'prepareForExport()' can be called to define one."
            )
        if image.size > 10e9:
            size_str = tqdm.format_sizeof(image.size, suffix='B')
            logger.warning(
                f"Consider adjusting the image bounds, resolution and/or data type with "
                f"'prepareForExport()' to reduce the export size: {size_str}."
            )

    def __enter__(self):
        self._env.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._env.__exit__(exc_type, exc_val, exc_tb)

    def _get_tile_shape(
        self,
        max_tile_size: float = _ee_max_tile_size,
        max_tile_dim: int = _ee_max_tile_dim,
        max_tile_bands: int = _ee_max_tile_bands,
    ) -> Sequence[int]:
        """Return a 3D tile shape (bands, height, width) that satisfies the ``max_*`` parameters."""
        if max_tile_size > Tiler._ee_max_tile_size:
            raise ValueError(
                f"'max_tile_size' must be less than or equal to the Earth Engine limit of "
                f"{Tiler._ee_max_tile_size} MB."
            )
        max_tile_size = int(max_tile_size) << 20  # convert MB to bytes
        if max_tile_dim > Tiler._ee_max_tile_dim:
            raise ValueError(
                f"'max_tile_dim' must be less than or equal to the Earth Engine limit of "
                f"{Tiler._ee_max_tile_dim}."
            )
        if max_tile_bands > Tiler._ee_max_tile_bands:
            raise ValueError(
                f"'max_tile_bands' must be less than or equal to the Earth Engine limit of "
                f"{Tiler._ee_max_tile_bands}."
            )

        # initialise loop vars
        dtype_size = np.dtype(self._im.dtype).itemsize
        if self._im.dtype.endswith('int8'):
            # workaround for apparent GEE overestimate of *int8 dtype download sizes
            dtype_size *= 2
        im_shape = np.array((self._im.count, *self._im.shape))
        tile_shape = im_shape
        tile_size = np.prod(tile_shape) * dtype_size
        num_tiles = np.array([1, 1, 1], dtype=int)  # num tiles along each dimension

        # increment the number of tiles the image is split into along the longest dimension of
        # the tile, until the tile size satisfies max_tile_size (aims for the largest possible
        # cube-ish shaped tiles that satisfy max_tile_size)
        while tile_size >= max_tile_size:
            num_tiles[np.argmax(tile_shape)] += 1
            tile_shape = np.ceil(im_shape / num_tiles).astype(int)
            tile_size = np.prod(tile_shape) * dtype_size

        # clip to max_tile_bands / max_tile_dim
        tile_shape = tile_shape.clip(None, [max_tile_bands, max_tile_dim, max_tile_dim])
        tile_shape = tuple(tile_shape.tolist())
        return tile_shape

    def _tiles(self) -> Generator[Tile]:
        """Generate tiles covering the image."""
        im_shape = (self._im.count, *self._im.shape)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            num_tiles = int(np.prod(np.ceil(np.array(im_shape) / self._tile_shape)))
            dtype_size = np.dtype(self._im.dtype).itemsize
            raw_tile_size = np.prod(self._tile_shape) * dtype_size
            logger.debug(f'Image shape (bands, height, width): {im_shape}')
            logger.debug(f"Raw image size: {tqdm.format_sizeof(self._im.size, suffix='B')}")
            logger.debug(f'Tile shape (bands, rows, cols): {self._tile_shape}')
            logger.debug(f"Raw tile size: {tqdm.format_sizeof(raw_tile_size, suffix='B')}")
            logger.debug(f'Number of tiles: {num_tiles}')

        # split the image into tiles, clipping tiles to image shape
        for tile_start in product(
            range(0, im_shape[0], self._tile_shape[0]),
            range(0, im_shape[1], self._tile_shape[1]),
            range(0, im_shape[2], self._tile_shape[2]),
        ):
            tile_stop = np.clip(np.add(tile_start, self._tile_shape), a_min=None, a_max=im_shape)
            tile_stop = tile_stop.tolist()
            yield Tile(*tile_start, *tile_stop, image_transform=self._im.transform)

    async def _download_tile(
        self,
        tile: Tile,
        masked: bool,
        session: aiohttp.ClientSession,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
    ) -> tuple[Tile, np.ndarray]:
        """Download a tile to a NumPy array with retries."""

        def get_tile_url() -> str:
            """Return a download URL for the tile."""
            return self._im._ee_image.slice(tile.band_start, tile.band_stop).getDownloadURL(
                dict(
                    crs=self._im.crs,
                    crs_transform=tile.tile_transform,
                    dimensions=tile.shape[::-1],
                    format='GEO_TIFF',
                )
            )

        async def download_url(url: str) -> BytesIO:
            """Download the GeoTIFF at the given URL into a buffer."""
            buf = BytesIO()
            async with session.get(url, chunked=True, raise_for_status=False) as response:
                if not response.ok:
                    # get a more detailed error message if possible
                    try:
                        response.reason = (await response.json())['error']['message']
                    except:
                        pass
                    response.raise_for_status()

                async for data in response.content.iter_chunked(102400):
                    buf.write(data)
            return buf

        def read_gtiff_buf(buf: BytesIO) -> np.ndarray:
            """Read the given GeoTIFF buffer into a NumPy array."""
            buf.seek(0)
            with rio.open(buf, 'r') as ds:
                return ds.read(masked=masked)

        loop = asyncio.get_running_loop()
        for retry in range(0, max_retries + 1):
            try:
                # limit concurrent EE requests to avoid exceeding quota
                async with self._limit_requests:
                    logger.debug(f'Getting URL for {tile!r}.')
                    url = await loop.run_in_executor(self._executor, get_tile_url)
                    logger.debug(f'Downloading {tile!r} from {url}.')
                    buf = await download_url(url)

                # limit concurrent tile reads to leave CPU capacity for the event loop
                async with self._limit_cpus:
                    logger.debug(f'Reading GeoTIFF buffer for {tile!r}.')
                    array = await loop.run_in_executor(self._executor, read_gtiff_buf, buf)
                return tile, array

            except self._retry_exceptions as ex:
                # raise an error on maximum retries or 'user memory limit exceeded'
                if retry == max_retries or 'memory limit' in getattr(ex, 'message', ''):
                    logger.debug(f'Tile download failed for {tile!r}.  Error: {ex!r}.')
                    raise
                # otherwise retry
                logger.debug(f'Retry {retry + 1} of {max_retries} for {tile!r}. Error: {ex!r}.')
                await asyncio.sleep(backoff_factor * (2**retry))

    def map_tiles(self, func: Callable[[Tile, np.ndarray], None], masked: bool = False) -> None:
        """
        Map a function over downloaded tiles.

        :param func:
            Thread-safe function that is called with :class:`Tile` instance and tile array
            parameters, for each tile.  The tile array is passed in Rasterio (bands, rows,
            columns) dimension ordering.
        :param masked:
            Whether to pass the tile array as a :class:`~numpy.ma.MaskedArray` (``True``) or
            :class:`~numpy.ndarray` (``False``).  If  ``False``, masked pixels are set to the image
            :attr:`nodata` value.
        """
        with logging_redirect_tqdm([logging.getLogger(__package__)], tqdm_class=tqdm):
            # set up progress bar kwargs
            desc = self._im.index or self._im.id or 'Image'
            tqdm_kwargs = utils.get_tqdm_kwargs(desc=desc, unit='tiles')

            async def _map_tiles() -> None:
                """Download tiles and apply the map function asynchronously."""
                loop = asyncio.get_running_loop()
                # persisting the aiohttp.ClientSession outside the event loop is not recommended,
                # so it is recreated for each export
                timeout = aiohttp.ClientTimeout(total=300, sock_connect=30, ceil_threshold=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # begin tile downloads
                    tasks = {
                        asyncio.create_task(self._download_tile(tile, masked, session))
                        for tile in self._tiles()
                    }
                    # apply func to tiles in the order they are downloaded
                    try:
                        for task in tqdm(
                            asyncio.as_completed(tasks), total=len(tasks), **tqdm_kwargs
                        ):
                            tile, tile_array = await task
                            await loop.run_in_executor(self._executor, func, tile, tile_array)

                    finally:
                        # cancel and await any incomplete tasks (except the current one)
                        logger.debug('Cleaning up export tasks...')
                        tasks = asyncio.all_tasks(loop)
                        tasks.remove(asyncio.current_task(loop))
                        [task.cancel() for task in tasks]
                        await asyncio.gather(*tasks, return_exceptions=True)
                        logger.debug('Clean up complete.')

                # https://docs.aiohttp.org/en/latest/client_advanced.html#graceful-shutdown
                await asyncio.sleep(0.25)

            _asyncio_run(_map_tiles(), self._executor)
