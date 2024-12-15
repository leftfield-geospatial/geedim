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
import operator
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, asynccontextmanager
from datetime import datetime, timezone
from functools import cached_property
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import Any, Generator, Sequence

import aiohttp
import ee
import numpy as np
import rasterio as rio
from rasterio import features
from rasterio.enums import Resampling as RioResampling
from rasterio.io import DatasetWriter
from rasterio.shutil import RasterioIOError
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geedim import utils
from geedim.enums import ExportType, ResamplingMethod
from geedim.stac import StacCatalog, StacItem
from geedim.tile import Tile

logger = logging.getLogger(__name__)
# import urllib3
#
# urllib3.add_stderr_logger()

_nodata_vals = dict(
    uint8=0,
    uint16=0,
    uint32=0,
    int8=np.iinfo('int8').min,
    int16=np.iinfo('int16').min,
    int32=np.iinfo('int32').min,
    float32=float('-inf'),
    float64=float('-inf'),
)
"""Nodata values for supported download / export dtypes. """
# Note:
# - while gdal >= 3.5 supports *int64 data type, there is a rasterio bug that casts the *int64
# nodata value to float, so geedim will not support these types for now.
# - the ordering of the keys above is relevant to the auto dtype and should be: unsigned ints
# smallest - largest, signed ints smallest to largest, float types smallest to largest.


class BaseImageAccessor:
    _default_resampling = ResamplingMethod.near
    _ee_max_tile_size = 32
    _ee_max_tile_dim = 10000
    _default_export_type = ExportType.drive
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
    except:
        pass

    def __init__(self, ee_image: ee.Image):
        """
        Accessor for describing and downloading an image.

        Provides download and export without size limits, download related methods,
        and client-side access to image properties.

        :param ee_image:
            Image to access.
        """
        self._ee_image = ee_image

    @cached_property
    def _min_projection(self) -> dict[str, Any]:
        """Projection information of the minimum scale band."""
        # TODO: some S2 images have 1x1 bands with scale of 1... e.g.
        #  'COPERNICUS/S2_SR_HARMONIZED/20170328T083601_20170328T084228_T35RNK'.  that will be an
        #  issue for BaseImageAccessor.projection() and its users too.
        proj_info = dict(crs=None, transform=None, shape=None, scale=None)

        # create a list of bands with fixed projections
        fixed_bands = [
            band_info
            for band_info in self.info.get('bands', [])
            if abs(band_info['crs_transform'][0]) != 1 or band_info['crs'] != 'EPSG:4326'
        ]

        # extract projection info from min scale fixed projection band
        if len(fixed_bands) > 0:
            fixed_bands.sort(key=operator.itemgetter('scale'))
            band_info = fixed_bands[0]

            proj_info['crs'] = band_info['crs']
            proj_info['transform'] = rio.Affine(*band_info['crs_transform'])
            if ('origin' in band_info) and not any(np.isnan(band_info['origin'])):
                proj_info['transform'] *= rio.Affine.translation(*band_info['origin'])
            proj_info['transform'] = proj_info['transform'][:6]
            if 'dimensions' in band_info:
                proj_info['shape'] = tuple(band_info['dimensions'][::-1])
            proj_info['scale'] = band_info['scale']

        return proj_info

    @cached_property
    def id(self) -> str | None:
        """Earth Engine ID."""
        return self.info.get('id', None)

    @cached_property
    def index(self) -> str | None:
        """Earth Engine index."""
        return self.properties.get('system:index', None)

    @cached_property
    def stac(self) -> StacItem | None:
        """STAC information.  ``None`` if there is no STAC entry for this image."""
        return StacCatalog().get_item(self.id)

    @cached_property
    def info(self) -> dict[str, Any]:
        """Earth Engine information as returned by :meth:`ee.Image.getInfo`, with scales in
        meters added to band dictionaries.
        """

        def band_scale(band_name):
            """Return scale in meters for ``band_name``."""
            return self._ee_image.select(ee.String(band_name)).projection().nominalScale()

        # combine ee.Image.getInfo() and band scale .getInfo() calls into one
        scales = self._ee_image.bandNames().map(band_scale)
        scales, ee_info = ee.List([scales, self._ee_image]).getInfo()

        # zip scales into ee_info band dictionaries
        for scale, bdict in zip(scales, ee_info.get('bands', [])):
            bdict['scale'] = scale
        return ee_info

    @property
    def date(self) -> datetime | None:
        """Acquisition date & time.  ``None`` if the ``system:time_start`` property is not present."""
        if 'system:time_start' in self.properties:
            return datetime.fromtimestamp(
                self.properties['system:time_start'] / 1000, tz=timezone.utc
            )
        else:
            return None

    @property
    def crs(self) -> str | None:
        """CRS of the minimum scale band. ``None`` if the image has no fixed projection."""
        return self._min_projection['crs']

    @property
    def scale(self) -> float | None:
        """Minimum scale of the image bands (meters). ``None`` if the image has no fixed projection."""
        return self._min_projection['scale']

    @property
    def geometry(self) -> dict | None:
        """GeoJSON geometry of the image extent.  ``None`` if the image has no fixed projection."""
        if 'properties' not in self.info or 'system:footprint' not in self.info['properties']:
            return None
        footprint = self.info['properties']['system:footprint']
        return ee.Geometry(footprint).toGeoJSON()

    @property
    def shape(self) -> tuple[int, int] | None:
        """Pixel dimensions of the minimum scale band (row, column). ``None`` if the image has no
        fixed projection.
        """
        return self._min_projection['shape']

    @property
    def count(self) -> int:
        """Number of image bands."""
        return len(self.info.get('bands', []))

    @property
    def transform(self) -> list[float] | None:
        """Geotransform of the minimum scale band. ``None`` if the image has no fixed projection."""
        return self._min_projection['transform']

    @cached_property
    def dtype(self) -> str:
        """Minimum size data type able to represent all image bands."""

        def get_min_int_dtype(data_types: list[dict]) -> str | None:
            """Return the minimum dtype able to represent integer bands."""
            # find min & max values across all integer bands
            int_data_types = [dt for dt in data_types if dt['precision'] == 'int']
            min_int_val = min(int_data_types, key=operator.itemgetter('min'))['min']
            max_int_val = max(int_data_types, key=operator.itemgetter('max'))['max']

            # find min integer type that can represent the min/max range (relies on ordering of
            # _nodata_vals)
            for dtype in list(_nodata_vals.keys())[:-2]:
                iinfo = np.iinfo(dtype)
                if (min_int_val >= iinfo.min) and (max_int_val <= iinfo.max):
                    return dtype
            return 'float64'

        # TODO: from gdal = 3.5, there is support *int64 in geotiffs, rasterio 1.3.0 had some
        #  nodata issues with this, int64 support should be added when these issues are resolved
        dtype = None
        data_types = [band_info['data_type'] for band_info in self.info.get('bands', [])]
        precisions = [data_type['precision'] for data_type in data_types]
        if 'double' in precisions:
            dtype = 'float64'
        elif 'float' in precisions:
            dtype = str(
                np.promote_types('float32', get_min_int_dtype(data_types))
                if 'int' in precisions
                else 'float32'
            )
        elif 'int' in precisions:
            dtype = get_min_int_dtype(data_types)

        return dtype

    @property
    def size(self) -> int | None:
        """Image size (bytes).  ``None`` if the image has no fixed projection."""
        if not self.shape:
            return None
        dtype_size = np.dtype(self.dtype).itemsize
        return self.shape[0] * self.shape[1] * self.count * dtype_size

    @property
    def properties(self) -> dict[str, Any]:
        """Earth Engine image properties."""
        return self.info.get('properties', {})

    @property
    def band_properties(self) -> list[dict]:
        """Merged STAC and Earth Engine band properties."""
        return self._get_band_properties()

    @property
    def reflBands(self) -> list[str] | None:
        """List of spectral / reflectance band names.  ``None`` if there is no :attr:`stac`
        entry, or no spectral / reflectance bands.
        """
        if not self.stac:
            return None
        return [
            bname for bname, bdict in self.stac.band_props.items() if 'center_wavelength' in bdict
        ]

    @property
    def profile(self) -> dict[str, Any]:
        """Rasterio image profile."""
        # TODO: allow setting a custom nodata value with ee.Image.unmask() - see #21
        return dict(
            crs=utils.rio_crs(self.crs),
            transform=self.transform,
            width=self.shape[1],
            height=self.shape[0],
            count=self.count,
            dtype=self.dtype,
            nodata=_nodata_vals[self.dtype],
        )

    @property
    def bounded(self) -> bool:
        """Whether the image is bounded."""
        return self.geometry is not None and (
            features.bounds(self.geometry) != (-180, -90, 180, 90)
        )

    @staticmethod
    def _build_overviews(ds: DatasetWriter, max_num_levels: int = 8, min_level_pixels: int = 256):
        """Build internal overviews for an open dataset.  Each overview level is downsampled by a
        factor of 2.  The number of overview levels is determined by whichever of the
        ``max_num_levels`` or ``min_level_pixels`` limits is reached first.
        """
        max_ovw_levels = int(np.min(np.log2(ds.shape)))
        min_level_shape_pow2 = int(np.log2(min_level_pixels))
        num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
        ovw_levels = [2**m for m in range(1, num_ovw_levels + 1)]
        ds.build_overviews(ovw_levels, resampling=RioResampling.average)

    @staticmethod
    def _get_tqdm_kwargs(desc: str = None, unit: str = None) -> dict:
        """Return a dictionary of kwargs for a tqdm progress bar."""
        tqdm_kwargs = dict(dynamic_ncols=True)
        if desc:
            desc_width = 50
            desc = '...' + desc[-desc_width:] if len(desc) > desc_width else desc
            tqdm_kwargs.update(desc=desc)

        if unit:
            bar_format = (
                '{desc}: |{bar}| {n_fmt}/{total_fmt} ({unit}) [{percentage:3.0f}%] in {elapsed} '
                '(eta: {remaining})'
            )
            tqdm_kwargs.update(bar_format=bar_format, unit=unit)
        else:
            bar_format = '{desc}: |{bar}| [{percentage:3.0f}%] in {elapsed} (eta: {remaining})'
            tqdm_kwargs.update(bar_format=bar_format)
        return tqdm_kwargs

    def _get_band_properties(self) -> list[dict]:
        """Merge Earth Engine and STAC band properties for this image."""
        band_ids = [bd['id'] for bd in self.info.get('bands', [])]
        if self.stac:
            stac_bands_props = self.stac.band_props
            band_props = [
                stac_bands_props[bid] if bid in stac_bands_props else dict(name=bid)
                for bid in band_ids
            ]
        else:  # just use the image band IDs
            band_props = [dict(name=bid) for bid in band_ids]
        return band_props

    def _prepare_for_export(
        self,
        crs: str = None,
        crs_transform: tuple[float, ...] = None,
        shape: tuple[int, int] = None,
        region: dict | ee.Geometry = None,
        scale: float = None,
        resampling: ResamplingMethod = _default_resampling,
        dtype: str = None,
        scale_offset: bool = False,
        bands: list[str] = None,
    ) -> BaseImageAccessor:
        """Prepare an image with the given export parameters."""
        # Create a new BaseImageAccessor if bands are specified.  This is done here so that crs,
        # scale etc parameters used below will have values specific to bands.
        exp_image = BaseImageAccessor(self._ee_image.select(bands)) if bands else self

        # Prevent exporting images with no fixed projection unless arguments defining the export
        # pixel grid and bounds are provided (EE allows this, but uses a 1 degree scale in
        # EPSG:4326 with worldwide bounds, which is an unlikely use case prone to memory limit
        # errors).
        if (
            (not crs or not region or not (scale or shape))
            and (not crs or not crs_transform or not shape)
            and not exp_image.scale
        ):
            raise ValueError(
                "This image does not have a fixed projection, you need to specify a 'crs', "
                "'region' & 'scale' / 'shape'; or a 'crs', 'crs_transform' & 'shape'."
            )

        # Prevent exporting unbounded images without arguments defining the bounds (EE also
        # raises an error in ee.Image.prepare_for_export() but with a less informative message).
        if (not region and (not crs_transform or not shape)) and not exp_image.bounded:
            raise ValueError(
                "This image is unbounded, you need to specify a 'region'; or a 'crs_transform' and "
                "'shape'."
            )

        if scale and shape:
            raise ValueError("You can specify one of 'scale' or 'shape', but not both.")

        # configure the export spatial parameters
        if not crs_transform and not shape:
            # Only pass crs to ee.Image.prepare_for_export() when it is different from the
            # source.  Passing same crs as source does not maintain the source pixel grid.
            crs = crs if crs is not None and crs != exp_image.crs else None
            # Default scale to the scale in meters of the minimum scale band.
            scale = scale or exp_image.projection().nominalScale()
        else:
            # crs argument is required with crs_transform
            crs = crs or exp_image.projection().crs()

        # apply export scale/offset, dtype and resampling
        if scale_offset:
            ee_image = exp_image.scaleOffset()
            im_dtype = 'float64'
        else:
            ee_image = exp_image._ee_image
            im_dtype = exp_image.dtype

        resampling = ResamplingMethod(resampling)
        if resampling != ResamplingMethod.near:
            if not exp_image.scale:
                raise ValueError(
                    'This image has no fixed projection and cannot be resampled.  If this image '
                    'is a composite, you can resample the component images used to create it.'
                )
            ee_image = BaseImageAccessor(ee_image).resample(resampling)

        # convert dtype (required for EE to set nodata correctly on download even if dtype is
        # unchanged)
        ee_image = BaseImageAccessor(ee_image).toDType(dtype=dtype or im_dtype)

        # apply export spatial parameters
        crs_transform = crs_transform[:6] if crs_transform else None
        dimensions = shape[::-1] if shape else None
        export_kwargs = dict(
            crs=crs, crs_transform=crs_transform, dimensions=dimensions, region=region, scale=scale
        )
        export_kwargs = {k: v for k, v in export_kwargs.items() if v is not None}
        ee_image, _ = ee_image.prepare_for_export(export_kwargs)

        return BaseImageAccessor(ee_image)

    def _get_tile_shape(self, max_tile_size: float, max_tile_dim: int) -> tuple[int, int]:
        """Returns a tile shape that satisfies ``max_tile_size`` and ``max_tile_dim``."""
        if max_tile_size > BaseImageAccessor._ee_max_tile_size:
            raise ValueError(
                f"'max_tile_size' must be less than or equal to the Earth Engine limit of "
                f"{BaseImageAccessor._ee_max_tile_size} MB."
            )
        max_tile_size = int(max_tile_size) << 20  # convert MB to bytes
        if max_tile_dim > BaseImageAccessor._ee_max_tile_dim:
            raise ValueError(
                f"'max_tile_dim' must be less than or equal to the Earth Engine limit of "
                f"{BaseImageAccessor._ee_max_tile_dim} pixels."
            )

        # initialise loop vars
        pixel_size = np.dtype(self.dtype).itemsize * self.count
        if self.dtype.endswith('int8'):
            # workaround for apparent GEE overestimate of *int8 dtype download sizes
            pixel_size *= 2
        tile_size = np.prod(self.shape) * pixel_size
        num_tiles = np.array([1, 1], dtype=int)
        tile_shape = np.array(self.shape)

        # increment the number of tiles the image is split into along the longest dimension of
        # the tile, until the tile size satisfies max_tile_size (aims for square-ish tiles & to
        # minimise the number of tiles)
        while tile_size >= max_tile_size:
            num_tiles[np.argmax(tile_shape)] += 1
            tile_shape = np.ceil(np.array(self.shape) / num_tiles).astype(int)
            tile_size = tile_shape[0] * tile_shape[1] * pixel_size

        # clip to max_tile_dim
        tile_shape[tile_shape > max_tile_dim] = max_tile_dim
        tile_shape = tuple(tile_shape.tolist())
        return tile_shape

    def _tiles(
        self, max_tile_size: float = _ee_max_tile_size, max_tile_dim: int = _ee_max_tile_dim
    ) -> Generator[Tile]:
        """Image tile generator."""
        # TODO: allow slicing along band dimension
        # get the dimensions of a tile that stays under the EE limits
        tile_shape = self._get_tile_shape(max_tile_size, max_tile_dim)

        if logger.getEffectiveLevel() <= logging.DEBUG:
            num_tiles = int(np.prod(np.ceil(np.array(self.shape) / tile_shape)))
            dtype_size = np.dtype(self.dtype).itemsize
            raw_tile_size = tile_shape[0] * tile_shape[1] * self.count * dtype_size
            logger.debug(f"Raw image size: {tqdm.format_sizeof(self.size, suffix='B')}")
            logger.debug(f'Num. tiles: {num_tiles}')
            logger.debug(f'Tile shape: {tile_shape}')
            logger.debug(f"Raw tile size: {tqdm.format_sizeof(raw_tile_size, suffix='B')}")

        # split the image into tiles, clipping tiles to image dimensions
        for tile_start in product(
            range(0, self.shape[0], tile_shape[0]), range(0, self.shape[1], tile_shape[1])
        ):
            tile_stop = np.clip(np.add(tile_start, tile_shape), a_min=None, a_max=self.shape)
            clip_tile_shape = (tile_stop - tile_start).tolist()
            yield Tile(*tile_start[::-1], *clip_tile_shape[::-1], transform=self.transform)

    @asynccontextmanager
    async def _tile_tasks(
        self,
        max_tile_size: float = _ee_max_tile_size,
        max_tile_dim: int = _ee_max_tile_dim,
        max_requests: int = 32,
        max_cpus: int = None,
    ) -> Generator[set[asyncio.Task]]:
        """Context manager for asynchronous tile download tasks."""

        def get_tile_url(tile: Tile) -> str:
            """Return an URL for the given tile."""
            return self._ee_image.getDownloadURL(
                dict(
                    crs=self.crs,
                    crs_transform=tile.transform,
                    dimensions=tile.shape[::-1],
                    format='GEO_TIFF',
                )
            )

        async def download_url(url: str, session: aiohttp.ClientSession) -> BytesIO:
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
            """Read the given GeoTIFF buffer into a numpy array."""
            buf.seek(0)
            with rio.open(buf, 'r') as ds:
                # Note: due to an Earth Engine bug (
                # https://issuetracker.google.com/issues/350528377), the tile is float64 for
                # uint32 and int64 image dtypes.  The tile nodata value is as it should be
                # though, so conversion to the image / GeoTIFF dtype works ok.
                return ds.read()

        async def download_tile(
            tile: Tile,
            limit_requests: asyncio.Semaphore,
            limit_cpus: asyncio.Semaphore,
            session: aiohttp.ClientSession,
            exec: ThreadPoolExecutor = None,
            max_retries: int = 5,
            backoff_factor: float = 2.0,
        ) -> tuple[Tile, np.ndarray]:
            """Download a tile to a numpy array, with retries."""
            loop = asyncio.get_running_loop()
            for retry in range(0, max_retries + 1):
                try:
                    # limit concurrent EE requests to avoid exceeding quota
                    async with limit_requests:
                        logger.debug(f'Getting URL for {tile!r}.')
                        url = await loop.run_in_executor(exec, get_tile_url, tile)
                        logger.debug(f'Downloading {tile!r} from {url}.')
                        buf = await download_url(url, session)

                    # limit concurrent tile reads to leave CPU capacity for the event loop
                    async with limit_cpus:
                        logger.debug(f'Reading GeoTIFF buffer for {tile!r}.')
                        array = await loop.run_in_executor(exec, read_gtiff_buf, buf)
                    return tile, array

                except self._retry_exceptions as ex:
                    # raise an error on maximum retries or 'user memory limit exceeded'
                    if retry == max_retries or 'memory limit' in getattr(ex, 'message', ''):
                        logger.debug(f'Tile download failed for {tile!r}.  Error: {ex!r}.')
                        raise
                    # otherwise retry
                    logger.debug(f'Retry {retry + 1} of {max_retries} for {tile!r}. Error: {ex!r}.')
                    await asyncio.sleep(backoff_factor * (2**retry))

        with ExitStack() as exit_stack:
            # use one thread per tile for reading GeoTIFF buffers so that CPU loading can be
            # controlled with max_cpus
            exit_stack.enter_context(rio.Env(GDAL_NUM_THREADS=1))
            # default max_cpus to two less than the number of CPUs (leaves capacity for one
            # thread to write tiles, and another to run the event loop)
            max_cpus = max_cpus or max((os.cpu_count() or 0) - 2, 1)

            # create thread pool for all synchronous tasks
            exec = exit_stack.enter_context(ThreadPoolExecutor(max_workers=max_requests + max_cpus))

            # create semaphores for limiting concurrency
            limit_requests = asyncio.Semaphore(max_requests)
            limit_cpus = asyncio.Semaphore(max_cpus)

            # yield async tasks for downloading tiles
            timeout = aiohttp.ClientTimeout(total=300, sock_connect=30, ceil_threshold=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tiles = [*self._tiles(max_tile_size=max_tile_size, max_tile_dim=max_tile_dim)]
                tasks = {
                    asyncio.create_task(
                        download_tile(tile, limit_requests, limit_cpus, session, exec)
                    )
                    for tile in tiles
                }
                try:
                    yield tasks

                finally:
                    # cancel and await any incomplete tasks
                    logger.debug('Cleaning up tile download tasks...')
                    [task.cancel() for task in tasks]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    logger.debug('Clean up complete.')

            # https://docs.aiohttp.org/en/latest/client_advanced.html#graceful-shutdown
            await asyncio.sleep(0.25)

    def _write_metadata(self, ds: DatasetWriter):
        """Write Earth Engine and STAC metadata to an open rasterio dataset."""
        # TODO: Xee with rioxarray writes an html description - can we do the same?
        # replace 'system:*' property keys with 'system-*', and remove footprint if its there
        properties = {k.replace(':', '-'): v for k, v in self.properties.items()}
        if 'system-footprint' in properties:
            properties.pop('system-footprint')
        ds.update_tags(**self.properties)

        if self.stac and self.stac.license:
            ds.update_tags(LICENSE=self.stac.license)

        def clean_text(text, width=80) -> str:
            """Return a shortened tidied string."""
            if not isinstance(text, str):
                return text
            text = text.split('.')[0] if len(text) > width else text
            text = text.strip()
            text = '-\n' + text if len(text) > width else text
            return text

        # populate band metadata
        for band_i, band_dict in enumerate(self.band_properties):
            # TODO: check how gdal/qgis handles scale/offset metadata and set here if / if not
            #  applied
            clean_band_dict = {k.replace(':', '-'): clean_text(v) for k, v in band_dict.items()}
            if 'name' in band_dict:
                ds.set_band_description(band_i + 1, clean_band_dict['name'])
            ds.update_tags(band_i + 1, **clean_band_dict)

    @staticmethod
    def monitor_export(task: ee.batch.Task, label: str = None) -> None:
        """
        Monitor and display the progress of an export task.

        :param task:
            Earth Engine task to monitor (as returned by :meth:`export`).
        :param label:
            Optional label for progress display.  Defaults to the task description.
        """
        pause = 0.1
        status = ee.data.getOperation(task.name)
        label = label or status["metadata"]["description"]
        tqdm_kwargs = BaseImageAccessor._get_tqdm_kwargs(desc=label)

        # poll EE until the export preparation is complete
        with utils.Spinner(label=f"Preparing {tqdm_kwargs['desc']}: ", leave='done'):
            while not status.get('done', False):
                time.sleep(5 * pause)
                status = ee.data.getOperation(task.name)
                if 'progress' in status['metadata']:
                    break
                elif status['metadata']['state'] == 'FAILED':
                    raise IOError(f'Export failed: {status}')

        # wait for export to complete, displaying a progress bar
        tqdm_kwargs.update(desc='Exporting ' + tqdm_kwargs['desc'])
        with tqdm(total=1, **tqdm_kwargs) as bar:
            while not status.get('done', False):
                time.sleep(10 * pause)
                status = ee.data.getOperation(task.name)
                bar.update(status['metadata']['progress'] - bar.n)

            if status['metadata']['state'] == 'SUCCEEDED':
                bar.update(bar.total - bar.n)
            else:
                raise IOError(f'Export failed: {status}')

    def projection(self, min_scale: bool = True) -> ee.Projection:
        """
        Return the projection of the minimum / maximum scale band.

        :param min_scale:
            Whether to return the projection of the minimum (``True``), or maximum (``False``)
            scale band.

        :return:
            Projection.
        """
        bands = self._ee_image.bandNames()
        scales = bands.map(
            lambda band: self._ee_image.select(ee.String(band)).projection().nominalScale()
        )
        projs = bands.map(lambda band: self._ee_image.select(ee.String(band)).projection())
        projs = projs.sort(scales)
        return ee.Projection(projs.get(0) if min_scale else projs.get(-1))

    def fixed(self) -> ee.Number:
        """Whether the image has a fixed projection."""
        proj = self.projection()
        # cannot use ee.String.compareTo or ee.String.equals with proj.crs() when it is null
        not_wgs84 = ee.List([proj.crs()]).indexOf(ee.String('EPSG:4326')).eq(-1)
        not_degree_scale = proj.nominalScale().toInt64().neq(111319)
        return not_wgs84.Or(not_degree_scale)

    def resample(self, method: ResamplingMethod | str) -> ee.Image:
        """
        Return a resampled image.

        Extends ``ee.Image.resample()`` by providing an
        :attr:`~geedim.enums.ResamplingMethod.average` method for downsampling, and only
        resampling when the image has a fixed projection.

        See https://developers.google.com/earth-engine/guides/resample for background information.

        :param method:
            Resampling method to use.  For the :attr:`~geedim.enums.ResamplingMethod.average`
            method, the image is reprojected to the minimum scale projection before resampling.

        :return:
            Resampled image.
        """
        method = ResamplingMethod(method)
        if method == ResamplingMethod.near:
            return self._ee_image

        # resample the image, if it has a fixed projection
        def _resample(ee_image: ee.Image) -> ee.Image:
            """Resample the given image, allowing for additional 'average' method."""
            if method == ResamplingMethod.average:
                # set the default projection to the minimum scale projection (required for e.g.
                # S2 images that have non-fixed projection bands)
                # TODO: test this works for different res S2 bands
                ee_image = ee_image.setDefaultProjection(self.projection())
                return ee_image.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
            else:
                return ee_image.resample(str(method.value))

        # TODO: refactor without If.  Perhaps  always resample and leave EE to raise an error if
        #  the image doesn't have a fixed projection.
        return ee.Image(ee.Algorithms.If(self.fixed(), _resample(self._ee_image), self._ee_image))

    def toDType(self, dtype: str) -> ee.Image:
        """
        Return an image with the data type converted.

        :param dtype:
            A recognised numpy / Rasterio data type to convert to.

        :return:
            Converted image.
        """
        conv_dict = dict(
            float32=ee.Image.toFloat,
            float64=ee.Image.toDouble,
            uint8=ee.Image.toUint8,
            int8=ee.Image.toInt8,
            uint16=ee.Image.toUint16,
            int16=ee.Image.toInt16,
            uint32=ee.Image.toUint32,
            int32=ee.Image.toInt32,
            # int64=ee.Image.toInt64,
        )
        dtype = str(dtype)
        if dtype not in conv_dict:
            raise ValueError(f"Unsupported dtype: '{dtype}'")

        return conv_dict[dtype](self._ee_image)

    def scaleOffset(self) -> ee.Image:
        """
        Return an image with STAC band scales and offsets applied.

        :return:
            Scaled and offset image.  If no STAC scales and offsets are available, the image is
            returned as is.
        """
        if self.band_properties is None:
            # TODO: raise error?
            logger.warning('Cannot scale and offset this image, there is no STAC band information.')
            return self._ee_image

        # create band scale and offset dicts
        scale_dict = {bp['name']: bp.get('scale', 1.0) for bp in self.band_properties}
        offset_dict = {bp['name']: bp.get('offset', 0.0) for bp in self.band_properties}

        # return if all scales are 1 and all offsets are 0
        if set(scale_dict.values()) == {1} and set(offset_dict.values()) == {0}:
            return self._ee_image

        # apply the scales and offsets to bands which have them
        adj_bands = self._ee_image.bandNames().filter(
            ee.Filter.inList('item', list(scale_dict.keys()))
        )
        non_adj_bands = self._ee_image.bandNames().removeAll(adj_bands)

        scale_im = ee.Dictionary(scale_dict).toImage().select(adj_bands)
        offset_im = ee.Dictionary(offset_dict).toImage().select(adj_bands)
        adj_im = self._ee_image.select(adj_bands).multiply(scale_im).add(offset_im)

        # add any unadjusted bands back to the adjusted image, and re-order bands to match
        # the original
        adj_im = adj_im.addBands(self._ee_image.select(non_adj_bands))
        adj_im = adj_im.select(self._ee_image.bandNames())

        # copy source image properties and return
        return ee.Image(adj_im.copyProperties(self._ee_image, self._ee_image.propertyNames()))

    def maskCoverage(
        self,
        region: dict | ee.Geometry = None,
        scale: float | ee.Number = None,
        **kwargs,
    ) -> ee.Dictionary:
        """
        Find the percentage of a region covered by each band of this image.  The image is treated
        as a mask image e.g. as returned by ``ee.Image.mask()``.

        :param region:
            Region over which to find coverage as a GeoJSON dictionary or ``ee.Geometry``.
            Defaults to the image geometry.
        :param scale:
            Scale at which to find coverage.  Defaults to the minimum scale of the image bands.
        :param kwargs:
            Optional keyword arguments for ``ee.Image.reduceRegion()``.  Defaults to
            ``bestEffort=True``.

        :return:
            Dictionary with band name keys, and band cover percentage values.
        """
        region = region or self._ee_image.geometry()
        proj = self.projection(min_scale=True)
        scale = scale or proj.nominalScale()
        kwargs = kwargs if len(kwargs) > 0 else dict(bestEffort=True)

        # Find the coverage as the sum over the region of the image divided by the sum over
        # the region of a constant (==1) image.  Note that a mean reducer does not find the
        # mean over the region, but the mean over the part of the region covered by the image.
        sums_image = ee.Image([self._ee_image.unmask(), ee.Image(1).rename('ttl')])
        # use crs=proj rather than crs=proj.crs() as some projections have no CRS EPSG string
        sums = sums_image.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, crs=proj, scale=scale, **kwargs
        )
        ttl = ee.Number(sums.values().get(-1))
        sums = sums.remove([ee.String('ttl')])

        def get_percentage(key: ee.String, value: ee.Number) -> ee.Number:
            return ee.Number(value).divide(ttl).multiply(100)

        return sums.map(get_percentage)

    def export(
        self,
        filename: str,
        type: ExportType = _default_export_type,
        folder: str = None,
        wait: bool = True,
        crs: str | ee.String = None,
        crs_transform: Sequence[float] = None,
        shape: tuple[int, int] = None,
        region: dict | ee.Geometry = None,
        scale: float | ee.Number = None,
        resampling: ResamplingMethod = _default_resampling,
        dtype: str = None,
        scale_offset: bool = False,
        bands: Sequence[str] = None,
    ) -> ee.batch.Task:
        """
        Export the image to Google Drive, Earth Engine asset or Google Cloud Storage.

        Export bounds and resolution can be specified with ``region`` and ``scale`` / ``shape``,
        or ``crs_transform`` and ``shape``.  If no bounds are specified (with either ``region``,
        or ``crs_transform`` & ``shape``), the entire image is exported.

        When ``crs``, ``scale``, ``crs_transform`` & ``shape`` are not specified, the pixel grids
        of the exported and source images will match.

        :param filename:
            Destination file or asset name.  Also used to form the task name.
        :param type:
            Export type.
        :param folder:
            Google Drive folder (when ``type`` is :attr:`~geedim.enums.ExportType.drive`),
            Earth Engine asset project (when ``type`` is :attr:`~geedim.enums.ExportType.asset`),
            or Google Cloud Storage bucket (when ``type`` is
            :attr:`~geedim.enums.ExportType.cloud`). If ``type`` is
            :attr:`~geedim.enums.ExportType.asset` and ``folder`` is not specified, ``filename``
            should be a valid Earth Engine asset ID. If ``type`` is
            :attr:`~geedim.enums.ExportType.cloud` then ``folder`` is required.
        :param wait:
            Whether to wait for the export to complete before returning.
        :param crs:
            WKT or EPSG specification of CRS to export to.  Where image bands have different
            CRSs, all are re-projected to this CRS. Defaults to the CRS of the minimum scale band
            if available.
        :param crs_transform:
            Sequence of 6 numbers specifying an affine transform in the specified CRS.  In
            row-major order: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation].
            All bands are re-projected to this transform.
        :param shape:
            (height, width) dimensions to export (pixels).
        :param region:
            Region to export as a GeoJSON dictionary or ``ee.Geometry``.  Defaults to the image
            geometry, if available.
        :param scale:
            Pixel scale (m) to export to.  Where image bands have different scales, all are
            re-projected to this scale. Ignored if ``crs`` and ``crs_transform`` are specified.
            Defaults to the minimum scale of the image bands if available.
        :param resampling:
            Resampling method.
        :param dtype:
            Export to this data type (``uint8``, ``int8``, ``uint16``, ``int16``, ``uint32``,
            ``int32``, ``float32`` or ``float64``). Defaults to the minimum size data type able
            to represent all image bands.
        :param scale_offset:
            Whether to apply any STAC band scales and offsets to the image.
        :param bands:
            Sequence of band names to export.  Defaults to all bands.

        :return:
            Export task, started if ``wait`` is False, or completed if ``wait`` is True.
        """
        # TODO: allow additional kwargs to ee.batch.Export.image.* & avoid setting maxPixels?
        # TODO: establish & document if folder/filename can be a path with sub-folders,
        #  or what the interaction between folder & filename is for the different export types.
        exp_image = self._prepare_for_export(
            crs=crs,
            crs_transform=crs_transform,
            shape=shape,
            region=region,
            scale=scale,
            resampling=resampling,
            dtype=dtype,
            scale_offset=scale_offset,
            bands=bands,
        )
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f"Uncompressed size: {tqdm.format_sizeof(exp_image.size, suffix='B')}")

        # create export task and start
        exp_kwargs = dict(
            image=exp_image._ee_image, description=filename.replace('/', '-')[:100], maxPixels=1e9
        )
        type = ExportType(type)
        if type == ExportType.drive:
            task = ee.batch.Export.image.toDrive(
                folder=folder,
                fileNamePrefix=filename,
                formatOptions=dict(cloudOptimized=True),
                **exp_kwargs,
            )
        elif type == ExportType.asset:
            # if folder is specified create an EE asset ID from it and filename, else treat
            # filename as a valid EE asset ID
            asset_id = utils.asset_id(filename, folder) if folder else filename
            task = ee.batch.Export.image.toAsset(assetId=asset_id, **exp_kwargs)
        else:
            if not folder:
                raise ValueError("'folder' is required for the 'cloud' export type.")
            # move sub-folders in 'folder' to parent folders in 'filename' ('bucket' arg should be
            # the bucket name only)
            filepath = Path(folder, filename)
            folder, filename = filepath.parts[0], '/'.join(filepath.parts[1:])
            task = ee.batch.Export.image.toCloudStorage(
                bucket=folder,
                fileNamePrefix=filename,
                formatOptions=dict(cloudOptimized=True),
                **exp_kwargs,
            )

        task.start()
        if wait:  # wait for completion
            BaseImageAccessor.monitor_export(task)
        return task

    def download(
        self,
        filename: os.PathLike | str,
        overwrite: bool = False,
        num_threads: int = None,
        max_tile_size: float = _ee_max_tile_size,
        max_tile_dim: int = _ee_max_tile_dim,
        crs: str | ee.String = None,
        crs_transform: Sequence[float] = None,
        shape: tuple[int, int] = None,
        region: dict | ee.Geometry = None,
        scale: float | ee.Number = None,
        resampling: ResamplingMethod = _default_resampling,
        dtype: str = None,
        scale_offset: bool = False,
        bands: Sequence[str] = None,
        max_requests: int = 32,
        max_cpus: int = None,
    ) -> None:
        """
        Download the image to a GeoTIFF file.

        Download bounds and resolution can be specified with ``region`` and ``scale`` / ``shape``,
        or ``crs_transform`` and ``shape``.  If no bounds are specified (with either ``region``,
        or ``crs_transform`` & ``shape``), the entire image is downloaded.

        When ``crs``, ``scale``, ``crs_transform`` & ``shape`` are not specified, the pixel grids
        of the downloaded and source images will match.

        The image is retrieved as separate GeoTIFF tiles which are downloaded and read
        concurrently.  Tile size can be controlled with ``max_tile_size`` and ``max_tile_dim``,
        and download / read concurrency with ``max_requests`` and ``max_cpus``.

        :param filename:
            Destination file name.
        :param overwrite:
            Whether to overwrite the destination file if it exists.
        :param num_threads:
            Deprecated and has no effect. ``max_requests`` and ``max_cpus`` can be used to
            limit concurrency.
        :param max_tile_size:
            Maximum tile size (MB).  Should be less than the `Earth Engine size limit
            <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`__ (32 MB).
        :param max_tile_dim:
            Maximum tile width/height (pixels).  Should be less than the `Earth Engine limit
            <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`__ (10000).
        :param crs:
            WKT or EPSG specification of CRS to download to.  Where image bands have different
            CRSs, all are re-projected to this CRS. Defaults to the CRS of the minimum scale band
            if available.
        :param crs_transform:
            Sequence of 6 numbers specifying an affine transform in the specified CRS.  In
            row-major order: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation].
            All bands are re-projected to this transform.
        :param shape:
            (height, width) dimensions to download (pixels).
        :param region:
            Region to download as a GeoJSON dictionary or ``ee.Geometry``.  Defaults to the image
            geometry, if available.
        :param scale:
            Pixel scale (m) to download to.  Where image bands have different scales, all are
            re-projected to this scale. Ignored if ``crs`` and ``crs_transform`` are specified.
            Defaults to the minimum scale of the image bands if available.
        :param resampling:
            Resampling method.
        :param dtype:
            Download to this data type (``uint8``, ``int8``, ``uint16``, ``int16``, ``uint32``,
            ``int32``, ``float32`` or ``float64``). Defaults to the minimum size data type able
            to represent all image bands.
        :param scale_offset:
            Whether to apply any STAC band scales and offsets to the image.
        :param bands:
            Sequence of band names to download.  Defaults to all bands.
        :param max_requests:
            Maximum number of concurrent tile downloads.  Should be less than the `max concurrent
            requests quota <https://developers.google.com/earth-engine/guides/usage
            #adjustable_quota_limits>`__.
        :param max_cpus:
            Maximum number of tile GeoTIFFs to read concurrently.  Defaults to two less than the
            number of CPUs, or one, whichever is greater.  Values larger than the default can
            stall the asynchronous event loop and are not recommended.
        """
        # TODO: add a note about nodata to the docstring
        # TODO: allow bands to be band indexes too
        if num_threads:
            raise DeprecationWarning(
                "'num_threads' is deprecated, has no effect and will be removed in a future "
                "release.  'max_requests' and 'max_cpus' can be used to limit concurrency."
            )
        filename = Path(filename)
        if not overwrite and filename.exists():
            raise FileExistsError(f'{filename} exists')

        # apply export parameters
        exp_image = self._prepare_for_export(
            crs=crs,
            crs_transform=crs_transform,
            shape=shape,
            region=region,
            scale=scale,
            resampling=resampling,
            dtype=dtype,
            scale_offset=scale_offset,
            bands=bands,
        )

        # create a rasterio profile for the download GeoTIFF
        profile = exp_image.profile
        profile.update(
            driver='GTiff', compress='deflate', interleave='band', tiled=True, bigtiff='if_safer'
        )
        # warn if the download is large
        if exp_image.size > 1e9:
            size_str = tqdm.format_sizeof(exp_image.size, suffix='B')
            logger.warning(
                f"Consider adjusting 'region', 'scale' and/or 'dtype' to reduce the "
                f"'{filename.name}' download size (raw:{size_str})."
            )

        with ExitStack() as exit_stack:
            # set up progress bar kwargs
            exit_stack.enter_context(
                logging_redirect_tqdm([logging.getLogger(__package__)], tqdm_class=tqdm)
            )
            tqdm_kwargs = self._get_tqdm_kwargs(desc=filename.name, unit='tiles')

            # Create & open the destination file. GDAL_NUM_THREADS='ALL_CPUS' is needed here for
            # building overviews once the download is complete.  It is overridden in nested
            # functions to control concurrency of the download itself.
            out_lock = threading.Lock()
            exit_stack.enter_context(rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False))
            out_ds = exit_stack.enter_context(rio.open(filename, 'w', **profile))

            # create a thread pool for writing tile arrays to file, and possibly for running the
            # async event loop (see below)
            exec = exit_stack.enter_context(ThreadPoolExecutor(max_workers=2))

            async def download_tiles(image: BaseImageAccessor):
                """Download and write tiles to file."""

                def write_tile(array: np.ndarray, tile: Tile):
                    """Write a tile array to file."""
                    with rio.Env(GDAL_NUM_THREADS=1), out_lock:
                        logger.debug(f'Writing {tile!r} to file.')
                        out_ds.write(array, window=tile.window)

                async with image._tile_tasks(
                    max_tile_size=max_tile_size,
                    max_tile_dim=max_tile_dim,
                    max_requests=max_requests,
                    max_cpus=max_cpus,
                ) as tasks:
                    loop = asyncio.get_running_loop()
                    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), **tqdm_kwargs):
                        tile, array = await task
                        await loop.run_in_executor(exec, write_tile, array, tile)

            # asyncio.run() cannot be called from a thread with an existing event loop, so test
            # if there is a loop running in this (the main) thread (see
            # https://stackoverflow.com/a/75341431)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # run in a separate thread if there is an existing loop (e.g. we are in a jupyter
                # notebook)
                exec.submit(lambda: asyncio.run(download_tiles(exp_image))).result()
            else:
                # run in this thread if there is no existing loop
                asyncio.run(download_tiles(exp_image))

            # populate GeoTIFF metadata
            exp_image._write_metadata(out_ds)
            # build overviews
            exp_image._build_overviews(out_ds)


class BaseImage(BaseImageAccessor):
    @classmethod
    def from_id(cls, image_id: str) -> BaseImage:
        """
        Create a BaseImage instance from an Earth Engine image ID.

        :param image_id:
           ID of earth engine image to wrap.

        :return:
            BaseImage instance.
        """
        ee_image = ee.Image(image_id)
        gd_image = cls(ee_image)
        return gd_image

    @property
    def ee_image(self) -> ee.Image:
        """Encapsulated Earth Engine image."""
        return self._ee_image

    @ee_image.setter
    def ee_image(self, value: ee.Image):
        for attr in ['info', '_min_projection', 'stac', 'dtype']:
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        self._ee_image = value

    @property
    def name(self) -> str | None:
        """Image name (the :attr:`id` with slashes replaced by dashes)."""
        return self.id.replace('/', '-') if self.id else None

    @property
    def transform(self) -> rio.Affine | None:
        transform = super().transform
        return rio.Affine(*transform) if transform else None

    @property
    def footprint(self) -> dict | None:
        """GeoJSON geometry of the image extent.  ``None`` if the image has no fixed projection."""
        return self.geometry

    @property
    def has_fixed_projection(self) -> bool:
        """Whether the image has a fixed projection."""
        return self.scale is not None

    @property
    def refl_bands(self) -> list[str] | None:
        """List of spectral / reflectance bands, if any."""
        return self.reflBands
