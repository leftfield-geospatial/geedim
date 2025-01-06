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
import json
import logging
import operator
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Coroutine, Sequence, TypeVar

import ee
import numpy as np
import rasterio as rio
from rasterio import features
from rasterio.enums import Resampling as RioResampling
from rasterio.io import DatasetWriter
from tqdm.auto import tqdm

from geedim import utils
from geedim.enums import ExportType, ResamplingMethod
from geedim.stac import StacCatalog, StacItem
from geedim.tile import Tile, Tiler

try:
    import xarray
except ImportError:
    xarray = None

logger = logging.getLogger(__name__)

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
# - There are a some problems with *int64: While gdal >= 3.5 supports it, rasterio casts the
# nodata value to float64 which cannot represent the int64 range.  Also, EE provides int64
# ee.Image's (via ee.Image.getDownloadUrl() or ee.data.computePixels()) as float64 with nodata
# advertised as -inf but actually zero.  So no geedim *int64 support for now...
# - See also https://issuetracker.google.com/issues/350528377, although EE provides uint32 images
# correctly now.
# - The ordering of the keys above is relevant to the auto dtype and should be: unsigned ints
# smallest - largest, signed ints smallest to largest, float types smallest to largest.

T = TypeVar('T')


class BaseImageAccessor:
    _default_resampling = ResamplingMethod.near
    _default_export_type = ExportType.drive

    def __init__(self, ee_image: ee.Image):
        """
        Accessor for describing and exporting an image.

        Provides methods for tiled export to various formats, and client-side access to image
        properties.

        :param ee_image:
            Image to access.
        """
        self._ee_image = ee_image

    @cached_property
    def _min_projection(self) -> dict[str, Any]:
        """Projection information of the minimum scale band."""
        # TODO: some S2 images have 1x1 bands with meter scale of 1... e.g.
        #  'COPERNICUS/S2_SR_HARMONIZED/20170328T083601_20170328T084228_T35RNK'.  that will be an
        #  issue for BaseImageAccessor.projection() and its users too.
        proj_info = dict(crs=None, transform=None, shape=None, scale=None, id=None)
        bands = self.info.get('bands', []).copy()
        if len(bands) > 0:
            bands.sort(key=operator.itemgetter('scale'))
            band_info = bands[0]

            proj_info['id'] = band_info['id']
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

    @cached_property
    def stac(self) -> StacItem | None:
        """STAC information.  ``None`` if there is no STAC entry for this image."""
        return StacCatalog().get_item(self.id)

    @cached_property
    def id(self) -> str | None:
        """Earth Engine ID."""
        return self.info.get('id', None)

    @cached_property
    def index(self) -> str | None:
        """Earth Engine index."""
        return self.properties.get('system:index', None)

    @property
    def date(self) -> datetime | None:
        """Acquisition date & time.  ``None`` if the ``system:time_start`` property is not present."""
        if 'system:time_start' in self.properties:
            timestamp = self.properties['system:time_start']
            return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        else:
            return None

    @property
    def properties(self) -> dict[str, Any]:
        """Earth Engine image properties."""
        return self.info.get('properties', {})

    @property
    def crs(self) -> str | None:
        """CRS of the minimum scale band."""
        return self._min_projection['crs']

    @property
    def transform(self) -> list[float] | None:
        """Geo-referencing transform of the minimum scale band."""
        return self._min_projection['transform']

    @property
    def shape(self) -> tuple[int, int] | None:
        """(height, width) dimensions of the minimum scale band in pixels. ``None`` if the image
        has no fixed projection.
        """
        return self._min_projection['shape']

    @property
    def count(self) -> int:
        """Number of image bands."""
        return len(self.info.get('bands', []))

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
    def nodata(self) -> float | int | None:
        """Masked pixel value used by Earth Engine when exporting.

        For integer :attr:`dtype`, this is the minimum possible value, and for floating point
        :attr:`dtype`, it is ``float('-inf')``.
        """
        return _nodata_vals[self.dtype]

    @property
    def size(self) -> int | None:
        """Image export size (bytes).  ``None`` if the image has no fixed projection."""
        if not self.shape:
            return None
        dtype_size = np.dtype(self.dtype).itemsize
        return self.shape[0] * self.shape[1] * self.count * dtype_size

    @property
    def profile(self) -> dict[str, Any] | None:
        """Export image profile for Rasterio.  ``None`` if the image has no fixed projection."""
        if not self.shape:
            return None
        return dict(
            crs=utils.rio_crs(self.crs),
            transform=self.transform,
            width=self.shape[1],
            height=self.shape[0],
            count=self.count,
            dtype=self.dtype,
        )

    @property
    def scale(self) -> float | None:
        """Minimum scale of the image bands (meters)."""
        return self._min_projection['scale']

    @property
    def geometry(self) -> dict | None:
        """GeoJSON geometry of the image extent.  ``None`` if the image has no fixed projection."""
        if 'properties' not in self.info or 'system:footprint' not in self.info['properties']:
            return None
        footprint = self.info['properties']['system:footprint']
        return ee.Geometry(footprint).toGeoJSON()

    @property
    def bounded(self) -> bool:
        """Whether the image is bounded."""
        return self.geometry is not None and (
            features.bounds(self.geometry) != (-180, -90, 180, 90)
        )

    @property
    def bandNames(self) -> list[str]:
        """List of the image band names."""
        bands = self.info.get('bands', [])
        return [bd['id'] for bd in bands]

    @property
    def band_properties(self) -> list[dict]:
        """Merged STAC and Earth Engine band properties."""
        # TODO: possibly remove after we've decided how to deal with STAC
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

    @staticmethod
    def _asyncio_run(coro: Coroutine[Any, Any, T], executor: ThreadPoolExecutor, **kwargs) -> T:
        """Run a coroutine and return the result, using a separate thread if an event loop is
        already running.
        """
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
            res = executor.submit(lambda: asyncio.run(coro, **kwargs)).result()
        else:
            # run in this thread if there is no existing loop
            res = asyncio.run(coro, **kwargs)
        return res

    def _get_band_properties(self) -> list[dict]:
        """Merge Earth Engine and STAC band properties for this image."""
        if self.stac:
            stac_bands_props = self.stac.band_props
            band_props = [
                stac_bands_props[bid] if bid in stac_bands_props else dict(name=bid)
                for bid in self.bandNames
            ]
        else:  # just use the image band IDs
            band_props = [dict(name=bid) for bid in self.bandNames]
        return band_props

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
    def monitorExport(task: ee.batch.Task, label: str = None) -> None:
        """
        Monitor and display the progress of an export task.

        :param task:
            Earth Engine task to monitor (as returned by :meth:`export`).
        :param label:
            Optional label for progress display.  Defaults to the task description.
        """
        pause = 0.1
        status = ee.data.getOperation(task.name)
        label = label or status['metadata']['description']
        tqdm_kwargs = utils.get_tqdm_kwargs(desc=label)

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
                raise IOError(f'Export failed: {status}.')

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
        return ee.Number(self._ee_image.propertyNames().contains('system:footprint'))

    def resample(self, method: ResamplingMethod | str) -> ee.Image:
        """
        Resample the image.

        Extends ``ee.Image.resample()`` by providing an
        :attr:`~geedim.enums.ResamplingMethod.average` method for downsampling, and returning
        images without fixed projections (e.g. composites) unaltered.

        Composites can be resampled by resampling their component images.

        See https://developers.google.com/earth-engine/guides/resample for background information.

        :param method:
            Resampling method to use.  With the :attr:`~geedim.enums.ResamplingMethod.average`
            method, the image is reprojected to the minimum scale projection before resampling.

        :return:
            Resampled image if the source has a fixed projection, otherwise the source image.
        """
        method = ResamplingMethod(method)
        if method is ResamplingMethod.near:
            return self._ee_image

        # resample the image, if it has a fixed projection
        def _resample(ee_image: ee.Image) -> ee.Image:
            """Resample the given image, allowing for additional 'average' method."""
            if method is ResamplingMethod.average:
                # set the default projection to the minimum scale projection (required for e.g.
                # S2 images that have non-fixed projection bands)
                ee_image = ee_image.setDefaultProjection(self.projection())
                return ee_image.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
            else:
                return ee_image.resample(str(method.value))

        # TODO: consider deprecating support for composites of composites and asset images i.e.
        #  composite images don't get the ID of their component image collection. Then
        #  composites can be made of composite components, or asset image components,
        #  but not both together.  then we can do away with the If here and in addMaskBands(),
        #  and generally simplify the design (e.g. maskClouds() can be done without addMaskBands()
        #  ).
        return ee.Image(ee.Algorithms.If(self.fixed(), _resample(self._ee_image), self._ee_image))

    def toDType(self, dtype: str) -> ee.Image:
        """
        Convert the image data dtype.

        :param dtype:
            A recognised NumPy / Rasterio data type to convert to.

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
        Apply any STAC scales and offsets to the image (e.g. for converting digital numbers to
        physical units).

        :return:
            Scaled and offset image if STAC scales and offsets are available, otherwise the
            source image.
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

    def prepareForExport(
        self,
        crs: str = None,
        crs_transform: Sequence[float] = None,
        shape: tuple[int, int] = None,
        region: dict | ee.Geometry = None,
        scale: float = None,
        resampling: str | ResamplingMethod = _default_resampling,
        dtype: str = None,
        scale_offset: bool = False,
        bands: list[str | int] | str | ee.List = None,
    ) -> ee.Image:
        """
        Prepare the image for export.

        Bounds and resolution of the prepared image can be specified with ``region`` and
        ``scale`` / ``shape``, or ``crs_transform`` and ``shape``.  Bounds default to those of
        the source image when they are not specified (with either ``region``,
        or ``crs_transform`` & ``shape``).

        When ``crs``, ``scale``, ``crs_transform`` & ``shape`` are not provided, the projections
        of the prepared and source images will match.

        ..warning::
            Depending on the provided arguments, the prepared image may be a reprojected and
            clipped version of the source.  This type of image is `not recommended
            <https://developers.google.com/earth-engine/guides/best_practices>`__ for use in map
            display or further computation.

        :param crs:
            CRS of the prepared image as an EPSG or WKT string.  All bands are re-projected to
            this CRS.  Defaults to the CRS of the minimum scale band.
        :param crs_transform:
            Geo-referencing transform of the prepared image, as a sequence of 6 numbers.  In
            row-major order: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation].
            All bands are re-projected to this transform.
        :param shape:
            (height, width) dimensions of the prepared image in pixels.
        :param region:
            Region defining the prepared image bounds as a GeoJSON dictionary or ``ee.Geometry``.
            Defaults to the image geometry.
        :param scale:
            Pixel scale (m) of the prepared image.  All bands are re-projected to this scale.
            Ignored if ``crs`` and ``crs_transform`` are provided.  Defaults to the minimum
            scale of the image bands.
        :param resampling:
            Resampling method to use for reprojecting.  Ignored for images without fixed
            projections e.g. composites.  Composites can be resampled by resampling their
            component images.
        :param dtype:
            Data type of the prepared image (``uint8``, ``int8``, ``uint16``, ``int16``, ``uint32``,
            ``int32``, ``float32`` or ``float64``).  Defaults to the minimum size data type able
            to represent all image bands.
        :param scale_offset:
            Whether to apply any STAC band scales and offsets to the image (e.g. for converting
            digital numbers to physical units).
        :param bands:
            Bands to include in the prepared image as a list of names / indexes, or a regex
            string.  Defaults to all bands.

        :return:
            Prepared image.
        """
        # Create a new BaseImageAccessor if bands are provided.  This is done here so that crs,
        # scale etc parameters used below will have values specific to bands.
        exp_image = BaseImageAccessor(self._ee_image.select(bands)) if bands else self

        # Prevent exporting images with no fixed projection unless arguments defining the export
        # pixel grid and bounds are provided (EE allows this in some cases, but uses a 1 degree
        # scale in EPSG:4326 with global bounds, which is an unlikely use case prone to memory
        # limit errors).
        if (
            (not crs or not region or not (scale or shape))
            and (not crs or not crs_transform or not shape)
            and not exp_image.shape
        ):
            raise ValueError(
                "This image does not have a fixed projection, you need to provide 'crs', "
                "'region' & 'scale' / 'shape'; or 'crs', 'crs_transform' & 'shape'."
            )

        if scale and shape:
            raise ValueError("You can provide one of 'scale' or 'shape', but not both.")

        # configure the export spatial parameters
        if not crs_transform and not shape:
            # Only pass crs to ee.Image.prepare_for_export() when it is different from the
            # source.  Passing same crs as source does not maintain the source pixel grid.
            crs = crs if crs is not None and crs != exp_image.crs else None
            # Default scale to the minimum scale in meters
            scale = scale or exp_image.projection().nominalScale()
        else:
            # crs argument is required with crs_transform
            crs = crs or exp_image.projection().crs()

        # apply export scale/offset, dtype and resampling
        if scale_offset:
            ee_image = exp_image.scaleOffset()
            exp_dtype = dtype or 'float64'  # avoid another getInfo() for default dtype
        else:
            ee_image = exp_image._ee_image
            exp_dtype = dtype or exp_image.dtype

        ee_image = BaseImageAccessor(ee_image).resample(resampling)

        # convert dtype (required for EE to set nodata correctly on download even if dtype is
        # unchanged)
        ee_image = BaseImageAccessor(ee_image).toDType(dtype=exp_dtype)

        # apply export spatial parameters
        crs_transform = crs_transform[:6] if crs_transform else None
        dimensions = shape[::-1] if shape else None
        export_kwargs = dict(
            crs=crs, crs_transform=crs_transform, dimensions=dimensions, region=region, scale=scale
        )
        export_kwargs = {k: v for k, v in export_kwargs.items() if v is not None}
        ee_image, _ = ee_image.prepare_for_export(export_kwargs)

        return ee_image

    def export(
        self,
        filename: str,
        type: ExportType = _default_export_type,
        folder: str = None,
        wait: bool = True,
        **kwargs,
    ) -> ee.batch.Task:
        """
        Export the image to Google Drive, Earth Engine asset or Google Cloud Storage.

        :meth:`prepareForExport` can be called before this method to apply export parameters.

        :param filename:
            Destination file or asset name.  Also used to form the task name.
        :param type:
            Export type.
        :param folder:
            Google Drive folder (when ``type`` is :attr:`~geedim.enums.ExportType.drive`),
            Earth Engine asset project (when ``type`` is :attr:`~geedim.enums.ExportType.asset`),
            or Google Cloud Storage bucket (when ``type`` is
            :attr:`~geedim.enums.ExportType.cloud`). If ``type`` is
            :attr:`~geedim.enums.ExportType.asset` and ``folder`` is not provided, ``filename``
            should be a valid Earth Engine asset ID. If ``type`` is
            :attr:`~geedim.enums.ExportType.cloud` then ``folder`` is required.
        :param wait:
            Whether to wait for the export to complete before returning.
        :param kwargs:
            Additional arguments to the ``type`` dependent Earth Engine function:
            ``Export.image.toDrive``, ``Export.image.toAsset`` or ``Export.image.toCloudStorage``.

        :return:
            Export task, started if ``wait`` is False, or completed if ``wait`` is True.
        """
        # TODO: establish & document if folder/filename can be a path with sub-folders,
        #  or what the interaction between folder & filename is for the different export types.
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f"Uncompressed size: {tqdm.format_sizeof(self.size, suffix='B')}")

        # update defaults with any provided **kwargs
        export_kwargs = dict(
            description=filename.replace('/', '-')[:100],
            maxPixels=1e9,
            formatOptions=dict(cloudOptimized=True),
        )
        export_kwargs.update(**kwargs)

        # create export task and start
        type = ExportType(type)
        if type == ExportType.drive:
            # move folders in 'filename' to sub-folders in 'folder' ('filename' should be the
            # filename only)
            filepath = Path(folder, filename)
            folder, filename = '/'.join(filepath.parts[:-1]), filepath.parts[-1]
            task = ee.batch.Export.image.toDrive(
                image=self._ee_image,
                folder=folder,
                fileNamePrefix=filename,
                **export_kwargs,
            )

        elif type == ExportType.asset:
            # if folder is provided create an EE asset ID from it and filename, else treat
            # filename as a valid EE asset ID
            asset_id = utils.asset_id(filename, folder) if folder else filename
            export_kwargs.pop('formatOptions')  # not used for asset export
            task = ee.batch.Export.image.toAsset(
                image=self._ee_image, assetId=asset_id, **export_kwargs
            )

        else:
            if not folder:
                raise ValueError("'folder' is required for the 'cloud' export type.")
            # move sub-folders in 'folder' to parent folders in 'filename' ('bucket' arg should be
            # the bucket name only)
            filepath = Path(folder, filename)
            folder, filename = filepath.parts[0], '/'.join(filepath.parts[1:])
            task = ee.batch.Export.image.toCloudStorage(
                image=self._ee_image,
                bucket=folder,
                fileNamePrefix=filename,
                **export_kwargs,
            )
        task.start()

        if wait:
            # wait for completion
            BaseImageAccessor.monitorExport(task)
        return task

    def toGeoTIFF(
        self,
        filename: os.PathLike | str,
        overwrite: bool = False,
        nodata: bool | int | float = True,
        max_tile_size: float = Tiler._ee_max_tile_size,
        max_tile_dim: int = Tiler._ee_max_tile_dim,
        max_tile_bands: int = Tiler._ee_max_tile_bands,
        max_requests: int = Tiler._max_requests,
        max_cpus: int = None,
    ) -> None:
        """
        Export the image to a GeoTIFF file.

        Export projection and bounds are defined by :attr:`crs`, :attr:`transform` and
        :attr:`shape`, and data type by :attr:`dtype`. :meth:`prepareForExport` can be called
        before this method to apply other export parameters.

        The image is retrieved as separate tiles which are downloaded and decompressed
        concurrently.  Tile size can be controlled with ``max_tile_size``, ``max_tile_dim`` and
        ``max_tile_bands``, and download / decompress concurrency with ``max_requests`` and
        ``max_cpus``.

        The downloaded file is masked with the :attr:`dtype` dependent ``nodata`` value provided by
        Earth Engine.  For integer types, this is the minimum value of the :attr:`dtype`,
        and for floating point types, it is ``float('-inf')``.

        :param filename:
            Destination file path.
        :param overwrite:
            Whether to overwrite the destination file if it exists.
        :param nodata:
            Set the GeoTIFF nodata tag to :attr:`nodata` (``True``), or leave the nodata tag
            unset (``False``).  If a custom integer or floating point value is provided,
            the nodata tag is set to this value.  Usually, a custom value would be provided when
            the image has been unmasked with ``ee.Image.unmask(nodata)``.
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
        filename = Path(filename)
        if not overwrite and filename.exists():
            raise FileExistsError(f'{filename} exists')

        # create a rasterio profile for the destination file
        profile = self.profile
        if nodata is True:
            nodata = _nodata_vals[self.dtype]
        elif nodata is False:
            nodata = None
        profile.update(
            driver='GTiff',
            compress='deflate',
            interleave='band',
            tiled=True,
            bigtiff='if_safer',
            nodata=nodata,
        )

        with ExitStack() as exit_stack:
            # Open the destination file. GDAL_NUM_THREADS='ALL_CPUS' is needed here for
            # building overviews once the download is complete.  It is overridden in write_tile
            # to control download concurrency.
            exit_stack.enter_context(rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False))
            out_ds = exit_stack.enter_context(rio.open(filename, 'w', **profile))
            out_lock = threading.Lock()

            # download and write tiles to file
            def write_tile(tile: Tile, tile_array: np.ndarray):
                """Write a tile array to file."""
                with rio.Env(GDAL_NUM_THREADS=1), out_lock:
                    logger.debug(f'Writing {tile!r} to file.')
                    out_ds.write(tile_array, indexes=tile.indexes, window=tile.window)

            tiler = exit_stack.enter_context(
                Tiler(
                    self,
                    max_tile_size=max_tile_size,
                    max_tile_dim=max_tile_dim,
                    max_tile_bands=max_tile_bands,
                    max_requests=max_requests,
                    max_cpus=max_cpus,
                )
            )
            tiler.map_tiles(write_tile)

            # populate GeoTIFF metadata
            self._write_metadata(out_ds)
            # build overviews
            self._build_overviews(out_ds)

    def toNumPy(
        self,
        masked: bool = False,
        structured: bool = False,
        max_tile_size: float = Tiler._ee_max_tile_size,
        max_tile_dim: int = Tiler._ee_max_tile_dim,
        max_tile_bands: int = Tiler._ee_max_tile_bands,
        max_requests: int = Tiler._max_requests,
        max_cpus: int = None,
    ) -> np.ndarray:
        """
        Export the image to a NumPy array.

        Export projection and bounds are defined by :attr:`crs`, :attr:`transform` and
        :attr:`shape`, and data type by :attr:`dtype`. :meth:`prepareForExport` can be called
        before this method to apply other export parameters.

        The image is retrieved as separate tiles which are downloaded and decompressed
        concurrently.  Tile size can be controlled with ``max_tile_size``, ``max_tile_dim`` and
        ``max_tile_bands``, and download / decompress concurrency with ``max_requests`` and
        ``max_cpus``.

        :param masked:
            Return a :class:`~numpy.ndarray` with masked pixels set to the :attr:`nodata` value
            (``False``), or a :class:`~numpy.ma.MaskedArray` (``True``).
        :param structured:
            Return a 3D array with (row, column, band) dimensions and a numerical ``dtype``
            (``False``), or a 2D array with (row, column) dimensions and a structured ``dtype``
            representing the image bands (``True``).
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

        :returns:
            NumPy array.
        """
        # TODO: convert float nodata to nan?
        im_shape = (*self.shape, self.count)
        if masked:
            array = np.ma.zeros(im_shape, dtype=self.dtype, fill_value=self.nodata)
        else:
            array = np.zeros(im_shape, dtype=self.dtype)

        # download and write tiles to array
        def write_tile(tile: Tile, tile_array: np.ndarray):
            """Write a tile to array."""
            # move band dimension from first to last
            tile_array = np.moveaxis(tile_array, 0, -1)
            array[tile.slices.row, tile.slices.col, tile.slices.band] = tile_array

        with Tiler(
            self,
            max_tile_size=max_tile_size,
            max_tile_dim=max_tile_dim,
            max_tile_bands=max_tile_bands,
            max_requests=max_requests,
            max_cpus=max_cpus,
        ) as tiler:
            tiler.map_tiles(write_tile, masked=masked)

        if structured:
            # TODO: return crs & transform or otherwise include them with the array?
            bands = [bd['name'] for bd in self.band_properties]
            dtype = np.dtype(dict(names=bands, formats=[self.dtype] * len(bands)))
            array = array.view(dtype=dtype).squeeze()
            if isinstance(array, np.ma.MaskedArray):
                # re-set masked array fill_value which is not copied in view
                array.fill_value = self.nodata

        return array

    def toXarray(
        self,
        masked: bool = False,
        max_tile_size: float = Tiler._ee_max_tile_size,
        max_tile_dim: int = Tiler._ee_max_tile_dim,
        max_tile_bands: int = Tiler._ee_max_tile_bands,
        max_requests: int = Tiler._max_requests,
        max_cpus: int = None,
    ) -> xarray.DataArray:
        """
        Export the image to an Xarray DataArray.

        Export projection and bounds are defined by :attr:`crs`, :attr:`transform` and
        :attr:`shape`, and data type by :attr:`dtype`. :meth:`prepareForExport` can be called
        before this method to apply other export parameters.

        The image is retrieved as separate tiles which are downloaded and decompressed
        concurrently.  Tile size can be controlled with ``max_tile_size``, ``max_tile_dim`` and
        ``max_tile_bands``, and download / decompress concurrency with ``max_requests`` and
        ``max_cpus``.

        DataArray attributes include the export :attr:`crs`, :attr:`transform` and ``nodata``
        values for compatibility with `rioxarray <https://github.com/corteva/rioxarray>`_,
        as well as ``ee`` and ``stac`` JSON strings of the Earth Engine property and STAC
        dictionaries.

        :param masked:
            Set masked pixels in the returned array to the :attr:`nodata` value (``False``),
            or to NaN (``True``).  If ``True``, the image :attr:`dtype` is integer, and one or
            more pixels are masked, the returned array is converted to a minimal floating point
            type able to represent :attr:`dtype`.
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

        :returns:
            Image DataArray.
        """
        if not xarray:
            raise ImportError("'toXarray()' requires the 'xarray' package to be installed.")

        if not self.transform[1] == self.transform[3] == 0:
            raise ValueError(
                "'The image cannot be exported to Xarray as its 'transform' is not aligned with "
                "its CRS axes.  It should be reprojected first."
            )

        array = self.toNumPy(
            masked=masked,
            max_tile_size=max_tile_size,
            max_tile_dim=max_tile_dim,
            max_tile_bands=max_tile_bands,
            max_requests=max_requests,
            max_cpus=max_cpus,
        )

        # create coordinates
        y = np.arange(0.5, array.shape[0] + 0.5) * self.transform[4] + self.transform[5]
        x = np.arange(0.5, array.shape[1] + 0.5) * self.transform[0] + self.transform[2]
        band = [bd['name'] for bd in self.band_properties]
        coords = dict(y=y, x=x, band=band)

        # create attributes dict
        attrs = dict(
            id=self.id, date=self.date.isoformat(timespec='milliseconds') if self.date else None
        )
        # add rioxarray required attributes
        attrs.update(
            crs=self.crs,
            transform=self.transform,
            nodata=_nodata_vals[self.dtype] if not masked else float('nan'),
        )
        # add EE / STAC attributes (use json strings here, then drop all Nones for serialisation
        # compatibility e.g. netcdf)
        attrs['ee'] = json.dumps(self.properties) if self.properties else None
        attrs['stac'] = json.dumps(self.stac._item_dict) if self.stac else None
        attrs = {k: v for k, v in attrs.items() if v is not None}

        return xarray.DataArray(data=array, coords=coords, dims=['y', 'x', 'band'], attrs=attrs)


class BaseImage(BaseImageAccessor):
    # TODO: deprecate, along with MaskedImage etc
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
        return self.shape is not None

    @property
    def refl_bands(self) -> list[str] | None:
        """List of spectral / reflectance bands, if any."""
        return self.reflBands

    @staticmethod
    def monitor_export(task: ee.batch.Task, label: str = None) -> None:
        """
        Monitor and display the progress of an export task.

        :param task:
            Earth Engine task to monitor (as returned by :meth:`export`).
        :param label:
            Optional label for progress display.  Defaults to the task description.
        """
        BaseImageAccessor.monitorExport(task, label)

    def export(
        self,
        filename: str,
        type: ExportType = BaseImageAccessor._default_export_type,
        folder: str = None,
        wait: bool = True,
        **export_kwargs,
    ) -> ee.batch.Task:
        """
        Export the image to Google Drive, Earth Engine asset or Google Cloud Storage.

        :param filename:
            Destination file or asset name.  Also used to form the task name.
        :param type:
            Export type.
        :param folder:
            Google Drive folder (when ``type`` is :attr:`~geedim.enums.ExportType.drive`),
            Earth Engine asset project (when ``type`` is :attr:`~geedim.enums.ExportType.asset`),
            or Google Cloud Storage bucket (when ``type`` is
            :attr:`~geedim.enums.ExportType.cloud`). If ``type`` is
            :attr:`~geedim.enums.ExportType.asset` and ``folder`` is not provided, ``filename``
            should be a valid Earth Engine asset ID. If ``type`` is
            :attr:`~geedim.enums.ExportType.cloud` then ``folder`` is required.
        :param wait:
            Whether to wait for the export to complete before returning.
        :param export_kwargs:
            Arguments to :meth:`BaseImageAccessor.prepareForExport`.

        :return:
            Export task, started if ``wait`` is False, or completed if ``wait`` is True.
        """
        export_image = BaseImageAccessor(self.prepareForExport(**export_kwargs))
        return export_image.export(filename, type=type, folder=folder, wait=wait)

    def download(
        self,
        filename: os.PathLike | str,
        overwrite: bool = False,
        num_threads: int = None,
        max_tile_size: float = Tiler._ee_max_tile_size,
        max_tile_dim: int = Tiler._ee_max_tile_dim,
        max_tile_bands: int = Tiler._ee_max_tile_bands,
        max_requests: int = Tiler._max_requests,
        max_cpus: int = None,
        **export_kwargs,
    ) -> None:
        """
        Download the image to a GeoTIFF file.

        The image is retrieved as separate tiles which are downloaded and decompressed
        concurrently.  Tile size can be controlled with ``max_tile_size``, ``max_tile_dim`` and
        ``max_tile_bands``, and download / decompress concurrency with ``max_requests`` and
        ``max_cpus``.

        The GeoTIFF nodata tag is set to the :attr:`nodata` value.

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
            Maximum tile width / height (pixels).  Should be less than the `Earth Engine limit
            <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`__ (10000).
        :param max_tile_bands:
            Maximum number of tile bands.  Should be less than the Earth Engine limit (1024).
        :param max_requests:
            Maximum number of concurrent tile downloads.  Should be less than the `max concurrent
            requests quota <https://developers.google.com/earth-engine/guides/usage
            #adjustable_quota_limits>`__.
        :param max_cpus:
            Maximum number of tile GeoTIFFs to read concurrently.  Defaults to two less than the
            number of CPUs, or one, whichever is greater.  Values larger than the default can
            stall the asynchronous event loop and are not recommended.
        :param export_kwargs:
            Arguments to :meth:`BaseImageAccessor.prepareForExport`.
        """
        if num_threads:
            raise DeprecationWarning(
                "'num_threads' is deprecated and has no effect.  'max_requests' and 'max_cpus' "
                "can be used to limit concurrency."
            )

        export_image = BaseImageAccessor(self.prepareForExport(**export_kwargs))
        export_image.toGeoTIFF(
            filename,
            overwrite,
            max_tile_size=max_tile_size,
            max_tile_dim=max_tile_dim,
            max_tile_bands=max_tile_bands,
            max_requests=max_requests,
            max_cpus=max_cpus,
        )
