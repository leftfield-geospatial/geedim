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

import json
import logging
import operator
import os
import threading
import time
import warnings
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from functools import cached_property
from math import sqrt
from pathlib import Path
from typing import Any, TypeVar

import ee
import fsspec
import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from fsspec.implementations.local import LocalFileSystem
from rasterio import features, warp
from rasterio.enums import Resampling as RioResampling
from rasterio.io import DatasetWriter
from tqdm.auto import tqdm

from geedim import mask, utils
from geedim.enums import Driver, ExportType, ResamplingMethod
from geedim.stac import STACClient
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
"""Nodata values for supported download / export dtypes."""
# Note:
# - There are a some problems with *int64: While gdal >= 3.5 supports it, rasterio casts the
# nodata value to float64 which cannot represent the int64 range.  Also, EE provides an int64
# ee.Image (via ee.Image.getDownloadUrl() or ee.data.computePixels()) as float64 with nodata
# advertised as -inf but actually zero.  So no geedim *int64 support for now...
# - See also https://issuetracker.google.com/issues/350528377, although EE provides uint32 images
# correctly now.
# - The ordering of the keys above is relevant to the auto dtype and should be: unsigned ints
# smallest - largest, signed ints smallest to largest, float types smallest to largest.

T = TypeVar('T')


@contextmanager
def _open_raster(ofile: OpenFile, **profile) -> DatasetWriter:
    """Open a local or remote Rasterio Dataset for writing."""
    if isinstance(ofile.fs, LocalFileSystem):
        # use the GDAL internal file system for local files
        with rio.open(ofile.path, 'w', **profile) as ds:
            yield ds
    else:
        # Otherwise, cache the raster in memory then write the cache to file with fsspec. Rather
        # than pass a file object to rio.open(), this manages MemoryFile and dataset contexts to
        # work around https://github.com/rasterio/rasterio/issues/3064.

        # open the destination file before yielding so we raise any errors early
        with rio.MemoryFile() as mem_file, ofile.open() as file_obj:
            with mem_file.open(**profile) as ds:
                yield ds
            logger.debug(f"Writing memory file to '{ofile.path}'.")
            file_obj.write(mem_file.read())


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


def _rio_crs(crs: str | rio.CRS) -> str | rio.CRS:
    """Convert a EE CRS string to a Rasterio compatible CRS string."""
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


def _scale_offset_image(ee_image: ee.Image, stac: dict[str, Any] | None) -> ee.Image:
    """Scale and offset an image using STAC information."""
    if stac is not None:
        # create band scale and offset dicts
        band_props = stac.get('summaries', {}).get('eo:bands', [])
        scale_dict = {bp['name']: bp.get('gee:scale', 1.0) for bp in band_props}
        offset_dict = {bp['name']: bp.get('gee:offset', 0.0) for bp in band_props}

    if not stac or (set(scale_dict.values()) == {1} and set(offset_dict.values()) == {0}):
        warnings.warn(
            'Cannot scale and offset this image, there is no STAC scale and offset information.',
            category=UserWarning,
            stacklevel=2,
        )
        return ee_image

    # apply the scales and offsets to bands which have them
    adj_bands = ee_image.bandNames().filter(ee.Filter.inList('item', list(scale_dict.keys())))
    non_adj_bands = ee_image.bandNames().removeAll(adj_bands)

    scale_im = ee.Dictionary(scale_dict).toImage().select(adj_bands)
    offset_im = ee.Dictionary(offset_dict).toImage().select(adj_bands)
    adj_im = ee_image.select(adj_bands).multiply(scale_im).add(offset_im)

    # add any unadjusted bands back to the adjusted image, and re-order bands to match
    # the original
    adj_im = adj_im.addBands(ee_image.select(non_adj_bands))
    adj_im = adj_im.select(ee_image.bandNames())

    # copy source image properties and return
    return ee.Image(adj_im.copyProperties(ee_image, ee_image.propertyNames()))


@utils.register_accessor('gd', ee.Image)
class ImageAccessor:
    _default_resampling = ResamplingMethod.near
    _default_export_type = ExportType.drive

    def __init__(self, ee_image: ee.Image):
        """
        Accessor for ``ee.Image``.

        Provides cloud/shadow masking, tiled export to various formats, and client-side access to
        image properties.

        :param ee_image:
            Image to access.
        """
        self._ee_image = ee_image

    @classmethod
    def _with_info(cls, ee_image: ee.Image, info: dict) -> ImageAccessor:
        """Create an accessor for ``ee_image`` with cached ``info``."""
        image = cls(ee_image)
        image.info = info
        return image

    @cached_property
    def _mi(self) -> type[mask._MaskedImage]:
        """Masking method container."""
        return mask._get_class_for_id(self.id)

    @cached_property
    def _min_projection(self) -> dict[str, Any]:
        """Projection information of the minimum scale band."""
        proj_info = dict(crs=None, transform=None, shape=None, scale=None, id=None)
        bands = self.info.get('bands', [])
        if len(bands) > 0:
            crss = [bi['crs'] for bi in bands]
            transforms = [
                (rio.Affine(*bi['crs_transform']) * rio.Affine.translation(*bi['origin']))[:6]
                if 'origin' in bi
                else bi['crs_transform']
                for bi in bands
            ]
            scales = [sqrt(abs(t[0] * t[4])) for t in transforms]

            # find minimum scale band
            if len(set(crss)) == 1:
                min_band = np.argmin(scales)
            else:
                # find band scales in CRS of first band so they can be compared
                scales_ = []
                for crs, tform, scale in zip(crss, transforms, scales, strict=True):
                    if crs != crss[0]:
                        xs, ys = warp.transform(
                            _rio_crs(crs),
                            _rio_crs(crss[0]),
                            (tform[2], tform[2] + tform[0]),
                            (tform[5], tform[5] + tform[4]),
                        )
                        scale_ = sqrt(abs((xs[1] - xs[0]) * (ys[1] - ys[0])))
                    else:
                        scale_ = scale
                    scales_.append(scale_)
                min_band = np.argmin(scales_)

            # set proj_info with the properties of the minimum scale band
            proj_info['id'] = bands[min_band]['id']
            proj_info['crs'] = crss[min_band]
            proj_info['transform'] = tuple(transforms[min_band])
            if 'dimensions' in bands[min_band]:
                proj_info['shape'] = tuple(bands[min_band]['dimensions'][::-1])
            proj_info['scale'] = scales[min_band]

        return proj_info

    @cached_property
    def info(self) -> dict[str, Any]:
        """Earth Engine information as returned by :meth:`ee.Image.getInfo`."""
        return self._ee_image.getInfo()

    @property
    def stac(self) -> dict[str, Any] | None:
        """STAC dictionary.  ``None`` if there is no STAC entry for this image."""
        return STACClient().get(self.id)

    @property
    def id(self) -> str | None:
        """Earth Engine ID."""
        return self.info.get('id', None)

    @property
    def index(self) -> str | None:
        """Earth Engine index."""
        return self.properties.get('system:index', None)

    @property
    def date(self) -> datetime | None:
        """Acquisition date & time.  ``None`` if the ``system:time_start`` property is not
        present.
        """
        if 'system:time_start' in self.properties:
            timestamp = self.properties['system:time_start']
            return datetime.fromtimestamp(timestamp / 1000, tz=UTC)
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
        """Georeferencing transform of the minimum scale band."""
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
    def dtype(self) -> str | None:
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
        return _nodata_vals[self.dtype] if self.dtype else None

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
            crs=_rio_crs(self.crs),
            transform=self.transform,
            width=self.shape[1],
            height=self.shape[0],
            count=self.count,
            dtype=self.dtype,
        )

    @property
    def scale(self) -> float | None:
        """Scale of the minimum scale band in units of its CRS."""
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
    def bandProps(self) -> list[dict[str, Any]]:
        """STAC band properties."""
        band_props = self.stac.get('summaries', {}).get('eo:bands', []) if self.stac else []
        band_props = {bp['name']: bp for bp in band_props}
        band_props = [band_props.get(bn, dict(name=bn)) for bn in self.bandNames]
        return band_props

    @property
    def specBands(self) -> list[str]:
        """List of spectral band names."""
        band_props = self.stac.get('summaries', {}).get('eo:bands', []) if self.stac else []
        band_props = {bp['name']: bp for bp in band_props}
        return [bn for bn in self.bandNames if 'center_wavelength' in band_props.get(bn, {})]

    @property
    def cloudShadowSupport(self) -> bool:
        """Whether this image has cloud/shadow support."""
        return issubclass(self._mi, mask._CloudlessImage)

    def _raise_cannot_export(self):
        """Raise an error if the image cannot be exported."""
        if not self.bandNames:
            raise ValueError("The image cannot be exported as it has no bands.")
        if not self.shape:
            raise ValueError(
                "The image cannot be exported as it doesn't have a fixed projection.  "
                "'prepareForExport()' can be called to define one."
            )

    def _write_metadata(self, ds: DatasetWriter):
        """Write Earth Engine and STAC metadata to an open rasterio dataset."""
        # populate the GEEDIM namespace with EE & STAC metadata JSON strings
        # populate dataset tags with Earth Engine properties and license
        properties = {k.replace(':', '-'): v for k, v in self.properties.items()}
        ds.update_tags(**properties)
        links = self.stac.get('links', []) if self.stac else []
        license_link = [lnk.get('href', None) for lnk in links if lnk.get('rel', '') == 'license']
        if license_link:
            ds.update_tags(LICENSE=license_link[0])

        def clean(value: Any) -> Any:
            """Strip and remove newlines from ``value`` if it is a string."""
            if isinstance(value, str):
                value = ' '.join(value.strip().splitlines())
            return value

        # populate band tags with STAC properties
        for bi, bp in enumerate(self.bandProps):
            clean_bp = {k.replace(':', '-'): clean(v) for k, v in bp.items()}
            ds.set_band_description(bi + 1, clean_bp.get('name', str(bi)))
            ds.update_tags(bi + 1, **clean_bp)

        # TODO: make writing scales/offsets an option and consistent with a similar xarray option
        #  for setting scale/offset/units in the encoding
        # populate band scales and offsets
        # if self.stac:
        #     ds.scales = [bp.get('gee:scale', 1.0) for bp in self.bandProps]
        #     ds.offsets = [bp.get('gee:offset', 0.0) for bp in self.bandProps]

    @staticmethod
    def monitorTask(task: ee.batch.Task, label: str | None = None) -> None:
        """
        Monitor and display the progress of a :meth:`toGoogleCloud` export task.

        :param task:
            Earth Engine task to monitor (as returned by :meth:`export`).
        :param label:
            Optional label for progress display.  Defaults to the task description.
        """
        status = ee.data.getOperation(task.name)
        label = label or status['metadata']['description']
        tqdm_kwargs = utils.get_tqdm_kwargs(desc=label)

        # poll EE until the export preparation is complete
        with utils.Spinner(desc=tqdm_kwargs['desc'] + ': ', leave=False):
            while not status.get('done', False) and 'progress' not in status['metadata']:
                time.sleep(0.5)
                status = ee.data.getOperation(task.name)

        # wait for export to complete, displaying a progress bar
        with tqdm(total=1, **tqdm_kwargs) as bar:
            while not status.get('done', False):
                bar.update(status['metadata']['progress'] - bar.n)
                time.sleep(1)
                status = ee.data.getOperation(task.name)

            if status['metadata']['state'] == 'SUCCEEDED':
                bar.update(bar.total - bar.n)
            else:
                msg = status.get('error', {}).get('message', status)
                raise OSError(f'Export failed: {msg}')

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
        return _scale_offset_image(self._ee_image, self.stac)

    def regionCoverage(
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
        kwargs = kwargs or dict(bestEffort=True)

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

    def addMaskBands(self, **kwargs) -> ee.Image:
        """
        Return this image with cloud/shadow masks and related bands added when supported,
        otherwise with fill (validity) mask added.

        Existing mask bands are overwritten except if this image has no fixed projection,
        in which case no bands are added or overwritten.

        :param bool mask_cirrus:
            Whether to mask cirrus clouds.  Valid for Landsat 8-9 images, and for Sentinel-2
            images with the :attr:`~geedim.enums.CloudMaskMethod.qa` ``mask_method``.  Defaults
            to ``True``.
        :param bool mask_shadows:
            Whether to mask cloud shadows.  Valid for Landsat images, and for Sentinel-2 images
            with the :attr:`~geedim.enums.CloudMaskMethod.qa` or
            :attr:`~geedim.enums.CloudMaskMethod.cloud_prob` ``mask_method``.  Defaults to ``True``.
        :param ~geedim.enums.CloudMaskMethod mask_method:
            Method used to mask clouds.  Valid for Sentinel-2 images.  See
            :class:`~geedim.enums.CloudMaskMethod` for details.  Defaults to
            :attr:`~geedim.enums.CloudMaskMethod.cloud_score`.
        :param float prob:
            Cloud probability threshold (%). Valid for Sentinel-2 images with the
            :attr:`~geedim.enums.CloudMaskMethod.cloud_prob` ``mask_method``.  Defaults to ``60``.
        :param float dark:
            NIR threshold [0-1]. NIR values below this threshold are potential cloud shadows.
            Valid for Sentinel-2 images with the :attr:`~geedim.enums.CloudMaskMethod.qa` or
            :attr:`~geedim.enums.CloudMaskMethod.cloud_prob` ``mask_method``.  Defaults to ``0.15``.
        :param int shadow_dist:
            Maximum distance (m) to look for cloud shadows from cloud edges.  Valid for Sentinel-2
            images with the :attr:`~geedim.enums.CloudMaskMethod.qa` or
            :attr:`~geedim.enums.CloudMaskMethod.cloud_prob` ``mask_method``.  Defaults to ``1000``.
        :param int buffer:
            Distance (m) to dilate cloud/shadow.  Valid for Sentinel-2 images with the
            :attr:`~geedim.enums.CloudMaskMethod.qa` or
            :attr:`~geedim.enums.CloudMaskMethod.cloud_prob` ``mask_method``.  Defaults to ``50``.
        :param float cdi_thresh:
            Cloud Displacement Index threshold. Values below this threshold are considered
            potential clouds. If this parameter is ``None`` (the default), the index is not used.
            Valid for Sentinel-2 images with the :attr:`~geedim.enums.CloudMaskMethod.qa` or
            :attr:`~geedim.enums.CloudMaskMethod.cloud_prob` ``mask_method``.  See
            https://developers.google.com/earth-engine/apidocs/ee-algorithms-sentinel2-cdi for
            details.
        :param int max_cloud_dist:
            Maximum distance (m) to look for clouds when forming the 'cloud distance' band.  Valid
            for Sentinel-2 images.  Defaults to ``5000``.
        :param float score:
            Cloud Score+ threshold.  Valid for Sentinel-2 images with the
            :attr:`~geedim.enums.CloudMaskMethod.cloud_score` ``mask_method``.  Defaults to ``0.6``.
        :param ~geedim.enums.CloudScoreBand cs_band:
            Cloud Score+ band to threshold.  Valid for Sentinel-2 images with the
            :attr:`~geedim.enums.CloudMaskMethod.cloud_score` ``mask_method``.  Defaults to
            :attr:`~geedim.enums.CloudScoreBand.cs`.

        :return:
            Image with added mask bands.
        """
        return self._mi.add_mask_bands(self._ee_image, **kwargs)

    def maskClouds(self) -> ee.Image:
        """
        Return this image with cloud/shadow masks applied when supported, otherwise return this
        image unaltered.

        Mask bands should be added with :meth:`addMaskBands` before calling this method.

        :return:
            Masked image.
        """
        return self._mi.mask_clouds(self._ee_image)

    def prepareForExport(
        self,
        crs: str | None = None,
        crs_transform: Sequence[float] | None = None,
        shape: tuple[int, int] | None = None,
        region: dict[str, Any] | ee.Geometry | None = None,
        scale: float | None = None,
        resampling: str | ResamplingMethod = _default_resampling,
        dtype: str | None = None,
        scale_offset: bool = False,
        bands: list[str | int] | str | ee.List = None,
    ) -> ee.Image:
        """
        Prepare the image for export.

        ..warning::
            Depending on the supplied arguments, the prepared image may be a reprojected and
            clipped version of the source.  This type of image is `not recommended
            <https://developers.google.com/earth-engine/guides/best_practices>`__ for use in map
            display or further computation.

        :param crs:
            CRS of the prepared image as a well-known authority (e.g. EPSG) or WKT string.
            Defaults to the CRS of the minimum scale band.
        :param crs_transform:
            Georeferencing transform of the prepared image, as a sequence of 6 numbers.  In
            row-major order: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation].
        :param shape:
            (height, width) dimensions of the prepared image in pixels.
        :param region:
            Region defining the prepared image bounds as a GeoJSON dictionary or ``ee.Geometry``.
            Defaults to the image geometry.  Ignored if ``crs_transform`` is supplied.
        :param scale:
            Pixel scale (m) of the prepared image.  Defaults to the minimum scale of the image
            bands.  Ignored if ``crs_transform`` is supplied.
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
        # Create a new ImageAccessor if bands are supplied.  This is done here so that crs,
        # scale etc parameters used below will have values specific to bands.
        exp_image = ImageAccessor(self._ee_image.select(bands)) if bands else self

        # Prevent exporting images with no fixed projection unless arguments defining the export
        # pixel grid and bounds are supplied (EE allows this in some cases, but uses a 1 degree
        # scale in EPSG:4326 with global bounds, which is an unlikely use case prone to memory
        # limit errors).
        if (
            (not crs or not region or not (scale or shape))
            and (not crs or not crs_transform or not shape)
            and not exp_image.shape
        ):
            raise ValueError(
                "The image does not have a fixed projection, you need to provide 'crs', "
                "'region' & 'scale' / 'shape'; or 'crs', 'crs_transform' & 'shape'."
            )

        if scale and shape:
            # Earth Engine raises an exception in this situation, but its message is obscure
            raise ValueError("You can provide one of 'scale' or 'shape', but not both.")

        # set a default scale and try to maintain pixel grid if no scaling params supplied
        if not crs_transform and not shape and not scale:
            # default to the minimum scale
            scale = exp_image.projection().nominalScale()
            # Only pass crs to ee.Image.prepare_for_export() when it is different from the
            # source.  Passing same crs as source does not maintain the source pixel grid.
            crs = crs if crs is not None and crs != exp_image.crs else None

        # set a default crs argument when crs_transform is provided (crs is required with
        # crs_transform)
        if crs_transform and not crs:
            crs = exp_image.projection().crs()

        # apply export scale/offset, dtype and resampling
        if scale_offset:
            ee_image = exp_image.scaleOffset()
            exp_dtype = dtype or 'float64'  # avoid another getInfo() for default dtype
        else:
            ee_image = exp_image._ee_image
            exp_dtype = dtype or exp_image.dtype

        ee_image = ImageAccessor(ee_image).resample(resampling)

        # convert dtype (required for EE to set nodata correctly on download even if dtype is
        # unchanged)
        ee_image = ImageAccessor(ee_image).toDType(dtype=exp_dtype)

        # apply export spatial parameters
        crs_transform = crs_transform[:6] if crs_transform else None
        dimensions = shape[::-1] if shape else None
        export_kwargs = dict(
            crs=crs, crs_transform=crs_transform, dimensions=dimensions, region=region, scale=scale
        )
        export_kwargs = {k: v for k, v in export_kwargs.items() if v is not None}
        ee_image, _ = ee_image.prepare_for_export(export_kwargs)

        return ee_image

    def toGoogleCloud(
        self,
        filename: str,
        type: ExportType = _default_export_type,
        folder: str | None = None,
        wait: bool = True,
        **kwargs,
    ) -> ee.batch.Task:
        """
        Export the image to Google Drive, Earth Engine asset or Google Cloud Storage using the
        Earth Engine batch environment.

        :meth:`prepareForExport` can be called before this method to apply export parameters.

        :param filename:
            Destination file or asset name (excluding extension).  Also used to form the task name.
        :param type:
            Export type.
        :param folder:
            Google Drive folder (when ``type`` is :attr:`~geedim.enums.ExportType.drive`),
            Earth Engine asset project (when ``type`` is :attr:`~geedim.enums.ExportType.asset`),
            or Google Cloud Storage bucket (when ``type`` is
            :attr:`~geedim.enums.ExportType.cloud`).  Can include sub-folders.  If ``type`` is
            :attr:`~geedim.enums.ExportType.asset` and ``folder`` is not supplied, ``filename``
            should be a valid Earth Engine asset ID. If ``type`` is
            :attr:`~geedim.enums.ExportType.cloud` then ``folder`` is required.
        :param wait:
            Whether to wait for the export to complete before returning.
        :param kwargs:
            Additional arguments to the ``type`` dependent Earth Engine function:
            ``Export.image.toDrive``, ``Export.image.toAsset`` or ``Export.image.toCloudStorage``.

        :return:
            Export task, started if ``wait`` is ``False``, or completed if ``wait`` is ``True``.
        """
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f"Uncompressed size: {tqdm.format_sizeof(self.size, suffix='B')}")

        # update defaults with any supplied **kwargs
        exp_kwargs = dict(
            description=filename.replace('/', '-')[:100],
            maxPixels=1e9,
            formatOptions=dict(cloudOptimized=True),
        )
        exp_kwargs.update(**kwargs)

        # create export task and start
        type = ExportType(type)
        if type == ExportType.drive:
            # move folders in 'filename' to sub-folders in 'folder' ('filename' should be the
            # filename only)
            filepath = Path(folder or '', filename)
            folder, filename = '/'.join(filepath.parts[:-1]), filepath.parts[-1]
            task = ee.batch.Export.image.toDrive(
                image=self._ee_image,
                folder=folder,
                fileNamePrefix=filename,
                **exp_kwargs,
            )

        elif type == ExportType.asset:
            # if folder is supplied create an EE asset ID from it and filename, else treat
            # filename as a valid EE asset ID
            asset_id = utils.asset_id(filename, folder) if folder else filename
            exp_kwargs.pop('formatOptions')  # not used for asset export
            task = ee.batch.Export.image.toAsset(
                image=self._ee_image, assetId=asset_id, **exp_kwargs
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
                **exp_kwargs,
            )
        task.start()

        if wait:
            # wait for completion
            ImageAccessor.monitorTask(task)
        return task

    def toGeoTIFF(
        self,
        file: os.PathLike | str | OpenFile,
        overwrite: bool = False,
        nodata: bool | int | float = True,
        driver: str | Driver = Driver.gtiff,
        max_tile_size: float = Tiler._default_max_tile_size,
        max_tile_dim: int = Tiler._ee_max_tile_dim,
        max_tile_bands: int = Tiler._ee_max_tile_bands,
        max_requests: int = Tiler._max_requests,
        max_cpus: int | None = None,
    ) -> None:
        """
        Export the image to a GeoTIFF file.

        Export projection and bounds are defined by :attr:`crs`, :attr:`transform` and
        :attr:`shape`, and data type by :attr:`dtype`. :meth:`prepareForExport` can be called
        before this method to apply export parameters.

        The image is retrieved as separate tiles which are downloaded and decompressed
        concurrently.  Tile size can be controlled with ``max_tile_size``, ``max_tile_dim`` and
        ``max_tile_bands``, and download / decompress concurrency with ``max_requests`` and
        ``max_cpus``.

        :param file:
            Destination file.  Can be a path or URI string, or an :class:`~fsspec.core.OpenFile`
            object in binary mode (``'wb'``).
        :param overwrite:
            Whether to overwrite the destination file if it exists.
        :param nodata:
            How to set the GeoTIFF nodata tag.  If ``True`` (the default), the nodata tag is set
            to :attr:`nodata` (the :attr:`dtype` dependent value provided by Earth Engine).
            Otherwise, if ``False``, the nodata tag is not set.  An integer or floating point
            value can also be provided, in which case the nodata tag is set to this value.
            Usually, a custom value would be supplied when the image has been unmasked with
            ``ee.Image.unmask(nodata)``.
        :param driver:
            File format driver.
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
            Maximum number of tiles to decompress concurrently.  Defaults to one less than the
            number of CPUs, or one, whichever is greater.  Values larger than the default can
            stall the asynchronous event loop and are not recommended.
        """
        driver = Driver(driver)
        ofile = fsspec.open(os.fspath(file), 'wb') if not isinstance(file, OpenFile) else file
        if not overwrite and ofile.fs.exists(ofile.path):
            raise FileExistsError(f"File exists: '{ofile.path}'.")
        self._raise_cannot_export()

        # create a rasterio profile for the destination file
        profile = self.profile
        if nodata is True:
            nodata = _nodata_vals[self.dtype]
        elif nodata is False:
            nodata = None
        profile.update(
            driver=driver.value,
            compress='deflate',
            bigtiff='if_safer',
            nodata=nodata,
        )
        # configure driver specific options
        if driver is Driver.gtiff:
            profile.update(interleave='band', tiled=True, blockxsize=512, blockysize=512)
        else:
            # use existing overviews built with DatasetWriter.build_overviews()
            profile.update(interleave='band', blocksize=512, overviews='force_use_existing')

        with ExitStack() as exit_stack:
            # Use GDAL_NUM_THREADS='ALL_CPUS' for building overviews once the download is
            # complete.  GDAL_NUM_THREADS is overridden in Tiler.map_tiles() to control download
            # concurrency.
            exit_stack.enter_context(rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False))

            # open the destination file
            out_ds = exit_stack.enter_context(_open_raster(ofile, **profile))

            # download and write tiles to file
            out_lock = threading.Lock()

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

            # populate metadata & build overviews
            self._write_metadata(out_ds)
            logger.debug(f"Building overviews for '{ofile.path}'.")
            _build_overviews(out_ds)

    def toNumPy(
        self,
        masked: bool = False,
        structured: bool = False,
        max_tile_size: float = Tiler._default_max_tile_size,
        max_tile_dim: int = Tiler._ee_max_tile_dim,
        max_tile_bands: int = Tiler._ee_max_tile_bands,
        max_requests: int = Tiler._max_requests,
        max_cpus: int | None = None,
    ) -> np.ndarray:
        """
        Export the image to a NumPy array.

        Export projection and bounds are defined by :attr:`crs`, :attr:`transform` and
        :attr:`shape`, and data type by :attr:`dtype`. :meth:`prepareForExport` can be called
        before this method to apply export parameters.

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
            Maximum number of tiles to decompress concurrently.  Defaults to one less than the
            number of CPUs, or one, whichever is greater.  Values larger than the default can
            stall the asynchronous event loop and are not recommended.

        :returns:
            NumPy array.
        """
        self._raise_cannot_export()
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
            dtype = np.dtype(dict(names=self.bandNames, formats=[self.dtype] * len(self.bandNames)))
            array = array.view(dtype=dtype).squeeze()
            if masked:
                # re-set masked array fill_value which is not copied in view
                array.fill_value = self.nodata

        return array

    def toXarray(
        self,
        masked: bool = False,
        max_tile_size: float = Tiler._default_max_tile_size,
        max_tile_dim: int = Tiler._ee_max_tile_dim,
        max_tile_bands: int = Tiler._ee_max_tile_bands,
        max_requests: int = Tiler._max_requests,
        max_cpus: int | None = None,
    ) -> xarray.DataArray:
        """
        Export the image to an Xarray DataArray.

        Export projection and bounds are defined by :attr:`crs`, :attr:`transform` and
        :attr:`shape`, and data type by :attr:`dtype`. :meth:`prepareForExport` can be called
        before this method to apply export parameters.

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
            Maximum number of tiles to decompress concurrently.  Defaults to one less than the
            number of CPUs, or one, whichever is greater.  Values larger than the default can
            stall the asynchronous event loop and are not recommended.

        :returns:
            Image DataArray.
        """
        if not xarray:
            raise ImportError("'toXarray()' requires the 'xarray' package to be installed.")

        self._raise_cannot_export()
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
        coords = dict(y=y, x=x, band=self.bandNames)

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
        attrs['stac'] = json.dumps(self.stac) if self.stac else None
        attrs = {k: v for k, v in attrs.items() if v is not None}

        return xarray.DataArray(data=array, coords=coords, dims=['y', 'x', 'band'], attrs=attrs)
