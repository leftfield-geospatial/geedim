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

##
import logging
import os
import pathlib
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from typing import Tuple, Dict, List, Union, Iterator, Optional

import ee
import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling as RioResampling
from rasterio import windows, features, warp
from tqdm import TqdmWarning
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geedim import utils
from geedim.enums import ResamplingMethod
from geedim.stac import StacCatalog, StacItem
from geedim.tile import Tile

logger = logging.getLogger(__name__)

supported_dtypes = ['uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32', 'float32', 'float64']
""" Supported image data types for downloading/exporting. """
# Note:
# - while gdal >= 3.5 supports *int64 data type, there is a rasterio bug retrieving the *int64 nodata value,
# so geedim will not support these types for now.
# - the ordering of the list above is relevant to the auto dtype and should be: unsigned ints smallest - largest,
# signed ints smallest to largest, float types smallest to largest.


class BaseImage:
    _float_nodata = float('nan')
    _desc_width = 50
    _default_resampling = ResamplingMethod.near
    _ee_max_tile_size = 32
    _ee_max_tile_dim = 10000

    def __init__(self, ee_image: ee.Image):
        """
        A class for describing and downloading an Earth Engine image.

        Allows download and export without size limits, and provides client-side access to image properties.

        Parameters
        ----------
        ee_image: ee.Image
            Earth Engine image to encapsulate.
        """
        if not isinstance(ee_image, ee.Image):
            raise TypeError('`ee_image` must be an instance of ee.Image.')
        self._ee_image = ee_image
        self.__ee_info = None
        self._id = None
        self.__min_projection = None
        self._min_dtype = None

    @classmethod
    def from_id(cls, image_id: str) -> 'BaseImage':
        """
        Create a BaseImage instance from an Earth Engine image ID.

        Parameters
        ----------
        image_id : str
           ID of earth engine image to wrap.

        Returns
        -------
        BaseImage
            BaseImage instance.
        """
        ee_image = ee.Image(image_id)
        gd_image = cls(ee_image)
        gd_image._id = image_id  # set the id attribute from image_id (avoids a call to getInfo() for .id property)
        return gd_image

    @property
    def _ee_info(self) -> Dict:
        """ Earth Engine image metadata. """
        if self.__ee_info is None:
            self.__ee_info = self._ee_image.getInfo()
        return self.__ee_info

    @property
    def _min_projection(self) -> Dict:
        """ Projection information corresponding to the minimum scale band. """
        if not self.__min_projection:
            self.__min_projection = self._get_projection(self._ee_info, min_scale=True)
        return self.__min_projection

    @property
    def _stac(self) -> Optional[StacItem]:
        """ Image STAC info.  None if there is no Earth Engine STAC entry for the image / image's collection. """
        return StacCatalog().get_item(self.id)

    @property
    def ee_image(self) -> ee.Image:
        """ Encapsulated Earth Engine image. """
        return self._ee_image

    @ee_image.setter
    def ee_image(self, value: ee.Image):
        self.__ee_info = None
        self.__min_projection = None
        self._min_dtype = None
        self._ee_image = value

    @property
    def id(self) -> Optional[str]:
        """ Earth Engine image ID.  None if the image ``system:id`` property is not set. """
        if self._id:  # avoid a call to getInfo() if _id is set
            return self._id
        else:
            return self._ee_info['id'] if 'id' in self._ee_info else None

    @property
    def date(self) -> Optional[datetime]:
        """ Image capture date & time.  None if the image ``system:time_start`` property is not set. """
        if 'system:time_start' in self.properties:
            return datetime.utcfromtimestamp(self.properties['system:time_start'] / 1000)
        else:
            return None

    @property
    def name(self) -> Optional[str]:
        """ Image name (the :attr:`id` with slashes replaced by dashes). """
        return self.id.replace('/', '-') if self.id else None

    @property
    def crs(self) -> Optional[str]:
        """
        Image CRS corresponding to minimum scale band, as an EPSG string. None if the image has no fixed projection.
        """
        return self._min_projection['crs']

    @property
    def scale(self) -> Optional[float]:
        """ Minimum scale (m) of image bands. None if the image has no fixed projection. """
        return self._min_projection['scale']

    @property
    def footprint(self) -> Optional[Dict]:
        """ Geojson polygon of the image extent.  None if the image is a composite. """
        if ('properties' not in self._ee_info) or ('system:footprint' not in self._ee_info['properties']):
            return None
        return self._ee_info['properties']['system:footprint']

    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        """ (row, column) pixel dimensions of the minimum scale band. None if the image has no fixed projection. """
        return self._min_projection['shape']

    @property
    def count(self) -> int:
        """ Number of image bands. """
        return len(self._ee_info['bands']) if 'bands' in self._ee_info else None

    @property
    def transform(self) -> Optional[rio.Affine]:
        """ Geo-transform of the minimum scale band. None if the image has no fixed projection. """
        return self._min_projection['transform']

    @property
    def dtype(self) -> str:
        """ Minimum size data type able to represent the image values. """
        if not self._min_dtype:
            self._min_dtype = self._get_min_dtype(self._ee_info)
        return self._min_dtype

    @property
    def size(self) -> Optional[int]:
        """ Image size (bytes).  None if the image has no fixed projection. """
        if not self.shape:
            return None
        dtype_size = np.dtype(self.dtype).itemsize
        return self.shape[0] * self.shape[1] * self.count * dtype_size

    @property
    def has_fixed_projection(self) -> bool:
        """ True if the image has a fixed projection, otherwise False. """
        return self.scale is not None

    @property
    def properties(self) -> Dict:
        """ Earth Engine image properties. """
        return self._ee_info['properties'] if 'properties' in self._ee_info else {}

    @property
    def band_properties(self) -> List[Dict]:
        """ Merged STAC and Earth Engine band properties. """
        return self._get_band_properties()

    @property
    def refl_bands(self) -> Optional[List[str]]:
        """ List of spectral / reflectance bands, if any. """
        if not self._stac:
            return None
        return [bname for bname, bdict in self._stac.band_props.items() if 'center_wavelength' in bdict]

    @property
    def profile(self) -> Dict:
        """ Rasterio image profile. """
        return dict(
            crs=self.crs, transform=self.transform, width=self.shape[1], height=self.shape[0], dtype=self.dtype,
            count=self.count
        )

    @property
    def bounded(self) -> bool:
        """ True if the image is bounded, otherwise False. """
        # TODO: an unbounded region could also have these bounds
        unbounded_bounds = (-180, -90, 180, 90)
        return (self.footprint is not None) and (features.bounds(self.footprint) != unbounded_bounds)

    @staticmethod
    def _get_projection(ee_info: Dict, min_scale=True) -> Dict:
        """
        Return the projection information corresponding to the min/max scale band of a given Earth Engine image info
        dictionary.
        """
        projection_info = dict(crs=None, transform=None, shape=None, scale=None)
        if 'bands' in ee_info:
            # get scale & crs corresponding to min/max scale band
            scales = np.array([abs(bd['crs_transform'][0]) for bd in ee_info['bands']])
            crss = np.array([bd['crs'] for bd in ee_info['bands']])
            fixed_idx = (crss != 'EPSG:4326') | (scales != 1)
            if sum(fixed_idx) > 0:
                idx = np.argmin(scales[fixed_idx]) if min_scale else np.argmax(scales[fixed_idx])
                band_info = np.array(ee_info['bands'])[fixed_idx][idx]
                projection_info['scale'] = abs(band_info['crs_transform'][0])
                projection_info['crs'] = band_info['crs']
                if 'dimensions' in band_info:
                    projection_info['shape'] = band_info['dimensions'][::-1]
                projection_info['transform'] = rio.Affine(*band_info['crs_transform'])
                if ('origin' in band_info) and not np.any(np.isnan(band_info['origin'])):
                    projection_info['transform'] *= rio.Affine.translation(*band_info['origin'])
        return projection_info

    @staticmethod
    def _get_min_dtype(ee_info: Dict) -> str:
        """ Return the minimum size rasterio data type corresponding to a given Earth Engine image info dictionary. """
        def get_min_int_dtype(band_info: List[Dict]) -> Optional[str]:
            """ Return the minimum integer dtype for the given band information. """
            # create a list of integer min/max values
            int_minmax = [
                minmax for band_dict in band_info if band_dict['data_type']['precision'] == 'int'
                for minmax in (int(band_dict['data_type']['min']), int(band_dict['data_type']['max']))
            ]  # yapf: disable

            if len(int_minmax) == 0:
                return None
            min_value = np.nanmin(int_minmax)
            max_value = np.nanmax(int_minmax)

            for dtype in supported_dtypes[:-2]:
                if (min_value >= np.iinfo(dtype).min) and (max_value <= np.iinfo(dtype).max):
                    return dtype
            return 'float64'

        # TODO: from gdal = 3.5, there is support *int64 in geotiffs, rasterio 1.3.0 had some nodata issues with this,
        #  int64 support should be added when these issues are resolved
        dtype = None
        if 'bands' in ee_info:
            precisions = np.array([bd['data_type']['precision'] for bd in ee_info['bands']])
            if any(precisions == 'double'):
                dtype = 'float64'
            elif any(precisions == 'float'):
                # if there are >= 32 integer bits, use float64 to accommodate them, otherwise float32
                int_dtype = get_min_int_dtype(ee_info['bands'])
                dtype = 'float32' if (not int_dtype or np.dtype(int_dtype).itemsize < 4) else 'float64'
            else:
                dtype = get_min_int_dtype(ee_info['bands'])
        return dtype

    @staticmethod
    def _str_format_size(byte_size: float, units=['bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']) -> str:
        """
        Returns a human readable string representation of a given byte size.
        Adapted from https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size.
        """
        if byte_size < 1000:
            return f'{byte_size:.2f} {units[0]}'
        else:
            return BaseImage._str_format_size(byte_size / 1000, units[1:])

    @staticmethod
    def _convert_dtype(ee_image: ee.Image, dtype: str) -> ee.Image:
        """ Convert the data type of an Earth Engine image to a specified type. """
        conv_dict = dict(
            float32=ee.Image.toFloat, float64=ee.Image.toDouble, uint8=ee.Image.toUint8, int8=ee.Image.toInt8,
            uint16=ee.Image.toUint16, int16=ee.Image.toInt16, uint32=ee.Image.toUint32, int32=ee.Image.toInt32,
        )
        if dtype not in conv_dict:
            raise TypeError(f'Unsupported dtype: {dtype}')

        return conv_dict[dtype](ee_image)

    def _get_band_properties(self) -> List[Dict]:
        """ Merge Earth Engine and STAC band properties for this image. """
        band_ids = [bd['id'] for bd in self._ee_info['bands']]
        if self._stac:
            stac_bands_props = self._stac.band_props
            band_props = [stac_bands_props[bid] if bid in stac_bands_props else dict(name=bid) for bid in band_ids]
        else:  # just use the image band IDs
            band_props = [dict(name=bid) for bid in band_ids]
        return band_props

    @staticmethod
    def _scale_offset(ee_image: ee.Image, band_properties: List[Dict]) -> ee.Image:
        """
        Apply any STAC band scales and offsets to an EE image.

        Applying STAC scales and offsets will convert pixel values from representative integer ranges to the floating
        point ranges of physical quantities (e.g. surface reflectance in the range 0-1).

        Parameters
        ----------
        ee_image: ee.Image
            The EE image to scale and offset.
        band_properties: list(dict)
            A list of dictionaries specifying band names and corresponding scale and or offset values e.g.
            :attr:`BaseImage.band_properties`.

        Returns
        -------
        ee.Image
            The scaled and offset image, with the same bands as ``ee_image``, and in the same order.
        """
        if band_properties is None:
            logger.warning('Cannot scale and offset this image, there is no STAC band information.')
            return ee_image

        # make band scale and offset dicts
        scale_dict = {bp['name']: bp['scale'] if 'scale' in bp else 1. for bp in band_properties}
        offset_dict = {bp['name']: bp['offset'] if 'offset' in bp else 0. for bp in band_properties}

        if all([s == 1 for s in scale_dict.values()]) and all([o == 0 for o in offset_dict.values()]):
            # all scales==1 and all offsets==0
            return ee_image

        adj_bands = ee_image.bandNames().filter(ee.Filter.inList('item', list(scale_dict.keys())))
        non_adj_bands = ee_image.bandNames().removeAll(adj_bands)

        # apply the scales and offsets
        scale_im = ee.Dictionary(scale_dict).toImage().select(adj_bands)
        offset_im = ee.Dictionary(offset_dict).toImage().select(adj_bands)
        adj_im = ee_image.select(adj_bands).multiply(scale_im).add(offset_im)

        # Typically, `band_properties` will contain items for every band, but there may be cases where there
        # are additional image bands in `ee_image` not in `band_properties`. Here, these additional bands are added
        # back to the adjusted image.
        adj_im = adj_im.addBands(ee_image.select(non_adj_bands))
        adj_im = adj_im.select(ee_image.bandNames())  # keep bands in original order
        # copy original ee_image properties and return
        return ee.Image(adj_im.copyProperties(ee_image, ee_image.propertyNames()))

    def _prepare_for_export(
        self, crs: str = None, crs_transform: Tuple[float] = None, shape: Tuple[int, int] = None, region: Dict =
        None, scale: float = None, resampling: ResamplingMethod = _default_resampling, dtype: str = None,
        scale_offset: bool = False
    ) -> 'BaseImage':
        """
        Prepare the encapsulated image for export/download.  Will reproject, resample, clip and convert the image
        according to the provided parameters.

        Export bounds and resolution can be specified with ``region`` and ``scale`` / ``shape``, or ``crs_transform``
        and ``shape``.  If no bounds are specified (with either ``region``, or ``crs_transform`` & ``shape``), the
        entire image granule is exported.

        When ``crs``, ``scale``, ``crs_transform`` & ``shape`` are not specified, the pixel grid of the exported
        image will coincide with that of the encapsulated image.

        Parameters
        ----------
        crs : str, optional
            WKT or EPSG specification of CRS to export to.  Where image bands have different CRSs, all are
            re-projected to this CRS. Defaults to use the CRS of the minimum scale band if available.
        crs_transform: tuple of float, list of float, rio.Affine, optional
            List of 6 numbers specifying an affine transform in the specified CRS.  In row-major order:
            [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation].  All bands are re-projected to
            this transform.
        shape: tuple of int, optional
            (height, width) dimensions to export (pixels).
        region : dict, geojson, ee.Geometry, optional
            Region of interest (WGS84) to export.  Defaults to the image footprint, when available.
        scale : float, optional
            Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this scale.
            Ignored if ``crs`` and ``crs_transform`` are specified.  Defaults to use the minimum scale of
            image bands if available.
        resampling : ResamplingMethod, optional
            Resampling method - see :class:`~geedim.enums.ResamplingMethod` for available options.
        dtype: str, optional
           Convert to this data type (`uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `float32` or
           `float64`). Defaults to auto select a minimal type that can represent the range of pixel values.
        scale_offset: bool, optional
            Whether to apply any EE band scales and offsets to the image.

        Returns
        -------
        BaseImage
            Prepared image.
        """

        # Check for parameter combination errors.
        # This necessarily duplicates some of what is done in ee.Image.getDownloadURL, so that errors are raised
        # before tile creation & download.
        if (not crs or not region or not scale) and (not crs or not crs_transform or not shape):
            if not self.has_fixed_projection:
                # if the image has no fixed projection, either crs, region, & scale; or crs, crs_transform and shape
                # must be specified
                raise ValueError(
                    f'This image does not have a fixed projection, you need to specify a crs, region & scale; or a '
                    f'crs, crs_transform & shape.'
                )

        if (not self.bounded) and (not region and (not crs or not crs_transform or not shape)):
            # if the image has no footprint (i.e. it is 'unbounded'), either region; or crs, crs_transform and shape
            # must be specified
            raise ValueError(
                f'This image is unbounded, you need to specify a region; or a crs, crs_transform and '
                f'shape.'
            )

        if self.crs == 'EPSG:4326' and not scale and not shape:
            # If the image is in EPSG:4326, either scale (in meters); or shape must be specified.
            # Note that ee.Image.prepare_for_export() expects a scale in meters, but if the image is EPSG:4326,
            # the default scale is in degrees.
            raise ValueError(f'This image is in EPSG:4326, you need to specify a scale (in meters); or a shape.')

        if scale and shape:
            # This error is raised in later calls to ee.Image.getInfo(), but is neater to raise here first
            raise ValueError('You can specify one of scale or shape, but not both.')

        # perform image scale/offset, dtype and resampling operations
        ee_image = self.ee_image
        if scale_offset:
            ee_image = self._scale_offset(ee_image, self.band_properties)
            im_dtype = 'float64'
        else:
            im_dtype = self.dtype

        resampling = ResamplingMethod(resampling)
        if resampling != self._default_resampling:
            if not self.has_fixed_projection:
                raise ValueError(
                    'This image has no fixed projection and cannot be resampled.  If this image is a composite, '
                    'you can resample the images used to create the composite.'
                )
            ee_image = utils.resample(ee_image, resampling)

        ee_image = self._convert_dtype(ee_image, dtype=dtype or im_dtype)

        # configure the export parameter values
        crs = crs or self.crs
        if not crs_transform and not shape:
            # if none of crs_transform, shape, region or scale are specified, then set region and scale to defaults
            region = region or self.footprint
            scale = scale or self.scale

            if (crs == self.crs) and (scale == self.scale):
                # if crs_transform & shape are not already specified, and crs & scale match this image's crs & scale,
                # then set crs_transform and shape to export on this image's (self) pixel grid
                if region == self.footprint:
                    crs_transform = self.transform
                    shape = self.shape
                else:
                    # find a crs_transform and shape that encompasses region
                    if isinstance(region, ee.Geometry):
                        region = region.getInfo()
                    region_crs = region['crs']['properties']['name'] if 'crs' in region else 'EPSG:4326'
                    region_bounds = warp.transform_bounds(region_crs, utils.rio_crs(crs), *features.bounds(region))
                    region_win = windows.from_bounds(*region_bounds, transform=self.transform)
                    region_win = utils.expand_window_to_grid(region_win)
                    crs_transform = self.transform * rio.Affine.translation(region_win.col_off, region_win.row_off)
                    shape = (region_win.height, region_win.width)

                # prevent exporting with region & scale now that crs_transform and shape are set
                region = None
                scale = None

        # create the export parameter dict
        crs_transform = crs_transform[:6] if crs_transform else None
        dimensions = shape[::-1] if shape else None
        export_kwargs = dict(
            crs=crs, crs_transform=crs_transform, dimensions=dimensions, region=region, scale=scale,
            fileFormat='GeoTIFF', filePerBand=False
        )
        # drop items with values==None
        export_kwargs = {k: v for k, v in export_kwargs.items() if v is not None}
        logger.debug(f'ee.Image.prepare_for_export() params: {export_kwargs}')
        ee_image, _ = ee_image.prepare_for_export(export_kwargs)
        return BaseImage(ee_image)

    def _prepare_for_download(self, set_nodata: bool = True, **kwargs) -> Tuple['BaseImage', Dict]:
        """
        Prepare the encapsulated image for tiled GeoTIFF download. Will reproject, resample, clip and convert the image
        according to the provided parameters.

        Returns the prepared image and a rasterio profile for the downloaded GeoTIFF.
        """
        # resample, convert, clip and reproject image according to download params
        exp_image = self._prepare_for_export(**kwargs)
        # see float nodata workaround note in Tile.download(...)
        nodata_dict = dict(
            float32=self._float_nodata,
            float64=self._float_nodata,
            uint8=0,
            int8=np.iinfo('int8').min,
            uint16=0,
            int16=np.iinfo('int16').min,
            uint32=0,
            int32=np.iinfo('int32').min,
        )  # yapf: disable
        nodata = nodata_dict[exp_image.dtype] if set_nodata else None
        profile = dict(
            driver='GTiff', dtype=exp_image.dtype, nodata=nodata, width=exp_image.shape[1], height=exp_image.shape[0],
            count=exp_image.count, crs=CRS.from_string(utils.rio_crs(exp_image.crs)), transform=exp_image.transform,
            compress='deflate', interleave='band', tiled=True, photometric=None,
        )
        # add BIGTIFF support if the uncompressed image is bigger than 4GB
        if exp_image.size >= 4e9:
            profile.update(bigtiff=True)
        return exp_image, profile

    @staticmethod
    def _get_tile_shape(
        exp_image: 'BaseImage', max_tile_size: Optional[float] = None, max_tile_dim: Optional[int] = None
    ) -> Tuple[Tuple[int, int], int]:  # yapf: disable
        """
        Return a tile shape and number of tiles for a given BaseImage, such that the tile shape satisfies GEE
        download limits, and is 'square-ish'.
        """
        # convert max_tile_size from MB to bytes & set to EE default if None
        if max_tile_size and (max_tile_size > BaseImage._ee_max_tile_size):
            raise ValueError(
                f'`max_tile_size` must be less than or equal to the Earth Engine download size limit of '
                f'{BaseImage._ee_max_tile_size} MB.'
            )
        max_tile_size = int((max_tile_size * (1 << 20)) if max_tile_size else (BaseImage._ee_max_tile_size << 20))
        if max_tile_dim and (max_tile_dim > BaseImage._ee_max_tile_dim):
            raise ValueError(
                f'`max_tile_dim` must be less than or equal to the Earth Engine download limit of '
                f'{BaseImage._ee_max_tile_size} pixels.'
            )
        max_tile_dim = max_tile_dim or BaseImage._ee_max_tile_dim   # set max_tile_dim to EE default if None

        # find the total number of tiles the image must be divided into to satisfy max_tile_size
        image_shape = np.array(exp_image.shape, dtype='int64')
        dtype_size = np.dtype(exp_image.dtype).itemsize
        image_size = exp_image.size
        if exp_image.dtype.endswith('int8'):
            # workaround for GEE overestimate of *int8 dtype download sizes
            dtype_size *= 2
            image_size *= 2

        pixel_size = dtype_size * exp_image.count

        num_tile_shape = np.array([1, 1], dtype='int64')
        tile_size = image_size
        tile_shape = image_shape
        while tile_size >= max_tile_size:
            div_axis = np.argmax(tile_shape)
            num_tile_shape[div_axis] += 1  # increase the num tiles down the longest dimension of tile_shape
            tile_shape = np.ceil(image_shape / num_tile_shape).astype('int64')
            tile_size = tile_shape[0] * tile_shape[1] * pixel_size

        tile_shape[tile_shape > max_tile_dim] = max_tile_dim
        num_tiles = int(np.product(np.ceil(image_shape / tile_shape)))
        tile_shape = tuple(tile_shape.tolist())
        return tile_shape, num_tiles

    @staticmethod
    def _build_overviews(dataset: rio.io.DatasetWriter, max_num_levels: int = 8, min_ovw_pixels: int = 256):
        """ Build internal overviews, downsampled by successive powers of 2, for an open rasterio dataset. """
        if dataset.closed:
            raise IOError('Image dataset is closed')

        # limit overviews so that the highest level has at least 2**8=256 pixels along the shortest dimension,
        # and so there are no more than 8 levels.
        max_ovw_levels = int(np.min(np.log2(dataset.shape)))
        min_level_shape_pow2 = int(np.log2(min_ovw_pixels))
        num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
        ovw_levels = [2**m for m in range(1, num_ovw_levels + 1)]
        dataset.build_overviews(ovw_levels, RioResampling.average)

    def _write_metadata(self, dataset: rio.io.DatasetWriter):
        """ Write Earth Engine and STAC metadata to an open rasterio dataset. """
        if dataset.closed:
            raise IOError('Image dataset is closed')

        # replace 'system:*' property keys with 'system-*', and remove footprint if its there
        properties = {k.replace(':', '-'):v for k, v in self.properties.items()}
        if 'system-footprint' in properties:
            properties.pop('system-footprint')
        dataset.update_tags(**properties)

        if self._stac and self._stac.license:
            dataset.update_tags(LICENSE=self._stac.license)

        def clean_text(text, width=80) -> str:
            """ Return a shortened tidied string. """
            if not isinstance(text, str):
                return text
            text = text.split('.')[0] if len(text) > width else text
            text = text.strip()
            text = '-\n' + text if len(text) > width else text
            return text

        # populate band metadata
        for band_i, band_dict in enumerate(self.band_properties):
            clean_band_dict = {k.replace(':', '-'): clean_text(v) for k, v in band_dict.items()}
            if 'name' in band_dict:
                dataset.set_band_description(band_i + 1, clean_band_dict['name'])
            dataset.update_tags(band_i + 1, **clean_band_dict)

    @staticmethod
    def _tiles(exp_image: 'BaseImage', tile_shape: Tuple[int, int]) -> Iterator[Tile]:
        """
        Iterator over downloadable image tiles.

        Divides an image into adjoining tiles no bigger than `tile_shape`.

        Parameters
        ----------
        exp_image: BaseImage
            Image to tile.
        tile_shape: Tuple[int, int]
            (row, column) tile shape to use (pixels). Use :meth:`BaseImage._get_tile_shape` to find a tile shape that
            satisfies the Earth Engine download limit for :param:`exp_image`.

        Yields
        -------
        Tile
            An image tile that can be downloaded.
        """

        # split the image up into tiles of at most `tile_shape` dimension
        image_shape = exp_image.shape
        start_range = product(range(0, image_shape[0], tile_shape[0]), range(0, image_shape[1], tile_shape[1]))
        for tile_start in start_range:
            tile_stop = np.clip(np.add(tile_start, tile_shape), a_min=None, a_max=image_shape)
            clip_tile_shape = (tile_stop - tile_start).tolist()  # tolist is just to convert to native int
            tile_window = windows.Window(tile_start[1], tile_start[0], clip_tile_shape[1], clip_tile_shape[0])
            yield Tile(exp_image, tile_window)

    @staticmethod
    def monitor_export(task: ee.batch.Task, label: str = None):
        """
        Monitor and display the progress of an export task.

        Parameters
        ----------
        task : ee.batch.Task
            Earth Engine task to monitor (as returned by :meth:`export`).
        label: str, optional
            Optional label for progress display.  Defaults to task description.
        """
        pause = 0.1
        status = ee.data.getOperation(task.name)

        if label is None:
            label = status["metadata"]["description"]
        label = label if (len(label) < BaseImage._desc_width) else f'*{label[-BaseImage._desc_width:]}'

        # poll EE until the export preparation is complete
        with utils.Spinner(label=f'Preparing {label}: ', leave='done'):
            while 'progress' not in status['metadata']:
                time.sleep(5 * pause)
                status = ee.data.getOperation(task.name)

        # wait for export to complete, displaying a progress bar
        bar_format = '{desc}: |{bar}| [{percentage:5.1f}%] in {elapsed:>5s} (eta: {remaining:>5s})'
        with tqdm(desc=f'Exporting {label}', total=1, bar_format=bar_format, dynamic_ncols=True) as bar:
            while ('done' not in status) or (not status['done']):
                time.sleep(10 * pause)
                status = ee.data.getOperation(task.name)  # get task status
                bar.update(status['metadata']['progress'] - bar.n)

            if status['metadata']['state'] == 'SUCCEEDED':
                bar.update(1 - bar.n)
            else:
                raise IOError(f"Export failed \n{status}")

    def export(self, filename: str, folder: str = None, wait: bool = True, **kwargs) -> ee.batch.Task:
        """
        Export the encapsulated image to Google Drive.

        Export bounds and resolution can be specified with ``region`` and ``scale`` / ``shape``, or ``crs_transform``
        and ``shape``.  If no bounds are specified (with either ``region``, or ``crs_transform`` & ``shape``), the
        entire image granule is exported.

        When ``crs``, ``scale``, ``crs_transform`` & ``shape`` are not specified, the pixel grid of the exported
        image will coincide with that of the encapsulated image.


        Parameters
        ----------
        filename : str
            Name of the task and destination file.
        folder : str, optional
            Google Drive folder to export to. Defaults to the root.
        wait : bool
            Wait for the export to complete before returning.
        crs : str, optional
            WKT or EPSG specification of CRS to export to.  Where image bands have different CRSs, all are
            re-projected to this CRS. Defaults to use the CRS of the minimum scale band if available.
        crs_transform: tuple of float, list of float, rio.Affine, optional
            List of 6 numbers specifying an affine transform in the specified CRS.  In row-major order:
            [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation].  All bands are re-projected to
            this transform.
        shape: tuple of int, optional
            (height, width) dimensions to export (pixels).
        region : dict, geojson, ee.Geometry, optional
            Region of interest (WGS84) to export.  Defaults to the image footprint, when available.
        scale : float, optional
            Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this scale.
            Ignored if ``crs`` and ``crs_transform`` are specified.  Defaults to use the minimum scale of
            image bands if available.
        resampling : ResamplingMethod, optional
            Resampling method - see :class:`~geedim.enums.ResamplingMethod` for available options.
        dtype: str, optional
           Convert to this data type (`uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `float32` or
           `float64`). Defaults to auto select a minimal type that can represent the range of pixel values.
        scale_offset: bool, optional
            Whether to apply any EE band scales and offsets to the image.

        Returns
        -------
        ee.batch.Task
            The Earth Engine export task, started if ``wait`` is False, or completed if ``wait`` is True.
        """

        exp_image = self._prepare_for_export(**kwargs)
        if exp_image.crs == 'SR-ORG:6974':
            logger.warning(
                'There is an earth engine bug exporting in SR-ORG:6974, you will need to edit the exported file to '
                'replace the CRS with SR-ORG:6842. See: https://issuetracker.google.com/issues/194561313.'
            )

        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f'Uncompressed size: {self._str_format_size(exp_image.size)}')

        # create export task and start
        task = ee.batch.Export.image.toDrive(
            image=exp_image.ee_image, description=filename[:100], folder=folder, fileNamePrefix=filename, maxPixels=1e9
        )
        task.start()
        if wait:  # wait for completion
            self.monitor_export(task)
        return task

    def download(
        self, filename: Union[pathlib.Path, str], overwrite: bool = False, num_threads: Optional[int] = None,
        max_tile_size: Optional[float] = None, max_tile_dim: Optional[int] = None, **kwargs
    ):
        """
        Download the encapsulated image to a GeoTiff file.

        Images larger than the `Earth Engine size limit
        <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`_ are split and downloaded as
        separate tiles, then re-assembled into a single GeoTIFF.  Downloaded image files are populated with metadata
        from the Earth Engine image and STAC.

        Image bounds and resolution can be specified with ``region`` and ``scale`` / ``shape``, or ``crs_transform``
        and ``shape``.  If no bounds are specified (with either ``region``, or ``crs_transform`` & ``shape``), the
        entire image granule is downloaded.

        When ``crs``, ``scale``, ``crs_transform`` & ``shape`` are not specified, the pixel grid of the exported
        image will coincide with that of the encapsulated image.

        Parameters
        ----------
        filename: pathlib.Path, str
            Name of the destination file.
        overwrite : bool, optional
            Overwrite the destination file if it exists.
        num_threads: int, optional
            Number of tiles to download concurrently.  Defaults to a sensible auto value.
        max_tile_size: int, optional
            Maximum tile size (MB).  If None, defaults to the Earth Engine download size limit (32 MB).
        max_tile_dim: int, optional
            Maximum tile width/height (pixels).  If None, defaults to Earth Engine download limit (10000).
        crs : str, optional
            WKT or EPSG specification of CRS to export to.  Where image bands have different CRSs, all are
            re-projected to this CRS. Defaults to use the CRS of the minimum scale band if available.
        crs_transform: tuple of float, list of float, rio.Affine, optional
            List of 6 numbers specifying an affine transform in the specified CRS.  In row-major order:
            [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation].  All bands are re-projected to
            this transform.
        shape: tuple of int, optional
            (height, width) dimensions to export (pixels).
        region : dict, geojson, ee.Geometry, optional
            Region of interest (WGS84) to export.  Defaults to the image footprint, when available.
        scale : float, optional
            Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this scale.
            Ignored if ``crs`` and ``crs_transform`` are specified.  Defaults to use the minimum scale of
            image bands if available.
        resampling : ResamplingMethod, optional
            Resampling method - see :class:`~geedim.enums.ResamplingMethod` for available options.
        dtype: str, optional
           Convert to this data type (`uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `float32` or
           `float64`). Defaults to auto select a minimal type that can represent the range of pixel values.
        scale_offset: bool, optional
            Whether to apply any EE band scales and offsets to the image.
        """

        max_threads = num_threads or min(32, (os.cpu_count() or 1) + 4)
        out_lock = threading.Lock()
        filename = pathlib.Path(filename)
        if filename.exists():
            if overwrite:
                os.remove(filename)
            else:
                raise FileExistsError(f'{filename} exists')

        # prepare (resample, convert, reproject) the image for download
        exp_image, profile = self._prepare_for_download(**kwargs)

        # get the dimensions of an image tile that will satisfy GEE download limits
        tile_shape, num_tiles = self._get_tile_shape(exp_image, max_tile_size=max_tile_size, max_tile_dim=max_tile_dim)

        # find raw size of the download data (less than the actual download size as the image data is zipped in a
        # compressed geotiff)
        raw_download_size = exp_image.size
        if logger.getEffectiveLevel() <= logging.DEBUG:
            dtype_size = np.dtype(exp_image.dtype).itemsize
            raw_tile_size = tile_shape[0] * tile_shape[1] * exp_image.count * dtype_size
            logger.debug(f'{filename.name}:')
            logger.debug(f'Uncompressed size: {self._str_format_size(raw_download_size)}')
            logger.debug(f'Num. tiles: {num_tiles}')
            logger.debug(f'Tile shape: {tile_shape}')
            logger.debug(f'Tile size: {self._str_format_size(int(raw_tile_size))}')

        if raw_download_size > 1e9:
            # warn if the download is large (>1GB)
            logger.warning(
                f'Consider adjusting `region`, `scale` and/or `dtype` to reduce the {filename.name}'
                f' download size (raw: {self._str_format_size(raw_download_size)}).'
            )

        # configure the progress bar to monitor raw/uncompressed download size
        desc = filename.name if (len(filename.name) < self._desc_width) else f'...{filename.name[-self._desc_width:]}'
        bar_format = (
            '{desc}: |{bar}| {n_fmt}/{total_fmt} (raw) [{percentage:5.1f}%] in {elapsed:>5s} (eta: {remaining:>5s})'
        )
        bar = tqdm(
            desc=desc, total=raw_download_size, bar_format=bar_format, dynamic_ncols=True, unit_scale=True, unit='B'
        )

        session = utils.retry_session(5)
        warnings.filterwarnings('ignore', category=TqdmWarning)
        redir_tqdm = logging_redirect_tqdm([logging.getLogger(__package__)])  # redirect logging through tqdm
        env = rio.Env(GDAL_NUM_THREADS='ALL_CPUs', GTIFF_FORCE_RGBA=False)
        with redir_tqdm, env, rio.open(filename, 'w', **profile) as out_ds, bar:

            def download_tile(tile):
                """Download a tile and write into the destination GeoTIFF. """
                tile_array = tile.download(session=session, bar=bar)
                with out_lock:
                    out_ds.write(tile_array, window=tile.window)

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                # Run the tile downloads in a thread pool
                tiles = self._tiles(exp_image, tile_shape=tile_shape)
                futures = [executor.submit(download_tile, tile) for tile in tiles]
                try:
                    for future in as_completed(futures):
                        future.result()
                except Exception as ex:
                    logger.info(f'Exception: {str(ex)}\nCancelling...')
                    executor.shutdown(wait=False)
                    raise ex

            bar.update(bar.total - bar.n)   # ensure the bar reaches 100%
            # populate GeoTIFF metadata and build overviews
            self._write_metadata(out_ds)
            self._build_overviews(out_ds)
