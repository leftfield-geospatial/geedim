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
import collections
import logging
import os
import pathlib
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Tuple, Dict, List, Union

import ee
import numpy as np
import rasterio as rio
from pip._vendor.progress.spinner import Spinner
from rasterio.crs import CRS
from rasterio.enums import Resampling as RioResampling
from rasterio.warp import transform_geom
from rasterio.windows import Window
from tqdm import TqdmWarning, tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geedim import info
from geedim.enums import ResamplingMethod
from geedim.errors import UnsupportedValueError, UnsupportedTypeError, IoError
from geedim.tile import Tile, _requests_retry_session

logger = logging.getLogger(__name__)


def split_id(image_id):
    """
    Split Earth Engine image ID into collection and index components.

    Parameters
    ----------
    image_id: str
              Earth engine image ID.

    Returns
    -------
    : Tuple[str, str]
        A tuple of strings: (collection name, image index).
    """
    index = image_id.split("/")[-1]
    ee_coll_name = "/".join(image_id.split("/")[:-1])
    return ee_coll_name, index


def get_bounds(filename, expand=5):  # pragma coverage
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
    try:
        # GEE sets tif colorinterp tags incorrectly, suppress rasterio warning relating to this:
        # 'Sum of Photometric type-related color channels and ExtraSamples doesn't match SamplesPerPixel'
        logging.getLogger("rasterio").setLevel(logging.ERROR)
        with rio.open(filename) as im:
            bbox = im.bounds
            if (im.crs.linear_units == "metre") and (expand > 0):  # expand the bounding box
                expand_x = (bbox.right - bbox.left) * expand / 100.0
                expand_y = (bbox.top - bbox.bottom) * expand / 100.0
                bbox_expand = rio.coords.BoundingBox(
                    bbox.left - expand_x, bbox.bottom - expand_y, bbox.right + expand_x, bbox.top + expand_y,
                )
            else:
                bbox_expand = bbox

            coordinates = [
                [bbox_expand.right, bbox_expand.bottom], [bbox_expand.right, bbox_expand.top],
                [bbox_expand.left, bbox_expand.top], [bbox_expand.left, bbox_expand.bottom],
                [bbox_expand.right, bbox_expand.bottom],
            ]

            bbox_expand_dict = dict(type="Polygon", coordinates=[coordinates])
            src_bbox_wgs84 = transform_geom(im.crs, "WGS84", bbox_expand_dict)  # convert to WGS84 geojson
    finally:
        logging.getLogger("rasterio").setLevel(logging.WARNING)
    return src_bbox_wgs84


class BaseImage:
    _float_nodata = float('nan')
    _desc_width = 70
    _default_resampling = ResamplingMethod.near
    _supported_collection_ids = ['*']

    def __init__(self, ee_image):
        """
        Create a BaseImage instance for encapsulating an Earth Engine image.  Allows download and export without size
        limits, and provides client-side access to image metadata.

        Parameters
        ----------
        ee_image: ee.Image
            The Earth Engine image to encapsulate.
        """
        if not isinstance(ee_image, ee.Image):
            raise UnsupportedTypeError('`ee_image` must be an instance of ee.Image.')
        self._ee_image = ee_image
        self._ee_info = None
        self._id = None
        self._min_projection = None
        self._min_dtype = None

    @classmethod
    def from_id(cls, image_id):
        """
        Create a BaseImage instance from an EE image ID.

        Parameters
        ----------
        image_id : str
           ID of earth engine image to wrap.

        Returns
        -------
        gd_image: BaseImage
            The BaseImage instance.
        """
        ee_image = ee.Image(image_id)
        gd_image = cls(ee_image)
        gd_image._id = image_id  # set the id attribute from image_id (avoids a call to getInfo() for .id property)
        return gd_image

    @property
    def ee_image(self) -> ee.Image:
        """The encapsulated EE image."""
        return self._ee_image

    @ee_image.setter
    def ee_image(self, value: ee.Image):
        self._ee_info = None
        self._min_projection = None
        self._min_dtype = None
        self._ee_image = value

    @property
    def id(self) -> str:
        """The EE image ID."""
        return self._id or self.ee_info['id']  # avoid a call to getInfo() if _id is set

    @property
    def name(self) -> str:
        """The image name (the ID with slashes replaces by dashes)."""
        return self.id.replace('/', '-')

    @property
    def ee_info(self) -> Union[Dict, None]:
        """The EE image metadata in a dict."""
        if self._ee_info is None:
            self._ee_info = self._ee_image.getInfo()
        return self._ee_info

    @property
    def properties(self) -> Union[Dict, None]:
        """The EE image properties in a dict."""
        return self.ee_info['properties'] if 'properties' in self.ee_info else None

    @property
    def min_projection(self) -> Union[Dict, None]:
        """A dict of the projection information corresponding to the minimum scale band."""
        if not self._min_projection:
            self._min_projection = self._get_projection(self.ee_info, min_scale=True)
        return self._min_projection

    @property
    def crs(self) -> str:
        """
        The image CRS corresponding to minimum scale band, as an EPSG string.
        Will return None if the image has no fixed projection.
        """
        return self.min_projection['crs']

    @property
    def scale(self) -> float:
        """The scale (m) corresponding to minimum scale band. Will return None if the image has no fixed projection."""
        return self.min_projection['scale']

    @property
    def shape(self) -> Tuple[int, int]:
        """
        The (row, column) dimensions of the minimum scale band.
        Will return None if the image has no fixed projection.
        """
        return self.min_projection['shape']

    @property
    def count(self) -> int:
        """The number of image bands."""
        return len(self.ee_info['bands']) if 'bands' in self.ee_info else None

    @property
    def transform(self) -> Union[rio.Affine, None]:
        """
        The geo-transform of the minimum scale band, as a rasterio Affine transform.
        Will return None if the image has no fixed projection.
        """
        return self.min_projection['transform']

    @property
    def has_fixed_projection(self) -> bool:
        """True if the image has a fixed projection, otherwise False."""
        return self.scale is not None

    @property
    def dtype(self) -> str:
        """The minimal size data type required to represent the values in this image (as a string)."""
        if not self._min_dtype:
            self._min_dtype = self._get_min_dtype(self.ee_info)
        return self._min_dtype

    @property
    def byte_size(self) -> int:
        """The size in bytes of this image."""
        dtype_size = np.dtype(self.dtype).itemsize
        return self.shape[0] * self.shape[1] * self.count * dtype_size

    @property
    def footprint(self) -> Union[Dict, None]:
        """A geojson polygon of the image extent."""
        if 'system:footprint' not in self.ee_info['properties']:
            return None
        return self.ee_info['properties']['system:footprint']

    @property
    def band_metadata(self) -> List:
        # TODO: replace with STAC
        """A list of dicts describing the image bands."""
        return self._get_band_metadata(self.ee_info)

    @property
    def info(self) -> Dict:
        # TODO can be removed after rewrite of tests?
        """All the BasicImage properties packed into a dictionary"""
        return dict(
            id=self.id, properties=self.properties, footprint=self.footprint, bands=self.band_metadata,
            **self.min_projection
        )

    @staticmethod
    def _get_projection(ee_info: Dict, min_scale=True) -> Dict:
        """
        Return the projection information corresponding to the min/max scale band of a given an EE image info
        dictionary.
        """
        projection_info = dict(crs=None, transform=None, shape=None, scale=None)
        if 'bands' in ee_info:
            # get scale & crs corresponding to min/max scale band
            scales = np.array([abs(bd['crs_transform'][0]) for bd in ee_info['bands']])
            crss = np.array([bd['crs'] for bd in ee_info['bands']])
            fixed_idx = (crss != 'EPSG:4326') & (scales != 1)
            if sum(fixed_idx) > 0:
                idx = np.argmin(scales[fixed_idx]) if min_scale else np.argmax(scales[fixed_idx])
                band_info = np.array(ee_info['bands'])[fixed_idx][idx]
                projection_info['scale'] = abs(band_info['crs_transform'][0])
                projection_info['crs'] = band_info['crs']
                projection_info['shape'] = band_info['dimensions'][::-1]
                projection_info['transform'] = rio.Affine(*band_info['crs_transform'])
                if ('origin' in band_info) and not np.any(np.isnan(band_info['origin'])):
                    projection_info['transform'] *= rio.Affine.translation(*band_info['origin'])
        return projection_info

    @staticmethod
    def _get_min_dtype(ee_info: Dict) -> str:
        """Return the minimal size data type corresponding to a given EE image info dictionary."""
        dtype = None
        if 'bands' in ee_info:
            precisions = np.array([bd['data_type']['precision'] for bd in ee_info['bands']])
            if all(precisions == 'int'):
                dtype_minmax = np.array(
                    [(bd['data_type']['min'], bd['data_type']['max']) for bd in ee_info['bands']], dtype=np.int64
                )
                dtype_min = int(dtype_minmax[:, 0].min())  # minimum image pixel value
                dtype_max = int(dtype_minmax[:, 1].max())  # maximum image pixel value

                # determine the number of integer bits required to represent the value range
                bits = 0
                for bound in [abs(dtype_max), abs(dtype_min)]:
                    bound_bits = 0 if bound == 0 else 2 ** np.ceil(np.log2(np.log2(abs(bound))))
                    bits += bound_bits
                bits = min(max(bits, 8), 32)  # clamp bits to allowed values
                dtype = f'{"u" if dtype_min >= 0 else ""}int{int(bits)}'
            elif any(precisions == 'double'):
                dtype = 'float64'
            else:
                dtype = 'float32'
        return dtype

    @staticmethod
    def _str_format_size(byte_size: float, units=['bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']) -> str:
        """
        Returns a human readable string representation of bytes.
        Adapted from https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size.
        """
        if byte_size < 1024:
            return f'{byte_size:.2f} {units[0]}'
        else:
            return BaseImage._str_format_size(byte_size / 1000, units[1:])

    @staticmethod
    def _get_band_metadata(ee_info: Dict) -> List[Dict]:
        """Return band metadata given an EE image info dict."""
        ee_coll_name, _ = split_id(ee_info['id'])
        band_ids = [bd['id'] for bd in ee_info['bands']]
        if ee_coll_name in info.collection_info:  # include SR band metadata if it exists
            # use DataFrame to concat SR band metadata from collection_info with band IDs from the image
            sr_band_list = info.collection_info[ee_coll_name]["bands"].copy()
            sr_band_dict = {bdict['id']: bdict for bdict in sr_band_list}
            band_metadata = [sr_band_dict[bid] if bid in sr_band_dict else dict(id=bid) for bid in band_ids]
        else:  # just use the image band IDs
            band_metadata = [dict(id=bid) for bid in band_ids]
        return band_metadata

    @staticmethod
    def _convert_dtype(ee_image: ee.Image, dtype: str) -> ee.Image:
        """ Convert the data type of an EE image to a specified type. """
        conv_dict = dict(
            float32=ee.Image.toFloat,
            float64=ee.Image.toDouble,
            uint8=ee.Image.toUint8,
            int8=ee.Image.toInt8,
            uint16=ee.Image.toUint16,
            int16=ee.Image.toInt16,
            uint32=ee.Image.toUint32,
            int32=ee.Image.toInt32,
        )
        if dtype not in conv_dict:
            raise UnsupportedTypeError(f'Unsupported dtype: {dtype}')

        return conv_dict[dtype](ee_image)

    def _prepare_for_export(self, region=None, crs=None, scale=None, resampling=_default_resampling, dtype=None):
        """
        Prepare the encapsulated image for export/download.  Will reproject, resample, clip and convert the image
        according to the provided parameters.

        Parameters
        ----------
        region : dict, geojson, ee.Geometry, optional
            Region of interest (WGS84) to export [default: export the entire image if it has a footprint].
        crs : str, optional
            WKT, EPSG etc specification of CRS to export to.  Where image bands have different CRSs, all are
            re-projected to this CRS.
            [default: use the CRS of the minimum scale band if available].
        scale : float, optional
            Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this scale.
            [default: use the minimum scale of image bands if available].
        resampling : ResamplingMethod, optional
            Resampling method: ('near'|'bilinear'|'bicubic') [default: 'near']
        dtype: str, optional
            Data type to export to ('uint8'|'int8'|'uint16'|'int16'|'uint32'|'int32'|'float32'|'float64')
            [default: auto select a minimal type]

        Returns
        -------
        exp_image: BaseImage
            The prepared image.
        """

        if not region or not crs or not scale:
            # One or more of region, crs and scale were not provided, so get the image values to use instead
            if not self.has_fixed_projection:
                # Raise an error if this image has no fixed projection
                raise UnsupportedValueError(
                    f'This image does not have a fixed projection, you need to specify a region, '
                    f'crs and scale.'
                )

        if not region and not self.footprint:
            raise UnsupportedValueError(f'This image does not have a footprint, you need to specify a region.')

        if self.crs == 'EPSG:4326' and not scale:
            # ee.Image.prepare_for_export() expects a scale in meters, but if the image is EPSG:4326, the default scale
            # is in degrees.
            raise UnsupportedValueError(f'This image is in EPSG:4326, you need to specify a scale in meters.')

        region = region or self.footprint  # TODO: test if this region is not in the download crs
        crs = crs or self.crs
        scale = scale or self.scale

        if crs == 'SR-ORG:6974':
            raise UnsupportedValueError(
                'There is an earth engine bug exporting in SR-ORG:6974, specify another CRS: '
                'https://issuetracker.google.com/issues/194561313'
            )

        resampling = ResamplingMethod(resampling)
        ee_image = self._ee_image
        if resampling != self._default_resampling:
            if not self.has_fixed_projection:
                raise UnsupportedValueError(
                    'This image has no fixed projection and cannot be resampled.  If this image is a composite, '
                    'you can resample the images used to create the composite.'
                )
            ee_image = ee_image.resample(resampling.value)

        ee_image = self._convert_dtype(ee_image, dtype=dtype or self.dtype)
        export_args = dict(region=region, crs=crs, scale=scale, fileFormat='GeoTIFF', filePerBand=False)
        ee_image, _ = ee_image.prepare_for_export(export_args)
        return BaseImage(ee_image)

    def _prepare_for_download(self, set_nodata=True, **kwargs) -> ('BaseImage', Dict):
        """
        Prepare the encapsulated image for tiled GeoTIFF download. Will reproject, resample, clip and convert the image
        according to the provided parameters.

        Returns the prepared image and a rasterio profile for the download GeoTIFF.
        """
        # resample, convert, clip and reproject image according to download params
        exp_image = self._prepare_for_export(**kwargs)
        nodata_dict = dict(
            float32=self._float_nodata,  # see workaround note in Tile.download(...)
            float64=self._float_nodata,  # ditto
            uint8=0,
            int8=np.iinfo('int8').min,
            uint16=0,
            int16=np.iinfo('int16').min,
            uint32=0,
            int32=np.iinfo('int32').min
        )
        nodata = nodata_dict[exp_image.dtype] if set_nodata else None
        profile = dict(
            driver='GTiff', dtype=exp_image.dtype, nodata=nodata, width=exp_image.shape[1],
            height=exp_image.shape[0], count=exp_image.count, crs=CRS.from_string(exp_image.crs),
            transform=exp_image.transform, compress='deflate', interleave='band', tiled=True
        )
        return exp_image, profile

    def _get_tile_shape(
        self, exp_image: 'BaseImage', max_download_size=32 << 20, max_grid_dimension=10000
    ) -> (Tuple[int, int], int):
        """
        Return a tile shape and number of tiles for a given BaseImage, such that the tile shape satisfies GEE
        download limits, and is 'square-ish'.
        """

        # find the total number of tiles we must divide the image into to satisfy max_download_size
        image_shape = exp_image.shape
        dtype_size = np.dtype(exp_image.dtype).itemsize
        image_size = exp_image.byte_size
        if exp_image.dtype.endswith('int8'):
            # workaround for GEE overestimate of *int8 dtype download sizes
            dtype_size *= 2
            image_size *= 2

        # here ceil_size is the worst case extra tile size due to np.ceil(image_shape / shape_num_tiles).astype('int')
        ceil_size = (image_shape[0] + image_shape[1]) * exp_image.count * dtype_size

        # TODO: the below is an approx and there is still the chance of tile size > max_download_size in unusual cases
        # adjust worst case ceil_size for the approx number of tiles in this case
        init_num_tiles = max(1, np.floor(image_size / max_download_size))
        ceil_size = ceil_size / np.sqrt(init_num_tiles)

        #  the total tile download size (tds) should be <= max_download_size, and
        #   tds <= image_size/num_tiles + ceil_size, which gives us:
        num_tiles = np.ceil(image_size / (max_download_size - ceil_size))

        def is_prime(x: int) -> bool:
            """Return True if x is prime else False."""
            for d in range(2, int(x ** 0.5) + 1):
                if x % d == 0:
                    return False
            return True

        # increment num_tiles if it is prime (This is so that we can factorize num_tiles into x & y dimension
        # components, and don't have all tiles along a single dimension.
        if num_tiles > 4 and is_prime(num_tiles):
            num_tiles += 1

        def factors(x: int) -> np.ndarray:
            """Return a Nx2 array of factors of x."""
            facts = np.arange(1, x + 1)
            facts = facts[np.mod(x, facts) == 0]
            return np.vstack((facts, x / facts)).transpose()

        # factorise num_tiles into the number of tiles down x,y axes
        fact_num_tiles = factors(num_tiles)

        # choose the factors that produce the most square-ish tile shape
        fact_aspect_ratios = fact_num_tiles[:, 0] / fact_num_tiles[:, 1]
        image_aspect_ratio = image_shape[0] / image_shape[1]
        fact_idx = np.argmin(np.abs(fact_aspect_ratios - image_aspect_ratio))
        shape_num_tiles = fact_num_tiles[fact_idx, :]

        # find the tile shape and clip to max_grid_dimension if necessary
        tile_shape = np.ceil(np.array(image_shape) / shape_num_tiles).astype('int')
        tile_shape[tile_shape > max_grid_dimension] = max_grid_dimension
        tile_shape = tuple(tile_shape.tolist())
        return tile_shape, num_tiles

    @staticmethod
    def _build_overviews(dataset: rio.io.DatasetWriter, max_num_levels=8, min_ovw_pixels=256):
        """Build internal overviews, downsampled by successive powers of 2, for an open rasterio dataset."""
        if dataset.closed:
            raise IoError('Image dataset is closed')

        # limit overviews so that the highest level has at least 2**8=256 pixels along the shortest dimension,
        # and so there are no more than 8 levels.
        max_ovw_levels = int(np.min(np.log2(dataset.shape)))
        min_level_shape_pow2 = int(np.log2(min_ovw_pixels))
        num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
        ovw_levels = [2 ** m for m in range(1, num_ovw_levels + 1)]
        dataset.build_overviews(ovw_levels, RioResampling.average)

    def _write_metadata(self, dataset: rio.io.DatasetWriter):
        """Write EE and geedim image metadata to an open rasterio dataset."""
        if dataset.closed:
            raise IoError('Image dataset is closed')

        dataset.update_tags(**self.properties)
        # populate band metadata
        for band_i, band_info in enumerate(self.band_metadata):
            if 'id' in band_info:
                dataset.set_band_description(band_i + 1, band_info['id'])
            dataset.update_tags(band_i + 1, **band_info)

    def tiles(self, exp_image, tile_shape=None):
        """
        Iterator over downloadable image tiles.

        Divides an image into adjoining tiles no bigger than `tile_shape`.

        Parameters
        ----------
        exp_image: BaseImage
            The image to tile.
        tile_shape: Tuple[int, int], optional
            The (row, column) tile shape to use (pixels) [default: calculate an auto tile shape that satisfies the EE
            download size limit.]

        Yields
        -------
        tile: Tile
            An image tile that can be downloaded.
        """
        if not tile_shape:
            tile_shape, num_tiles = self._get_tile_shape(exp_image)

        # split the image up into tiles of at most `tile_shape` dimension
        image_shape = exp_image.shape
        start_range = product(range(0, image_shape[0], tile_shape[0]), range(0, image_shape[1], tile_shape[1]))
        for tile_start in start_range:
            tile_stop = np.clip(np.add(tile_start, tile_shape), a_min=None, a_max=image_shape)
            clip_tile_shape = (tile_stop - tile_start).tolist()  # tolist is just to convert to native int
            tile_window = Window(tile_start[1], tile_start[0], clip_tile_shape[1], clip_tile_shape[0])
            yield Tile(exp_image, tile_window)

    @staticmethod
    def monitor_export_task(task, label=None):
        """
        Monitor and display the progress of an export task.

        Parameters
        ----------
        task : ee.batch.Task
            EE task to monitor.
        label: str, optional
            Optional label for progress display [default: use task description].
        """
        pause = 0.1
        status = ee.data.getOperation(task.name)

        if label is None:
            label = status["metadata"]["description"]
        label = label if (len(label) < BaseImage._desc_width) else f'*{label[-BaseImage._desc_width:]}'

        class Spin(threading.Thread):
            stop = False

            def run(self):
                """Wait for export preparation to complete, displaying a spin toggle"""
                with Spinner(f'Preparing {label}: ') as spinner:
                    while 'progress' not in status['metadata'] and not self.stop:
                        time.sleep(pause)
                        spinner.next()
                    spinner.writeln(f'Preparing {label}: done') if not self.stop else spinner.writeln('')

        # run the spinner in a separate thread so it does not hang on EE calls
        spin_thread = Spin()
        spin_thread.start()
        try:
            # poll EE until the export preparation is complete
            while 'progress' not in status['metadata']:
                time.sleep(5 * pause)
                status = ee.data.getOperation(task.name)
            spin_thread.join()
        except KeyboardInterrupt:
            spin_thread.stop = True
            raise

        # wait for export to complete, displaying a progress bar
        bar_format = '{desc}: |{bar}| [{percentage:5.1f}%] in {elapsed:>5s} (eta: {remaining:>5s})'
        with tqdm(desc=f'Exporting {label}', total=1, bar_format=bar_format, dynamic_ncols=True) as bar:
            while ('done' not in status) or (not status['done']):
                time.sleep(10 * pause)
                status = ee.data.getOperation(task.name)  # get task status
                bar.update(status['metadata']['progress'] - bar.n)

        if status['metadata']['state'] != 'SUCCEEDED':
            raise IoError(f"Export failed \n{status}")

    def export(self, filename, folder='', wait=True, **kwargs):
        """
        Export the encapsulated image to Google Drive.

        Parameters
        ----------
        filename : str
            The name of the task and destination file.
        folder : str, optional
            Google Drive folder to export to [default: root].
        wait : bool
            Wait for the export to complete before returning [default: True].
        kwargs: optional

            region : dict, geojson, ee.Geometry, optional
                Region of interest (WGS84) to export [default: export the entire image footprint if it has one].
            crs : str, optional
                WKT, EPSG etc specification of CRS to export to.  Where image bands have different CRSs, all are
                re-projected to this CRS.
                [default: use the CRS of the minimum scale band if available].
            scale : float, optional
                Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this
                scale. [default: use the minimum scale of image bands if available].
            resampling : ResamplingMethod, optional
                Resampling method: ('near'|'bilinear'|'bicubic') [default: 'near'].
            dtype: str, optional
                Data type to export to ('uint8'|'int8'|'uint16'|'int16'|'uint32'|'int32'|'float32'|'float64')
                [default: auto select a minimal type].
        """

        exp_image = self._prepare_for_export(**kwargs)

        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f'Uncompressed size: {self._str_format_size(exp_image.byte_size)}')

        # create export task and start
        task = ee.batch.Export.image.toDrive(
            image=exp_image.ee_image, description=filename[:100], folder=folder, fileNamePrefix=filename, maxPixels=1e9
        )
        task.start()
        if wait:  # wait for completion
            self.monitor_export_task(task)
        return task

    def download(self, filename: pathlib.Path, overwrite=False, num_threads=None, **kwargs):
        """
        Download the encapsulated image to a GeoTiff file.

        There is no size limit on the EE image - it is split into tiles, and re-assembled locally, to work around the
        EE download size limit.

        Parameters
        ----------
        filename: pathlib.Path, str
            Name of the destination file.
        overwrite : bool, optional
            Overwrite the destination file if it exists, otherwise prompt the user [default: True].
        num_threads: int, optional
            Number of tiles to download concurrently [default: use a sensible auto value].
        kwargs: optional

            region : dict, geojson, ee.Geometry, optional
                Region of interest (WGS84) to export [default: export the entire image footprint if it has one].
            crs : str, optional
                WKT, EPSG etc specification of CRS to export to.  Where image bands have different CRSs, all are
                re-projected to this CRS.
                [default: use the CRS of the minimum scale band if available].
            scale : float, optional
                Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this
                scale. [default: use the minimum scale of image bands if available].
            resampling : ResamplingMethod, optional
                Resampling method: ('near'|'bilinear'|'bicubic') [default: 'near'].
            dtype: str, optional
                Data type to export to ('uint8'|'int8'|'uint16'|'int16'|'uint32'|'int32'|'float32'|'float64')
                [default: auto select a minimal type].
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
        tile_shape, num_tiles = self._get_tile_shape(exp_image)

        # find raw size of the download data (less than the actual download size as the image data is zipped in a
        # compressed geotiff)
        raw_download_size = exp_image.byte_size
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
        desc = filename.name if (len(filename.name) < self._desc_width) else f'*{filename.name[-self._desc_width:]}'
        bar_format = (
            '{desc}: |{bar}| {n_fmt}/{total_fmt} (raw) [{percentage:5.1f}%] in {elapsed:>5s} (eta: {remaining:>5s})'
        )
        bar = tqdm(
            desc=desc, total=raw_download_size, bar_format=bar_format, dynamic_ncols=True, unit_scale=True, unit='B'
        )

        session = _requests_retry_session(5, status_forcelist=[500, 502, 503, 504])
        warnings.filterwarnings('ignore', category=TqdmWarning)
        redir_tqdm = logging_redirect_tqdm([logging.getLogger(__package__)])  # redirect logging through tqdm
        out_ds = rio.open(filename, 'w', **profile)  # create output geotiff

        with redir_tqdm, rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), out_ds, bar:
            def download_tile(tile):
                """Download a tile and write into the destination GeoTIFF."""
                tile_array = tile.download(session=session, bar=bar)
                with out_lock:
                    out_ds.write(tile_array, window=tile.window)

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                # Run the tile downloads in a thread pool
                tiles = self.tiles(exp_image, tile_shape=tile_shape)
                futures = [executor.submit(download_tile, tile) for tile in tiles]
                try:
                    for future in as_completed(futures):
                        future.result()
                except Exception as ex:
                    logger.info('Cancelling...')
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise ex

            # populate GeoTIFF metadata and build overviews
            self._write_metadata(out_ds)
            self._build_overviews(out_ds)
