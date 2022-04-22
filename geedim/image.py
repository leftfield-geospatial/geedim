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
# Functions to download and export Earth Engine images
import collections
import logging
import os
import pathlib
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Tuple, Dict, List

import ee
import numpy as np
import pandas as pd
import rasterio as rio
from pip._vendor.progress.spinner import Spinner
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform_geom
from rasterio.windows import Window
from tqdm import TqdmWarning
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geedim import info
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
    tuple
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
    filename :  str, pathlib.Path
                Path of the image file whose bounds to find.
    expand :    int
                Percentage (0-100) by which to expand the bounds (default: 5).

    Returns
    -------
    bounds : dict
             Geojson polygon.
    crs : str
          Image CRS as EPSG string.
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
            src_bbox_wgs84 = transform_geom(im.crs, "WGS84", bbox_expand_dict)  # convert to WGS84 geojson
    finally:
        logging.getLogger("rasterio").setLevel(logging.WARNING)

    image_bounds = collections.namedtuple('ImageBounds', ['bounds', 'crs'])
    return image_bounds(src_bbox_wgs84, im.crs.to_epsg())


class BaseImage:
    """
    Base class for encapsulating an EE image.

    Provides client-side access to metadata, download and export functionality.
    """
    _float_nodata = float('nan')
    _desc_width = 70
    _footprint_key = 'system:footprint'
    _default_resampling = 'near'
    _supported_collection_ids = ['*']

    def __init__(self, ee_image: ee.Image, num_threads=None, **kwargs):
        # TODO: get rid of kwargs, rather pass num_threads to download, and in MaskedImage, remove internal score band
        #  and associated cloud_dist argument, and remove mask from constructor, rather make a method to apply it

        if not isinstance(ee_image, ee.Image):
            raise TypeError('ee_image must be an instance of ee.Image')
        self._ee_image = ee_image
        self._ee_info = None
        self._id = None
        self._min_projection = None
        self._min_dtype = None
        # TODO: some collections e.g. LANDSAT/LC08/C01/T1_8DAY_EVI, won't allow get('system:id')
        self._ee_coll_name = ee.String(ee_image.get('system:id')).split('/').slice(0, -1).join('/')
        self._out_lock = threading.Lock()
        self._max_threads = num_threads or min(32, (os.cpu_count() or 1) + 4)

    @classmethod
    def from_id(cls, image_id, **kwargs):
        """
        Construct a *Image object from an EE image ID.

        Parameters
        ----------
        image_id : str
           ID of earth engine image to wrap.
        kwargs : optional
            Any keyword arguments to pass to cls.__init__()

        Returns
        -------
        gd_image: BaseImage
            The image object.
        """
        ee_coll_name = split_id(image_id)[0]
        if (cls._supported_collection_ids != ['*']) and (ee_coll_name not in cls._supported_collection_ids):
            raise ValueError(f"Unsupported collection: {ee_coll_name}.  "
                             f"{cls.__name__} supports images from {cls._supported_collection_ids}")
        ee_image = ee.Image(image_id)
        gd_image = cls(ee_image, **kwargs)
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
    def ee_info(self) -> Dict:
        """The EE image metadata in a dict."""
        if self._ee_info is None:
            self._ee_info = self._ee_image.getInfo()
        return self._ee_info

    @property
    def properties(self) -> Dict:
        return self.ee_info['properties'] if 'properties' in self.ee_info else None

    @property
    def id(self) -> str:
        """The EE image ID."""
        return self._id or self.ee_info["id"]  # avoid a call to getInfo() if _id is set

    @property
    def name(self) -> str:
        """The image name (the ID with slashes replaces by dashes)."""
        return self.id.replace('/', '-')

    @property
    def collection_id(self) -> str:
        """The EE collection ID for this image."""
        return split_id(self.id)[0]

    @property
    def min_projection(self) -> Dict:
        """A dict of the projection information corresponding to the minimum scale band."""
        if not self._min_projection:
            self._min_projection = self._get_projection(self.ee_info, min=True)
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
        """
        The scale (m) corresponding to minimum scale band.
        Will return None if the image has no fixed projection.
        """
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
        """The number of image bands"""
        return len(self.ee_info['bands']) if 'bands' in self.ee_info else None

    @property
    def transform(self) -> rio.Affine:
        """
        The geo-transform of the minimim scale band, as a rasterio Affine transform.
        Will return None if the image has no fixed projection.
        """
        return self.min_projection['transform']

    @property
    def has_fixed_projection(self) -> bool:
        return self.scale is not None

    @property
    def dtype(self) -> str:
        """The minimal size data type required to represent the values in this image."""
        if not self._min_dtype:
            self._min_dtype = self._get_min_dtype(self.ee_info)
        return self._min_dtype

    @property
    def footprint(self) -> Dict:
        """A geojson polygon of the image extent."""
        if 'system:footprint' not in self.ee_info['properties']:
            return None
        return self.ee_info['properties'][self._footprint_key]

    @property
    def band_metadata(self) -> List:
        """A list of dicts describing the image bands."""
        return self._get_band_metadata(self.ee_info)

    @property
    def info(self) -> Dict:
        """All the BasicImage properties packed into a dictionary"""
        return dict(id=self.id, properties=self.properties, footprint=self.footprint, bands=self.band_metadata,
                    **self.min_projection)

    @staticmethod
    def human_size(bytes, units=['bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']):
        """
        Returns a human readable string representation of bytes -
        see https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
        """
        return f'{bytes:.2f} {units[0]}' if bytes < 1024 else BaseImage.human_size(bytes / 1000, units[1:])

    @staticmethod
    def _get_projection(ee_info: Dict, min=True) -> Dict:
        """
        Return the projection information corresponding to the min/max scale band, for an EE image info dictionary
        """
        projection_info = dict(crs=None, transform=None, shape=None, scale=None)
        if 'bands' in ee_info:
            # get scale & crs corresponding to min/max scale band (exclude 'EPSG:4326' (composite/constant) bands)
            band_df = pd.DataFrame(ee_info['bands'])
            scales = pd.DataFrame(band_df['crs_transform'].tolist())[0].abs().astype(float)
            band_df['scale'] = scales
            filt_band_df = band_df[~((band_df.crs == 'EPSG:4326') & (band_df.scale == 1))]
            if filt_band_df.shape[0] > 0:
                idx = filt_band_df.scale.idxmin() if min else filt_band_df.scale.idxmax()
                sel_band_df = filt_band_df.loc[idx]
                projection_info['crs'], projection_info['scale'] = sel_band_df[['crs', 'scale']]
                if 'dimensions' in sel_band_df:
                    projection_info['shape'] = sel_band_df['dimensions'][::-1]
                projection_info['transform'] = rio.Affine(*sel_band_df['crs_transform'])
                if ('origin' in sel_band_df) and not np.any(np.isnan(sel_band_df['origin'])):
                    projection_info['transform'] *= rio.Affine.translation(*sel_band_df['origin'])
        return projection_info

    @staticmethod
    def _get_min_dtype(ee_info: Dict):
        """Return the minimal size data type for an EE image info dictionary"""
        dtype = None
        if 'bands' in ee_info:
            band_df = pd.DataFrame(ee_info['bands'])
            dtype_df = pd.DataFrame(band_df.data_type.tolist(), index=band_df.id)
            if all(dtype_df.precision == 'int'):
                dtype_min = dtype_df['min'].min()  # minimum image value
                dtype_max = dtype_df['max'].max()  # maximum image value

                # determine the number of integer bits required to represent the value range
                bits = 0
                for bound in [abs(dtype_max), abs(dtype_min)]:
                    bound_bits = 0 if bound == 0 else 2 ** np.ceil(np.log2(np.log2(abs(bound))))
                    bits += bound_bits
                bits = min(max(bits, 8), 32)  # clamp bits to allowed values
                dtype = f'{"u" if dtype_min >= 0 else ""}int{int(bits)}'
            elif any(dtype_df.precision == 'double'):
                dtype = 'float64'
            else:
                dtype = 'float32'
        return dtype

    @staticmethod
    def _get_band_metadata(ee_info) -> List[Dict,]:
        """Return band metadata given an EE image info dict."""
        ee_coll_name, _ = split_id(ee_info['id'])
        band_df = pd.DataFrame(ee_info['bands'])
        if ee_coll_name in info.collection_info:  # include SR band metadata if it exists
            # use DataFrame to concat SR band metadata from collection_info with band IDs from the image
            sr_band_list = info.collection_info[ee_coll_name]["bands"].copy()
            sr_band_dict = {bdict['id']: bdict for bdict in sr_band_list}
            band_metadata = [sr_band_dict[id] if id in sr_band_dict else dict(id=id) for id in band_df.id]
        else:  # just use the image band IDs
            band_metadata = band_df[["id"]].to_dict("records")
        return band_metadata

    def _convert_dtype(self, ee_image, dtype):
        """
        Converts the data type of an image.

        Parameters
        ----------
        ee_image: ee.Image
            The image to convert.
        dtype: str
            The data type to convert the image to, as a valid rasterio dtype string.

        Returns
        -------
        ee_image: ee.Image
            The converted image.
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
        )
        if dtype not in conv_dict:
            raise ValueError(f'Unsupported dtype: {dtype}')

        return conv_dict[dtype](ee_image)

    def _prepare_for_export(self, region=None, crs=None, scale=None, resampling=_default_resampling, dtype=None):
        """
        Prepare the encapsulated image for export to Google Drive.  Will reproject, resample, clip and convert the image
        according to the provided parameters.

        Returns the prepared image, and it's data type.
        Parameters
        ----------
        region : dict, geojson, ee.Geometry, optional
            Region of interest (WGS84) to export (default: export the entire image granule if it has one).
        crs : str, optional
            WKT, EPSG etc specification of CRS to export to.  Where image bands have different CRSs, all are
            re-projected to this CRS.
            (default: use the CRS of the minimum scale band if available).
        scale : float, optional
            Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this scale.
            (default: use the minimum scale of image bands if available).
        resampling : str, optional
            Resampling method: ("near"|"bilinear"|"bicubic") (default: "near")
        dtype: str, optional
            Data type to export to ("uint8"|"int8"|"uint16"|"int16"|"uint32"|"int32"|"float32"|"float64")
            (default: auto select a minimal type)

        Returns
        -------
        exp_image: BaseImage
            The prepared image.
        """

        if not region or not crs or not scale:
            # One or more of region, crs and scale were not provided, so get the image values to use instead
            if not self.scale:
                # Raise an error if this image is a composite (or similar)
                raise ValueError(f'This image does not have a fixed projection, you need to specify a region, '
                                 f'crs and scale.')
        if not region and not self.footprint:
            raise ValueError(f'This image does not have a footprint, you need to specify a region.')

        region = region or self.footprint  # TODO: test if this region is not in the download crs
        crs = crs or self.crs
        scale = scale or self.scale

        if crs == 'SR-ORG:6974':
            raise ValueError(
                'There is an earth engine bug exporting in SR-ORG:6974, specify another CRS: '
                'https://issuetracker.google.com/issues/194561313'
            )

        ee_image = self._ee_image.resample(resampling) if resampling != self._default_resampling else self._ee_image
        ee_image = self._convert_dtype(ee_image, dtype=dtype or self.dtype)
        export_args = dict(region=region, crs=crs, scale=scale, fileFormat='GeoTIFF', filePerBand=False)
        ee_image, _ = ee_image.prepare_for_export(export_args)
        return BaseImage(ee_image)

    def _prepare_for_download(self, set_nodata=True, **kwargs) -> ('BaseImage', Dict):
        """
        Prepare the encapsulated image for tiled downloading to local GeoTIFF. Will reproject, resample, clip and
        convert the image according to the provided parameters.

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
        profile = dict(driver='GTiff', dtype=exp_image.dtype, nodata=nodata, width=exp_image.shape[1],
                       height=exp_image.shape[0],
                       count=exp_image.count, crs=CRS.from_string(exp_image.crs), transform=exp_image.transform,
                       compress='deflate', interleave='band', tiled=True)
        return exp_image, profile

    @staticmethod
    def _get_image_size(exp_image: 'BaseImage'):
        dtype_size = np.dtype(exp_image.dtype).itemsize
        return exp_image.shape[0] * exp_image.shape[1] * exp_image.count * dtype_size


    def _get_tile_shape(self, exp_image: 'BaseImage', max_download_size=32<<20,
                        max_grid_dimension=10000) -> (Tuple[int, int], int):
        """Return a tile shape for provided BaseImage that satisfies GEE download limits, and is 'square-ish'."""

        # find the total number of tiles we must divide the image into to satisfy max_download_size
        image_shape = np.int64(exp_image.shape)
        dtype_size = np.dtype(exp_image.dtype).itemsize
        if exp_image.dtype.endswith('int8'):
            dtype_size *= 2  # workaround for GEE overestimate of *int8 dtype download sizes

        image_size = self._get_image_size(exp_image)
        # ceil_size is the worst case extra tile size due to np.ceil(image_shape / shape_num_tiles).astype('int')
        ceil_size = (image_shape[0] + image_shape[1]) * exp_image.count * dtype_size
        #  the total tile download size (tds) should be <= max_download_size, and
        #   tds <= image_size/num_tiles + ceil_size, which gives us:
        num_tiles = np.ceil(image_size / (max_download_size - ceil_size))

        # increment num_tiles if it is prime (This is so that we can factorize num_tiles into x & y dimension
        # components, and don't have all tiles along a single dimension.
        def is_prime(x):
            for d in range(2, int(x ** 0.5) + 1):
                if x % d == 0:
                    return False
            return True

        if num_tiles > 4 and is_prime(num_tiles):
            num_tiles += 1

        # factorise num_tiles into the number of tiles down x,y axes
        def factors(x):
            facts = np.arange(1, x + 1)
            facts = facts[np.mod(x, facts) == 0]
            return np.vstack((facts, x / facts)).transpose()

        fact_num_tiles = factors(num_tiles)

        # choose the factors that produce a near-square tile shape
        fact_aspect_ratios = fact_num_tiles[:, 0] / fact_num_tiles[:, 1]
        image_aspect_ratio = image_shape[0] / image_shape[1]
        fact_idx = np.argmin(np.abs(fact_aspect_ratios - image_aspect_ratio))
        shape_num_tiles = fact_num_tiles[fact_idx, :]

        # find the tile shape and clip to max_grid_dimension if necessary
        tile_shape = np.ceil(image_shape / shape_num_tiles).astype('int')
        tile_shape[tile_shape > max_grid_dimension] = max_grid_dimension
        tile_shape = tuple(tile_shape.tolist())
        return tile_shape, num_tiles

    def _build_overviews(self, dataset: rio.io.DatasetWriter, max_num_levels=8, min_ovw_pixels=256):
        """Build internal overviews, downsampled by successive powers of 2, for an open rasterio dataset."""
        if dataset.closed:
            raise IOError('Image dataset is closed')

        # limit overviews so that the highest level has at least 2**8=256 pixels along the shortest dimension,
        # and so there are no more than 8 levels.
        max_ovw_levels = int(np.min(np.log2(dataset.shape)))
        min_level_shape_pow2 = int(np.log2(min_ovw_pixels))
        num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
        ovw_levels = [2 ** m for m in range(1, num_ovw_levels + 1)]
        dataset.build_overviews(ovw_levels, Resampling.average)

    def _write_metadata(self, dataset: rio.io.DatasetWriter):
        """Write GEE and geedim image metadata to an open rasterio dataset"""
        if dataset.closed:
            raise IOError('Image dataset is closed')

        dataset.update_tags(**self.properties)
        # populate band metadata
        for band_i, band_info in enumerate(self.band_metadata):
            if 'id' in band_info:
                dataset.set_band_description(band_i + 1, band_info['id'])
            dataset.update_tags(band_i + 1, **band_info)

    def tiles(self, exp_image: 'BaseImage', tile_shape=None):
        """
        Iterator over the image tiles.

        Yields:
        -------
        tile: DownloadTile
            A tile of the encapsulated image that can be downloaded.
        """
        if not tile_shape:
            tile_shape, num_tiles, download_size = self._get_tile_shape(exp_image)

        # split the image up into tiles of at most `tile_shape` dimension
        image_shape = exp_image.shape
        start_range = product(range(0, image_shape[0], tile_shape[0]), range(0, image_shape[1], tile_shape[1]))
        for tile_start in start_range:
            tile_stop = np.clip(np.add(tile_start, tile_shape), a_min=None, a_max=image_shape)
            clip_tile_shape = (tile_stop - tile_start).tolist()  # tolist is just to convert to native int
            tile_window = Window(tile_start[1], tile_start[0], clip_tile_shape[1], clip_tile_shape[0])
            yield Tile(exp_image, tile_window)

    @staticmethod
    def monitor_export_task(task, label: str = None):
        """
        Monitor and display the progress of an export task

        Parameters
        ----------
        task : ee.batch.Task
               EE task to monitor
        label: str, optional
               Optional label for progress display (default: use task description)
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
            raise IOError(f"Export failed \n{status}")

    def export(self, filename, folder='', wait=True, **kwargs):
        """
        Export the encapsulated image to Google Drive.

        Parameters
        ----------
        filename : str
                   The name of the task and destination file
        folder : str, optional
                 Google Drive folder to export to (default: root).
        wait : bool
               Wait for the export to complete before returning (default: True)
        kwargs:
            region : dict, geojson, ee.Geometry, optional
                Region of interest (WGS84) to export (default: export the entire image granule if it has one).
            crs : str, optional
                WKT, EPSG etc specification of CRS to export to.  Where image bands have different CRSs, all are
                re-projected to this CRS.
                (default: use the CRS of the minimum scale band if available).
            scale : float, optional
                Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this
                scale. (default: use the minimum scale of image bands if available).
            resampling : str, optional
                Resampling method: ("near"|"bilinear"|"bicubic") (default: "near")
            dtype: str, optional
                Data type to export to ("uint8"|"int8"|"uint16"|"int16"|"uint32"|"int32"|"float32"|"float64")
                (default: auto select a minimal type)
        """

        # TODO: test composite of resampled images and resampled composite
        exp_image = self._prepare_for_export(**kwargs)
        raw_download_size = self._get_image_size(exp_image)

        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f'Uncompressed size: {self.human_size(raw_download_size)}')

        # create export task and start
        task = ee.batch.Export.image.toDrive(image=exp_image.ee_image, description=filename[:100], folder=folder,
                                             fileNamePrefix=filename, maxPixels=1e9)
        task.start()
        if wait:  # wait for completion
            self.monitor_export_task(task)
        return task

    def download(self, filename: pathlib.Path, overwrite=False, **kwargs):
        """
        Download the encapsulated image to a GeoTiff file.

        There is no size limit on the EE image - it is split into tiles, and re-assembled locally, to work around the
        EE download size limit.

        Parameters
        ----------
        filename: pathlib.Path, str
           Name of the destination file.
        overwrite : bool, optional
            Overwrite the destination file if it exists, otherwise prompt the user (default: True)
        kwargs:
            region : dict, geojson, ee.Geometry, optional
                Region of interest (WGS84) to export (default: export the entire image granule if it has one).
            crs : str, optional
                WKT, EPSG etc specification of CRS to export to.  Where image bands have different CRSs, all are
                re-projected to this CRS.
                (default: use the CRS of the minimum scale band if available).
            scale : float, optional
                Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this
                scale. (default: use the minimum scale of image bands if available).
            resampling : str, optional
                Resampling method: ("near"|"bilinear"|"bicubic") (default: "near")
            dtype: str, optional
                Data type to export to ("uint8"|"int8"|"uint16"|"int16"|"uint32"|"int32"|"float32"|"float64")

        """

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
        raw_download_size = self._get_image_size(exp_image)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            dtype_size = np.dtype(exp_image.dtype).itemsize
            raw_tile_size = tile_shape[0] * tile_shape[1] * exp_image.count * dtype_size
            logger.debug(f'{filename.name}:')
            logger.debug(f'Uncompressed size: {self.human_size(raw_download_size)}')
            logger.debug(f'Num. tiles: {num_tiles}')
            logger.debug(f'Tile shape: {tile_shape}')
            logger.debug(f'Tile size: {self.human_size(int(raw_tile_size))}')

        if raw_download_size > 1e9:
            # warn if the download is large (>1GB)
            logger.warning(f'Consider adjusting `region`, `scale` and/or `dtype` to reduce the {filename.name}'
                           f' download size (raw: {self.human_size(raw_download_size)}).')

        # configure the progress bar to monitor raw/uncompressed download size
        desc = filename.name if (len(filename.name) < self._desc_width) else f'*{filename.name[-self._desc_width:]}'
        bar_format = ('{desc}: |{bar}| {n_fmt}/{total_fmt} (raw) [{percentage:5.1f}%] in {elapsed:>5s} '
                      '(eta: {remaining:>5s})')
        bar = tqdm(desc=desc, total=raw_download_size, bar_format=bar_format, dynamic_ncols=True,
                   unit_scale=True, unit='B')

        session = _requests_retry_session(5, status_forcelist=[500, 502, 503, 504])
        warnings.filterwarnings('ignore', category=TqdmWarning)
        redir_tqdm = logging_redirect_tqdm([logging.getLogger(__package__)])  # redirect logging through tqdm
        out_ds = rio.open(filename, 'w', **profile)  # create output geotiff

        with redir_tqdm, rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), out_ds, bar:
            def download_tile(tile):
                """Download a tile and write into the destination GeoTIFF."""
                tile_array = tile.download(session=session, bar=bar)
                with self._out_lock:
                    out_ds.write(tile_array, window=tile.window)

            with ThreadPoolExecutor(max_workers=self._max_threads) as executor:
                # Run the tile downloads in a thread pool
                tiles = self.tiles(exp_image, tile_shape=tile_shape)
                futures = [executor.submit(download_tile, tile) for tile in tiles]
                try:
                    for future in as_completed(futures):
                        future.result()
                except:
                    logger.info('Cancelling...')
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
            # TODO parse specific expections like ee.ee_exception.EEException: "Total request size (55039924 bytes) must be less than or equal to 50331648 bytes."

            # populate GeoTIFF metadata and build overviews
            self._write_metadata(out_ds)
            self._build_overviews(out_ds)
