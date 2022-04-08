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
import multiprocessing
import os
import pathlib
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Tuple, Dict

import ee
import numpy as np
import pandas as pd
import rasterio as rio
from pip._vendor.progress.spinner import Spinner
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform_geom
from rasterio.windows import Window
from tqdm import tqdm, TqdmWarning

from geedim import info
from geedim.tile import _requests_retry_session, Tile

"""EE image property key for image region"""
_footprint_key = "system:footprint"
"""Default EE image resampling method"""
_default_resampling = 'near'


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


def get_info(ee_image, min=True):
    """
    Retrieve Earth Engine image metadata

    Parameters
    ----------
    ee_image : ee.Image
               The image whose information to retrieve.
    min : bool, optional
          Retrieve the crs, crs_transform & scale corresponding to the band with the minimum (True) or maximum (False)
          scale.(default: True)

    Returns
    -------
    dict
        Dictionary of image information with 'id', 'properties', 'bands', 'crs' and 'scale' keys.
    """
    # TODO: lose the need for ee_info below, perhaps change Image class to return ee_info with its .info property
    gd_info = dict(id=None, properties={}, bands=[], crs=None, crs_transform=None, scale=None, dimensions=None,
                   footprint=None, ee_info=None)
    ee_info = ee_image.getInfo()  # retrieve image info from cloud
    gd_info['ee_info'] = ee_info

    if "id" in ee_info:
        gd_info["id"] = ee_info["id"]

    if "properties" in ee_info:
        gd_info["properties"] = ee_info["properties"]
        if 'system:footprint' in ee_info["properties"]:
            gd_info['footprint'] = ee_info['properties']['system:footprint']

    if "bands" in ee_info:
        # get scale & crs corresponding to min/max scale band (exclude 'EPSG:4326' (composite/constant) bands)
        band_df = pd.DataFrame(ee_info["bands"])
        scales = pd.DataFrame(band_df["crs_transform"].tolist())[0].abs().astype(float)
        band_df["scale"] = scales
        filt_band_df = band_df[~((band_df.crs == "EPSG:4326") & (band_df.scale == 1))]
        if filt_band_df.shape[0] > 0:
            idx = filt_band_df.scale.idxmin() if min else filt_band_df.scale.idxmax()
            sel_band_df = filt_band_df.loc[idx]
            gd_info['crs'], gd_info['scale'] = sel_band_df[['crs', 'scale']]
            if 'dimensions' in sel_band_df:
                gd_info['dimensions'] = sel_band_df['dimensions'][::-1]
            gd_info['crs_transform'] = rio.Affine(*sel_band_df['crs_transform'])
            if ('origin' in sel_band_df) and not np.any(np.isnan(sel_band_df['origin'])):
                gd_info['crs_transform'] *= rio.Affine.translation(*sel_band_df['origin'])


        # populate band metadata
        ee_coll_name = split_id(str(gd_info["id"]))[0]
        if ee_coll_name in info.ee_to_gd:  # include SR band metadata if it exists
            # TODO: don't assume all the bands are in band_df, there could have been a select
            # use DataFrame to concat SR band metadata from collection_info with band IDs from the image
            sr_band_list = info.collection_info[ee_coll_name]["bands"].copy()
            sr_band_dict = {bdict['id']: bdict for bdict in sr_band_list}
            gd_info["bands"] = [sr_band_dict[id] if id in sr_band_dict else dict(id=id) for id in band_df.id]
        else:  # just use the image band IDs
            gd_info["bands"] = band_df[["id"]].to_dict("records")

    return gd_info


class BaseImage:
    """
    Base class for encapsulating an EE image.

    Provides access to metadata, download and export functionality.
    """
    float_nodata = float('nan')

    def __init__(self, image: ee.Image, num_threads=None):

        if not isinstance(image, ee.Image):
            raise TypeError('image must be an instance of ee.Image')
        self._ee_image = image
        self._ee_coll_name = ee.String(image.get('system:id')).split('/').slice(0, -1).join('/')
        self._info = None
        self._out_lock = threading.Lock()
        self._num_threads = num_threads or max(multiprocessing.cpu_count(), 4)

    @property
    def ee_image(self) -> ee.Image:
        """The encapsulated EE image."""
        return self._ee_image

    @property
    def info(self) -> Dict:
        """The image metadata in a dict."""
        if self._info is None:
            self._info = get_info(self._ee_image)
        return self._info

    @property
    def id(self) -> str:
        """The EE image ID."""
        return self.info["id"]

    @property
    def name(self) -> str:
        """The image name (the ID with slashes replaces by dashes)."""
        return self.id.replace('/', '-')

    @property
    def crs(self) -> str:
        """
        The image CRS corresponding to minimum scale band, as an EPSG string.
        Will return None if the image has no fixed projection.
        """
        return self.info["crs"]

    @property
    def scale(self) -> float:
        """
        The scale (m) corresponding to minimum scale band.
        Will return None if the image has no fixed projection.
        """
        return self.info["scale"]

    @property
    def dimensions(self) -> Tuple[int, int]:
        """
        The (row, column) dimensions of this image.
        Will return None if the image has no fixed projection.
        """
        return self.info["dimensions"]

    @property
    def transform(self) -> rio.Affine:
        """
        The geo-transform of this image as a rasterio Affine transform.
        Will return None if the image has no fixed projection.
        """
        return self.info["dimensions"]

    @property
    def footprint(self) -> Dict:
        """A geojson polygon of the image extent."""
        return self.info["footprint"]

    def _auto_dtype(self):
        """Return the minimum (lowest memory) data type to represent the values of the encapsulated image."""

        band_df = pd.DataFrame(self.info['ee_info']['bands'])
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

    def _convert_dtype(self, image, dtype=None):
        """
        Converts the data type of an image, choosing a minimal target type, if one is not specified.

        Parameters
        ----------
        image: ee.Image
            The image to convert.
        dtype: str, optional
            The data type to convert the image to, as a valid rasterio dtype string.  If not specified, a miminal
            dtype that can represent the `image` values will be chosen automatically.

        Returns
        -------
        image: ee.Image
            The converted image.
        dtype: str
            The automatically selected dtype
        """
        if dtype is None:
            dtype = self._auto_dtype()

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

        return conv_dict[dtype](image), dtype

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
        image: ee.Image
            The prepared image.
        dtype: str
            The dtype of the converted image.
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

        if crs == "SR-ORG:6974":
            raise ValueError(
                "There is an earth engine bug exporting in SR-ORG:6974, specify another CRS: "
                "https://issuetracker.google.com/issues/194561313"
            )

        ee_image = self._ee_image.resample(resampling) if resampling != _default_resampling else self._ee_image
        ee_image, dtype = self._convert_dtype(ee_image, dtype=dtype)
        export_args = dict(region=region, crs=crs, scale=scale, fileFormat='GeoTIFF', filePerBand=False)
        ee_image, _ = ee_image.prepare_for_export(export_args)
        return ee_image, dtype

    def _prepare_for_download(self, mask=True, **kwargs):
        """
        Prepare the encapsulated image for tiled downloading to local GeoTIFF. Will reproject, resample, clip and
        convert the image according to the provided parameters.

        Returns the prepared image and a rasterio profile for the download GeoTIFF.
        """
        # resample, convert, clip and reproject image according to download params
        ee_image, dtype = self._prepare_for_export(**kwargs)
        # _prepare_for_export adjusts crs, dimensions, crs_transform etc so get the latest image info for populating
        # the rasterio profile
        info = get_info(ee_image)

        # get transform, shape and band count of the prepared image to configure up the output image profile
        # band_info = info['bands'][0]  # all bands are same crs & scale after prepare_for_export

        shape = info['dimensions']
        count = len(info['bands'])
        nodata_dict = dict(
            float32=self.float_nodata,  # see workaround note in Tile.download(...)
            float64=self.float_nodata,  # ditto
            uint8=0,
            int8=np.iinfo('int8').min,
            uint16=0,
            int16=np.iinfo('int16').min,
            uint32=0,
            int32=np.iinfo('int32').min
        )
        nodata = nodata_dict[dtype] if mask else None
        profile = dict(driver='GTiff', dtype=dtype, nodata=nodata, width=shape[1], height=shape[0], count=count,
                       crs=CRS.from_string(info['crs']), transform=info['crs_transform'], compress='deflate',
                       interleave='band', tiled=True)
        return ee_image, profile

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

    def _get_tile_shape(self, profile: Dict, max_download_size=33554432, max_grid_dimension=10000) -> Tuple[int, int]:
        """Return a tile shape for the encapsulated image that satisfies GEE download limits and is near-square."""
        # find the total number of tiles we must divide the image into to satisfy max_download_size
        image_shape = np.int64((profile['height'], profile['width']))
        dtype_size = np.dtype(profile['dtype']).itemsize
        if profile['dtype'].endswith('int8'):
            dtype_size *= 2     # workaround for GEE overestimate of *int8 dtype download sizes

        image_size = np.prod(image_shape) * profile['count'] * dtype_size
        # ceil_size is the worst case extra tile size due to np.ceil(image_shape / shape_num_tiles).astype('int')
        ceil_size = np.sum(image_shape) * profile['count'] * dtype_size
        #  the total tile download size (tds) should be <= max_download_size, and
        #   tds <= image_size/num_tiles + ceil_size, which gives us:
        num_tiles = np.ceil(image_size / (max_download_size - ceil_size))

        # TODO: warn if num_tiles is big, and add debug message about the approx size of the file
        #  and use the tile_shape to report raw size downloaded rather than num tiles.

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
        return tuple(tile_shape.tolist())

    def tiles(self, image, profile):
        """
        Iterator over the image tiles.

        Yields:
        -------
        tile: DownloadTile
            A tile of the encapsulated image that can be downloaded.
        """
        tile_shape = self._get_tile_shape(profile)

        # split the image up into tiles of at most `tile_shape` dimension
        image_shape = (profile['height'], profile['width'])
        start_range = product(range(0, image_shape[0], tile_shape[0]), range(0, image_shape[1], tile_shape[1]))
        for tile_start in start_range:
            tile_stop = np.clip(np.add(tile_start, tile_shape), a_min=None, a_max=image_shape)
            clip_tile_shape = (tile_stop - tile_start).tolist()  # tolist is just to convert to native int
            tile_window = Window(tile_start[1], tile_start[0], clip_tile_shape[1], clip_tile_shape[0])
            yield Tile(image, profile['transform'], tile_window)

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
            label = f'{status["metadata"]["description"][:80]}'

        def spin():
            """Wait for export preparation to complete, displaying a spin toggle"""
            with Spinner(f'Preparing {label}: ') as spinner:
                while 'progress' not in status['metadata']:
                    time.sleep(pause)
                    spinner.next()
                spinner.writeln(f'Preparing {label}: done')

        # run the spinner in a separate thread so it does not hang on EE calls
        spin_thread = threading.Thread(target=spin)
        spin_thread.start()

        # poll EE until the export preparation is complete
        while 'progress' not in status['metadata']:
            time.sleep(5 * pause)
            status = ee.data.getOperation(task.name)
        spin_thread.join()

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
        image, _ = self._prepare_for_export(**kwargs)
        # create export task and start
        task = ee.batch.Export.image.toDrive(image=image, description=filename[:100], folder=folder,
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

        image, profile = self._prepare_for_download(**kwargs)
        session = _requests_retry_session(5, status_forcelist=[500, 502, 503, 504])

        # TODO avoid retrieving all the tiles up front - there can be 1000s
        tiles = list(self.tiles(image, profile))
        bar_format = '{desc}: |{bar}| {n:4.1f}/{total_fmt} tile(s) [{percentage:5.1f}%] in {elapsed:>5s} (eta: {remaining:>5s})'
        bar = tqdm(desc=filename.name, total=len(tiles), bar_format=bar_format, dynamic_ncols=True)
        out_ds = rio.open(filename, 'w', **profile)
        warnings.filterwarnings('ignore', category=TqdmWarning)

        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), out_ds, bar:
            def download_tile(tile):
                """Download a tile and write into the destination GeoTIFF."""
                tile_array = tile.download(session, bar)
                with self._out_lock:
                    out_ds.write(tile_array, window=tile.window)

            with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
                # Run the tile downloads in a thread pool
                futures = [executor.submit(download_tile, tile) for tile in tiles]
                try:
                    for future in as_completed(futures):
                        future.result()
                except:
                    # TODO: log message that we are cancelling download
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
            # TODO parse specific expections like ee.ee_exception.EEException: "Total request size (55039924 bytes) must be less than or equal to 50331648 bytes."

            # populate GeoTIFF metadata and build overviews
            out_ds.update_tags(**self.info['properties'])
            for band_i, band_info in enumerate(self.info['bands']):
                if 'id' in band_info:
                    out_ds.set_band_description(band_i + 1, band_info['id'])
                out_ds.update_tags(band_i + 1, **band_info)
            self._build_overviews(out_ds)
