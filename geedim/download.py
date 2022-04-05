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

import multiprocessing
import os
import pathlib
import threading
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from itertools import product
from typing import Tuple

import ee
import numpy as np
import pandas as pd
import rasterio as rio
import requests
from pip._vendor.progress.spinner import Spinner
from rasterio import Affine
from rasterio import MemoryFile
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.windows import Window
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm, TqdmWarning

from geedim import info


def requests_retry_session(
        retries=3,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504),
        session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

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
    gd_info = dict(id=None, properties={}, bands=[], crs=None, crs_transform=None, scale=None, footprint=None,
                   ee_info=None)
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
        filt_band_df = band_df[(band_df.crs != "EPSG:4326") & (band_df.scale != 1)]
        if filt_band_df.shape[0] > 0:
            idx = filt_band_df.scale.idxmin() if min else filt_band_df.scale.idxmax()
            gd_info["crs"], gd_info["scale"]  = filt_band_df.loc[idx, ["crs", "scale"]]
            # gd_info["crs_transform"] = Affine(*filt_band_df.loc[idx, "crs_transform"])
            # if 'origin' in filt_band_df and not np.isnan(filt_band_df.loc[idx, 'origin']):
            #     gd_info["crs_transform"] *= Affine.translation(*filt_band_df.loc[idx, 'origin'])

        # populate band metadata
        ee_coll_name = split_id(str(gd_info["id"]))[0]
        if ee_coll_name in info.ee_to_gd:  # include SR band metadata if it exists
            # TODO: don't assume all the bands are in band_df, there could have been a select
            # use DataFrame to concat SR band metadata from collection_info with band IDs from the image
            sr_band_list = info.collection_info[info.ee_to_gd[ee_coll_name]]["bands"].copy()
            sr_band_dict = {bdict['id']: bdict for bdict in sr_band_list}
            gd_info["bands"] = [sr_band_dict[id] if id in sr_band_dict else dict(id=id) for id in band_df.id]
        else:  # just use the image band IDs
            gd_info["bands"] = band_df[["id"]].to_dict("records")

    return gd_info

class TileDownload:
    def __init__(self, image: ee.Image, transform: Affine, window: Window):
        self._image = image
        self._window = window
        self._transform = transform * Affine.translation(window.col_off, window.row_off)
        self._shape = (window.height, window.width)

    @property
    def transform(self) -> Affine:
        return self._transform

    @property
    def shape(self) -> Affine:
        return self._shape

    @property
    def window(self) -> Window:
        return self._window

    def download(self, session, bar: tqdm):
        # TODO: get image crs from DownloadImage where it is found for the output dataset, rather than reget it here
        session = session if session else requests

        # get image download url
        url = self._image.getDownloadURL(
            dict(crs=self._image.projection().crs(), crs_transform=tuple(self._transform)[:6],
                 dimensions=self._shape[::-1], filePerBand=False, fileFormat='GeoTIFF'))

        # download zip into buffer
        zip_buffer = BytesIO()
        resp = session.get(url, stream=True)
        download_size = int(resp.headers.get('content-length', 0))
        for data in resp.iter_content(chunk_size=10240):
            zip_buffer.write(data)
            bar.update(len(data) / download_size)
        zip_buffer.flush()

        # extract geotiff from zipped buffer into another buffer
        zip_file = zipfile.ZipFile(zip_buffer)
        ext_buffer = BytesIO(zip_file.read(zip_file.filelist[0]))

        # read the geotiff with a rasterio memory file
        with MemoryFile(ext_buffer) as mem_file:
            with mem_file.open() as ds:
                array = ds.read()
                if (array.dtype == np.dtype('float32')) or (array.dtype == np.dtype('float64')):
                    # GEE sets nodata to -inf for float data types, (but does not populate the nodata field)
                    # rasterio won't allow nodata=-inf, so this is a workaround to change nodata to nan at source
                    array[np.isinf(array)] = np.nan

        return array

class Image(object):
    def __init__(self, ee_image):
        """
        Base class to wrap any ee.Image and provide access to metadata.

        Parameters
        ----------
        ee_image : ee.Image
                   Image to wrap.
        """
        if not isinstance(ee_image, ee.Image):
            raise TypeError('image must be an instance of ee.Image')
        self._ee_image = ee_image
        self._info = None

    @property
    def ee_image(self):
        """ ee.Image: The wrapped image. """
        return self._ee_image

    @property
    def info(self):
        """ dict: Image information as from get_info(). """
        if self._info is None:
            self._info = get_info(self._ee_image)
        return self._info

    @property
    def id(self):
        """ str: Earth Engine image ID. """
        return self.info["id"]

    @property
    def crs(self):
        """ str, None: Image CRS corresponding to minimum scale band, as EPSG string. None if all bands are in
        EPSG:4326. """
        return self.info["crs"]

    @property
    def scale(self):
        """ float, None: Scale (m) corresponding to minimum scale band.  None if all bands are in EPSG:4326. """
        return self.info["scale"]

    @property
    def footprint(self):
        """ geojson: Polygon of the image extent. """
        return self.info["footprint"]


class ImageDownload(Image):
    float_nodata = float('nan')
    def __init__(self, image: ee.Image):
        Image.__init__(self, ee_image=image)
        self._out_lock = threading.Lock()
        self._threads = max(multiprocessing.cpu_count(), 4)

    def _auto_dtype(self):
        band_df = pd.DataFrame(self.info['ee_info']['bands'])
        dtype_df = pd.DataFrame(band_df.data_type.tolist(), index=band_df.id)
        if all(dtype_df.precision == 'int'):
            # TODO: rasterio.dtypes.get_minimum_dtype
            dtype_min = dtype_df['min'].min()
            dtype_max = dtype_df['max'].max()
            bits = 0
            for bound in [abs(dtype_max), abs(dtype_min)]:
                bound_bits = 0 if bound == 0 else 2 ** np.ceil(np.log2(np.log2(abs(bound))))
                bits += bound_bits
            bits = min(max(bits, 8), 32)
            dtype = f'{"u" if dtype_min >= 0 else ""}int{int(bits)}'
        elif any(dtype_df.precision == 'double'):
            dtype = 'float64'
        else:
            dtype = 'float32'
        return dtype

    def _convert_dtype(self, image, dtype=None):
        if dtype is None:
            dtype = self._auto_dtype()

        # TODO: check the nodata vals that GEE uses for each dtype
        conv_dict = dict(
            float32=dict(conv=ee.Image.toFloat, nodata=self.float_nodata),
            float64=dict(conv=ee.Image.toDouble, nodata=self.float_nodata),
            uint8=dict(conv=ee.Image.toUint8, nodata=0),
            int8=dict(conv=ee.Image.toInt8, nodata=np.iinfo('int8').min),
            uint16=dict(conv=ee.Image.toUint16, nodata=0),
            int16=dict(conv=ee.Image.toInt16, nodata=np.iinfo('int16').min),
            uint32=dict(conv=ee.Image.toUint32, nodata=0),
            int32=dict(conv=ee.Image.toInt32, nodata=np.iinfo('int32').min),
            # int64=dict(conv=ee.Image.toInt64, nodata=np.iinfo('int64').min),
        )
        if dtype not in conv_dict:
            raise ValueError(f'Unsupported dtype: {dtype}')

        return conv_dict[dtype]['conv'](image), dtype, conv_dict[dtype]['nodata']

    def _prepare_for_export(self, region=None, crs=None, scale=None, resampling='near', dtype=None):
        if not region or not crs or not scale:
            # check if this image is a composite (also retrieve this image's min crs, min scale, and footprint)
            if not self.scale:
                raise ValueError(f'Image appears to be a composite, specify a region, crs and scale')
        region = region or self.footprint  # TODO: what happens if this is not in the download crs
        crs = crs or self.crs
        scale = scale or self.scale

        if crs == "SR-ORG:6974":
            raise ValueError(
                "There is an earth engine bug exporting in SR-ORG:6974, specify another CRS: "
                "https://issuetracker.google.com/issues/194561313"
            )
        ee_image = self._ee_image.resample(resampling) if resampling != 'near' else self._ee_image
        ee_image, dtype, nodata = self._convert_dtype(ee_image, dtype=dtype)
        export_args = dict(region=region, crs=crs, scale=scale, fileFormat='GeoTIFF', filePerBand=False)
        ee_image, _ = ee_image.prepare_for_export(export_args)
        return ee_image, dtype, nodata

    def _prepare_for_download(self, mask=True, **kwargs):
        # resample, convert, clip and reproject image according to download params
        ee_image, dtype, nodata = self._prepare_for_export(**kwargs)
        # _prepare_for_export adjusts crs, dimensions, crs_transform etc so get the latest image info
        info = ee_image.getInfo()

        # get transform, shape and band count of the prepared image to configure up the output image profile
        band_info = info['bands'][0]  # all bands are same crs & scale after prepare_for_export
        transform = Affine(*band_info['crs_transform'])
        if 'origin' in band_info:
            transform *= Affine.translation(*band_info['origin'])

        shape = band_info['dimensions'][::-1]
        count = len(info['bands'])
        profile = dict(driver='GTiff', dtype=dtype, nodata=nodata if mask else None, width=shape[1], height=shape[0],
                       count=count, crs=CRS.from_string(band_info['crs']), transform=transform, compress='deflate',
                       interleave='band', tiled=True)
        return ee_image, profile

    def _build_overviews(self, im, max_num_levels=8, min_level_pixels=256):
        """
        Build internal overviews, downsampled by successive powers of 2, for a rasterio dataset.

        Parameters
        ----------
        im: rasterio.io.DatasetWriter
            An open rasterio dataset to write the metadata to.
        max_num_levels: int, optional
            Maximum number of overview levels to build.
        min_level_pixels: int, pixel
            Minimum width/height (in pixels) of any overview level.
        """

        # limit overviews so that the highest level has at least 2**8=256 pixels along the shortest dimension,
        # and so there are no more than 8 levels.
        if im.closed:
            raise ValueError('Image dataset is closed')

        max_ovw_levels = int(np.min(np.log2(im.shape)))
        min_level_shape_pow2 = int(np.log2(min_level_pixels))
        num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
        ovw_levels = [2 ** m for m in range(1, num_ovw_levels + 1)]
        im.build_overviews(ovw_levels, Resampling.average)

    def _get_tile_shape(self, profile, max_download_size=33554432, max_grid_dimension=10000) -> Tuple[int, int]:
        """Internal function to find a tile shape that satisfies download limits and is roughly square"""
        # find the ttl number of tiles needed to satisfy max_download_size
        image_shape = (profile['height'], profile['width'])
        dtype_size = np.dtype(profile['dtype']).itemsize
        image_size = np.prod(np.int64((profile['count'], dtype_size, *image_shape)))
        num_tiles = np.ceil(image_size / max_download_size)

        # increment num_tiles if it is prime
        # (this is so that we can factorize num_tiles into x,y dimension components, and don't have all tiles along
        # one dimension)
        def is_prime(x):
            for d in range(2, int(x ** 0.5) + 1):
                if x % d == 0:
                    return False
            return True

        if num_tiles > 4 and is_prime(num_tiles):
            num_tiles += 1

        # factorise num_tiles into num of tiles down x,y axes
        def factors(x):
            facts = np.arange(1, x + 1)
            facts = facts[np.mod(x, facts) == 0]
            return np.vstack((facts, x / facts)).transpose()

        fact_num_tiles = factors(num_tiles)

        # choose the factors that produce a roughly square (as close as possible) tile shape
        fact_aspect_ratios = fact_num_tiles[:, 0] / fact_num_tiles[:, 1]
        aspect_ratio = image_shape[0] / image_shape[1]
        fact_idx = np.argmin(np.abs(fact_aspect_ratios - aspect_ratio))
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
            An image tile that can be downloaded.
        """
        tile_shape = self._get_tile_shape(profile)

        # split the image up into tiles of at most tile_shape
        image_shape = (profile['height'], profile['width'])
        start_range = product(range(0, image_shape[0], tile_shape[0]), range(0, image_shape[1], tile_shape[1]))
        for tile_start in start_range:
            tile_stop = np.clip(np.add(tile_start, tile_shape), a_min=None, a_max=image_shape)
            clip_tile_shape = (tile_stop - tile_start).tolist()  # tolist is just to convert to native int
            tile_window = Window(tile_start[1], tile_start[0], clip_tile_shape[1], clip_tile_shape[0])
            yield TileDownload(image, profile['transform'], tile_window)

    @staticmethod
    def monitor_export_task(task, label=None):
        """
        Monitor and display the progress of an export task

        Parameters
        ----------
        task : ee.batch.Task
               WW task to monitor
        label: str, optional
               Optional label for progress display (default: use task description)
        """
        pause = 0.1
        status = ee.data.getOperation(task.name)

        if label is None:
            label = f'{status["metadata"]["description"][:80]}'

        # TODO: make this into an iterator?
        # wait for export preparation to complete, displaying a spin toggle
        def spin():
            with Spinner(f'Preparing {label}: ') as spinner:
                while 'progress' not in status['metadata']:
                    time.sleep(pause)
                    spinner.next()
                spinner.writeln(f'Preparing {label}: done')

        spin_thread = threading.Thread(target=spin)
        spin_thread.start()
        while 'progress' not in status['metadata']:
            time.sleep(5 * pause)
            status = ee.data.getOperation(task.name)  # get task status
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
        # TODO: resampling.  in __init__ where we know if it is composite or not?
        # TODO: write metadata and build overviews

        image, _, _ = self._prepare_for_export(**kwargs)
        # create export task and start
        task = ee.batch.Export.image.toDrive(image=image, description=filename[:100], folder=folder,
                                             fileNamePrefix=filename, maxPixels=1e9)
        task.start()

        if wait:  # wait for completion
            self.monitor_export_task(task)

        return task

    def download(self, filename: pathlib.Path, overwrite=False, **kwargs):
        # TODO: resampling.  in __init__ where we know if it is composite or not?
        # TODO: write metadata and build overviews
        filename = pathlib.Path(filename)
        if filename.exists():
            if overwrite:
                os.remove(filename)
            else:
                raise FileExistsError(f'{filename} exists')

        image, profile = self._prepare_for_download(**kwargs)
        session = requests_retry_session(5, status_forcelist=[500, 502, 503, 504])
        tiles = list(self.tiles(image, profile))
        bar_format = '{desc}: |{bar}| {n:4.1f}/{total_fmt} tile(s) [{percentage:5.1f}%] in {elapsed:>5s} (eta: {remaining:>5s})'
        bar = tqdm(desc=filename.name, total=len(tiles), bar_format=bar_format, dynamic_ncols=True)
        out_ds = rio.open(filename, 'w', **profile)
        warnings.filterwarnings('ignore', category=TqdmWarning)

        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), out_ds, bar:
            # threaded downloading of tiles into output dataset
            def download_tile(tile):
                tile_array = tile.download(session, bar)
                with self._out_lock:
                    out_ds.write(tile_array, window=tile.window)

            with ThreadPoolExecutor(max_workers=self._threads) as executor:
                futures = [executor.submit(download_tile, tile) for tile in tiles]
                for future in as_completed(futures):
                    future.result()

            # populate metadata
            out_ds.update_tags(**self.info['properties'])
            for band_i, band_info in enumerate(self.info['bands']):
                if 'id' in band_info:
                    out_ds.set_band_description(band_i + 1, band_info['id'])
                out_ds.update_tags(band_i + 1, **band_info)
            self._build_overviews(out_ds)
