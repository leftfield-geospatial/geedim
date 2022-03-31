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
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from itertools import product
from typing import Tuple, Dict

import ee
import numpy as np
import rasterio as rio
import requests
from geojson import Point, Polygon
from rasterio import Affine
from rasterio import MemoryFile
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.warp import transform_bounds, transform_geom
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm


def open_url(url: str, session, pbar):
    # get the session
    session = session if session is not None else requests

    mem_file = BytesIO()

    r = session.get(url, stream=True)
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            mem_file.write(chunk)
            mem_file.flush()
            if pbar is not None:
                pbar.update(1024)

    if pbar is not None and (pbar.n < pbar.total):
        pbar.update(pbar.total - pbar.n)

    return mem_file


def open_zip_url(url: str, session=None, pbar=None):
    # get the contents from the url
    content = open_url(url, session, pbar)

    # create a zipfile with the url content
    z = zipfile.ZipFile(content)

    # extract the content of the zipfile
    extracted = BytesIO(z.read(z.filelist[0]))

    return extracted


def open_url_dataset(url: str, session, pbar: tqdm):
    # get the extracted file at url
    extracted = open_zip_url(url, session, pbar)
    # open the file in memory with rasterio
    with MemoryFile(extracted) as mem_file:
        with mem_file.open() as ds:
            array = ds.read()
    return array


def plot_url(url: str):
    plt.figure(figsize=(10, 10))
    ds = open_url_dataset(url)
    img = ds.read().squeeze()

    plt.imshow(img)


def create_geometry(pts, logger=None):
    if isinstance(pts, tuple):
        geometry = Point(coordinates=pts)

    elif isinstance(pts, (Point, Polygon)):
        geometry = pts

    else:
        # check if the polygon is correctly closed. If it is not, close it.
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        geometry = Polygon(coordinates=[pts])

    # if the geometry is not valid, return None
    if geometry.is_valid:
        return geometry
    else:
        # get the context logger
        msg = 'Informed points do not correspond to a valid polygon.'

        if logger is not None:
            logger.error(msg)
        else:
            print(msg)


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


class DownloadTile:
    def __init__(self, image: ee.Image, shape: Tuple[int, int], transform: Affine, slices: Tuple[slice, slice]):
        # find the
        self._image = image
        self._transform = transform * Affine.translation(slices[1].start, slices[0].start)
        self._shape = (int(slices[0].stop - slices[0].start - 1), int(slices[1].stop - slices[1].start - 1))
        self._window = Window.from_slices(*slices)
        self._lock = threading.Lock()
        # xmin, ymin = self._transform * (slices[1].start, slices[0].start)
        # xmax, ymax = self._transform * (slices[1].stop - 1, slices[0].stop - 1)
        # coordinates = [[xmax, ymax], [xmax, ymin], [xmin, ymin], [xmin, ymax], [xmax, ymax]]
        # region_dict = dict(type='Polygon', coordinates=[coordinates])
        # self._region = transform_geom(src_crs=CRS.from_string(self._image.projection().crs().getInfo()),
        #                               dst_crs=CRS({'init': 'EPSG:4326'}), geom=region_dict)
        # self._region = ee.Geometry(region_dict, opt_proj=self._image.projection(), opt_geodesic=False)


    @classmethod
    def from_download_image(cls, image, slices: Tuple[slice, slice]):
        return cls(image.image, image.shape, image.transform, slices)

    @property
    def transform(self) -> Affine:
        return self._transform

    @property
    def shape(self) -> Affine:
        return self._shape

    @property
    def window(self) -> Window:
        return self._window

    def download(self, session):
        session = session if session else requests

        with self._lock:
            url = self._image.getDownloadURL(dict(crs=self._image.projection().crs(), crs_transform=tuple(self._transform)[:6],
                                                  dimensions=self._shape[::-1], filePerBand=False, fileFormat='GeoTIFF'))
            # url = self._image.getDownloadURL(dict(region=self._region, filePerBand=False, fileFormat='GeoTIFF'))
        return open_url_dataset(url, session, None)


class DownloadImage:
    def __init__(self, image: ee.Image, **kwargs):
        # TODO: masking
        kwargs.update(fileFormat='GeoTIFF', filePerBand=False)
        self._image, _ = image.prepare_for_export(kwargs)
        self._info = self._image.getInfo()
        self._band_info = self._info['bands'][0]
        self._transform = Affine(*self._band_info['crs_transform'])
        if 'origin' in self._band_info:
            self._transform *= Affine.translation(*self._band_info['origin'])
        self._shape = self._band_info['dimensions'][::-1]
        self._count = len(self._info['bands'])
        self._dtype = self._get_dtype()
        self._threads = multiprocessing.cpu_count() - 1
        rio_crs = CRS.from_string(kwargs['crs'] if 'crs' in kwargs else self._band_info['crs'])
        self._profile = dict(driver='GTiff', dtype=self._dtype, nodata=0, width=self._shape[1], height=self._shape[0],
                             count=self.count, crs=rio_crs, transform=self._transform, compress='deflate',
                             interleave='band', tiled=True)
        self._out_lock = threading.Lock()

    @property
    def image(self) -> ee.Image:
        return self._image

    @property
    def info(self) -> Dict:
        return self._im_info

    @property
    def band_info(self) -> Dict:
        return self._band_info

    @property
    def transform(self) -> Affine:
        return self._transform

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def count(self) -> int:
        return self._count

    @property
    def profile(self) -> Dict:
        return self._profile

    def _get_dtype(self):
        # TODO: each band has different min/max vals
        ee_data_type = self._band_info['data_type']

        if ee_data_type['precision'] == 'int':
            # TODO: dtype_bits will not be correct for cases where part of the range is <0, but the range is not symm about 0
            dtype_range = ee_data_type['max'] - ee_data_type['min']
            dtype_bits = 2 ** np.ceil(np.log2(np.log2(dtype_range))).astype('int')
            poss_int_bits = [8, 16, 32, 64]
            if not dtype_bits in poss_int_bits:
                return 'int64'  # revert to int64 in unusual cases
            else:
                return f'int{dtype_bits}' if ee_data_type['min'] < 0 else f'uint{dtype_bits}'
        elif ee_data_type['precision'] == 'double':
            return 'float64'
        else:
            raise ValueError(f'Unknown image data type: {ee_data_type}')

    def _get_tile_shape(self, max_download_size=33554432, max_grid_dimension=10000):
        # TODO incorporate max_grid_dimension
        dtype_size = np.dtype(self._dtype).itemsize
        image_size = np.prod(np.int64((self.count, dtype_size, *self._shape)))
        num_tiles = np.ceil(image_size / max_download_size)

        def is_prime(x):
            for d in range(2, int(x ** 0.5) + 1):
                if x % d == 0:
                    return False
            return True

        if is_prime(num_tiles):
            num_tiles += 1

        def factors(x):
            facts = np.arange(1, x + 1)
            facts = facts[np.mod(x, facts) == 0]
            return np.vstack((facts, x / facts)).transpose()

        fact_num_tiles = factors(num_tiles)
        fact_aspect_ratios = fact_num_tiles[:, 0] / fact_num_tiles[:, 1]
        aspect_ratio = self._shape[0] / self._shape[1]
        fact_idx = np.argmin(np.abs(fact_aspect_ratios - aspect_ratio))
        shape_num_tiles = fact_num_tiles[fact_idx, :]
        return tuple(np.ceil(self._shape / shape_num_tiles).astype('int').tolist())

    def tiles(self):
        tile_shape = self._get_tile_shape()
        ul_row_range = range(0, self._shape[0], tile_shape[0])
        ul_col_range = range(0, self._shape[1], tile_shape[1])
        for ul_row, ul_col in product(ul_row_range, ul_col_range):
            br_row = min((ul_row + tile_shape[0], self._shape[0]))
            br_col = min((ul_col + tile_shape[1], self._shape[1]))
            tile_slices = (slice(ul_row, br_row), slice(ul_col, br_col))
            yield DownloadTile.from_download_image(self, tile_slices)

    def download(self, filename, resampling='near', overwrite=False):
        # TODO: resampling.  in __init__ where we know if it is composite or not?

        session = requests_retry_session(5, status_forcelist=[500, 502, 503, 504])
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **self._profile) as out_ds:
            def download_tile(tile):
                tile_array = tile.download(session)
                with self._out_lock:
                    out_ds.write(tile_array, window=tile.window)
            with ThreadPoolExecutor(max_workers=self._threads) as executor:
                futures = [executor.submit(download_tile, tile) for tile in self.tiles()]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    future.result()
