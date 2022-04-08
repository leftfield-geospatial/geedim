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

import zipfile
from io import BytesIO

import ee
import numpy as np
import requests
from affine import Affine
from rasterio import MemoryFile
from rasterio.windows import Window
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm


def _requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    """A persistent requests session configured for retries"""
    session = session or requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class Tile:
    """Class for encapsulating and downloading a GEE image tile (i.e. a rectangular region of interest in the image)"""

    def __init__(self, image: ee.Image, transform: Affine, window: Window):
        """
        Create an instance of Tile.

        Parameters
        ----------
        image: ee.Image
            An EE image to derive the tile from.
        transform: Affine
            A rasterio geo-transform for `image`.
        window: Window
            A rasterio window into `image`, specifying the region of interest for this tile.
        """
        self._image = image
        self._window = window
        # offset the image geo-transform origin so that it corresponds to the UL corner of the tile.
        self._transform = transform * Affine.translation(window.col_off, window.row_off)
        self._shape = (window.height, window.width)

    @property
    def window(self) -> Window:
        """The rasterio window into the source image."""
        return self._window

    def download(self, session=None, bar: tqdm = None):
        """

        Parameters
        ----------
        session: requests.Session, optional
            Session to use for downloading.
        bar: tqdm, optional
            A tqdm progress bar to update with the download progress.

        Returns
        -------
        array: numpy.ndarray
            The tile pixel data in a 3D array (bands down the first dimension).
        """
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
        if download_size == 0 or not resp.ok:
            raise IOError(resp.json())
        for data in resp.iter_content(chunk_size=10240):
            zip_buffer.write(data)
            if bar:
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