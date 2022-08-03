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
import threading

import numpy as np
import requests
import rasterio as rio
from rasterio import Affine, MemoryFile
from rasterio.windows import Window
from tqdm.auto import tqdm


class Tile:
    # lock to prevent concurrent calls to ee.Image.getDownloadURL(), which can cause a seg fault in the standard
    # python networking libraries.
    _ee_lock = threading.Lock()

    def __init__(self, exp_image, window: Window):
        """
        Class for downloading an Earth Engine image tile (a rectangular region of interest in the image).

        Parameters
        ----------
        exp_image: BaseImage
            BaseImage instance to derive the tile from.
        window: Window
            rasterio window into `exp_image`, specifying the region of interest for this tile.
        """
        self._exp_image = exp_image
        self._window = window
        # offset the image geo-transform origin so that it corresponds to the UL corner of the tile.
        self._transform = exp_image.transform * Affine.translation(window.col_off, window.row_off)
        self._shape = (window.height, window.width)

    @property
    def window(self) -> Window:
        """ rasterio tile window into the source image. """
        return self._window

    def _get_download_url_response(self, session=None):
        """ Get tile download url and response. """
        session = session if session else requests
        with self._ee_lock:
            url = self._exp_image.ee_image.getDownloadURL(
                dict(
                    crs=self._exp_image.crs, crs_transform=tuple(self._transform)[:6], dimensions=self._shape[::-1],
                    filePerBand=False, fileFormat='GeoTIFF'
                )
            )
        return session.get(url, stream=True), url

    def download(self, session=None, response=None, bar: tqdm = None):
        """
        Download the image tile into a numpy array.

        Parameters
        ----------
        session: requests.Session, optional
            requests session to use for downloading
        response: requests.Response, optional
            Response to a get request on the tile download url.
        bar: tqdm, optional
            tqdm propgress bar instance to update with incremental (0-1) download progress.

        Returns
        -------
        array: numpy.ndarray
            3D numpy array of the tile pixel data with bands down the first dimension.
        """

        # get image download url and response
        if response is None:
            response, url = self._get_download_url_response(session=session)

        # find raw and actual download sizes
        dtype_size = np.dtype(self._exp_image.dtype).itemsize
        raw_download_size = self._shape[0] * self._shape[1] * self._exp_image.count * dtype_size
        download_size = int(response.headers.get('content-length', 0))

        if download_size == 0 or not response.ok:
            resp_dict = response.json()
            if 'error' in resp_dict and 'message' in resp_dict['error']:
                msg = resp_dict['error']['message']
                ex_msg = f'Error downloading tile: {msg}'
                if 'user memory limit exceeded' in msg.lower():
                    ex_msg += (
                        '\nThe `max_tile_size` or `max_tile_dim` parameters can be decreased to work around this error.'
                    )
            else:
                ex_msg = str(response.json())
            raise IOError(ex_msg)

        # download zip into buffer
        zip_buffer = BytesIO()
        for data in response.iter_content(chunk_size=10240):
            zip_buffer.write(data)
            if bar is not None:
                # update with raw download progress (0-1)
                bar.update(raw_download_size * (len(data) / download_size))
        zip_buffer.flush()

        # extract geotiff from zipped buffer into another buffer
        zip_file = zipfile.ZipFile(zip_buffer)
        ext_buffer = BytesIO(zip_file.read(zip_file.filelist[0]))

        # read the geotiff with a rasterio memory file
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs', GTIFF_FORCE_RGBA=False), MemoryFile(ext_buffer) as mem_file:
            with mem_file.open() as ds:
                array = ds.read()
                if (array.dtype == np.dtype('float32')) or (array.dtype == np.dtype('float64')):
                    # GEE sets nodata to -inf for float data types, (but does not populate the nodata field).
                    # rasterio won't allow nodata=-inf, so this is a workaround to change nodata to nan at source.
                    array[np.isinf(array)] = np.nan

        return array
