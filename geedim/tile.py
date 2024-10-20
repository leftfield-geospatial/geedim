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

import logging
import threading
import time
from io import BytesIO

import numpy as np
import rasterio as rio
import requests
from rasterio import Affine
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from requests.exceptions import JSONDecodeError, RequestException
from tqdm.auto import tqdm

from geedim import utils
from geedim.errors import TileError

logger = logging.getLogger(__name__)


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
        """rasterio tile window into the source image."""
        return self._window

    @staticmethod
    def _raise_for_status(response: requests.Response):
        """Raise a TileError if the tile cannot be downloaded."""
        download_size = int(response.headers.get('content-length', 0))
        if download_size == 0 or not response.status_code == 200:
            msg = f'Error downloading tile: {response.status_code} - {response.reason}. URL: {response.url}.'
            try:
                resp_dict = response.json()
                if 'error' in resp_dict and 'message' in resp_dict['error']:
                    # raise an exception with the response error message
                    msg = resp_dict['error']['message']
                    msg = f'Error downloading tile: {msg} URL: {response.url}.'
                    if 'user memory limit exceeded' in msg.lower():
                        msg += (
                            '\nThe `max_tile_size` or `max_tile_dim` parameters can be decreased to work around this '
                            'error.  Alternatively you can export to Earth Engine asset, and then download the asset '
                            'image.'
                        )
            except JSONDecodeError:
                pass

            raise TileError(msg)

    def _download_to_array(self, url: str, session: requests.Session = None, bar: tqdm = None) -> np.ndarray:
        """Download the image tile into a numpy array."""
        # get image download response
        session = session or requests
        response = session.get(url, stream=True, timeout=(30, 300))

        # raise a TileError if the tile cannot be downloaded
        self._raise_for_status(response)

        # find raw download size
        download_size = int(response.headers.get('content-length', 0))
        dtype_size = np.dtype(self._exp_image.dtype).itemsize
        raw_download_size = self._shape[0] * self._shape[1] * self._exp_image.count * dtype_size

        # download & read the tile
        downloaded_size = 0
        buf = BytesIO()
        try:
            # download gtiff into buffer
            for data in response.iter_content(chunk_size=10240):
                if bar:
                    # update with raw download progress (0-1)
                    bar.update(raw_download_size * (len(data) / download_size))
                buf.write(data)
                downloaded_size += len(data)
            buf.flush()

            # read the tile array from the GeoTIFF buffer
            buf.seek(0)
            env = rio.Env(GDAL_NUM_THREADS='ALL_CPUs', GTIFF_FORCE_RGBA=False)
            with utils.suppress_rio_logs(), env, rio.open(buf, 'r') as ds:
                array = ds.read()
            return array

        except (RequestException, RasterioIOError):
            if bar:
                # reverse progress bar
                bar.update(-raw_download_size * (downloaded_size / download_size))
                pass
            raise

    def download(
        self,
        session: requests.Session = None,
        bar: tqdm = None,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
    ) -> np.ndarray:
        """
        Download the image tile into a numpy array.

        Parameters
        ----------
        session: requests.Session, optional
            requests session to use for downloading
        bar: tqdm, optional
            tqdm progress bar instance to update with incremental (0-1) download progress.
        max_retries: int, optional
            Number of times to retry downloading the tile.  This is independent of the ``session``, which may have its
            own retry configuration.
        backoff_factor: float, optional
            Backoff factor to apply between tile download retries.  The delay between retries is: {backoff_factor} *
            (2 ** ({number of previous retries})) seconds.  This is independent of the ``session``, which may have its
            own retry configuration.

        Returns
        -------
        array: numpy.ndarray
            3D numpy array of the tile pixel data with bands down the first dimension.
        """
        session = session or requests

        # get download URL
        url = self._exp_image.ee_image.getDownloadURL(
            dict(
                crs=self._exp_image.crs,
                crs_transform=tuple(self._transform)[:6],
                dimensions=self._shape[::-1],
                format='GEO_TIFF',
            )
        )

        # download and read the tile, with retries
        for retry in range(max_retries + 1):
            try:
                return self._download_to_array(url, session=session, bar=bar)
            except (RequestException, RasterioIOError) as ex:
                if retry < max_retries:
                    time.sleep(backoff_factor * (2**retry))
                    logger.warning(f'Tile download failed, retry {retry + 1} of {max_retries}.  URL: {url}. {str(ex)}.')
                else:
                    raise TileError(f'Tile download failed, reached the maximum retries.  URL: {url}.') from ex
