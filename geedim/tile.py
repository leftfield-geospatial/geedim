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
import time
from io import BytesIO

import numpy as np
import rasterio as rio
import requests
from rasterio import Affine
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from requests import exceptions, HTTPError
from requests.exceptions import RetryError
from tqdm.auto import tqdm

from geedim import utils

logger = logging.getLogger(__name__)


class Tile:
    _retry_exceptions = (exceptions.RequestException, RasterioIOError)

    def __init__(self, exp_image, window: Window):
        """
        Image tile downloader.

        :param exp_image:
            BaseImageAccessor instance to derive the tile from.
        :param window:
            rasterio window into ``exp_image``, specifying the tile bounds.
        """
        self._exp_image = exp_image
        self._window = window
        # offset the image geo-transform origin so that it corresponds to the UL corner of the tile.
        self._transform = rio.Affine(*exp_image.transform) * Affine.translation(
            window.col_off, window.row_off
        )
        self._shape = (window.height, window.width)

    @property
    def window(self) -> Window:
        """rasterio tile window into the source image."""
        return self._window

    @staticmethod
    def _raise_for_status(response: requests.Response):
        """Raise an HTTPError if the tile cannot be downloaded."""
        download_size = int(response.headers.get('content-length', 0))
        if download_size == 0 or not response.status_code == 200:
            try:
                msg = response.json()['error']['message']
                msg = f'Error downloading tile: {response.status_code} - {msg} URL: {response.url}.'
                if 'user memory limit exceeded' in msg.lower():
                    # TODO: it would better design sense to move this to the BaseImageAccessor
                    msg += (
                        "\nThe 'max_tile_size' or 'max_tile_dim' parameters can be decreased "
                        "to work around this error. Alternatively you can export to Earth "
                        "Engine asset, then download the asset image."
                    )
            except (KeyError, exceptions.JSONDecodeError):
                msg = (
                    f'Error downloading tile: {response.status_code} - {response.reason}. URL: '
                    f'{response.url}.'
                )
            raise HTTPError(msg, response=response)

    def _download_to_array(
        self, url: str, session: requests.Session = None, bar: tqdm = None
    ) -> np.ndarray:
        """Download the image tile into a numpy array."""
        downloaded_size = download_size = 0
        dtype_size = np.dtype(self._exp_image.dtype).itemsize
        raw_download_size = self._shape[0] * self._shape[1] * self._exp_image.count * dtype_size
        session = session or requests
        try:
            # get image download response
            response = session.get(url, stream=True, timeout=(30, 300))

            # raise a TileError if the tile cannot be downloaded
            self._raise_for_status(response)

            # find raw download size
            download_size = int(response.headers.get('content-length', 0))

            # download & read the tile
            buf = BytesIO()
            # download gtiff into buffer
            for data in response.iter_content(chunk_size=10240):
                if bar:
                    # TODO: avoid needing to convert to byte size here or otherwise simplify
                    #  updating of progress bar...
                    # update with raw download progress (0-1)
                    bar.update(raw_download_size * (len(data) / download_size))
                buf.write(data)
                downloaded_size += len(data)
            buf.flush()

            # read the tile array from the GeoTIFF buffer
            buf.seek(0)
            env = rio.Env(GDAL_NUM_THREADS='ALL_CPUs', GTIFF_FORCE_RGBA=False)
            with utils.suppress_rio_logs(), env, rio.open(buf, 'r') as ds:
                # TODO: assert nodata is set correctly.  and add a test that checks that
                #  downloaded nodata masks match EE masks
                array = ds.read()
            return array
        # TODO: avoid duplicating except clauses with download below
        except self._retry_exceptions:
            if bar and downloaded_size > 0:
                # reverse progress bar
                bar.update(-raw_download_size * (downloaded_size / download_size))
            raise

    def download(
        self,
        session: requests.Session = None,
        bar: tqdm = None,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
    ) -> np.ndarray:
        """
        Download the image tile into a numpy array, with optional retries.

        :param session:
            requests session to use for downloading.
        :param bar:
            tqdm progress bar instance to update with raw download progress (bytes).
        :param max_retries:
            Number of times to retry downloading the tile.  If greater than zero (the default),
            ``session`` should not be configured for retries.
        :param backoff_factor:
            Backoff factor to apply between tile download retries.  The delay between retries is:
            {backoff_factor} * (2 ** ({number of previous retries})) seconds.

        :return:
            numpy array of the tile data with bands along the first dimension.
        """
        session = session or requests

        # get download URL
        # TODO: include getDownloadURL in custom retries as EE session is not configured for retries
        url = self._exp_image._ee_image.getDownloadURL(
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
            except self._retry_exceptions as ex:
                if (
                    isinstance(ex, HTTPError)
                    and ex.response is not None
                    and ex.response.status_code not in (429, 500, 502, 503, 504)
                ):
                    raise
                elif retry < max_retries:
                    time.sleep(backoff_factor * (2**retry))
                    logger.debug(
                        f'\nTile download failed, retry {retry + 1} of {max_retries}.  URL:'
                        f' {url}. Error: {repr(ex)}.\n'
                    )
                else:
                    raise RetryError(
                        f'Tile download failed, reached the maximum retries.  URL: {url}.'
                    ) from ex
