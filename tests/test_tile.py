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

import io
import json
import logging
import zipfile
from collections import namedtuple

import ee
import numpy as np
import pytest
import rasterio as rio
import requests
from rasterio import Affine
from rasterio.windows import Window
from tqdm.auto import tqdm

from geedim.errors import TileError
from geedim.tile import Tile
from geedim.utils import retry_session

BaseImageLike = namedtuple('BaseImageLike', ['ee_image', 'crs', 'transform', 'shape', 'count', 'dtype'])


@pytest.fixture(scope='module')
def mock_base_image(region_25ha: dict) -> BaseImageLike:
    """A BaseImage mock containing a synthetic ee.Image."""
    ee_image = ee.Image([1, 2, 3]).reproject(crs='EPSG:4326', scale=30).clip(region_25ha)
    ee_info = ee_image.getInfo()
    band_info = ee_info['bands'][0]
    transform = Affine(*band_info['crs_transform']) * Affine.translation(*band_info['origin'])
    return BaseImageLike(ee_image, 'EPSG:3857', transform, tuple(band_info['dimensions'][::-1]), 3, 'uint8')


@pytest.fixture(scope='module')
def synth_tile(mock_base_image: BaseImageLike) -> Tile:
    """A tile representing the whole of ``mock_base_image``."""
    window = Window(0, 0, *mock_base_image.shape[::-1])
    return Tile(mock_base_image, window)


@pytest.fixture(scope='function')
def mock_ee_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ee.Image.getDownloadURL() to return None."""

    def getDownloadURL(*args, **kwargs):
        return None

    monkeypatch.setattr(ee.Image, 'getDownloadURL', getDownloadURL)


@pytest.fixture(scope='module')
def zipped_gtiff_bytes(mock_base_image: BaseImageLike) -> bytes:
    """Zipped GeoTIFF bytes for ``mock_base_image``."""
    zip_buffer = io.BytesIO()
    array = np.ones((mock_base_image.count, *mock_base_image.shape)) * np.array([1, 2, 3]).reshape(-1, 1, 1)

    with rio.MemoryFile() as mem_file:
        with mem_file.open(
            **rio.default_gtiff_profile,
            width=mock_base_image.shape[1],
            height=mock_base_image.shape[0],
            count=mock_base_image.count
        ) as ds:
            ds.write(array)
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zf:
            zf.writestr('test.tif', mem_file.read())

    return zip_buffer.getvalue()


def test_create(mock_base_image: BaseImageLike):
    """Test creation of a Tile object."""
    window = Window(0, 0, *mock_base_image.shape[::-1])
    tile = Tile(mock_base_image, window)
    assert tile.window == window
    assert tile._transform == mock_base_image.transform
    assert tile._shape == mock_base_image.shape


@pytest.mark.parametrize('session', [None, retry_session()])
def test_download(synth_tile: Tile, session):
    """Test downloading the synthetic image tile."""
    dtype_size = np.dtype(synth_tile._exp_image.dtype).itemsize
    raw_download_size = synth_tile._shape[0] * synth_tile._shape[1] * synth_tile._exp_image.count * dtype_size
    bar = tqdm(total=float(raw_download_size))
    array = synth_tile.download(session=session, bar=bar)

    assert array is not None
    assert array.shape == (synth_tile._exp_image.count, *synth_tile._exp_image.shape)
    assert array.dtype == np.dtype(synth_tile._exp_image.dtype)
    for i in range(array.shape[0]):
        assert np.all(array[i] == i + 1)
    assert bar.n == pytest.approx(raw_download_size, rel=0.01)


def test_mem_limit_error(synth_tile: Tile, mock_ee_image: None):
    """Test downloading raises the 'user memory limit exceeded' error with a mock response."""
    # patch session.get() to return a mock response with EE memory limit error
    session = retry_session()
    msg = 'User memory limit exceeded.'

    def get(url, **kwargs):
        response = requests.Response()
        response.status_code = 400
        response.headers = {'content-length': '1'}
        response._content = json.dumps({'error': {'message': msg}}).encode()
        return response

    session.get = get

    # test memory limit error is raised on download
    with pytest.raises(TileError) as ex:
        synth_tile.download(session=session)
    assert msg in str(ex.value)


def test_retry(synth_tile: Tile, mock_ee_image: None, zipped_gtiff_bytes: bytes, caplog: pytest.LogCaptureFixture):
    """Test downloading retries invalid tiles until it succeeds."""
    # create progress bar
    dtype_size = np.dtype(synth_tile._exp_image.dtype).itemsize
    raw_download_size = synth_tile._shape[0] * synth_tile._shape[1] * synth_tile._exp_image.count * dtype_size
    bar = tqdm(total=float(raw_download_size))

    # create mock invalid responses for each retry
    responses = []
    for _ in range(5):
        response = requests.Response()
        response.status_code = 200
        response.headers = {'content-length': str(len(zipped_gtiff_bytes))}
        response.raw = io.BytesIO(b'error')
        responses.append(response)

    # make the last response valid
    responses[-1].raw = io.BytesIO(zipped_gtiff_bytes)

    # patch session.get() to pop and return a mocked response from the list
    session = retry_session()

    def get(url, **kwargs):
        return responses.pop(0)

    session.get = get

    # test the tile is downloaded correctly, after retries
    with caplog.at_level(logging.WARNING):
        array = synth_tile.download(session=session, bar=bar, backoff_factor=0)

    assert array.shape == (synth_tile._exp_image.count, *synth_tile._exp_image.shape)
    assert array.dtype == np.dtype(synth_tile._exp_image.dtype)
    for i in range(array.shape[0]):
        assert np.all(array[i] == i + 1)

    # test progress bar is adjusted for retries
    assert bar.n == pytest.approx(raw_download_size, rel=0.01)

    # test retry logs
    assert 'retry' in caplog.text and 'zip' in caplog.text


def test_retry_error(synth_tile: Tile, mock_ee_image: None, zipped_gtiff_bytes: bytes):
    """Test downloading raises an error when the maximum retries are reached."""
    # create progress bar
    dtype_size = np.dtype(synth_tile._exp_image.dtype).itemsize
    raw_download_size = synth_tile._shape[0] * synth_tile._shape[1] * synth_tile._exp_image.count * dtype_size
    bar = tqdm(total=float(raw_download_size))

    # patch session.get() to return a mock response with invalid bytes
    session = retry_session()

    def get(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.headers = {'content-length': '10'}
        response.raw = io.BytesIO(b'error')
        return response

    session.get = get

    # test max retries error is raised on download
    with pytest.raises(TileError) as ex:
        synth_tile.download(session=session, bar=bar, backoff_factor=0)
    assert 'maximum retries' in str(ex.value)

    # test progress bar is adjusted for retries
    assert bar.n == pytest.approx(0)
