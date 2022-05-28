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
from collections import namedtuple

import ee
import numpy as np
import pytest
from rasterio import Affine
from rasterio.windows import Window
from tqdm import tqdm

from geedim.tile import Tile, requests_retry_session

BaseImageLike = namedtuple('BaseImageLike', ['ee_image', 'crs', 'transform', 'shape', 'count', 'dtype'])


@pytest.fixture(scope='module')
def base_image_like(region_25ha):
    """ Create a synthetic image object to emulate BaseImage. """
    ee_image = ee.Image([1, 2, 3]).reproject(crs='EPSG:4326', scale=30).clip(region_25ha)
    ee_info = ee_image.getInfo()
    band_info = ee_info['bands'][0]
    transform = Affine(*band_info['crs_transform']) * Affine.translation(*band_info['origin'])
    return BaseImageLike(ee_image, 'EPSG:3857', transform, tuple(band_info['dimensions'][::-1]), 3, 'uint8')


def test_create(base_image_like):
    """ Test creation of a Tile object that refers to the whole of `base_image_like`. """
    window = Window(0, 0, *base_image_like.shape[::-1])
    tile = Tile(base_image_like, window)
    assert tile.window == window
    assert tile._transform == base_image_like.transform
    assert tile._shape == base_image_like.shape


@pytest.mark.parametrize('session', [None, requests_retry_session()])
def test_download(base_image_like, session):
    """ Test downloading the synthetic image tile.  """
    window = Window(0, 0, *base_image_like.shape[::-1])
    tile = Tile(base_image_like, window)
    dtype_size = np.dtype(tile._exp_image.dtype).itemsize
    raw_download_size = tile._shape[0] * tile._shape[1] * tile._exp_image.count * dtype_size
    bar = tqdm(total=float(raw_download_size))
    array = tile.download(session=session, bar=bar)

    assert array is not None
    assert array.shape == (base_image_like.count, *base_image_like.shape)
    assert array.dtype == np.dtype(base_image_like.dtype)
    for i in range(3):
        assert np.all(array[i] == i + 1)
    assert bar.n == pytest.approx(raw_download_size, rel=0.01)
