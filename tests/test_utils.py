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

import time
from typing import Dict

import ee
import pytest
from rasterio.features import bounds

from geedim import MaskedImage
from geedim.enums import ResamplingMethod
from geedim.utils import asset_id, get_bounds, get_projection, resample, Spinner, split_id


@pytest.mark.parametrize('id, exp_split', [('A/B/C', ('A/B', 'C')), ('ABC', ('', 'ABC')), (None, (None, None))])
def test_split_id(id, exp_split):
    """Test split_id()."""
    assert split_id(id) == exp_split


def test_get_bounds(const_image_25ha_file, region_25ha):
    """Test get_bounds()."""
    raster_bounds = bounds(get_bounds(const_image_25ha_file, expand=0))
    test_bounds = bounds(region_25ha)
    assert raster_bounds == pytest.approx(test_bounds, abs=0.001)


def test_get_projection(s2_sr_masked_image):
    """Test get_projection()."""
    min_proj = get_projection(s2_sr_masked_image.ee_image, min_scale=True)
    min_crs = min_proj.crs().getInfo()
    min_scale = min_proj.nominalScale().getInfo()
    max_proj = get_projection(s2_sr_masked_image.ee_image, min_scale=False)
    max_crs = max_proj.crs().getInfo()
    max_scale = max_proj.nominalScale().getInfo()

    assert min_crs.startswith('EPSG:')
    assert min_crs == max_crs
    assert max_scale == 60
    assert min_scale == 10


def test_spinner():
    """Test Spinner class."""
    spinner = Spinner(label='test', interval=0.1)
    assert not spinner.is_alive()
    with spinner:
        assert spinner._run
        assert spinner.is_alive()
        time.sleep(0.5)
    assert not spinner._run
    assert not spinner.is_alive()


@pytest.mark.parametrize(
    'image_id, method, scale',
    [
        ('l9_image_id', ResamplingMethod.bilinear, 15),
        ('s2_sr_hm_image_id', ResamplingMethod.average, 25),
        ('modis_nbar_image_id', ResamplingMethod.bicubic, 100),
    ],
)
def test_resample_fixed(
    image_id: str, method: ResamplingMethod, scale: float, region_100ha: Dict, request: pytest.FixtureRequest
):
    """Test that resample() smooths images with a fixed projection."""
    image_id = request.getfixturevalue(image_id)
    source_im = ee.Image(image_id)
    resampled_im = resample(source_im, method)

    # find mean of std deviations of bands for each image
    crs = source_im.select(0).projection().crs()
    stds = []
    for im in [source_im, resampled_im]:
        im = im.reproject(crs=crs, scale=scale)  # required to resample at scale
        std = im.reduceRegion('stdDev', geometry=region_100ha).values().reduce('mean')
        stds.append(std)
    stds = ee.List(stds).getInfo()

    # test resampled_im is smoother than source_im
    assert stds[1] < stds[0]


@pytest.mark.parametrize(
    'masked_image, method, scale',
    [
        ('user_masked_image', ResamplingMethod.bilinear, 50),
        ('landsat_ndvi_masked_image', ResamplingMethod.average, 50),
    ],
)
def test_resample_comp(
    masked_image: str, method: ResamplingMethod, scale: float, region_100ha: Dict, request: pytest.FixtureRequest
):
    """Test that resample() leaves composite images unaltered."""
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    source_im = masked_image.ee_image
    resampled_im = resample(source_im, method)

    # find mean of std deviations of bands for each image
    crs = source_im.select(0).projection().crs()
    stds = []
    for im in [source_im, resampled_im]:
        im = im.reproject(crs=crs, scale=scale)  # required to resample at scale
        std = im.reduceRegion('stdDev', geometry=region_100ha).values().reduce('mean')
        stds.append(std)
    stds = ee.List(stds).getInfo()

    # test no change between resampled_im and source_im
    assert stds[1] == stds[0]


@pytest.mark.parametrize(
    'filename, folder, exp_id',
    [
        ('file', 'folder', 'projects/folder/assets/file'),
        ('fp1/fp2/fp3', 'folder', 'projects/folder/assets/fp1-fp2-fp3'),
        ('file', 'folder/sub-folder', 'projects/folder/assets/sub-folder/file'),
        ('file', None, 'file'),
        ('projects/folder/assets/file', None, 'projects/folder/assets/file'),
    ],
)
def test_asset_id(filename: str, folder: str, exp_id: str):
    """Test asset_id() works as expected."""
    id = asset_id(filename, folder)
    assert id == exp_id
