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
from geedim import MaskedImage
from geedim.enums import ResamplingMethod
from geedim.utils import split_id, get_projection, get_bounds, Spinner, resample, asset_id
from rasterio.features import bounds

from .conftest import get_image_std


@pytest.mark.parametrize('id, exp_split', [('A/B/C', ('A/B', 'C')), ('ABC', ('', 'ABC')), (None, (None, None))])
def test_split_id(id, exp_split):
    """ Test split_id(). """
    assert split_id(id) == exp_split


def test_get_bounds(const_image_25ha_file, region_25ha):
    """ Test get_bounds(). """
    raster_bounds = bounds(get_bounds(const_image_25ha_file, expand=0))
    test_bounds = bounds(region_25ha)
    assert raster_bounds == pytest.approx(test_bounds, abs=.001)


def test_get_projection(s2_sr_masked_image):
    """ Test get_projection().  """
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
    """ Test Spinner class. """
    spinner = Spinner(label='test', interval=0.1)
    assert not spinner.is_alive()
    with spinner:
        assert spinner._run
        assert spinner.is_alive()
        time.sleep(0.5)
    assert not spinner._run
    assert not spinner.is_alive()


# yapf: disable
@pytest.mark.parametrize(
    'image_id, method, std_scale', [
        ('l9_image_id', ResamplingMethod.bilinear, 30),
        ('s2_sr_image_id', ResamplingMethod.average, 60),
        ('modis_nbar_image_id', ResamplingMethod.bicubic, 500),
    ]
)
# yapf: enable
def test_resample_fixed(
    image_id: str, method: ResamplingMethod, std_scale: float, region_10000ha: Dict, request: pytest.FixtureRequest
):
    """ Test that resample() smooths images with a fixed projection. """
    image_id = request.getfixturevalue(image_id)
    before_image = ee.Image(image_id)
    after_image = resample(before_image, method)

    assert (
        get_image_std(after_image, region_10000ha, std_scale) < get_image_std(before_image, region_10000ha, std_scale)
    )


@pytest.mark.parametrize(
    'masked_image, method, std_scale', [
        ('user_masked_image', ResamplingMethod.bilinear, 100),
        ('landsat_ndvi_masked_image', ResamplingMethod.average, 60)
    ]
)
def test_resample_comp(
    masked_image: str, method: ResamplingMethod, std_scale: float, region_10000ha: Dict, request: pytest.FixtureRequest
):
    """ Test that resample() leaves composite images unaltered. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    before_image = masked_image.ee_image
    after_image = resample(before_image, method)

    assert (
        get_image_std(after_image, region_10000ha, std_scale) == get_image_std(before_image, region_10000ha, std_scale)
    )


@pytest.mark.parametrize(
    'filename, folder, exp_id', [
        ('file', 'folder', 'projects/folder/assets/file'),
        ('fp1/fp2/fp3', 'folder', 'projects/folder/assets/fp1-fp2-fp3'),
        ('file', 'folder/sub-folder', 'projects/folder/assets/sub-folder/file'),
        ('file', None, 'file'),
        ('projects/folder/assets/file', None, 'projects/folder/assets/file'),
    ]
)
def test_asset_id(filename: str, folder: str, exp_id: str):
    """ Test asset_id() works as expected. """
    id = asset_id(filename, folder)
    assert id == exp_id
