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

import pytest
from rasterio.features import bounds

from geedim.utils import split_id, get_projection, get_bounds


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
