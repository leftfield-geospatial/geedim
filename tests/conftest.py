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
from typing import Dict

import ee
import pytest

from geedim import _ee_init


class TestImage:
    def __init__(self, ee_id, region=None, scale=None):
        if isinstance(ee_id, str):
            self.ee_image = ee.Image(ee_id)
            self.id = ee_id
        else:
            self.ee_image = ee_id
            self.id = 'None'
        self.region = region
        self.scale = scale


@pytest.fixture(scope='session', autouse=True)
def ee_init() -> None:
    _ee_init()
    return None


@pytest.fixture(scope='session')
def small_region() -> Dict:
    return {
        'type': 'Polygon',
        'coordinates': [[[24.3885, -33.6659],
                         [24.3885, -33.6601],
                         [24.3947, -33.6601],
                         [24.3947, -33.6659],
                         [24.3885, -33.6659]]]
    }


@pytest.fixture
def big_region() -> Dict:
    return {
        'type': 'Polygon',
        'coordinates': [[[22.5, -34.],
                         [22.5, -33.5],
                         [23.5, -33.5],
                         [23.5, -34.],
                         [22.5, -34.]]]
    }


@pytest.fixture
def s2_sr_small_image(small_region) -> TestImage:
    return TestImage('COPERNICUS/S2_SR/20220114T080159_20220114T082124_T35HKC', region=small_region)


@pytest.fixture
def s2_sr_small_image(small_region) -> TestImage:
    return TestImage('COPERNICUS/S2/20220114T080159_20220114T082124_T35HKC', region=small_region)


@pytest.fixture
def l9_small_image(small_region) -> TestImage:
    return TestImage('LANDSAT/LC09/C02/T1_L2/LC09_171084_20220427', region=small_region)  # no cloud


@pytest.fixture
def l8_small_image(small_region) -> TestImage:
    return TestImage('LANDSAT/LC08/C02/T1_L2/LC08_171084_20220113', region=small_region)  # no cloud


@pytest.fixture
def l7_small_image(small_region) -> TestImage:
    return TestImage('LANDSAT/LE07/C02/T1_L2/LE07_171083_20210118', region=small_region)  # no cloud


@pytest.fixture
def l5_small_image(small_region) -> TestImage:
    return TestImage('LANDSAT/LT05/C02/T1_L2/LT05_172083_20110207', region=small_region)  # no cloud


@pytest.fixture
def l4_small_image(small_region) -> TestImage:
    return TestImage('LANDSAT/LT04/C02/T1_L2/LT04_172083_19890306', region=small_region)  # no cloud


@pytest.fixture
def s2_sr_big_image(big_region) -> TestImage:
    return TestImage('COPERNICUS/S2_SR/20220226T080909_20220226T083100_T34HFH', region=big_region)


@pytest.fixture
def s2_toa_big_image(big_region) -> TestImage:
    return TestImage('COPERNICUS/S2/20220226T080909_20220226T083100_T34HFH', region=big_region)


@pytest.fixture
def l9_big_image(big_region) -> TestImage:
    return TestImage('LANDSAT/LC09/C02/T1_L2/LC09_172083_20220128', region=big_region)


@pytest.fixture
def l8_big_image(big_region) -> TestImage:
    return TestImage('LANDSAT/LC08/C02/T1_L2/LC08_172083_20220104', region=big_region)


@pytest.fixture
def l7_big_image(big_region) -> TestImage:
    return TestImage('LANDSAT/LE07/C02/T1_L2/LE07_172083_20220128', region=big_region)


@pytest.fixture
def l5_big_image(big_region) -> TestImage:
    return TestImage('LANDSAT/LT05/C02/T1_L2/LT05_172083_20100204', region=big_region)


@pytest.fixture
def l4_big_image(big_region) -> TestImage:
    return TestImage('LANDSAT/LT04/C02/T1_L2/LT04_172083_19880319', region=big_region)

@pytest.fixture(scope='session')
def synth_unfixed_ee_image() -> ee.Image:
    return ee.Image([1, 2, 3])

@pytest.fixture(scope='session')
def synth_fixed_ee_image(synth_unfixed_ee_image, small_region) -> ee.Image:
    return synth_unfixed_ee_image.reproject(crs='EPSG:3857', scale=30).clip(small_region)


@pytest.fixture(scope='session')
def synth_fixed_ee_info(synth_fixed_ee_image) -> Dict:
    return synth_fixed_ee_image.getInfo()
