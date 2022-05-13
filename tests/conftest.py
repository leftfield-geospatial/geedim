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
from typing import Dict, List

import ee
import pytest

from geedim import _ee_init, MaskedImage


@pytest.fixture(scope='session', autouse=True)
def ee_init() -> None:
    _ee_init()
    return None


@pytest.fixture(scope='session')
def region_25ha() -> Dict:
    """ A geojson polygon defining a 500x500m region. """
    return {
        "type": "Polygon", "coordinates": [
            [[21.6389, -33.4520], [21.6389, -33.4474], [21.6442, -33.4474], [21.6442, -33.4520], [21.6389, -33.4520]]
        ]
    }


@pytest.fixture(scope='session')
def region_100ha() -> Dict:
    """ A geojson polygon defining a 1x1km region. """
    return {
        "type": "Polygon", "coordinates": [
            [[21.6374, -33.4547], [21.6374, -33.4455], [21.6480, -33.4455], [21.6480, -33.4547], [21.6374, -33.4547]]
        ]
    }


@pytest.fixture(scope='session')
def region_10000ha() -> Dict:
    """ A geojson polygon defining a 10x10km region. """
    return {
        "type": "Polygon", "coordinates": [
            [[21.5893, -33.4964], [21.5893, -33.4038], [21.6960, -33.4038], [21.6960, -33.4964], [21.5893, -33.4964]]
        ]
    }


@pytest.fixture(scope='session')
def l4_image_id() -> str:
    """ Landsat-4 EE ID for image that covers `region_*ha`, with partial cloud cover only for `region10000ha`.  """
    return 'LANDSAT/LT04/C02/T1_L2/LT04_173083_19880310'


@pytest.fixture(scope='session')
def l5_image_id() -> str:
    """ Landsat-5 EE ID for image that covers `region_*ha` with partial cloud cover.  """
    return 'LANDSAT/LT05/C02/T1_L2/LT05_173083_20051112'  # 'LANDSAT/LT05/C02/T1_L2/LT05_173083_20070307'


@pytest.fixture(scope='session')
def l7_image_id() -> str:
    """ Landsat-7 EE ID for image that covers `region_*ha` with partial cloud cover.  """
    return 'LANDSAT/LE07/C02/T1_L2/LE07_173083_20220119'  # 'LANDSAT/LE07/C02/T1_L2/LE07_173083_20200521'


@pytest.fixture(scope='session')
def l8_image_id() -> str:
    """ Landsat-8 EE ID for image that covers `region_*ha` with partial cloud cover.  """
    return 'LANDSAT/LC08/C02/T1_L2/LC08_173083_20180217'  # 'LANDSAT/LC08/C02/T1_L2/LC08_173083_20171113'


@pytest.fixture(scope='session')
def l9_image_id() -> str:
    """ Landsat-9 EE ID for image that covers `region_*ha` with partial cloud cover. """
    return 'LANDSAT/LC09/C02/T1_L2/LC09_173083_20220308'  # 'LANDSAT/LC09/C02/T1_L2/LC09_173083_20220103'


@pytest.fixture(scope='session')
def landsat_image_ids(l4_image_id, l5_image_id, l7_image_id, l8_image_id, l9_image_id) -> List[str]:
    """ Landsat4-9 EE IDs for images that covers `region_*ha` with partial cloud cover. """
    return [l4_image_id, l5_image_id, l7_image_id, l8_image_id, l9_image_id]


@pytest.fixture(scope='session')
def s2_sr_image_id() -> str:
    """ Sentinel-2 SR EE ID for image that covers `region_*ha` with partial cloud cover. """
    # 'COPERNICUS/S2_SR/20211123T081241_20211123T083704_T34HEJ'
    # 'COPERNICUS/S2_SR/20211123T081241_20211123T083704_T34HEH',  #no shadow
    return 'COPERNICUS/S2_SR/20211004T080801_20211004T083709_T34HEJ'


@pytest.fixture(scope='session')
def s2_toa_image_id() -> str:
    """ Sentinel-2 TOA EE ID for image that covers `region_*ha` with partial cloud cover. """
    # 'COPERNICUS/S2/20211123T081241_20211123T083704_T34HEJ'
    # 'COPERNICUS/S2/20211123T081241_20211123T083704_T34HEH'
    return 'COPERNICUS/S2/20211004T080801_20211004T083709_T34HEJ'


@pytest.fixture(scope='session')
def s2_image_ids(s2_sr_image_id, s2_toa_image_id) -> List[str]:
    """ Sentinel-2 TOA/SR EE IDs for images that covers `region_*ha` with partial cloud cover. """
    return [s2_sr_image_id, s2_toa_image_id]


@pytest.fixture(scope='session')
def l4_masked_image(l4_image_id) -> MaskedImage:
    """ Landsat-4 MaskedImage that covers `region_*ha`, with partial cloud cover only for `region10000ha`. """
    return MaskedImage.from_id(l4_image_id)


@pytest.fixture(scope='session')
def l5_masked_image(l5_image_id) -> MaskedImage:
    """ Landsat-5 MaskedImage that covers `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(l5_image_id)


@pytest.fixture(scope='session')
def l7_masked_image(l7_image_id) -> MaskedImage:
    """ Landsat-7 MaskedImage that covers `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(l7_image_id)


@pytest.fixture(scope='session')
def l8_masked_image(l8_image_id) -> MaskedImage:
    """ Landsat-8 MaskedImage that cover `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(l8_image_id)


@pytest.fixture(scope='session')
def l9_masked_image(l9_image_id) -> MaskedImage:
    """ Landsat-9 MaskedImage that covers `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(l9_image_id)


@pytest.fixture(scope='session')
def landsat_masked_images(
    l4_masked_image, l5_masked_image, l7_masked_image, l8_masked_image, l9_masked_image
) -> List[MaskedImage]:
    """ Landsat4-9 MaskedImage's that cover `region_*ha` with partial cloud cover. """
    return [l4_masked_image, l5_masked_image, l7_masked_image, l8_masked_image, l9_masked_image]


@pytest.fixture(scope='session')
def s2_sr_masked_image(s2_sr_image_id) -> MaskedImage:
    """ Sentinel-2 SR MaskedImage that covers `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(s2_sr_image_id)


@pytest.fixture(scope='session')
def s2_toa_masked_image(s2_toa_image_id) -> MaskedImage:
    """ Sentinel-2 TOA MaskedImage that covers `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(s2_toa_image_id)


@pytest.fixture(scope='session')
def s2_masked_images(s2_sr_masked_image, s2_toa_masked_image) -> List[MaskedImage]:
    """ Sentinel-2 TOA and SRR MaskedImage's that cover `region_*ha` with partial cloud cover. """
    return [s2_sr_masked_image, s2_toa_masked_image]


@pytest.fixture(scope='session')
def user_masked_image() -> MaskedImage:
    """ A MaskedImage instance where the encapsulated image has no fixed projection or ID.  """
    return MaskedImage(ee.Image([1, 2, 3]))


@pytest.fixture(scope='session')
def modis_nbar_masked_image(region_10000ha) -> MaskedImage:
    """ A list of MaskedImage's from non cloud/shadow masked collections.  """
    return MaskedImage(
        ee.Image('MODIS/006/MCD43A4/2022_01_01').clip(region_10000ha).
            reproject('EPSG:3857', scale=500)
    )
