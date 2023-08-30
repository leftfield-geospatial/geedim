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
import re

import pytest
from geedim.stac import StacCatalog, StacItem
from geedim.utils import split_id


@pytest.fixture(scope='session')
def stac_catalog() -> StacCatalog:
    """ The StacCatalog instance.  """
    return StacCatalog()


@pytest.fixture(scope='session')
def s2_sr_stac_item(stac_catalog, s2_sr_image_id) -> StacItem:
    """ A StacItem for the Sentinel-2 SR collection.  """
    return stac_catalog.get_item(s2_sr_image_id)


def test_singleton(landsat_ndvi_image_id: str):
    """ Test StacCatalog is a singleton. """
    coll_name, _ = split_id(landsat_ndvi_image_id)
    stac1 = StacCatalog()
    stac2 = StacCatalog()
    _ = stac1.url_dict
    assert len(stac2._url_dict) > 0
    _ = stac1.get_item(coll_name)
    assert coll_name in stac2._cache


def test_traverse_stac(stac_catalog: StacCatalog):
    """ Test _traverse_stac() on the root of the COPERNICUS subtree. """
    url_dict = {}
    url_dict = stac_catalog._traverse_stac(
        'https://storage.googleapis.com/earthengine-stac/catalog/COPERNICUS/catalog.json', url_dict
    )
    assert len(url_dict) > 0
    assert 'COPERNICUS/S2_SR' in url_dict


@pytest.mark.parametrize(
    'image_id', [
        'l4_image_id', 'l5_image_id', 'l7_image_id', 'l8_image_id', 'l9_image_id', 'landsat_ndvi_image_id',
        's2_sr_image_id', 's2_toa_image_id', 's1_sar_image_id', 'modis_nbar_image_id', 'gedi_cth_image_id',
    ]
)
def test_known_get_item(image_id: str, stac_catalog: StacCatalog, request: pytest.FixtureRequest):
    """  Test that stac_catalog contains expected 'items'.  """
    image_id = request.getfixturevalue(image_id)
    coll_name, _ = split_id(image_id)
    assert coll_name in stac_catalog.url_dict
    stac_item = stac_catalog.get_item(coll_name)
    assert stac_item is not None


def test_unknown_get_item(stac_catalog: StacCatalog):
    """  Test that stac_catalog returns None for unknown entries.  """
    assert stac_catalog.get_item_dict('unknown') is None
    assert stac_catalog.get_item('unknown') is None


@pytest.mark.parametrize(
    'image_id', [
        'l4_image_id', 'l5_image_id', 'l7_image_id', 'l8_image_id', 'l9_image_id', 's2_sr_image_id', 's2_toa_image_id',
        's2_sr_hm_image_id', 's2_toa_hm_image_id', 'modis_nbar_image_id',
    ]
)
def test_refl_stac_item(image_id: str, stac_catalog: StacCatalog, request: pytest.FixtureRequest):
    """ Test reflectance collectionStacItem properties are as expected. """
    image_id = request.getfixturevalue(image_id)
    coll_name, _ = split_id(image_id)
    stac_item = stac_catalog.get_item(coll_name)
    assert stac_item is not None
    if coll_name:
        assert stac_item.name == coll_name
    assert len(stac_item.license) > 0
    assert stac_item.band_props is not None
    for key in ['gsd', 'description']:
        has_key = [key in bd for bd in stac_item.band_props.values()]
        assert all(has_key)
    has_center_wavelength = ['center_wavelength' in bd for bd in stac_item.band_props.values()]
    assert sum(has_center_wavelength) >= 7
    for band_dict in stac_item.band_props.values():
        if re.search(r'^B\d|^SR_B\d|^Nadir_Reflectance_Band\d', band_dict['name']):
            assert 'center_wavelength' in band_dict
            assert 'scale' in band_dict


@pytest.mark.parametrize('image_id', ['l4_image_id', 's2_sr_image_id', 's1_sar_image_id'])
def test_stac_item_descriptions(image_id: str, stac_catalog: StacCatalog, request: pytest.FixtureRequest):
    """ Test StacItem.descriptions. """
    image_id = request.getfixturevalue(image_id)
    coll_name, _ = split_id(image_id)
    stac_item = stac_catalog.get_item(coll_name)
    assert stac_item is not None
    assert stac_item.descriptions is not None
    assert len(stac_item.descriptions) > 0
    assert len(list(stac_item.descriptions.values())[0]) > 0
