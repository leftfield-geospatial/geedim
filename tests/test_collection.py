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
from typing import List, Union

import ee
import pytest
import numpy as np
from datetime import datetime

from geedim.collection import MaskedCollection, split_id
from geedim.errors import UnfilteredError, ComponentImageError
from geedim.mask import MaskedImage


@pytest.fixture()
def l4_5_image_list(l4_image_id, l5_masked_image) -> List[Union[str, MaskedImage]]:
    """ A list of landsat 4 & 5 image IDs/ MaskedImage's """
    return [l4_image_id, l5_masked_image]


@pytest.fixture()
def l8_9_image_list(l8_image_id, l9_masked_image) -> List[Union[str, MaskedImage]]:
    """ A list of landsat 8 & 9 image IDs/ MaskedImage's """
    return [l8_image_id, l9_masked_image]


@pytest.fixture()
def s2_sr_image_list() -> List[Union[str, MaskedImage]]:
    """ A list of Sentinel-2 SR image IDs/ MaskedImage's """
    return [
        'COPERNICUS/S2_SR/20211004T080801_20211004T083709_T34HEJ',
        'COPERNICUS/S2_SR/20211123T081241_20211123T083704_T34HEJ',
        MaskedImage.from_id('COPERNICUS/S2_SR/20220107T081229_20220107T083059_T34HEJ')
    ]


@pytest.fixture()
def gedi_image_list() -> List[Union[str, MaskedImage]]:
    """ A list of canopy top height IDs/ MaskedImage's """
    return [
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202009_018E_036S', 'LARSE/GEDI/GEDI02_A_002_MONTHLY/202010_018E_036S',
        MaskedImage.from_id('LARSE/GEDI/GEDI02_A_002_MONTHLY/202112_018E_036S')
    ]


def test_split_id():
    """ Test split_id(). """
    coll_name, im_id = split_id('A/B/C')
    assert coll_name == 'A/B'
    assert im_id == 'C'
    coll_name, im_id = split_id('ABC')
    assert coll_name == ''
    assert im_id == 'ABC'


@pytest.mark.parametrize('name', ['l9_image_id', 'gch_image_id'])
def test_from_name(name: str, request):
    """ Test MaskedCollection.from_name() for non Sentinel-2 collections. """
    name = request.getfixturevalue(name)
    name, _ = split_id(name)
    gd_collection = MaskedCollection.from_name(name)
    assert gd_collection._name == name
    assert gd_collection.info is not None and len(gd_collection.info) > 0
    assert gd_collection.properties_key is not None and len(gd_collection.properties_key) > 0
    assert gd_collection.ee_collection == ee.ImageCollection(name)


@pytest.mark.parametrize('name', ['s2_sr_image_id', 's2_toa_image_id'])
def test_from_name_s2(name: str, request):
    """
    Test MaskedCollection.from_name() filters out images w/o matching cloud probability for Sentinel-2
    collections.
    """
    name = request.getfixturevalue(name)
    name, _ = split_id(name)
    gd_collection = MaskedCollection.from_name(name)
    assert gd_collection._name == name
    assert gd_collection.info is not None and len(gd_collection.info) > 0
    assert gd_collection.properties_key is not None and len(gd_collection.properties_key) > 0
    assert gd_collection.ee_collection != ee.ImageCollection(name)
    # check that one of the problem images is not in the collection
    # 0220122T081241_20220122T083135_T34HEJ, 20220226T080909_20220226T083100_T34HEH are other options
    filt = ee.Filter.eq('system:index', '20220305T075809_20220305T082125_T35HKD')
    filt_collection = gd_collection.ee_collection.filter(filt)
    assert filt_collection.size().getInfo() == 0


def test_unfiltered_error(s2_sr_image_id):
    """ Test UnfilteredError is raised when calling `properties` or `composite` on an unfiltered collection. """
    gd_collection = MaskedCollection.from_name(split_id(s2_sr_image_id)[0])
    with pytest.raises(UnfilteredError):
        _ = gd_collection.properties
    with pytest.raises(UnfilteredError):
        _ = gd_collection.composite()


def test_from_list_errors(landsat_image_ids, s2_image_ids, user_masked_image):
    """ Test MaskedCollection.from_list() error cases. """
    with pytest.raises(ComponentImageError):
        # test an error is raised when an image has no 'system:id' property
        MaskedCollection.from_list([landsat_image_ids[0], user_masked_image])

    with pytest.raises(ComponentImageError):
        # test an error is raised when images are not from compatible collections
        MaskedCollection.from_list([*landsat_image_ids])

    with pytest.raises(TypeError):
        # test an error is raised when an image type is not recognised
        MaskedCollection.from_list([landsat_image_ids[0], {}])

    with pytest.raises(ValueError):
        # test an error is raised when image list is empty
        MaskedCollection.from_list([])


@pytest.mark.parametrize('image_list', ['l4_5_image_list', 'l8_9_image_list'])
def test_from_list_landsat(image_list: str, request):
    """ Test MaskedCollection.from_list() with compatible landsat images. """
    image_list: List = request.getfixturevalue(image_list)
    gd_collection = MaskedCollection.from_list(image_list)
    assert 'LANDSAT' in gd_collection.name
    assert gd_collection.properties is not None
    comp_im = gd_collection.composite()
    assert comp_im.ee_info is not None


@pytest.mark.parametrize('image_list', ['s2_sr_image_list', 'gedi_image_list'])
def test_from_list(image_list: str, request):
    """ Test MaskedCollection.from_list() with cloud/shadow maskable, and generic images. """
    image_list: List = request.getfixturevalue(image_list)
    image_ids = [im_obj if isinstance(im_obj, str) else im_obj.id for im_obj in image_list]
    gd_collection = MaskedCollection.from_list(image_list)
    assert gd_collection.properties is not None
    assert gd_collection.properties_key is not None
    assert gd_collection.info is not None and len(gd_collection.info) > 0
    assert gd_collection.name == split_id(image_ids[0])[0]
    assert len(gd_collection.properties) == len(image_list)
    assert list(gd_collection.properties.keys()) == image_ids
    assert set(gd_collection.properties_key.keys()) > set(list(gd_collection.properties.values())[0].keys())
    assert gd_collection.properties_table is not None
    assert gd_collection.key_table is not None


@pytest.mark.parametrize('image_list', ['s2_sr_image_list', 'gedi_image_list'])
def test_from_list_order(image_list: str, request):
    """ Test MaskedCollection.from_list() maintains the order of the image list. """
    image_list: List = request.getfixturevalue(image_list)[::-1]
    image_ids = [im_obj if isinstance(im_obj, str) else im_obj.id for im_obj in image_list]
    gd_collection = MaskedCollection.from_list(image_list)
    assert list(gd_collection.properties.keys()) == image_ids


# TODO: if testing is separated well, set_region_stats would be tests in test_mask.py, and we shouldn't need to test
#  searching with so many collections here.
@pytest.mark.parametrize('name, start_date, end_date, region, cloudless_portion, is_csmask', [
    ('LANDSAT/LC09/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha', 50, True),
    ('LANDSAT/LE07/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha', 0, True),
    ('LANDSAT/LT05/C02/T1_L2', '2005-01-01', '2006-02-01', 'region_100ha', 50, True),
    ('COPERNICUS/S2_SR', '2022-01-01', '2022-01-15', 'region_100ha', 50, True),
    ('COPERNICUS/S2', '2022-01-01', '2022-01-15', 'region_100ha', 50, True),
    ('COPERNICUS/S2', '2022-01-01', '2022-01-15', 'region_100ha', 50, True),
    ('LARSE/GEDI/GEDI02_A_002_MONTHLY', '2021-11-01', '2022-01-01', 'region_100ha', 1, False)
    ])
def test_search(name, start_date:str, end_date:str, region:str, cloudless_portion:float, is_csmask, request):
    """ Test MaskedCollection.search() results with different cloud/shadow maskable collections. """
    region: dict = request.getfixturevalue(region)
    gd_collection = MaskedCollection.from_name(name)
    searched_collection = gd_collection.search(start_date, end_date, region, cloudless_portion=cloudless_portion)
    properties = searched_collection.properties
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    im_dates = np.array(
        [datetime.utcfromtimestamp(im_props['system:time_start'] / 1000) for im_props in properties.values()]
    )
    im_fill_portions = np.array([im_props['FILL_PORTION'] for im_props in properties.values()])
    assert np.all(im_fill_portions >= cloudless_portion) and np.all(im_fill_portions <= 100)
    if is_csmask:
        im_cl_portions = np.array([im_props['CLOUDLESS_PORTION'] for im_props in properties.values()])
        assert np.all(im_cl_portions >= cloudless_portion) and np.all(im_cl_portions <= 100)
        assert np.all(im_cl_portions <= im_fill_portions)
    assert np.all(im_dates >= start_date) and np.all(im_dates < end_date)
    assert set(searched_collection.properties_key.keys()) == set(list(properties.values())[0].keys())
    assert np.all(sorted(im_dates) == im_dates)

def test_empty_search(region_100ha):
    """ Test MaskedCollection.search() when it returns no results. """
    gd_collection = MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2')
    searched_collection = gd_collection.search('2022-01-01', '2022-01-02', region_100ha, cloudless_portion=100)
    assert searched_collection.properties is not None
    assert len(searched_collection.properties) == 0
    assert searched_collection.properties_table is not None




# To Test
# --------------
# - from_name() has name and ee_collection set.
# - from_name() for S2, filters out problem images.
# - from_name() & unfiltered exception on composite or properties
# - from_list() exceptions:
#   - List of ids from incompatible collections
#   - List includes image w/o ID
#   - List includes suspect object
#   - len of list == 0
#   - name property is as expected
# - from_list():
#   - test we can get properties for S2/landsat/generic images and they makes sense
#   - test size of collection matches size of list
#   - test we can combine different landsat images
# - image_type matches correct class for different ids
# - name & __init__, name & from_list as expected
# - properties: can be retrieved for filtered collections, and includes correct field names.  Test with for different
# collections.  Also test they are in correct order.  E.g. a time sorted collection has time sorted properties.
# - info & properties key.  for now just test they exist.
# - key_table & properties_table.  check they exist.
# - search (for each cloud masked type, and for some generic types):
#   - return images are in correct date range, and ordering
#   - return images are in correct region
#   - return images have correct CLOUDLESS_PORTION
#   - return images are marked filtered.
# - composite:
#   -
