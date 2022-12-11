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
from datetime import datetime
from typing import List, Union, Dict

import ee
import numpy as np
import pytest
from geedim import schema
from geedim.collection import MaskedCollection
from geedim.enums import CompositeMethod, ResamplingMethod
from geedim.errors import UnfilteredError, InputImageError
from geedim.mask import MaskedImage
from geedim.utils import split_id, get_projection

from .conftest import get_image_std


@pytest.fixture()
def l4_5_image_list(l4_image_id, l5_masked_image) -> List[Union[str, MaskedImage]]:
    """ A list of landsat 4 & 5 image ID / MaskedImage's """
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
    """ A list of GEDI canopy top height IDs/ MaskedImage's """
    return [
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202008_018E_036S', 'LARSE/GEDI/GEDI02_A_002_MONTHLY/202009_018E_036S',
        MaskedImage.from_id('LARSE/GEDI/GEDI02_A_002_MONTHLY/202005_018E_036S')
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
    assert gd_collection.schema is not None
    assert len(gd_collection.schema) >= len(schema.default_prop_schema)
    assert gd_collection.ee_collection == ee.ImageCollection(name)


@pytest.mark.parametrize('name', ['s2_sr_image_id', 's2_toa_image_id'])
def test_from_name_s2(name: str, request):
    """
    Test MaskedCollection.from_name() filters out images that don't have matching cloud probability (for Sentinel-2
    collections).
    """
    name = request.getfixturevalue(name)
    name, _ = split_id(name)
    gd_collection = MaskedCollection.from_name(name)
    assert gd_collection._name == name
    assert gd_collection.schema is not None
    assert gd_collection.schema == schema.s2_prop_schema
    # check ee_collection is not the full unfiltered collection
    assert gd_collection.ee_collection != ee.ImageCollection(name)
    # check one of the problem images is not in the collection
    # 20220305T075809_20220305T082125_T35HKD, 0220122T081241_20220122T083135_T34HEJ,
    # 20220226T080909_20220226T083100_T34HEH are other options
    filt = ee.Filter.eq('system:index', '0220122T081241_20220122T083135_T34HEJ')
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
    """ Test MaskedCollection.from_list() various error cases. """
    with pytest.raises(InputImageError):
        # test an error is raised when an image has no 'id'/'system:time_start' property
        MaskedCollection.from_list([landsat_image_ids[0], user_masked_image])

    with pytest.raises(InputImageError):
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
    """
    Test MaskedCollection.from_list() works with landsat images from different, but spectrally compatible
    collections.
    """
    image_list: List = request.getfixturevalue(image_list)
    gd_collection = MaskedCollection.from_list(image_list)
    assert 'LANDSAT' in gd_collection.name
    assert gd_collection.properties is not None


@pytest.mark.parametrize('image_list', ['s2_sr_image_list', 'gedi_image_list'])
def test_from_list(image_list: str, request):
    """
    Test MaskedCollection.from_list() generates a valid MaskedCollection object from lists of cloud/shadow
    maskable, and generic images.
    """
    image_list: List = request.getfixturevalue(image_list)
    image_ids = [im_obj if isinstance(im_obj, str) else im_obj.id for im_obj in image_list]
    gd_collection = MaskedCollection.from_list(image_list)
    assert gd_collection.properties is not None
    assert gd_collection.schema is not None
    assert len(gd_collection.schema) >= len(schema.default_prop_schema)
    assert gd_collection.name == split_id(image_ids[0])[0]
    assert len(gd_collection.properties) == len(image_list)
    assert list(gd_collection.properties.keys()) == image_ids
    assert set(gd_collection.schema.keys()) > set(list(gd_collection.properties.values())[0].keys())
    assert gd_collection.properties_table is not None
    assert gd_collection.schema_table is not None


@pytest.mark.parametrize('image_list', ['s2_sr_image_list', 'gedi_image_list'])
def test_from_list_order(image_list: str, request):
    """ Test MaskedCollection.from_list() maintains the order of the provided image list. """
    image_list: List = request.getfixturevalue(image_list)[::-1]
    image_ids = [im_obj if isinstance(im_obj, str) else im_obj.id for im_obj in image_list]
    gd_collection = MaskedCollection.from_list(image_list)
    assert list(gd_collection.properties.keys()) == image_ids


def test_from_list_ee_image(gedi_image_list: List):
    """ Test MaskedCollection.from_list() with an ee.Image in the list. """
    image_ids = [im_obj if isinstance(im_obj, str) else im_obj.id for im_obj in gedi_image_list]
    image_list = gedi_image_list
    image_list[1] = ee.Image(image_list[1])
    gd_collection = MaskedCollection.from_list(image_list)
    assert list(gd_collection.properties.keys()) == image_ids

@pytest.mark.parametrize(
    'image_list, add_props', [
        ('s2_sr_image_list', ['AOT_RETRIEVAL_ACCURACY', 'CLOUDY_PIXEL_PERCENTAGE']),
        ('l8_9_image_list', ['CLOUD_COVER', 'GEOMETRIC_RMSE_VERIFY'])
    ]
)  # yapf: disable
def test_from_list_add_props(image_list: str, add_props: List, request: pytest.FixtureRequest):
    """
    Test MaskedCollection.from_list(add_props=...) contains the add_props in properties and schema.
    """
    image_list: List = request.getfixturevalue(image_list)
    gd_collection = MaskedCollection.from_list(image_list, add_props=add_props)
    assert gd_collection.properties is not None
    assert gd_collection.schema is not None
    assert len(gd_collection.schema) > len(schema.default_prop_schema)
    assert all([add_prop in gd_collection.schema.keys() for add_prop in add_props])
    assert all([gd_collection.schema[add_prop]['abbrev'] is not None for add_prop in add_props])
    assert all([add_prop in list(gd_collection.properties.values())[0].keys() for add_prop in add_props])
    assert gd_collection.properties_table is not None
    assert gd_collection.schema_table is not None


@pytest.mark.parametrize(
    'name, start_date, end_date, region, fill_portion, cloudless_portion, is_csmask', [
        ('LANDSAT/LC09/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha', 0, 50, True),
        ('LANDSAT/LE07/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha', 0, 0, True),
        ('LANDSAT/LT05/C02/T1_L2', '2005-01-01', '2006-02-01', 'region_100ha', 40, 50, True),
        ('COPERNICUS/S2_SR', '2022-01-01', '2022-01-15', 'region_100ha', 0, 50, True),
        ('COPERNICUS/S2_HARMONIZED', '2022-01-01', '2022-01-15', 'region_100ha', 50, 40, True),
        ('LARSE/GEDI/GEDI02_A_002_MONTHLY', '2021-08-01', '2021-09-01', 'region_100ha', .1, 0, False)
    ]
)
def test_search(
    name, start_date: str, end_date: str, region: str, fill_portion: float, cloudless_portion: float, is_csmask, request
):
    """
    Test MaskedCollection.search() with fill / cloudless portion filters gives valid results for different cloud/shadow
    maskable, and generic collections.
    """
    region: Dict = request.getfixturevalue(region)
    gd_collection = MaskedCollection.from_name(name)
    searched_collection = gd_collection.search(
        start_date, end_date, region, fill_portion=fill_portion, cloudless_portion=cloudless_portion
    )

    properties = searched_collection.properties
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    im_dates = np.array(
        [datetime.utcfromtimestamp(im_props['system:time_start'] / 1000) for im_props in properties.values()]
    )
    # test FILL_PORTION in expected range
    im_fill_portions = np.array([im_props['FILL_PORTION'] for im_props in properties.values()])
    assert np.all(im_fill_portions >= fill_portion) and np.all(im_fill_portions <= 100)
    if is_csmask:  # is a cloud/shadow masked collection
        # test CLOUDLESS_PORTION in expected range
        im_cl_portions = np.array([im_props['CLOUDLESS_PORTION'] for im_props in properties.values()])
        assert np.all(im_cl_portions >= cloudless_portion) and np.all(im_cl_portions <= 100)
    # test search result image dates lie between `start_date` and `end_date`
    assert np.all(im_dates >= start_date) and np.all(im_dates < end_date)
    assert set(searched_collection.schema.keys()) == set(list(properties.values())[0].keys())
    # test search result image dates are sorted
    assert np.all(sorted(im_dates) == im_dates)


def test_empty_search(region_100ha):
    """ Test MaskedCollection.search() for empty search results. """
    gd_collection = MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2')
    searched_collection = gd_collection.search('2022-01-01', '2022-01-02', region_100ha, cloudless_portion=100)
    assert searched_collection.properties is not None
    assert len(searched_collection.properties) == 0
    assert searched_collection.properties_table is not None


def test_search_no_end_date(region_100ha):
    """ Test MaskedCollection.search() with ``end_date=None`` searches for a day from ``start_date`. """
    gd_collection = MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2')
    searched_collection = gd_collection.search('2022-01-03', None, region_100ha)

    start_date = datetime.strptime('2022-01-03', '%Y-%m-%d')
    end_date = datetime.strptime('2022-01-04', '%Y-%m-%d')
    properties = searched_collection.properties
    im_dates = np.array(
        [datetime.utcfromtimestamp(im_props['system:time_start'] / 1000) for im_props in properties.values()]
    )
    assert len(properties) > 0
    assert np.all(im_dates >= start_date) and np.all(im_dates < end_date)


def test_search_date_error(region_100ha):
    """ Test MaskedCollection.search() raises an error when end date is on or before start date. """
    gd_collection = MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2')
    with pytest.raises(ValueError):
        _ = gd_collection.search('2022-01-02', '2022-01-01', region_100ha)


def test_search_mult_kwargs(region_100ha):
    """
    When a search filtered collection is searched again, test that masks change with different cloud/shadow kwargs.
    """
    start_date = '2022-01-01'
    end_date = '2022-01-10'
    gd_collection = MaskedCollection.from_name('COPERNICUS/S2_SR')

    def get_cloudless_portion(properties: Dict) -> List[float]:
        return [prop_dict['CLOUDLESS_PORTION'] for prop_dict in properties.values()]

    filt_collection = gd_collection.search(start_date, end_date, region_100ha, prob=80, fill_portion=0)
    filt_coll_prob80 = filt_collection.search(start_date, end_date, region_100ha, prob=80, fill_portion=0)
    filt_coll_prob40 = filt_collection.search(start_date, end_date, region_100ha, prob=40, fill_portion=0)

    cp_ref = get_cloudless_portion(filt_collection.properties)
    cp_prob80 = get_cloudless_portion(filt_coll_prob80.properties)
    cp_prob40 = get_cloudless_portion(filt_coll_prob40.properties)

    assert cp_ref == pytest.approx(cp_prob80, abs=1e-3)
    assert cp_ref != pytest.approx(cp_prob40, abs=1e-1)


def test_search_custom_filter(region_25ha):
    """
    Test that a CLOUDLESS_PORTION custom filter gives the same search results as the equivalent cloudless_portion
    kwarg specification.
    """
    start_date = '2022-01-01'
    end_date = '2022-02-01'
    gd_collection = MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2')
    kwarg_coll = gd_collection.search(start_date, end_date, region_25ha, cloudless_portion=90)
    cust_filt_coll = gd_collection.search(start_date, end_date, region_25ha, custom_filter='CLOUDLESS_PORTION>=90')
    assert (kwarg_coll.properties is not None) and (len(kwarg_coll.properties) > 0)
    assert kwarg_coll.properties == cust_filt_coll.properties


def test_search_add_props(region_25ha):
    """
    Test that specified add_props are added to the search results.
    """
    add_props = ['CLOUD_COVER', 'GEOMETRIC_RMSE_VERIFY']
    gd_collection = MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2', add_props=add_props)
    searched_coll = gd_collection.search('2022-01-01', '2022-02-01', region_25ha)
    assert all([add_prop in searched_coll.schema.keys() for add_prop in add_props])
    assert all([add_prop in list(searched_coll.properties.values())[0].keys() for add_prop in add_props])
    assert searched_coll.properties_table is not None
    assert searched_coll.schema_table is not None


@pytest.mark.parametrize(
    'name, start_date, end_date, region', [
        ('LANDSAT/LC09/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha'),
        ('COPERNICUS/S2_HARMONIZED', '2022-01-01', '2022-01-15', 'region_100ha'),
        ('LARSE/GEDI/GEDI02_A_002_MONTHLY', '2021-08-01', '2021-09-01', 'region_100ha')
    ]
)
def test_search_no_fill_or_cloudless_portion(
    name: str, start_date: str, end_date: str, region: str, request: pytest.FixtureRequest
):
    """
    Test MaskedCollection.search() without fill / cloudless portion filters gives valid results for different
    collections.
    """
    region: Dict = request.getfixturevalue(region)
    gd_collection = MaskedCollection.from_name(name)
    searched_collection = gd_collection.search(start_date, end_date, region)

    properties = searched_collection.properties
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    im_dates = np.array(
        [datetime.utcfromtimestamp(im_props['system:time_start'] / 1000) for im_props in properties.values()]
    )
    # test FILL_PORTION and CLOUDLESS_PORTION are not in properties
    prop_keys = list(properties.values())[0].keys()
    assert 'FILL_PORTION' not in prop_keys
    assert 'CLOUDLESS_PORTION' not in prop_keys
    # test search result image dates lie between `start_date` and `end_date`
    assert np.all(im_dates >= start_date) and np.all(im_dates < end_date)
    assert set(searched_collection.schema.keys()) >= set(list(properties.values())[0].keys())
    # test search result image dates are sorted
    assert np.all(sorted(im_dates) == im_dates)


@pytest.mark.parametrize(
    'image_list, method, region, date', [
        ('s2_sr_image_list', CompositeMethod.q_mosaic, 'region_10000ha', None),
        ('s2_sr_image_list', CompositeMethod.q_mosaic, None, '2021-10-01'),
        ('gedi_image_list', CompositeMethod.mosaic, 'region_10000ha', None),
        ('gedi_image_list', CompositeMethod.mosaic, None, '2020-09-01'),
        ('l8_9_image_list', CompositeMethod.medoid, 'region_10000ha', None),
        ('l8_9_image_list', CompositeMethod.medoid, None, '2021-10-01'),
    ]
)
def test_composite_region_date_ordering(image_list, method, region, date, request):
    """
    In MaskedCollection.composite(), test the component images are ordered correctly, according to `region`/`date`
    parameters.
    """
    image_list: List = request.getfixturevalue(image_list)
    region: Dict = request.getfixturevalue(region) if region else None
    gd_collection = MaskedCollection.from_list(image_list)
    ee_collection = gd_collection._prepare_for_composite(method=method, date=date, region=region)
    properties = gd_collection._get_properties(ee_collection)
    assert len(properties) == len(image_list)
    if region:
        # test images are ordered by CLOUDLESS/FILL_PORTION
        schema_keys = list(gd_collection.schema.keys())
        # CLOUDLESS_PORTION is not in MaskedCollection.schema for generic images, so use FILL_PORTION instead
        portion_key = 'CLOUDLESS_PORTION' if 'CLOUDLESS_PORTION' in schema_keys else 'FILL_PORTION'
        im_portions = [im_props[portion_key] for im_props in properties.values()]
        assert sorted(im_portions) == im_portions
    elif date:
        # test images are ordered by time difference with `date`
        im_dates = np.array(
            [datetime.utcfromtimestamp(im_props['system:time_start'] / 1000) for im_props in properties.values()]
        )
        comp_date = datetime.strptime(date, '%Y-%m-%d')
        im_date_diff = np.abs(comp_date - im_dates)
        assert all(sorted(im_date_diff, reverse=True) == im_date_diff)


@pytest.mark.parametrize(
    'image_list, method, mask', [
        ('s2_sr_image_list', CompositeMethod.q_mosaic, True), ('s2_sr_image_list', CompositeMethod.mosaic, False),
        ('l8_9_image_list', CompositeMethod.medoid, True), ('l8_9_image_list', CompositeMethod.median, False),
        ('s2_sr_image_list', CompositeMethod.medoid, True), ('s2_sr_image_list', CompositeMethod.medoid, False),
        ('l8_9_image_list', CompositeMethod.medoid, False),
    ]
)
def test_composite_mask(image_list, method, mask, region_100ha, request):
    """
    Test the MaskedImage.composite() `mask` parameter results in a masked composite image, comprised of masked
    component images.
    """

    image_list: List = request.getfixturevalue(image_list)
    gd_collection = MaskedCollection.from_list(image_list)
    proj = get_projection(gd_collection.ee_collection.first(), min_scale=True)
    ee_collection = gd_collection._prepare_for_composite(method=method, mask=mask)
    properties = gd_collection._get_properties(ee_collection)
    assert len(properties) == len(image_list)

    def count_masked_pixels(ee_image: ee.Image, count_list: ee.List):
        """ Return the pixel count (area) of the EE mask / valid image area.  """
        ee_mask = ee_image.select('SR_B.*|B.*|rh.*').mask().reduce(ee.Reducer.allNonZero()).rename('EE_MASK')
        count = ee_mask.reduceRegion(reducer='sum', crs=proj.crs(), scale=proj.nominalScale(), geometry=region_100ha)
        return ee.List(count_list).add(count.get('EE_MASK'))

    # get the mask pixel counts for the component images
    component_mask_counts = ee_collection.iterate(count_masked_pixels, ee.List([])).getInfo()
    # get the mask pixel count for the composite image
    composite_im = gd_collection.composite(method=method, mask=mask)
    composite_mask_count = count_masked_pixels(composite_im.ee_image, []).get(0).getInfo()
    if mask:
        # test the composite mask (valid area) is smaller than combined area of the component image masks
        assert composite_mask_count <= np.sum(component_mask_counts)
        # test the composite mask is larger than the smallest of the component image masks
        assert composite_mask_count >= np.min(component_mask_counts)
    else:
        # test that the component mask areas are ~ equal to composite mask area (i.e. the image area)
        assert np.array(component_mask_counts) == pytest.approx(composite_mask_count, rel=.01)


@pytest.mark.parametrize(
    'image_list, resampling, std_scale', [
        ('s2_sr_image_list', ResamplingMethod.bilinear, 60), ('s2_sr_image_list', ResamplingMethod.bicubic, 60),
        ('s2_sr_image_list', ResamplingMethod.average, 120), ('l8_9_image_list', ResamplingMethod.bilinear, 30),
        ('l8_9_image_list', ResamplingMethod.bicubic, 30), ('l8_9_image_list', ResamplingMethod.average, 120),
    ]
)
def test_composite_resampling(
    image_list: str, resampling: ResamplingMethod, std_scale: float, region_10000ha: Dict,
    request: pytest.FixtureRequest
):
    """ Test that resampling smooths the composite image. """

    image_list: List = request.getfixturevalue(image_list)
    gd_collection = MaskedCollection.from_list(image_list)
    comp_im_before = gd_collection.composite(method=CompositeMethod.mosaic, mask=False)
    comp_im_after = gd_collection.composite(method=CompositeMethod.mosaic, resampling=resampling, mask=False)

    # test the resampled composite is smoother than the default composite
    std_before = get_image_std(comp_im_before.ee_image, region_10000ha, std_scale)
    std_after = get_image_std(comp_im_after.ee_image, region_10000ha, std_scale)
    assert std_before > std_after


def test_composite_s2_cloud_mask_params(s2_sr_image_list, region_10000ha):
    """
    Test cloud/shadow mask **kwargs are passed from MaskedCollection.composite() through to
    Sentinel2ClImage._aux_image().
    """
    gd_collection = MaskedCollection.from_list(s2_sr_image_list)
    comp_im_prob80 = gd_collection.composite(prob=80)
    comp_im_prob80._set_region_stats(region_10000ha, scale=gd_collection.stats_scale)
    comp_im_prob40 = gd_collection.composite(prob=40)
    comp_im_prob40._set_region_stats(region_10000ha, scale=gd_collection.stats_scale)
    prob80_portion = comp_im_prob80.properties['FILL_PORTION']
    prob40_portion = comp_im_prob40.properties['FILL_PORTION']
    assert prob80_portion > prob40_portion


def test_composite_landsat_cloud_mask_params(l8_9_image_list, region_10000ha):
    """
    Test cloud/shadow mask **kwargs are passed from MaskedCollection.composite() through to
    LandsatImage._aux_image().
    """
    gd_collection = MaskedCollection.from_list(l8_9_image_list)
    comp_im_wshadows = gd_collection.composite(mask_shadows=False)
    comp_im_wshadows._set_region_stats(region_10000ha, scale=gd_collection.stats_scale)
    comp_im_woshadows = gd_collection.composite(mask_shadows=True)
    comp_im_woshadows._set_region_stats(region_10000ha, scale=gd_collection.stats_scale)
    with_shadows_portion = comp_im_wshadows.properties['FILL_PORTION']
    without_shadows_portion = comp_im_woshadows.properties['FILL_PORTION']
    assert with_shadows_portion > without_shadows_portion


@pytest.mark.parametrize(
    'image_list, method, mask, region, date, cloud_kwargs', [
        ('s2_sr_image_list', CompositeMethod.q_mosaic, True, 'region_100ha', None, {}),
        ('s2_sr_image_list', CompositeMethod.mosaic, True, None, '2021-10-01', {}),
        ('s2_sr_image_list', CompositeMethod.medoid, False, None, None, {}),
        (
            's2_sr_image_list', CompositeMethod.median, True, None, None,
            dict(
                mask_method='qa', mask_cirrus=False, mask_shadows=False, prob=60, dark=0.2, shadow_dist=500, buffer=500,
                cdi_thresh=None, max_cloud_dist=2000
            )
        ),
        ('l8_9_image_list', CompositeMethod.q_mosaic, True, 'region_100ha', None, {}),
        ('l8_9_image_list', CompositeMethod.mosaic, False, None, '2022-03-01', {}),
        ('l8_9_image_list', CompositeMethod.medoid, True, None, None, dict(mask_cirrus=False, mask_shadows=False)),
        ('l8_9_image_list', CompositeMethod.median, True, None, None, {}),
        ('l4_5_image_list', CompositeMethod.q_mosaic, False, 'region_100ha', None, {}),
        (
            'l4_5_image_list', CompositeMethod.mosaic, True, None, '1988-01-01',
            dict(mask_cirrus=False, mask_shadows=False)
        ),
        ('l4_5_image_list', CompositeMethod.medoid, True, None, None, {}),
        ('l4_5_image_list', CompositeMethod.median, True, None, None, {}),
        ('gedi_image_list', CompositeMethod.mosaic, True, 'region_100ha', None, {}),
        ('gedi_image_list', CompositeMethod.mosaic, True, None, '2020-09-01', {}),
        ('gedi_image_list', CompositeMethod.medoid, True, None, None, {}),
    ]
)  # yapf: disable
def test_composite(image_list, method, mask, region, date, cloud_kwargs, request):
    """ Test MaskedCollection.composite() runs successfully with a variety of `method` and other parameters. """
    image_list: List = request.getfixturevalue(image_list)
    region: Dict = request.getfixturevalue(region) if region else None
    gd_collection = MaskedCollection.from_list(image_list)
    comp_im = gd_collection.composite(method=method, mask=mask, region=region, date=date, **cloud_kwargs)
    assert comp_im._ee_info is not None and len(comp_im._ee_info) > 0
    assert 'INPUT_IMAGES' in comp_im.properties


def test_composite_errors(gedi_image_list, region_100ha):
    """ Test MaskedCollection.composite() error conditions. """
    gedi_collection = MaskedCollection.from_list(gedi_image_list)
    with pytest.raises(ValueError):
        # q-mosaic is only supported on cloud/shadow maskable images
        gedi_collection.composite(method=CompositeMethod.q_mosaic)
    with pytest.raises(ValueError):
        # unknown method
        gedi_collection.composite(method='unknown')
    with pytest.raises(ValueError):
        # date format error
        gedi_collection.composite(method=CompositeMethod.mosaic, date='not a date')
    with pytest.raises(ValueError):
        # composite of empty collection
        empty_collection = gedi_collection.search('2000-01-01', '2000-01-02', region_100ha, 100)
        empty_collection.composite(method=CompositeMethod.mosaic)


@pytest.mark.parametrize('image_list', ['s2_sr_image_list', 'l8_9_image_list'])
def test_composite_date(image_list: str, request: pytest.FixtureRequest):
    """ Test the composite date is the same as the first input image date. """

    image_list: List = request.getfixturevalue(image_list)
    gd_collection = MaskedCollection.from_list(image_list)
    # assumes the image_list's are in date order
    first_date = datetime.utcfromtimestamp(
        gd_collection.ee_collection.first().get('system:time_start').getInfo() / 1000
    ) # yapf: disable
    comp_im = gd_collection.composite()
    assert comp_im.date == first_date


def test_composite_mult_kwargs(region_100ha):
    """
    When a search filtered collection is composited, test that masks change with different cloud/shadow kwargs i.e.
    test that image *_MASK bands are overwritten in the encapsulated collection.
    """
    gd_collection = MaskedCollection.from_name('COPERNICUS/S2_SR')
    filt_collection = gd_collection.search('2022-01-01', '2022-01-10', region_100ha)

    comp_im_prob80 = filt_collection.composite(prob=80)
    comp_im_prob80._set_region_stats(region_100ha, scale=filt_collection.stats_scale)
    comp_im_prob40 = filt_collection.composite(prob=40)
    comp_im_prob40._set_region_stats(region_100ha, scale=filt_collection.stats_scale)

    cp_prob80 = comp_im_prob80.properties['FILL_PORTION']
    cp_prob40 = comp_im_prob40.properties['FILL_PORTION']

    assert cp_prob80 != pytest.approx(cp_prob40, abs=1e-1)


@pytest.mark.parametrize(
    'name', ['FAO/WAPOR/2/L1_RET_E', 'MODIS/006/MCD43A4']
)
def test_unbounded_search_no_region(name):
    """
    Test searching an unbounded collection without a region raises an exception.
    """
    start_date = '2022-01-01'
    end_date = '2022-01-02'
    gd_collection = MaskedCollection.from_name(name)
    gd_collection = gd_collection.search(start_date, end_date, fill_portion=0)
    with pytest.raises(ValueError) as ex:
        _ = gd_collection.properties
    assert 'unbounded' in str(ex) and 'region' in str(ex)


def test_unknown_collection_error(region_25ha):
    """
    Test searching a non-existent collection gives an error.
    """
    start_date = '2022-01-01'
    end_date = '2022-01-02'
    gd_collection = MaskedCollection.from_name('Unknown')
    with pytest.raises(ee.EEException) as ex:
        gd_collection = gd_collection.search(start_date, end_date, region_25ha)
        _ = gd_collection.properties
    assert 'not found' in str(ex)

