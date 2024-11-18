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

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Union

import ee
import numpy as np
import pytest

from geedim import schema
from geedim.collection import MaskedCollection
from geedim.enums import CompositeMethod, ResamplingMethod
from geedim.errors import InputImageError
from geedim.mask import MaskedImage
from geedim.utils import get_projection, split_id


@pytest.fixture()
def l4_5_images(l4_image_id, l5_masked_image) -> List[Union[str, MaskedImage]]:
    """A list of landsat 4 & 5 image ID / MaskedImage's"""
    return [l4_image_id, l5_masked_image]


@pytest.fixture()
def l8_9_images(l8_image_id, l9_masked_image) -> List[Union[str, MaskedImage]]:
    """A list of landsat 8 & 9 image IDs/ MaskedImage's"""
    return [l8_image_id, l9_masked_image]


@pytest.fixture()
def s2_sr_hm_images(s2_sr_hm_image_ids: list[str]) -> list[str | MaskedImage]:
    """A list of harmonised Sentinel-2 SR image IDs / MaskedImage's with QA* data, covering `region_*ha` with partial
    cloud/shadow.
    """
    # TODO: to get a named ImageCollection that can mask clouds with just these images,
    #  use ee.ImageCollection.filter(ee.Filter.inList).  this would be more for testing the new
    #  ImageCollectionAccessor
    image_list = s2_sr_hm_image_ids.copy()
    image_list[-1] = MaskedImage.from_id(image_list[-1])
    return image_list


@pytest.fixture()
def gedi_image_list() -> List[Union[str, MaskedImage]]:
    """A list of GEDI canopy top height IDs/ MaskedImage's"""
    return [
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202008_018E_036S',
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202009_018E_036S',
        MaskedImage.from_id('LARSE/GEDI/GEDI02_A_002_MONTHLY/202005_018E_036S'),
    ]


def test_split_id():
    """Test split_id()."""
    coll_name, im_id = split_id('A/B/C')
    assert coll_name == 'A/B'
    assert im_id == 'C'
    coll_name, im_id = split_id('ABC')
    assert coll_name == ''
    assert im_id == 'ABC'


@pytest.mark.parametrize('name', ['s2_sr_hm_image_id', 'l9_image_id'])
def test_from_name(name: str, request):
    """Test MaskedCollection.from_name()."""
    name = request.getfixturevalue(name)
    name, _ = split_id(name)
    gd_collection = MaskedCollection.from_name(name)
    assert gd_collection.id == name
    assert gd_collection.schema is not None
    assert len(gd_collection.schema) >= len(schema.default_prop_schema)
    assert gd_collection.ee_collection == ee.ImageCollection(name)


def test_from_list_errors(landsat_image_ids, user_masked_image):
    """Test MaskedCollection.from_list() raises an error with images from incompatible
    collections.
    """
    with pytest.raises(InputImageError):
        MaskedCollection.from_list([landsat_image_ids[0], user_masked_image])

    with pytest.raises(InputImageError):
        MaskedCollection.from_list([*landsat_image_ids])


@pytest.mark.parametrize('image_list', ['l4_5_images', 'l8_9_images'])
def test_from_list_landsat(image_list: str, request):
    """Test MaskedCollection.from_list() works with landsat images from different, but spectrally
    compatible collections.
    """
    image_list: List = request.getfixturevalue(image_list)
    gd_collection = MaskedCollection.from_list(image_list)
    assert 'LANDSAT' in gd_collection.id
    assert gd_collection.properties is not None


@pytest.mark.parametrize('image_list', ['s2_sr_hm_images', 'gedi_image_list'])
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
    assert gd_collection.id == split_id(image_ids[0])[0]
    assert len(gd_collection.properties) == len(image_list)
    assert list(gd_collection.properties.keys()) == image_ids
    assert set(gd_collection.schema.keys()) > set(list(gd_collection.properties.values())[0].keys())
    assert gd_collection.properties_table is not None
    assert gd_collection.schema_table is not None


@pytest.mark.parametrize('image_list', ['s2_sr_hm_images', 'gedi_image_list'])
def test_from_list_order(image_list: str, request):
    """Test MaskedCollection.from_list() maintains the order of the provided image list."""
    image_list: List = request.getfixturevalue(image_list)[::-1]
    image_ids = [im_obj if isinstance(im_obj, str) else im_obj.id for im_obj in image_list]
    gd_collection = MaskedCollection.from_list(image_list)
    assert list(gd_collection.properties.keys()) == image_ids


def test_from_list_ee_image(gedi_image_list: List):
    """Test MaskedCollection.from_list() with an ee.Image in the list."""
    image_ids = [im_obj if isinstance(im_obj, str) else im_obj.id for im_obj in gedi_image_list]
    image_list = gedi_image_list
    image_list[1] = ee.Image(image_list[1])
    gd_collection = MaskedCollection.from_list(image_list)
    assert list(gd_collection.properties.keys()) == image_ids


@pytest.mark.parametrize(
    'image_list, add_props',
    [
        ('s2_sr_hm_images', ['AOT_RETRIEVAL_ACCURACY', 'CLOUDY_PIXEL_PERCENTAGE']),
        ('l8_9_images', ['CLOUD_COVER', 'GEOMETRIC_RMSE_VERIFY']),
    ],
)
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
    assert all(
        [add_prop in list(gd_collection.properties.values())[0].keys() for add_prop in add_props]
    )
    assert gd_collection.properties_table is not None
    assert gd_collection.schema_table is not None


@pytest.mark.parametrize(
    'name, start_date, end_date, region, fill_portion, cloudless_portion, is_csmask',
    [
        ('LANDSAT/LC09/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha', 0, 50, True),
        ('LANDSAT/LE07/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha', 0, 0, True),
        ('LANDSAT/LT05/C02/T1_L2', '2005-01-01', '2006-02-01', 'region_100ha', 40, 50, True),
        ('COPERNICUS/S2_SR', '2022-01-01', '2022-01-15', 'region_100ha', 0, 50, True),
        ('COPERNICUS/S2_HARMONIZED', '2022-01-01', '2022-01-15', 'region_100ha', 50, 40, True),
        (
            'LARSE/GEDI/GEDI02_A_002_MONTHLY',
            '2021-08-01',
            '2021-09-01',
            'region_100ha',
            0.1,
            0,
            False,
        ),
    ],
)
def test_search(
    name,
    start_date: str,
    end_date: str,
    region: str,
    fill_portion: float,
    cloudless_portion: float,
    is_csmask,
    request,
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
        [
            datetime.fromtimestamp(im_props['system:time_start'] / 1000)
            for im_props in properties.values()
        ]
    )
    # test FILL_PORTION in expected range
    im_fill_portions = np.array([im_props['FILL_PORTION'] for im_props in properties.values()])
    assert np.all(im_fill_portions >= fill_portion) and np.all(im_fill_portions <= 100)
    if is_csmask:  # is a cloud/shadow masked collection
        # test CLOUDLESS_PORTION in expected range
        im_cl_portions = np.array(
            [im_props['CLOUDLESS_PORTION'] for im_props in properties.values()]
        )
        assert np.all(im_cl_portions >= cloudless_portion) and np.all(im_cl_portions <= 100)
    # test search result image dates lie between `start_date` and `end_date`
    assert np.all(im_dates >= start_date) and np.all(im_dates < end_date)
    assert set(searched_collection.schema.keys()) == set(list(properties.values())[0].keys())
    # test search result image dates are sorted
    assert np.all(sorted(im_dates) == im_dates)


def test_empty_search(region_100ha):
    """Test MaskedCollection.search() for empty search results."""
    gd_collection = MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2')
    searched_collection = gd_collection.search(
        '2022-01-01', '2022-01-02', region_100ha, cloudless_portion=100
    )
    assert searched_collection.properties is not None
    assert len(searched_collection.properties) == 0
    assert searched_collection.properties_table is not None


def test_search_no_end_date(region_100ha):
    """Test MaskedCollection.search() with ``end_date=None`` searches for a day from ``start_date``."""
    gd_collection = MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2')
    searched_collection = gd_collection.search('2022-01-03', None, region_100ha)

    start_date = datetime.strptime('2022-01-03', '%Y-%m-%d')
    end_date = datetime.strptime('2022-01-04', '%Y-%m-%d')
    properties = searched_collection.properties
    im_dates = np.array(
        [
            datetime.fromtimestamp(im_props['system:time_start'] / 1000)
            for im_props in properties.values()
        ]
    )
    assert len(properties) > 0
    assert np.all(im_dates >= start_date) and np.all(im_dates < end_date)


def test_search_mult_kwargs(region_100ha):
    """When a search filtered collection is searched again, test that masks change with different
    cloud/shadow kwargs.
    """
    start_date = '2022-01-01'
    end_date = '2022-01-10'
    gd_collection = MaskedCollection.from_name('COPERNICUS/S2_SR_HARMONIZED')

    def get_cloudless_portion(properties: Dict) -> List[float]:
        return [prop_dict['CLOUDLESS_PORTION'] for prop_dict in properties.values()]

    cl_portions = []
    for score in [0.5, 0.5, 0.2]:
        gd_collection = gd_collection.search(
            start_date,
            end_date,
            region_100ha,
            mask_method='cloud-score',
            score=score,
            fill_portion=0,
        )
        cl_portions.append(get_cloudless_portion(gd_collection.properties))

    assert cl_portions[0] == pytest.approx(cl_portions[1], abs=1e-3)
    assert cl_portions[0] != pytest.approx(cl_portions[2], abs=1e-1)


def test_search_custom_filter(region_25ha):
    """
    Test that a CLOUDLESS_PORTION custom filter gives the same search results as the equivalent cloudless_portion
    kwarg specification.
    """
    start_date = '2022-01-01'
    end_date = '2022-02-01'
    gd_collection = MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2')
    kwarg_coll = gd_collection.search(start_date, end_date, region_25ha, cloudless_portion=90)
    cust_filt_coll = gd_collection.search(
        start_date, end_date, region_25ha, custom_filter='CLOUDLESS_PORTION>=90'
    )
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
    assert all(
        [add_prop in list(searched_coll.properties.values())[0].keys() for add_prop in add_props]
    )
    assert searched_coll.properties_table is not None
    assert searched_coll.schema_table is not None


@pytest.mark.parametrize(
    'name, start_date, end_date, region',
    [
        ('LANDSAT/LC09/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha'),
        ('COPERNICUS/S2_HARMONIZED', '2022-01-01', '2022-01-15', 'region_100ha'),
        ('LARSE/GEDI/GEDI02_A_002_MONTHLY', '2021-08-01', '2021-09-01', 'region_100ha'),
    ],
)
def test_search_no_fill_or_cloudless_portion(
    name: str, start_date: str, end_date: str, region: str, request: pytest.FixtureRequest
):
    """Test MaskedCollection.search() without fill / cloudless portion filters gives valid
    results for different collections.
    """
    region: Dict = request.getfixturevalue(region)
    gd_collection = MaskedCollection.from_name(name)
    searched_collection = gd_collection.search(start_date, end_date, region)

    properties = searched_collection.properties
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    im_dates = np.array(
        [
            datetime.fromtimestamp(im_props['system:time_start'] / 1000)
            for im_props in properties.values()
        ]
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


def test_search_errors():
    """Test MaskedCollection.search() argument combination errors."""
    gd_coll = MaskedCollection.from_name('COPERNICUS/S2_HARMONIZED')

    # no start_date or region args
    with pytest.raises(ValueError) as ex:
        gd_coll.search()
    assert 'region' in str(ex.value) and 'start_date' in str(ex.value)

    # fill_portion/cloudless_portion but no region arg
    with pytest.raises(ValueError) as ex:
        gd_coll.search(start_date='2020-01-01', fill_portion=0)
    assert 'fill_portion' in str(ex.value) and 'region' in str(ex.value)


@pytest.mark.parametrize(
    'image_list, method, region, date',
    [
        ('s2_sr_hm_images', CompositeMethod.q_mosaic, 'region_10000ha', None),
        ('s2_sr_hm_images', CompositeMethod.q_mosaic, None, '2021-10-01'),
        ('gedi_image_list', CompositeMethod.mosaic, 'region_10000ha', None),
        ('gedi_image_list', CompositeMethod.mosaic, None, '2020-09-01'),
        ('l8_9_images', CompositeMethod.medoid, 'region_10000ha', None),
        ('l8_9_images', CompositeMethod.medoid, None, '2021-10-01'),
    ],
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
    properties = MaskedCollection(ee_collection).properties
    assert len(properties) == len(image_list)

    if region:
        # test images are ordered by CLOUDLESS/FILL_PORTION, and that portions are not the same
        schema_keys = list(gd_collection.schema.keys())
        portion_key = 'CLOUDLESS_PORTION' if 'CLOUDLESS_PORTION' in schema_keys else 'FILL_PORTION'
        im_portions = [im_props[portion_key] for im_props in properties.values()]
        assert sorted(im_portions) == im_portions
        assert len(set(im_portions)) == len(im_portions)

    elif date:
        # test images are ordered by time difference with `date`
        im_dates = np.array(
            [
                datetime.fromtimestamp(im_props['system:time_start'] / 1000)
                for im_props in properties.values()
            ]
        )
        comp_date = datetime.strptime(date, '%Y-%m-%d')
        im_date_diff = np.abs(comp_date - im_dates)
        assert all(sorted(im_date_diff, reverse=True) == im_date_diff)


def _get_masked_portion(
    ee_image: ee.Image, proj: ee.Projection = None, region: dict = None
) -> ee.Number:
    """Return the valid portion of the ``ee_image`` inside ``region``.  Assumes the ``region`` is
    completely covered by ``ee_image``.
    """
    proj = proj or get_projection(ee_image)
    ee_mask = (
        ee_image.select('SR_B.*|B.*|rh.*').mask().reduce(ee.Reducer.allNonZero()).rename('EE_MASK')
    )
    mean = ee_mask.reduceRegion(
        reducer='mean', crs=proj, scale=proj.nominalScale(), geometry=region
    )
    return ee.Number(mean.get('EE_MASK')).multiply(100)


@pytest.mark.parametrize(
    'image_list, method, mask',
    [
        ('s2_sr_hm_images', CompositeMethod.q_mosaic, True),
        ('s2_sr_hm_images', CompositeMethod.mosaic, False),
        ('l8_9_images', CompositeMethod.medoid, True),
        ('l8_9_images', CompositeMethod.median, False),
        ('s2_sr_hm_images', CompositeMethod.medoid, True),
        ('s2_sr_hm_images', CompositeMethod.medoid, False),
        ('l8_9_images', CompositeMethod.medoid, False),
    ],
)
def test_composite_mask(image_list, method, mask, region_100ha, request):
    """In MaskedImage.composite(), test masking of component and composite images with the `mask` parameter."""
    # TODO: combine >1 getInfo() calls into 1 for all tests
    # form the composite collection and image
    image_list: list = request.getfixturevalue(image_list)
    gd_collection = MaskedCollection.from_list(image_list)
    ee_collection = gd_collection._prepare_for_composite(method=method, mask=mask)
    composite_im = gd_collection.composite(method=method, mask=mask)
    proj = get_projection(ee_collection.first(), min_scale=True)

    def get_masked_portions(ee_image: ee.Image, portions: ee.List) -> ee.List:
        """Append the valid portion of ``ee_image`` to  ``portions``."""
        portion = _get_masked_portion(ee_image, proj=proj, region=region_100ha)
        return ee.List(portions).add(portion)

    # get the mask portions for the component and composite images
    component_portions = ee_collection.iterate(get_masked_portions, ee.List([]))
    composite_portion = _get_masked_portion(composite_im.ee_image, proj=proj, region=region_100ha)
    component_portions, composite_portion = ee.List(
        [component_portions, composite_portion]
    ).getInfo()
    component_portions = np.array(component_portions)
    assert len(component_portions) == len(image_list)

    # test masking of components and composite image
    if mask:
        assert np.all(component_portions > 0) and np.all(component_portions < 100)
        assert composite_portion <= np.sum(component_portions)
        assert composite_portion >= np.min(component_portions)
    else:
        assert component_portions == pytest.approx(100, abs=2)
        assert composite_portion >= component_portions.max()


@pytest.mark.parametrize(
    'masked_image, method, mask_kwargs',
    [
        ('s2_sr_hm_nocp_masked_image', 'q-mosaic', dict(mask_method='cloud-prob')),
        ('s2_sr_hm_qa_zero_masked_image', 'medoid', dict(mask_method='qa')),
        ('s2_sr_hm_nocs_masked_image', 'median', dict(mask_method='cloud-score')),
    ],
)
def test_composite_s2_mask_missing_data(
    masked_image: str, method: str, mask_kwargs: dict, region_100ha, request
):
    """In MaskedImage.composite(), test when an S2 component image is fully masked due to missing cloud data,
    the composite image is also fully masked.
    """
    # form the composite collection and image
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    image_list = [masked_image]
    gd_collection = MaskedCollection.from_list(image_list)

    ee_collection = gd_collection._prepare_for_composite(method=method, mask=True, **mask_kwargs)
    composite_im = gd_collection.composite(method=method, mask=True, **mask_kwargs)
    proj = get_projection(ee_collection.first(), min_scale=True)

    # get the mask portions for the component and composite images
    component_portion = _get_masked_portion(
        ee_collection.first(), proj=proj, region=region_100ha
    ).getInfo()
    composite_portion = _get_masked_portion(
        composite_im.ee_image, proj=proj, region=region_100ha
    ).getInfo()

    # test component and composite images are fully masked
    assert component_portion == composite_portion == 0


def test_composite_s2_q_mosaic_missing_data(
    s2_sr_hm_nocs_masked_image: MaskedImage, region_100ha: dict
):
    """In MaskedImage.composite(), test when an S2 component image is unmasked, but has masked CLOUD_DIST band due to
    missing cloud data, the composite image is fully masked with 'q-mosaic' method.
    """
    # form the composite collection and image
    image_list = [s2_sr_hm_nocs_masked_image]
    gd_collection = MaskedCollection.from_list(image_list)

    kwargs = dict(method='q-mosaic', mask_method='cloud-score', mask=False)
    ee_collection = gd_collection._prepare_for_composite(**kwargs)
    composite_im = gd_collection.composite(**kwargs)
    proj = get_projection(ee_collection.first(), min_scale=True)

    # get and test the mask portions for the component and composite images
    component_portion = _get_masked_portion(
        ee_collection.first(), proj=proj, region=region_100ha
    ).getInfo()
    composite_portion = _get_masked_portion(
        composite_im.ee_image, proj=proj, region=region_100ha
    ).getInfo()
    assert component_portion == pytest.approx(100, abs=1)
    assert composite_portion == 0


@pytest.mark.parametrize(
    'image_list, resampling, scale',
    [
        ('s2_sr_hm_images', ResamplingMethod.bilinear, 7.5),
        ('s2_sr_hm_images', ResamplingMethod.bicubic, 7.5),
        ('s2_sr_hm_images', ResamplingMethod.average, 30),
        ('l8_9_images', ResamplingMethod.bilinear, 20),
        ('l8_9_images', ResamplingMethod.bicubic, 20),
        ('l8_9_images', ResamplingMethod.average, 50),
    ],
)
def test_composite_resampling(
    image_list: str,
    resampling: ResamplingMethod,
    scale: float,
    region_100ha: Dict,
    request: pytest.FixtureRequest,
):
    """Test that resampling smooths the composite image."""
    image_list: List = request.getfixturevalue(image_list)
    gd_collection = MaskedCollection.from_list(image_list)
    comp_im = gd_collection.composite(method=CompositeMethod.mosaic, mask=False)
    comp_im_resampled = gd_collection.composite(
        method=CompositeMethod.mosaic, resampling=resampling, mask=False
    )

    # find mean of std deviations of reflectance bands for each composite image
    crs = gd_collection.ee_collection.first().select(0).projection().crs()
    stds = []
    for im in [comp_im, comp_im_resampled]:
        im = im.ee_image.select(gd_collection.refl_bands)
        im = im.reproject(crs=crs, scale=scale)  # required to resample at scale
        std = im.reduceRegion('stdDev', geometry=region_100ha).values().reduce('mean')
        stds.append(std)
    stds = ee.List(stds).getInfo()

    # test comp_im_resampled is smoother than comp_im
    assert stds[1] < stds[0]


def test_composite_s2_cloud_mask_params(s2_sr_hm_images, region_100ha):
    """
    Test cloud/shadow mask **kwargs are passed from MaskedCollection.composite() through to
    Sentinel2ClImage._aux_image().
    """
    gd_collection = MaskedCollection.from_list(s2_sr_hm_images)
    proj = get_projection(gd_collection.ee_collection.first())
    mask_portions = []
    for score in [0.3, 0.5]:
        comp_im = gd_collection.composite(mask_method='cloud-score', score=score)
        portion = _get_masked_portion(comp_im.ee_image, proj=proj, region=region_100ha)
        mask_portions.append(portion)

    mask_portions = ee.List(mask_portions).getInfo()
    assert mask_portions[0] > mask_portions[1] > 0


def test_composite_landsat_cloud_mask_params(l8_9_images, region_10000ha):
    """
    Test cloud/shadow mask **kwargs are passed from MaskedCollection.composite() through to
    LandsatImage._aux_image().
    """
    gd_collection = MaskedCollection.from_list(l8_9_images)
    proj = gd_collection.ee_collection.first().projection()
    mask_portions = []
    for mask_shadows in [False, True]:
        comp_im = gd_collection.composite(mask_shadows=mask_shadows)
        mask_portion = _get_masked_portion(comp_im.ee_image, proj=proj, region=region_10000ha)
        mask_portions.append(mask_portion)

    mask_portions = ee.List(mask_portions).getInfo()
    assert mask_portions[0] > mask_portions[1]


@pytest.mark.parametrize(
    'image_list, method, mask, region, date, cloud_kwargs',
    [
        ('s2_sr_hm_images', CompositeMethod.q_mosaic, True, 'region_100ha', None, {}),
        ('s2_sr_hm_images', CompositeMethod.mosaic, True, None, '2021-10-01', {}),
        ('s2_sr_hm_images', CompositeMethod.medoid, False, None, None, {}),
        (
            's2_sr_hm_images',
            CompositeMethod.median,
            True,
            None,
            None,
            dict(mask_method='cloud-score', score=0.4),
        ),
        ('l8_9_images', CompositeMethod.q_mosaic, True, 'region_100ha', None, {}),
        ('l8_9_images', CompositeMethod.mosaic, False, None, '2022-03-01', {}),
        (
            'l8_9_images',
            CompositeMethod.medoid,
            True,
            None,
            None,
            dict(mask_cirrus=False, mask_shadows=False),
        ),
        ('l8_9_images', CompositeMethod.median, True, None, None, {}),
        ('l4_5_images', CompositeMethod.q_mosaic, False, 'region_100ha', None, {}),
        (
            'l4_5_images',
            CompositeMethod.mosaic,
            True,
            None,
            '1988-01-01',
            dict(mask_cirrus=False, mask_shadows=False),
        ),
        ('l4_5_images', CompositeMethod.medoid, True, None, None, {}),
        ('l4_5_images', CompositeMethod.median, True, None, None, {}),
        ('gedi_image_list', CompositeMethod.mosaic, True, 'region_100ha', None, {}),
        ('gedi_image_list', CompositeMethod.mosaic, True, None, '2020-09-01', {}),
        ('gedi_image_list', CompositeMethod.medoid, True, None, None, {}),
    ],
)
def test_composite(
    image_list: str,
    method: CompositeMethod,
    mask: bool,
    region: str | None,
    date: str | None,
    cloud_kwargs: dict,
    request: pytest.FixtureRequest,
):
    """Test MaskedCollection.composite() runs successfully with a variety of `method` and other parameters."""
    image_list: list = request.getfixturevalue(image_list)
    region: dict = request.getfixturevalue(region) if region else None
    gd_collection = MaskedCollection.from_list(image_list)
    comp_im = gd_collection.composite(
        method=method, mask=mask, region=region, date=date, **cloud_kwargs
    )
    assert comp_im._ee_info is not None and len(comp_im._ee_info) > 0
    assert gd_collection.id in comp_im.id
    assert f'{method.value.upper()}-COMP' in comp_im.id


def test_composite_errors(gedi_image_list, region_100ha):
    """Test MaskedCollection.composite() error conditions."""
    gedi_collection = MaskedCollection.from_list(gedi_image_list)
    with pytest.raises(ValueError):
        # q-mosaic is only supported on cloud/shadow maskable images
        gedi_collection.composite(method=CompositeMethod.q_mosaic)
    with pytest.raises(ValueError):
        # unknown method
        gedi_collection.composite(method='unknown')
    with pytest.raises(ee.EEException):
        # date format error
        gedi_collection.composite(method=CompositeMethod.mosaic, date='not a date')


@pytest.mark.parametrize('image_list', ['s2_sr_hm_images', 'l8_9_images'])
def test_composite_date(image_list: str, request: pytest.FixtureRequest):
    """Test the composite date is the same as the first input image date."""
    image_list: List = request.getfixturevalue(image_list)
    gd_collection = MaskedCollection.from_list(image_list)
    first_date = datetime.fromtimestamp(
        gd_collection.ee_collection.sort('system:time_start')
        .first()
        .get('system:time_start')
        .getInfo()
        / 1000,
        tz=timezone.utc,
    )
    comp_im = gd_collection.composite()
    assert comp_im.date == first_date


def test_composite_mult_kwargs(region_100ha):
    """When a search filtered collection is composited, test that masks change with different
    cloud/shadow kwargs i.e. test that image *_MASK bands are overwritten in the encapsulated
    collection.
    """
    gd_collection = MaskedCollection.from_name('COPERNICUS/S2_SR_HARMONIZED')
    filt_collection = gd_collection.search('2022-01-01', '2022-01-10', region_100ha)
    proj = get_projection(filt_collection.ee_collection.first())
    mask_portions = []
    for score in [0.3, 0.5]:
        comp_im = filt_collection.composite(mask_method='cloud-score', score=score)
        mask_portion = _get_masked_portion(comp_im.ee_image, proj=proj, region=region_100ha)
        mask_portions.append(mask_portion)

    mask_portions = ee.List(mask_portions).getInfo()
    assert mask_portions[0] != pytest.approx(mask_portions[1], abs=1e-1)


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
