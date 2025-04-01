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

import ee
import numpy as np
import pytest
from pandas import to_datetime

from geedim import schema
from geedim.collection import (
    ImageCollectionAccessor,
    MaskedCollection,
    _compatible_collections,
)
from geedim.download import BaseImage
from geedim.enums import CompositeMethod, ResamplingMethod
from geedim.errors import InputImageError
from geedim.mask import MaskedImage, _MaskedImage
from geedim.stac import STACClient
from geedim.utils import split_id

# TODO: ImageCollectionAccessor testing
#  - fromImages()
#    - Works with e.g. s2 ims or landsat ims.  Collection gets correct ID and images keep their
#    indexes.
#    - Raises error on incompatible ims or ids=None.
#  - schemaPropertyNames:
#    - test it updates the schema correctly with stac schema, geedim schema or unknown property
#    names
#    - test errors and empty behaviour
#  - schemaTable
#    - contains schemaPropertyNames, and works with unknown properties which have none vals
#    - empty schemaPropertyNames behaviour
#  - propertiesTable
#    - contains schemaPropertyNames abbreviations & >= num rows as images in collection.
#    - behaviour when schemaPropertyNames contains names not present in image properties
#    - empty schemaPropertyNames and empty collection behaviour
#    - conversion of datetime
#    - images with different properties?
#  - filter
#    - start_date and no end_date (or revert to EE default?)
#    - start_date, end_date & region
#    - start_date, end_date & region, fill_portion & cloudless_portion
#    - custom_filter
#  - prepare_for_composite
#    - mask bands present
#    - image ordering based on region / date
#    - masking based on mask
#    - errors
#  - composite
#    - test id, index & time properties
#    - test mask bands always present
#    - test when an S2 component image is has missing cloud data, the q-mosaic composite image is
#    fully masked with mask=True/False
#    - test mask param changes masked portion of the composite
# TODO: MaskedCollection


def accessors_from_collections(ee_colls: list[ee.ImageCollection]) -> list[ImageCollectionAccessor]:
    """Return a list of ImageCollectionAccessor objects, with cached info properties,
    for the given list of ee.ImageCollection objects, using a single getInfo() call.
    """

    def aggregate_images(image: ee.Image, images: ee.List) -> ee.List:
        return ee.List(images).add(image)

    coll_images = [ee_coll.iterate(aggregate_images, ee.List([])) for ee_coll in ee_colls]
    infos = ee.List([ee_colls, coll_images]).getInfo()

    colls = []
    for ee_coll, coll_info, images_info in zip(ee_colls, infos[0], infos[1]):
        coll = ImageCollectionAccessor(ee_coll)
        info = dict(**coll_info, features=images_info)
        coll._info = info
        colls.append(coll)

    return colls


@pytest.fixture(scope='session')
def s2_sr_hm_coll(s2_sr_hm_image_ids: list[str]) -> ImageCollectionAccessor:
    coll = ee.ImageCollection(s2_sr_hm_image_ids)
    coll = coll.set('system:id', 'COPERNICUS/S2_SR_HARMONIZED')
    return ImageCollectionAccessor(coll)


@pytest.fixture(scope='session')
def l9_sr_coll() -> ImageCollectionAccessor:
    # LC09_173083_20220308, LC09_173083_20221205, LC09_173083_20230106, LC09_173083_20240414
    image_ids = [
        'LANDSAT/LC09/C02/T1_L2/LC09_173083_20220308',
        'LANDSAT/LC09/C02/T1_L2/LC09_173083_20221205',
        'LANDSAT/LC09/C02/T1_L2/LC09_173083_20230106',
    ]  # TODO: make these a fixture like s2_sr_hm_image_ids?
    coll = ee.ImageCollection(image_ids)
    coll = coll.set('system:id', 'LANDSAT/LC09/C02/T1_L2')
    return ImageCollectionAccessor(coll)


@pytest.fixture(scope='session')
def modis_nbar_coll() -> ImageCollectionAccessor:
    coll = ee.ImageCollection('MODIS/061/MCD43A4')
    coll = coll.filterDate('2024-01-01', '2024-03-01')
    return ImageCollectionAccessor(coll.limit(3))


@pytest.fixture()
def gedi_cth_coll() -> ImageCollectionAccessor:
    image_ids = [
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202112_018E_036S',
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202205_018E_036S',
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202207_018E_036S',
    ]
    coll = ee.ImageCollection(image_ids)
    coll = coll.set('system:id', 'LARSE/GEDI/GEDI02_A_002_MONTHLY')
    return ImageCollectionAccessor(coll)


@pytest.fixture()
def l4_5_images(l4_image_id, l5_masked_image) -> list[str | MaskedImage]:
    """A list of landsat 4 & 5 image ID / MaskedImage's"""
    return [l4_image_id, l5_masked_image]


@pytest.fixture()
def l8_9_images(l8_image_id, l9_masked_image) -> list[str | MaskedImage]:
    """A list of landsat 8 & 9 image IDs/ MaskedImage's"""
    return [l8_image_id, l9_masked_image]


@pytest.fixture()
def s2_sr_hm_images(s2_sr_hm_image_ids: list[str]) -> list[str | MaskedImage]:
    """A list of harmonised Sentinel-2 SR image IDs / MaskedImage's with QA* data, covering
    `region_*ha` with partial cloud/shadow.
    """
    # TODO: to get a named ImageCollection that can mask clouds with just these images,
    #  use ee.ImageCollection.filter(ee.Filter.inList).  this would be more for testing the new
    #  ImageCollectionAccessor
    image_list = s2_sr_hm_image_ids.copy()
    image_list[-1] = MaskedImage.from_id(image_list[-1])
    return image_list


@pytest.fixture()
def gedi_image_list() -> list[str | MaskedImage]:
    """A list of GEDI canopy top height IDs/ MaskedImage's"""
    return [
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202008_018E_036S',
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202009_018E_036S',
        MaskedImage.from_id('LARSE/GEDI/GEDI02_A_002_MONTHLY/202005_018E_036S'),
    ]


def test_compatible_collections(
    l4_image_id: str,
    l5_image_id: str,
    l7_image_id: str,
    l8_image_id: str,
    l9_image_id: str,
    s2_sr_hm_image_ids: list[str],
):
    """Test _compatible_collections()."""

    def compatible_images(*im_ids) -> list[str]:
        """Extract collection IDs and return the result of _compatible_collections()."""
        coll_ids = [split_id(im_id)[0] for im_id in im_ids]
        return _compatible_collections(coll_ids)

    assert compatible_images(l4_image_id, l5_image_id) is True
    assert compatible_images(l8_image_id, l9_image_id) is True
    assert compatible_images(*s2_sr_hm_image_ids) is True
    assert compatible_images(l4_image_id, l5_image_id, l7_image_id) is False
    assert compatible_images(l4_image_id, l5_image_id, l8_image_id) is False
    assert compatible_images(l7_image_id, l8_image_id, l9_image_id) is False
    assert compatible_images(*s2_sr_hm_image_ids, l9_image_id) is False


def test_from_images(s2_sr_hm_image_ids: list[str]):
    """Test ImageCollectionAccessor.fromImages()."""
    # TODO: test the ID when passed composite image(s)
    coll = ImageCollectionAccessor.fromImages(s2_sr_hm_image_ids)
    info = coll.getInfo()
    assert info['id'] == 'COPERNICUS/S2_SR_HARMONIZED'
    assert set([im_props['id'] for im_props in info['features']]) == set(s2_sr_hm_image_ids)


def test_from_images_error(l7_image_id: str, l8_image_id: str):
    """Test ImageCollectionAccessor.fromImages() raises an error with incompatible images."""
    with pytest.raises(ValueError, match='spectrally compatible'):
        ImageCollectionAccessor.fromImages([l7_image_id, l8_image_id])


@pytest.mark.parametrize(
    'stac, exp_val',
    [
        ({'summaries': {'gsd': [10, 40], 'eo:bands': [{}]}}, 20),
        ({'summaries': {'gsd': [10], 'eo:bands': [{}]}}, 10),
        ({'summaries': {'eo:bands': [{'gsd': 10}, {'gsd': 40}]}}, 40),
        ({'summaries': {'eo:bands': [{'gsd': 10}, {'gsd': 400}]}}, 10),
    ],
)
def test_portion_scale(stac: dict, exp_val: float, monkeypatch: pytest.MonkeyPatch):
    """Test the _portion_scale property with different STAC dictionaries."""
    coll = ImageCollectionAccessor(None)
    # patch coll and STACClient so that coll's stac property returns the mock stac
    coll.id = None
    monkeypatch.setitem(STACClient()._cache, coll.id, stac)

    assert coll._portion_scale == exp_val


def test_properties(s2_sr_hm_coll: ImageCollectionAccessor):
    """Test properties that don't have their own specific tests."""
    # TODO: symmetry with test_image.py
    info = s2_sr_hm_coll._ee_coll.getInfo()
    props = {ip['properties']['system:index']: ip['properties'] for ip in info['features']}

    assert s2_sr_hm_coll.id == info['id']
    assert s2_sr_hm_coll.info == info
    assert s2_sr_hm_coll.properties == props

    assert s2_sr_hm_coll.stac is not None
    band_props = s2_sr_hm_coll.stac['summaries']['eo:bands']
    spec_bands = [bp['name'] for bp in band_props if 'center_wavelength' in bp]
    assert len(s2_sr_hm_coll.specBands) > 0
    assert s2_sr_hm_coll.specBands == spec_bands

    assert s2_sr_hm_coll.schema == schema.collection_schema[info['id']]['prop_schema']


@pytest.mark.parametrize('coll_id', ['COPERNICUS/S2_SR_HARMONIZED', 'LANDSAT/LC09/C02/T1_L2', None])
def test_schema_property_names_default(coll_id: str):
    """Test the schemaPropertyNames and schema property defaults for different collections."""
    # patch collection ID to avoid getInfo()
    coll = ImageCollectionAccessor(None)
    coll.id = coll_id

    schema_ = (
        schema.collection_schema[coll_id]['prop_schema']
        if (coll_id in schema.collection_schema)
        else schema.default_prop_schema
    )
    assert coll.schemaPropertyNames == list(schema_.keys())
    assert coll.schema == schema_


def test_schema_property_names_set():
    """Test setting the schemaPropertyNames property."""
    # patch collection ID to avoid getInfo()
    coll = ImageCollectionAccessor(None)
    coll.id = 'COPERNICUS/S2_SR_HARMONIZED'

    # set schemaPropertyNames
    schema_prop_names = ('CLOUDLESS_PORTION', 'CLOUD_COVERAGE_ASSESSMENT', 'unknownPropertyName')
    coll.schemaPropertyNames = schema_prop_names
    assert coll.schemaPropertyNames == schema_prop_names

    # test duplicate property names are removed
    coll.schemaPropertyNames += ('CLOUDLESS_PORTION', 'CLOUD_COVERAGE_ASSESSMENT')
    assert coll.schemaPropertyNames == schema_prop_names

    # test the schema property contains the correct new items
    assert tuple(coll.schema.keys()) == schema_prop_names
    for prop, abbrev, has_descr in zip(
        schema_prop_names, ['CLOUDLESS', 'CCA', 'UPN'], [True, True, False]
    ):
        assert coll.schema[prop]['abbrev'] == abbrev
        descr = coll.schema[prop]['description']
        if has_descr:
            assert len(descr) > 0
            assert '.' not in descr and '\n' not in descr
        else:
            assert descr is None


def test_schema_set_error():
    """Test setting the schemaPropertyNames property with a value that is not a list of strings
    raises an error.
    """
    # patch collection ID to avoid getInfo()
    coll = ImageCollectionAccessor(None)
    coll.id = None

    with pytest.raises(ValueError, match='iterable of strings'):
        coll.schemaPropertyNames = [123]


def test_schema_table():
    """Test the schemaTable property."""
    coll = ImageCollectionAccessor(None)
    # patch collection ID to a known collection
    coll.id = 'COPERNICUS/S2_SR_HARMONIZED'
    # add a property without a stac description
    coll.schemaPropertyNames += ['unknownPropertyName']
    assert len(coll.schemaTable.splitlines()) >= len(coll.schema) + 2
    assert all(pn in coll.schemaTable for pn in coll.schemaPropertyNames)

    # test empty schema
    coll.schemaPropertyNames = []
    assert coll.schemaTable == ''


def test_properties_table():
    """Test the propertiesTable property."""
    # mock a collection
    coll = ImageCollectionAccessor(None)
    coll.id = None
    # mock properties with names that vary between images & some values that are None
    coll._properties = {
        '1': {'system:index': '1', 'system:time_start': 0, 'propName': 'value'},
        '2': {'system:index': '2', 'system:time_start': 1e9, 'propName': 1.23},
        '3': {'system:index': '3', 'system:time_start': 1e9, 'propName': None},
        '4': {'system:index': '4', 'system:time_start': 2e9, 'anotherPropName': 'value'},
    }
    # mock a schema with properties that exist in all images, properties that exist in some
    # images, and properties that exist in none
    coll.schemaPropertyNames = [
        'system:index',
        'system:time_start',
        'FILL_PORTION',
        'propName',
        'unknownPropertyName',
    ]
    assert len(coll.propertiesTable.splitlines()) == len(coll.properties) + 2

    # test schema properties that exist in one or more images are included in the table
    incl_props = ['system:index', 'system:time_start', 'propName']
    incl_abbrevs = [coll.schema[ip]['abbrev'] for ip in incl_props]
    assert all(ia in coll.propertiesTable for ia in incl_abbrevs)

    # test schema properties that don't exist in any image are not included in the table
    excl_props = set(coll.schemaPropertyNames).difference(incl_props)
    excl_abbrevs = [coll.schema[ep]['abbrev'] for ep in excl_props]
    assert all(ea not in coll.propertiesTable for ea in excl_abbrevs)

    # test conversion of timestamps to date strings
    first_time_start = datetime.fromtimestamp(
        coll.properties['1']['system:time_start'] / 1000, tz=timezone.utc
    )
    first_time_start = datetime.strftime(first_time_start, '%Y-%m-%d %H:%M')
    assert first_time_start in coll.propertiesTable

    # test empty collection
    coll = ImageCollectionAccessor(None)
    coll.id = None
    coll._properties = {}
    assert coll.propertiesTable == ''

    # test empty schema
    coll = ImageCollectionAccessor(None)
    coll.id = None
    coll._properties = {'1': {'system:index': '1'}, '2': {'system:index': '2'}}
    coll._schema = {}
    assert coll.propertiesTable == ''


@pytest.mark.parametrize(
    'coll, exp_support',
    [('l9_sr_coll', True), ('s2_sr_hm_coll', True), ('modis_nbar_coll', False)],
)
def test_cs_support(coll: str, exp_support: bool, request: pytest.FixtureRequest):
    """Test the cloudShadowSupport property."""
    coll: ImageCollectionAccessor = request.getfixturevalue(coll)
    assert coll.cloudShadowSupport == exp_support


def test_add_mask_bands(l9_sr_coll: ImageCollectionAccessor):
    """Test addMaskBands()."""
    # This just tests bands exist and kwargs were passed. Detailed mask testing is done in
    # test_mask.py.
    cs_coll = l9_sr_coll.addMaskBands(mask_shadows=False)
    info = cs_coll.getInfo()
    assert len(info['features']) == len(l9_sr_coll.properties)
    for im_info in info['features']:
        band_names = [bi['id'] for bi in im_info['bands']]
        assert all(bn in band_names for bn in ['CLOUDLESS_MASK', 'CLOUD_DIST', 'FILL_MASK'])
        assert 'SHADOW_MASK' not in band_names


def test_mask_clouds(l9_sr_coll: ImageCollectionAccessor, region_100ha: dict):
    """Test maskClouds()."""
    # This just tests the masked area increases. Detailed mask testing is done in test_mask.py.

    def aggregate_mask_sum(image: ee.Image, sums: ee.List) -> ee.List:
        """Add the sum of the image masks to the sums list."""
        sum_ = image.mask().reduceRegion('sum', geometry=region_100ha).values().reduce('mean')
        return ee.List(sums).add(sum_)

    # find & compare image mask sums before and after masking
    masked_coll = ImageCollectionAccessor(l9_sr_coll.addMaskBands()).maskClouds()
    unmasked_sums = l9_sr_coll._ee_coll.iterate(aggregate_mask_sum, ee.List([]))
    masked_sums = masked_coll.iterate(aggregate_mask_sum, ee.List([]))
    # combine getInfo() calls into one
    sums = ee.Dictionary(dict(unmasked=unmasked_sums, masked=masked_sums)).getInfo()

    assert len(sums['unmasked']) == len(l9_sr_coll.properties)
    for masked_sum, unmasked_sum in zip(sums['masked'], sums['unmasked']):
        assert masked_sum < unmasked_sum


def test_medoid(l9_sr_coll: ImageCollectionAccessor, region_100ha: dict):
    """Test medoid()."""
    # relative difference of median and medoid (without bands arg)
    median_im = l9_sr_coll._ee_coll.median()
    medoid_im = l9_sr_coll.medoid()
    median_diff_im = medoid_im.subtract(median_im).divide(median_im)
    # relative difference of medoid without and with bands arg
    medoid_bands_im = l9_sr_coll.medoid(bands=l9_sr_coll.specBands[2:-2])
    bands_diff_im = medoid_bands_im.subtract(medoid_im).divide(medoid_im)

    # find means of differences, combining all getInfo() calls into one
    diffs = [
        im.reduceRegion('mean', geometry=region_100ha, scale=30)
        for im in [median_diff_im, bands_diff_im]
    ]
    diffs = ee.List(diffs).getInfo()

    # test medoid is similar to median
    assert all(abs(diffs[0][bn]) < 0.05 for bn in l9_sr_coll.specBands)
    # test bands argument changes medoid
    assert all(diffs[1][bn] != 0 for bn in l9_sr_coll.specBands)


def test_filter(region_10000ha: dict):
    """Test filter() with different parameters."""
    # get image properties for testing filter parameters (note that an ee.List or ee.Dictionary
    # containing nested image collections doesn't return collection image info on getInfo(), so
    # image properties are retrieved by other means):
    # start_date, end_date & region
    coll = ImageCollectionAccessor(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
    props = {}
    ref_kwargs = dict(start_date='2023-01-01', end_date='2024-01-01', region=region_10000ha)
    filt_coll = coll.filter(**ref_kwargs)
    props['date_region'] = dict(
        time_start=filt_coll.aggregate_array('system:time_start'),
        intersections=filt_coll.iterate(
            lambda im, inters: ee.List(inters).add(im.geometry().intersects(region_10000ha)),
            ee.List([]),
        ),
    )

    # start_date without end_date
    filt_coll = coll.filter(start_date=filt_coll.first().date().advance(-0.001, 'second'))
    props['start_date'] = filt_coll.aggregate_array('system:time_start')

    # fill_portion
    fill_portion = 99.95
    filt_coll = coll.filter(**ref_kwargs, fill_portion=fill_portion)
    props['fill_portion'] = [
        coll.filter(**ref_kwargs, fill_portion=fp).aggregate_array('FILL_PORTION')
        for fp in [0, fill_portion]
    ]

    # cloudless_portion
    cloudless_portion = 90
    filt_coll = coll.filter(**ref_kwargs, cloudless_portion=cloudless_portion)
    props['cloudless_portion'] = [
        coll.filter(**ref_kwargs, cloudless_portion=cp).aggregate_array('CLOUDLESS_PORTION')
        for cp in [0, cloudless_portion]
    ]

    # custom_filter (without FILL_PORTION or CLOUDLESS_PORTION)
    cloud_cover = 50
    props['custom_filter_nop'] = [
        coll.filter(**ref_kwargs, custom_filter=f'CLOUD_COVER<={cc}').aggregate_array('CLOUD_COVER')
        for cc in [100, cloud_cover]
    ]

    # custom_filter (with FILL_PORTION or CLOUDLESS_PORTION)
    props['custom_filter_p'] = [
        coll.filter(**ref_kwargs, custom_filter=f'CLOUDLESS_PORTION<{cp}').aggregate_array(
            'CLOUDLESS_PORTION'
        )
        for cp in [100, cloudless_portion]
    ]

    # cloud / shadow kwargs
    props['cs_kwargs'] = [
        coll.filter(**ref_kwargs, fill_portion=0, mask_shadows=ms).aggregate_array(
            'CLOUDLESS_PORTION'
        )
        for ms in [False, True]
    ]

    # combine getInfo() calls into one
    props = ee.Dictionary(props).getInfo()

    # test start_date, end_date & region filtering
    ref_dates = to_datetime(props['date_region']['time_start'], unit='ms')
    assert all(ref_dates >= to_datetime(ref_kwargs['start_date']))
    assert all(ref_dates <= to_datetime(ref_kwargs['end_date']))
    assert all(sorted(ref_dates) == ref_dates)
    assert all(props['date_region']['intersections'])

    # test start_date without end_date (end_date should default to a millisecond after start_date)
    assert len(props['start_date']) == 0

    # test fill_portion
    assert any(fp < fill_portion for fp in props['fill_portion'][0])
    assert all(fp >= fill_portion for fp in props['fill_portion'][1])

    # test cloudless_portion
    assert any(cp < cloudless_portion for cp in props['cloudless_portion'][0])
    assert all(cp >= cloudless_portion for cp in props['cloudless_portion'][1])

    # test custom_filter without FILL_PORTION / CLOUDLESS_PORTION
    assert any(cc >= cloud_cover for cc in props['custom_filter_nop'][0])
    assert all(cc < cloud_cover for cc in props['custom_filter_nop'][1])

    # test custom_filter with FILL_PORTION / CLOUDLESS_PORTION
    assert any(cp >= cloudless_portion for cp in props['custom_filter_p'][0])
    assert all(cp < cloudless_portion for cp in props['custom_filter_p'][1])

    # cloud / shadow kwargs (changing mask_shadows from False to True reduces the cloudless_portion)
    assert any(cp_nms > cp_ms for cp_nms, cp_ms in zip(*props['cs_kwargs'], strict=True))


def test_filter_error(l9_sr_coll: ImageCollectionAccessor, region_10000ha: dict):
    """Test filter() raises an error when fill_portion or cloudless_portion are supplied without
    region.
    """
    with pytest.raises(ValueError, match="'region' is required"):
        l9_sr_coll.filter(start_date='2023-07-01', fill_portion=50)
    with pytest.raises(ValueError, match="'region' is required"):
        l9_sr_coll.filter(start_date='2023-07-01', cloudless_portion=50)


def test_prepare_for_composite_date_region(
    l9_sr_coll: ImageCollectionAccessor, gedi_cth_coll: ImageCollectionAccessor, region_100ha: dict
):
    """Test sorting of the _prepare_for_composite() collection with date and region parameters."""

    def set_mask_portions(image: ee.Image, mi: _MaskedImage) -> ee.Image:
        """Set FILL_PORTION and CLOUDLESS_PORTION image properties."""
        image = mi.add_mask_bands(image)
        return mi.set_mask_portions(image, region=region_100ha)

    # create property lists for testing prepared collections
    # date:
    infos = {}
    date = '2022-12-05'
    comp_coll = l9_sr_coll._prepare_for_composite('q-mosaic', date=date)
    infos['date'] = [
        coll.aggregate_array('system:time_start') for coll in [l9_sr_coll._ee_coll, comp_coll]
    ]

    # region with cloud/shadow supported collection:
    src_coll = l9_sr_coll._ee_coll.map(lambda im: set_mask_portions(im, l9_sr_coll._mi))
    src_coll = ImageCollectionAccessor(src_coll)
    src_coll.id = l9_sr_coll.id  # patch id to avoid getInfo()
    comp_coll = src_coll._prepare_for_composite('q-mosaic', region=region_100ha)
    infos['region_cp'] = [
        coll.aggregate_array('CLOUDLESS_PORTION') for coll in [src_coll._ee_coll, comp_coll]
    ]

    # region with non-cloud/shadow supported collection:
    src_coll = gedi_cth_coll._ee_coll.map(lambda im: set_mask_portions(im, gedi_cth_coll._mi))
    src_coll = ImageCollectionAccessor(src_coll)
    src_coll.id = gedi_cth_coll.id  # patch id to avoid getInfo()
    comp_coll = src_coll._prepare_for_composite('mosaic', region=region_100ha)
    infos['region_fp'] = [
        coll.aggregate_array('FILL_PORTION') for coll in [src_coll._ee_coll, comp_coll]
    ]

    # combine getInfo() calls into one
    infos = ee.Dictionary(infos).getInfo()

    # test date:
    assert infos['date'][1] != infos['date'][0]
    date = to_datetime(date)
    date_dist = [abs(date - d) for d in to_datetime(infos['date'][1], unit='ms')]
    sorted_dates = [d for _, d in sorted(zip(date_dist, infos['date'][1]), reverse=True)]
    assert infos['date'][1] == sorted_dates

    # test region with cloud/shadow supported collection:
    assert infos['region_cp'][1] != infos['region_cp'][0]
    assert sorted(infos['region_cp'][1]) == infos['region_cp'][1]

    # test region with non-cloud/shadow supported collection:
    assert infos['region_fp'][1] != infos['region_fp'][0]
    assert sorted(infos['region_fp'][1]) == infos['region_fp'][1]


def test_prepare_for_composite_errors(gedi_cth_coll: ImageCollectionAccessor, region_100ha: dict):
    """Test _prepare_for_composite() error and warning conditions."""
    with pytest.raises(ValueError, match='cloud / shadow masking support'):
        gedi_cth_coll._prepare_for_composite('q-mosaic')
    with pytest.raises(ValueError, match="'date' or 'region'"):
        gedi_cth_coll._prepare_for_composite('mosaic', date='2020-01-01', region=region_100ha)

    with pytest.warns(UserWarning, match="'date' is valid"):
        gedi_cth_coll._prepare_for_composite('mean', date='2020-01-01')
    with pytest.warns(UserWarning, match="'region' is valid"):
        gedi_cth_coll._prepare_for_composite('mean', region=region_100ha)


def test_composite_params(l9_sr_coll: ImageCollectionAccessor, region_100ha: dict):
    """Test composite() parameters."""

    def get_refl_stat(image: ee.Image, stat: str = 'mean') -> ee.Number:
        """Return the mean of stat over the reflectance bands."""
        image = image.select('SR_B.*')
        return image.reduceRegion(stat, geometry=region_100ha, scale=30).values().reduce('mean')

    # create stats for testing each parameter
    infos = {}
    infos['method'] = [
        get_refl_stat(l9_sr_coll.composite(method), stat='mean') for method in CompositeMethod
    ]
    infos['mask'] = [
        get_refl_stat(l9_sr_coll.composite(mask=mask).mask(), stat='sum') for mask in [False, True]
    ]
    infos['resampling'] = {
        resampling.value: get_refl_stat(l9_sr_coll.composite(resampling=resampling), stat='stdDev')
        for resampling in ResamplingMethod
    }
    infos['date'] = [
        get_refl_stat(l9_sr_coll.composite('mosaic', date=date), stat='mean')
        for date in [None, '2022-12-05']
    ]
    infos['region'] = [
        get_refl_stat(l9_sr_coll.composite('mosaic', region=region), stat='mean')
        for region in [None, region_100ha]
    ]
    # cloud / shadow kwargs
    infos['cs_kwargs'] = [
        get_refl_stat(l9_sr_coll.composite(mask_shadows=mask_shadows).mask(), stat='sum')
        for mask_shadows in [False, True]
    ]

    # combine getInfo() calls into one
    infos = ee.Dictionary(infos).getInfo()

    # test each method gives a unique mean reflectance
    assert len(infos['method']) == len(set(infos['method'])) > 0
    # test mask=True reduces the mask=False masked area
    assert infos['mask'][0] > infos['mask'][1]
    # test each resampling method gives a unique reflectance std dev
    assert len(infos['resampling'].values()) == len(set(infos['resampling'].values())) > 0
    # test nearest resampling gives a larger std dev than the other methods
    assert all(
        infos['resampling']['near'] > infos['resampling'][k]
        for k in infos['resampling']
        if k != 'near'
    )
    # test date ordering affects the mean reflectance
    assert infos['date'][0] != infos['date'][1]
    # test region (CLOUDLESS_PORTION) ordering affects the mean reflectance
    assert infos['region'][0] != infos['region'][1]
    # test cloud / shadow kwargs have an effect (i.e. mask_shadows=True reduces the
    # mask_shadows=False masked area)
    assert infos['cs_kwargs'][0] > infos['cs_kwargs'][1]


def test_composite_properties(l9_sr_coll: ImageCollectionAccessor):
    """Test composite() image properties and bands."""
    method = 'mosaic'
    comp_image = l9_sr_coll.composite(method)
    info = comp_image.getInfo()

    exp_index = f'{method.upper()}-COMP'
    assert info['properties']['system:index'] == exp_index
    assert info['id'] == l9_sr_coll.id + '/' + exp_index
    coll_time_starts = [prop['system:time_start'] for prop in l9_sr_coll.properties.values()]
    assert info['properties']['system:time_start'] == min(coll_time_starts)
    assert info['properties']['system:time_end'] == max(coll_time_starts)

    band_names = [b['id'] for b in info['bands']]
    assert set(band_names).issuperset(
        ['CLOUD_MASK', 'SHADOW_MASK', 'CLOUD_DIST', 'CLOUDLESS_MASK', 'FILL_MASK']
    )


# old tests
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
    image_list: list = request.getfixturevalue(image_list)
    gd_collection = MaskedCollection.from_list(image_list)
    assert 'LANDSAT' in gd_collection.id
    assert gd_collection.properties is not None


@pytest.mark.parametrize('image_list', ['s2_sr_hm_images', 'gedi_image_list'])
def test_from_list(image_list: str, request):
    """
    Test MaskedCollection.from_list() generates a valid MaskedCollection object from lists of cloud/shadow
    maskable, and generic images.
    """
    image_list: list = request.getfixturevalue(image_list)
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
    image_list: list = request.getfixturevalue(image_list)[::-1]
    image_ids = [im_obj if isinstance(im_obj, str) else im_obj.id for im_obj in image_list]
    gd_collection = MaskedCollection.from_list(image_list)
    assert list(gd_collection.properties.keys()) == image_ids


def test_from_list_ee_image(gedi_image_list: list):
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
def test_from_list_add_props(image_list: str, add_props: list, request: pytest.FixtureRequest):
    """
    Test MaskedCollection.from_list(add_props=...) contains the add_props in properties and schema.
    """
    image_list: list = request.getfixturevalue(image_list)
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
    region: dict = request.getfixturevalue(region)
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

    def get_cloudless_portion(properties: dict) -> list[float]:
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
    region: dict = request.getfixturevalue(region)
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
    image_list: list = request.getfixturevalue(image_list)
    region: dict = request.getfixturevalue(region) if region else None
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
    proj = proj or BaseImage(ee_image).projection()
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
    proj = BaseImage(ee_collection.first()).projection()

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
    proj = BaseImage(ee_collection.first()).projection()

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
    proj = BaseImage(ee_collection.first()).projection()

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
    region_100ha: dict,
    request: pytest.FixtureRequest,
):
    """Test that resampling smooths the composite image."""
    image_list: list = request.getfixturevalue(image_list)
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
    proj = BaseImage(gd_collection.ee_collection.first()).projection()
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
def _test_composite(
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
    assert comp_im.info is not None and len(comp_im.info) > 0
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
    image_list: list = request.getfixturevalue(image_list)
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
    proj = BaseImage(filt_collection.ee_collection.first()).projection()
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
