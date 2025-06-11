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

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import ee
import numpy as np
import pytest
import rasterio as rio
from pandas import to_datetime
from rasterio.enums import Compression

from geedim import ExportType, schema
from geedim.collection import ImageCollectionAccessor, MaskedCollection, _compatible_collections
from geedim.download import BaseImage
from geedim.enums import CompositeMethod, ResamplingMethod, SplitType
from geedim.image import ImageAccessor, _nodata_vals
from geedim.mask import MaskedImage, _MaskedImage
from geedim.stac import STACClient
from geedim.utils import split_id
from tests.conftest import accessors_from_collections, transform_bounds


@pytest.fixture(scope='session')
def s2_sr_hm_coll(s2_sr_hm_image_ids: list[str]) -> ImageCollectionAccessor:
    """Sentinel-2 harmonized surface reflectance collection with three images."""
    coll = ee.ImageCollection(s2_sr_hm_image_ids)
    coll = coll.set('system:id', 'COPERNICUS/S2_SR_HARMONIZED')
    return ImageCollectionAccessor(coll)


@pytest.fixture(scope='session')
def l9_sr_coll(l9_sr_image_ids: list[str]) -> ImageCollectionAccessor:
    """Landsat-9 surface reflectance collection with three images."""
    coll = ee.ImageCollection(l9_sr_image_ids)
    coll = coll.set('system:id', 'LANDSAT/LC09/C02/T1_L2')
    return ImageCollectionAccessor(coll)


@pytest.fixture(scope='session')
def modis_nbar_coll() -> ImageCollectionAccessor:
    """MODIS NBAR collection with three images."""
    coll = ee.ImageCollection('MODIS/061/MCD43A4')
    coll = coll.filterDate('2024-01-01', '2024-03-01')
    return ImageCollectionAccessor(coll.limit(3))


@pytest.fixture(scope='session')
def gedi_cth_coll(gedi_cth_image_id: str) -> ImageCollectionAccessor:
    """GEDI canopy top height collection with three images."""
    image_ids = [
        gedi_cth_image_id,
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202205_018E_036S',
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202207_018E_036S',
    ]
    coll = ee.ImageCollection(image_ids)
    coll = coll.set('system:id', 'LARSE/GEDI/GEDI02_A_002_MONTHLY')
    return ImageCollectionAccessor(coll)


@pytest.fixture(scope='session')
def s2_sr_hm_masked_coll(s2_sr_hm_image_ids: list[str]) -> MaskedCollection:
    """Sentinel-2 harmonized surface reflectance MaskedCollection with three images."""
    coll = ee.ImageCollection(s2_sr_hm_image_ids)
    coll = coll.set('system:id', 'COPERNICUS/S2_SR_HARMONIZED')
    return MaskedCollection(coll)


@pytest.fixture(scope='session')
def l9_sr_masked_coll(l9_sr_image_ids) -> ImageCollectionAccessor:
    """Landsat-9 surface reflectance MaskedCollection with three images."""
    coll = ee.ImageCollection(l9_sr_image_ids)
    coll = coll.set('system:id', 'LANDSAT/LC09/C02/T1_L2')
    return MaskedCollection(coll)


_s2_b1_info = {
    'id': 'B1',
    'data_type': {'type': 'PixelType', 'precision': 'int', 'min': 0, 'max': 65535},
    'dimensions': [1830, 1830],
    'crs': 'EPSG:32734',
    'crs_transform': [60, 0, 499980, 0, -60, 6400000],
}
"""Example EE band info dictionary for B1 of a Sentinel-2 image."""


def test_xarray():
    import xarray
    from pandas import to_datetime  # noqa

    assert xarray is not None
    if not xarray:
        raise ImportError()


def test_compatible_collections(
    l4_sr_image_id: str,
    l5_sr_image_id: str,
    l7_sr_image_id: str,
    l8_sr_image_id: str,
    l9_sr_image_id: str,
    s2_sr_hm_image_ids: list[str],
):
    """Test _compatible_collections()."""

    def compatible_images(*im_ids) -> list[str]:
        """Extract collection IDs and return the result of _compatible_collections()."""
        coll_ids = [split_id(im_id)[0] for im_id in im_ids]
        return _compatible_collections(coll_ids)

    assert compatible_images(l4_sr_image_id, l5_sr_image_id) is True
    assert compatible_images(l8_sr_image_id, l9_sr_image_id) is True
    assert compatible_images(*s2_sr_hm_image_ids) is True
    assert compatible_images(l4_sr_image_id, l5_sr_image_id, l7_sr_image_id) is False
    assert compatible_images(l7_sr_image_id, l8_sr_image_id, l9_sr_image_id) is False
    assert compatible_images(*s2_sr_hm_image_ids, l9_sr_image_id) is False
    assert compatible_images(l9_sr_image_id, None) is False


def test_from_images(s2_sr_hm_image_ids: list[str]):
    """Test ImageCollectionAccessor.fromImages()."""
    coll = ImageCollectionAccessor.fromImages(s2_sr_hm_image_ids)
    info = coll.getInfo()
    assert info['id'] == 'COPERNICUS/S2_SR_HARMONIZED'
    assert set([im_props['id'] for im_props in info['features']]) == set(s2_sr_hm_image_ids)


def test_from_images_error(l7_sr_image_id: str, l8_sr_image_id: str):
    """Test ImageCollectionAccessor.fromImages() raises an error with incompatible images."""
    with pytest.raises(ValueError, match='spectrally compatible'):
        ImageCollectionAccessor.fromImages([l7_sr_image_id, l8_sr_image_id])


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


@pytest.mark.parametrize(
    'coll, exp_support', [('s2_sr_hm_coll', True), ('l9_sr_coll', True), ('modis_nbar_coll', False)]
)
def test_properties(coll: str, exp_support: bool, request: pytest.FixtureRequest):
    """Test properties that don't have their own specific tests."""
    coll: ImageCollectionAccessor = request.getfixturevalue(coll)
    props = {ip['properties']['system:index']: ip['properties'] for ip in coll.info['features']}

    # EE info dependent properties
    assert coll.id == coll.info['id']
    assert coll.properties == props
    assert coll._first.id == coll.info['features'][0]['id']

    # STAC dependent properties
    assert coll.stac is not None
    band_props = coll.stac['summaries']['eo:bands']
    spec_bands = [bp['name'] for bp in band_props if 'center_wavelength' in bp]
    if exp_support:
        assert len(spec_bands) > 0
    assert coll.specBands == spec_bands

    # cloud/shadow support
    assert coll.cloudShadowSupport == exp_support


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
    assert coll.schemaPropertyNames == tuple(schema_.keys())
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
        schema_prop_names, ['CLOUDLESS', 'CCA', 'UPN'], [True, True, False], strict=True
    ):
        assert coll.schema[prop]['abbrev'] == abbrev
        descr = coll.schema[prop]['description']
        if has_descr:
            assert len(descr) > 0
            assert '.' not in descr and '\n' not in descr
        else:
            assert descr is None


def test_schema_set_error():
    """Test setting the schemaPropertyNames property with a value that is not an iterable of strings
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
    coll.schemaPropertyNames += ('unknownPropertyName',)
    assert len(coll.schemaTable.splitlines()) >= len(coll.schema) + 2
    assert all(pn in coll.schemaTable for pn in coll.schemaPropertyNames)

    # test empty schema
    coll.schemaPropertyNames = ()
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
        coll.properties['1']['system:time_start'] / 1000, tz=UTC
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
    sorted_dates = [
        d for _, d in sorted(zip(date_dist, infos['date'][1], strict=True), reverse=True)
    ]
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
    for masked_sum, unmasked_sum in zip(sums['masked'], sums['unmasked'], strict=True):
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


@pytest.mark.parametrize('features', [([{'bands': [_s2_b1_info, dict(_s2_b1_info, id='B2')]}] * 2)])
def test_raise_image_consistency(features: list):
    """Test _raise_image_consistency() with a consistent collection."""
    coll = ImageCollectionAccessor(None)
    # patch with mock EE collection info
    coll._info = dict(features=features)
    coll._raise_image_consistency()


@pytest.mark.parametrize(
    'features, match',
    [
        # inconsistent number of bands
        (
            [{'bands': [_s2_b1_info, dict(_s2_b1_info, id='B2')]}, {'bands': [_s2_b1_info]}],
            'number of bands or band names',
        ),
        # inconsistent band names
        (
            [{'bands': [_s2_b1_info]}, {'bands': [dict(_s2_b1_info, id='B2')]}],
            'number of bands or band names',
        ),
        # non fixed projection band
        (
            [{'bands': [{k: v for k, v in _s2_b1_info.items() if k != 'dimensions'}]}],
            'fixed projection',
        ),
        # inconsistent band dimensions
        (
            [{'bands': [_s2_b1_info, dict(_s2_b1_info, id='B2', dimensions=[1, 1])]}],
            'band projections, bounds or data types',
        ),
        # inconsistent band data type
        (
            [
                {
                    'bands': [
                        _s2_b1_info,
                        dict(
                            _s2_b1_info,
                            id='B2',
                            data_type={'type': 'PixelType', 'precision': 'float'},
                        ),
                    ]
                }
            ],
            'band projections, bounds or data types',
        ),
        # inconsistent band crs
        (
            [{'bands': [_s2_b1_info, dict(_s2_b1_info, id='B2', crs='EPSG:3857')]}],
            'band projections, bounds or data types',
        ),
        # inconsistent band crs_transform
        (
            [
                {
                    'bands': [
                        _s2_b1_info,
                        dict(_s2_b1_info, id='B2', crs_transform=[1, 0, 0, 0, 1, 0]),
                    ]
                }
            ],
            'band projections, bounds or data types',
        ),
    ],
)
def test_raise_image_consistency_error(features: list, match: str):
    """Test _raise_image_consistency() error conditions."""
    coll = ImageCollectionAccessor(None)
    # patch with mock EE collection info
    coll._info = dict(features=features)
    with pytest.raises(ValueError, match=match):
        coll._raise_image_consistency()


def test_prepare_for_export(
    s2_sr_hm_coll: ImageCollectionAccessor, s2_sr_hm_image: ImageAccessor, region_100ha: dict
):
    """Test prepareForExport()."""
    # adapted from test_image.test_prepare_for_export()
    crs = 'EPSG:3857'
    prep_kwargs_list = [
        dict(crs=crs, region=region_100ha, scale=60),
        dict(crs=crs, region=region_100ha, shape=(300, 400)),
        dict(crs=crs, crs_transform=(60.0, 0.0, 500000.0, 0.0, -30.0, 6400000.0), shape=(600, 400)),
        dict(region=region_100ha),
        dict(crs=crs, region=region_100ha, scale=60, dtype='int16', bands=['B4', 'B3', 'B2']),
        # maintain pixel grid
        dict(crs=s2_sr_hm_image.crs, region=region_100ha),
        dict(region=region_100ha, scale=s2_sr_hm_image.scale),
        dict(region=region_100ha),
        dict(),
    ]
    prep_colls = [s2_sr_hm_coll.prepareForExport(**prep_kwargs) for prep_kwargs in prep_kwargs_list]
    prep_colls = accessors_from_collections(prep_colls)

    # test prepared collection properties
    for prep_coll, prep_kwargs in zip(prep_colls, prep_kwargs_list, strict=True):
        prep_coll._raise_image_consistency()

        prep_first = prep_coll._first
        assert prep_first.crs == prep_kwargs.get('crs', prep_first.crs)
        assert prep_first.dtype == prep_kwargs.get('dtype', prep_first.dtype)
        assert prep_first.bandNames == prep_kwargs.get('bands', prep_first.bandNames)
        if 'shape' in prep_kwargs:
            assert prep_first.shape == prep_kwargs['shape']
        if 'scale' in prep_kwargs:
            assert prep_first.scale == prep_kwargs['scale']
        if 'crs_transform' in prep_kwargs:
            assert prep_first.transform == prep_kwargs['crs_transform']

        # region is a special case that is approximate & needs transformation between CRSs
        if 'region' in prep_kwargs:
            region_bounds = transform_bounds(ee.Geometry(prep_kwargs['region']).toGeoJSON(), crs)
            image_bounds = transform_bounds(prep_first.geometry, crs)
            assert image_bounds == pytest.approx(region_bounds, abs=60)

    # test pixel grid is maintained when arguments allow
    src_transform = rio.Affine(*s2_sr_hm_image.transform)
    for prep_coll in prep_colls[-4:]:
        prep_first = prep_coll._first
        prep_transform = rio.Affine(*prep_first.transform)
        assert (prep_transform[0], prep_transform[4]) == (src_transform[0], src_transform[4])
        pixel_offset = ~src_transform * (prep_transform[2], prep_transform[5])
        assert pixel_offset == (int(pixel_offset[0]), int(pixel_offset[1]))


def test_prepare_for_export_scale_offset(
    s2_sr_hm_coll: ImageCollectionAccessor, region_100ha: dict
):
    """Test the prepareForExport() scale_offset parameter."""
    # This just tests the scale_offset parameter was acted on. Detailed scale / offset testing is
    # done in test_image.test_scale_offset(). (Adapted from
    # test_image.test_prepare_for_export_scale_offset())
    prep_colls = [
        s2_sr_hm_coll.prepareForExport(region=region_100ha, scale_offset=scale_offset)
        for scale_offset in [False, True]
    ]
    maxs = [
        prep_coll.first()
        .select('B.*')
        .reduceRegion(reducer='max', geometry=region_100ha, bestEffort=True)
        .values()
        .reduce('max')
        for prep_coll in prep_colls
    ]
    maxs = ee.List(maxs).getInfo()
    assert maxs[1] < maxs[0]


@pytest.mark.parametrize('split', SplitType)
def test_split_images(prepared_coll: ImageCollectionAccessor, split: SplitType):
    """Test _split_images() with different split types."""
    images = prepared_coll._split_images(split)

    # test index & band names
    if split is SplitType.images:
        assert images.keys() == prepared_coll.properties.keys()
        for key, image in images.items():
            assert image.index == key
            assert image.bandNames == prepared_coll._first.bandNames
    else:
        assert list(images.keys()) == prepared_coll._first.bandNames
        for key, image in images.items():
            assert image.index == key
            assert image.bandNames == list(prepared_coll.properties.keys())

    # test georeferencing & dtype
    for image in images.values():
        assert image.crs == prepared_coll._first.crs
        assert image.transform == prepared_coll._first.transform
        assert image.shape == prepared_coll._first.shape
        assert image.dtype == prepared_coll._first.dtype


@pytest.mark.parametrize(
    'etype, split, patch_export_task',
    itertools.product(ExportType, SplitType, ['export_task_success']),
    indirect=['patch_export_task'],
)
def test_to_google_cloud(
    prepared_coll: ImageCollectionAccessor,
    etype: ExportType,
    split: SplitType,
    patch_export_task,
    capsys: pytest.CaptureFixture,
):
    """Test toGoogleCloud()."""
    exp_num_tasks = (
        len(prepared_coll.properties) if split is SplitType.images else prepared_coll._first.count
    )
    tasks = prepared_coll.toGoogleCloud(type=etype, folder='geedim', wait=False, split=split)
    assert len(tasks) == exp_num_tasks
    # test monitorTask is not called with wait=False
    assert capsys.readouterr().err == ''

    tasks = prepared_coll.toGoogleCloud(type=etype, folder='geedim', wait=True, split=split)
    assert len(tasks) == exp_num_tasks
    # test monitorTask is called with wait=True
    captured = capsys.readouterr()
    assert prepared_coll.id in captured.err and '100%' in captured.err


@pytest.mark.parametrize(
    'split, kwargs',
    [(SplitType.bands, dict(driver='gtiff')), (SplitType.images, dict(driver='cog', nodata=False))],
)
def test_to_geotiff(
    prepared_coll: ImageCollectionAccessor,
    prepared_coll_array: np.ndarray,
    tmp_path: Path,
    split: SplitType,
    kwargs: dict,
):
    """Test toGeoTIFF()"""
    # adapted from test_image.test_to_geotiff()
    prepared_coll.toGeoTIFF(tmp_path, split=split, **kwargs)

    first = prepared_coll._first
    exp_file_stems = (
        list(prepared_coll.properties.keys()) if split is SplitType.images else first.bandNames
    )
    files = list(tmp_path.glob('*'))
    assert set([f.stem for f in files]) == set(exp_file_stems)

    nodata = kwargs.get('nodata', True)
    for fi, file_stem in enumerate(exp_file_stems):
        with rio.open(tmp_path.joinpath(file_stem + '.tif')) as ds:
            # format
            assert ds.crs == first.crs
            assert ds.transform[:6] == first.transform
            assert ds.shape == first.shape
            assert ds.count == (
                first.count if split is SplitType.images else len(prepared_coll.properties)
            )
            assert ds.dtypes[0] == first.dtype
            assert ds.compression == Compression.deflate
            if nodata is True:
                assert ds.nodata == _nodata_vals[first.dtype]
            elif nodata is False:
                assert ds.nodata is None
            else:
                assert ds.nodata == nodata

            # contents
            array = ds.read()
            array = np.moveaxis(array, 0, -1)
            # masked pixels will always == _nodata_vals[image.dtype], irrespective of the nodata
            # value
            mask = array != _nodata_vals[first.dtype]
            axis = 2 if split is SplitType.bands else 3
            assert (mask == ~prepared_coll_array.mask.take(fi, axis=axis)).all()
            assert (array == prepared_coll_array.take(fi, axis=axis)).all()

            # metadata
            metadata = ds.tags()
            if split is SplitType.images:
                props = {
                    k.replace(':', '-'): str(v)
                    for k, v in prepared_coll.properties[file_stem].items()
                }
                assert all([metadata.get(k) == v for k, v in props.items()])
                assert metadata.get('LICENSE') is not None
                assert ds.descriptions == tuple(prepared_coll._first.bandNames)
                for bi in range(ds.count):
                    band_props = {
                        k.replace(':', '-'): str(v)
                        for k, v in prepared_coll._first.bandProps[bi].items()
                    }
                    assert ds.tags(bi + 1) == band_props
            else:
                assert metadata['system-index'] == file_stem
                assert ds.descriptions == tuple(prepared_coll.properties.keys())


@pytest.mark.parametrize(
    'masked, structured, split',
    [
        (False, False, SplitType.bands),
        (True, False, SplitType.images),
        (False, True, SplitType.bands),
        (True, True, SplitType.images),
    ],
)
def test_to_numpy(
    prepared_coll: ImageCollectionAccessor,
    prepared_coll_array: np.ndarray,
    masked: bool,
    structured: bool,
    split: SplitType,
):
    """Test toNumpy()."""
    # adapted from test_image.test_to_numpy()
    array = prepared_coll.toNumPy(masked=masked, structured=structured, split=split)

    # dimensions and dtype
    first = prepared_coll._first
    if structured:
        indexes = list(prepared_coll.properties.keys())
        assert array.shape == first.shape
        assert len(array.dtype) == len(indexes) if split is SplitType.images else first.count

        # construct reference structured dtype
        date_strings = [
            datetime.fromtimestamp(p['system:time_start'] / 1000).isoformat(timespec='seconds')
            for p in prepared_coll.properties.values()
        ]
        # last dimension dtype
        names = (
            first.bandNames
            if split is SplitType.images
            else list(zip(date_strings, indexes, strict=True))
        )
        dtype = np.dtype(list(zip(names, [first.dtype] * len(names), strict=True)))
        # nested (second last dimension) dtype
        names = (
            list(zip(date_strings, indexes, strict=True))
            if split is SplitType.images
            else first.bandNames
        )
        dtype = np.dtype(list(zip(names, [dtype] * len(names), strict=True)))

        assert array.dtype == dtype
    else:
        num_ims = len(prepared_coll.properties)
        shape = (
            (*first.shape, num_ims, first.count)
            if split is SplitType.images
            else (*first.shape, first.count, num_ims)
        )
        assert array.shape == shape
        assert array.dtype == first.dtype

    # masking
    if masked:
        assert isinstance(array, np.ma.MaskedArray)
        assert array.fill_value == np.array(first.nodata, array.dtype)
    else:
        assert not isinstance(array, np.ma.MaskedArray)

    # contents
    ref_array = (
        np.swapaxes(prepared_coll_array, 2, 3) if split is SplitType.images else prepared_coll_array
    )
    array_ = array.view(first.dtype).reshape(ref_array.shape) if structured else array
    mask = ~array_.mask if masked else array_ != first.nodata
    assert (mask == ~ref_array.mask).all()
    assert (array_ == ref_array).all()


@pytest.mark.parametrize('masked, split', [(False, SplitType.bands), (True, SplitType.images)])
def test_to_xarray(
    prepared_coll: ImageCollectionAccessor,
    prepared_coll_array: np.ndarray,
    masked: bool,
    split: SplitType,
):
    """Test toXarray()."""
    # adapted from test_image.test_to_xarray()
    first = prepared_coll._first
    ds = prepared_coll.toXarray(masked=masked, split=split)

    # variables
    if split is SplitType.bands:
        assert list(ds.keys()) == first.bandNames
    else:
        assert ds.keys() == prepared_coll.properties.keys()

    # coordinates
    if split is SplitType.bands:
        timestamps = [p['system:time_start'] for p in prepared_coll.properties.values()]
        datetimes = to_datetime(timestamps, unit='ms')
        assert (ds.coords['time'] == datetimes).all()
    else:
        assert all(ds.coords['band'] == first.bandNames)
    y = np.arange(0.5, first.shape[1] + 0.5) * first.transform[4] + first.transform[5]
    x = np.arange(0.5, first.shape[0] + 0.5) * first.transform[0] + first.transform[2]
    assert (ds.coords['x'] == x).all()
    assert (ds.coords['y'] == y).all()

    # dtype & nodata
    if masked:
        dtype = np.promote_types(first.dtype, 'float32')
        assert all([da.dtype == dtype for da in ds.values()])
        assert np.isnan(ds.attrs['nodata'])
    else:
        assert all([da.dtype == first.dtype for da in ds.values()])
        assert ds.attrs['nodata'] == first.nodata

    # attributes
    assert ds.attrs['id'] == prepared_coll.id
    for attr in ['crs', 'transform']:
        assert ds.attrs[attr] == getattr(first, attr), attr
    assert ds.attrs['ee'] == json.dumps(prepared_coll.info['properties'])
    assert ds.attrs['stac'] == json.dumps(prepared_coll.stac)

    # contents
    axis = 2 if split is SplitType.bands else 3
    for vi, da in enumerate(ds.values()):
        mask = ~da.isnull() if masked else da != first.nodata
        assert (mask == ~prepared_coll_array.mask.take(vi, axis=axis)).all()
        assert (da == prepared_coll_array.take(vi, axis=axis)).all()


# MaskedCollection tests
def test_init_deprecation():
    """Test MaskedCollection.__init__() issues a deprecation warning."""
    with pytest.warns(FutureWarning, match='deprecated'):
        _ = MaskedCollection(ee.ImageCollection([]))


def test_masked_init():
    """Test MaskedCollection.__init__()."""
    ee_coll = ee.ImageCollection([])
    add_props = ['CLOUD_COVERAGE_ASSESSMENT', 'unknownPropertyName']
    coll = MaskedCollection(ee_coll, add_props=add_props)
    # patch to avoid getInfo()
    coll.id = 'COPERNICUS/S2_SR_HARMONIZED'

    assert coll.ee_collection == ee_coll
    assert set(coll.schema.keys()).issuperset(add_props)


def test_masked_from_name():
    """Test MaskedCollection.from_name()."""
    ee_id = 'COPERNICUS/S2_SR_HARMONIZED'
    add_props = ['CLOUD_COVERAGE_ASSESSMENT', 'unknownPropertyName']
    coll = MaskedCollection.from_name(ee_id, add_props=add_props)
    # patch to avoid getInfo()
    coll.id = ee_id

    assert coll.ee_collection == ee.ImageCollection(ee_id)
    assert set(coll.schema.keys()).issuperset(add_props)


def test_masked_from_list(s2_sr_hm_image_ids: list[str]):
    """Test MaskedCollection.from_list()."""
    images = [
        s2_sr_hm_image_ids[0],
        ee.Image(s2_sr_hm_image_ids[1]),
        BaseImage(ee.Image(s2_sr_hm_image_ids[2])),
    ]
    add_props = ['CLOUD_COVERAGE_ASSESSMENT', 'unknownPropertyName']
    coll = MaskedCollection.from_list(images, add_props=add_props)

    assert set([im_props['id'] for im_props in coll.info['features']]) == set(
        s2_sr_hm_image_ids[: len(images)]
    )
    assert coll.id == 'COPERNICUS/S2_SR_HARMONIZED'
    assert set(coll.schema.keys()).issuperset(add_props)


@pytest.mark.parametrize(
    'masked_coll, accessor',
    [('s2_sr_hm_masked_coll', 's2_sr_hm_coll'), ('l9_sr_masked_coll', 'l9_sr_coll')],
)
def test_masked_properties(masked_coll: str, accessor: str, request: pytest.FixtureRequest):
    """Test MaskedCollection specific properties against a matching ImageCollectionAccessor."""
    masked_coll: MaskedCollection = request.getfixturevalue(masked_coll)
    accessor: ImageCollectionAccessor = request.getfixturevalue(accessor)
    props = {
        info['id']: {
            k: info['properties'][k]
            for k in accessor.schemaPropertyNames
            if k in info['properties']
        }
        for info in accessor.info['features']
    }

    assert masked_coll.ee_collection == accessor._ee_coll
    assert masked_coll.name == accessor.id
    assert masked_coll.image_type == MaskedImage
    assert masked_coll.stats_scale == accessor._portion_scale
    assert masked_coll.schema == accessor.schema
    assert masked_coll.schema_table == accessor.schemaTable
    assert masked_coll.properties == props
    assert masked_coll.properties_table == accessor.propertiesTable
    assert masked_coll.refl_bands == accessor.specBands


def test_masked_search(region_10000ha: dict):
    """Test MaskedCollection.search()."""
    # this just tests args are passed through and add_props are maintained, detailed testing is done
    # in test_filter()
    add_props = ['CLOUD_COVER']
    coll = MaskedCollection(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'), add_props=add_props)
    kwargs = dict(start_date='2023-01-01', end_date='2024-01-01', region=region_10000ha)
    filt_coll = coll.search(**kwargs)

    assert isinstance(filt_coll, MaskedCollection)
    dates = to_datetime([p['system:time_start'] for p in filt_coll.properties.values()], unit='ms')
    # test add_props is maintained in filtered collection
    assert set(filt_coll.schema.keys()).issuperset(add_props)
    # test search kwargs are passed through
    assert all(dates >= to_datetime(kwargs['start_date']))
    assert all(dates <= to_datetime(kwargs['end_date']))


def test_masked_composite(l9_sr_masked_coll: MaskedCollection):
    """Test MaskedCollection.composite()"""
    # this just tests args are passed through, detailed testing is done in
    # test_prepare_for_composite_date_region() and test_composite_params()
    method = 'mosaic'
    comp_image = l9_sr_masked_coll.composite(method)

    assert isinstance(comp_image, MaskedImage)
    exp_index = f'{method.upper()}-COMP'
    assert comp_image.properties['system:index'] == exp_index
