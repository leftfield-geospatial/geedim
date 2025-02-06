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

from collections.abc import Iterable

import ee
import numpy as np
import pytest
import rasterio as rio

from geedim import CloudMaskMethod, schema, utils
from geedim.enums import CloudScoreBand
from geedim.mask import MaskedImage, _get_class_for_id


def test_class_from_id(landsat_image_ids, s2_sr_image_id, s2_toa_hm_image_id, generic_image_ids):
    """Test class_from_id()."""
    from geedim.mask import _LandsatImage, _MaskedImage, _Sentinel2SrImage, _Sentinel2ToaImage

    assert all([_get_class_for_id(im_id) == _LandsatImage for im_id in landsat_image_ids])
    assert all([_get_class_for_id(im_id) == _MaskedImage for im_id in generic_image_ids])
    assert _get_class_for_id(s2_sr_image_id) == _Sentinel2SrImage
    assert _get_class_for_id(s2_toa_hm_image_id) == _Sentinel2ToaImage


@pytest.mark.parametrize(
    'masked_image',
    [
        'user_masked_image',
        'modis_nbar_masked_image',
        'gch_masked_image',
        's1_sar_masked_image',
        'gedi_agb_masked_image',
        'gedi_cth_masked_image',
        'landsat_ndvi_masked_image',
    ],
)
def test_gen_aux_bands_exist(masked_image: str, request: pytest.FixtureRequest):
    """Test the presence of auxiliary band (i.e. FILL_MASK) in generic masked images."""
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    band_names = masked_image.ee_image.bandNames().getInfo()
    assert 'FILL_MASK' in band_names


@pytest.mark.parametrize(
    'masked_image',
    [
        's2_sr_masked_image',
        's2_toa_masked_image',
        's2_sr_hm_masked_image',
        's2_toa_hm_masked_image',
        'l9_masked_image',
        'l8_masked_image',
        'l7_masked_image',
        'l5_masked_image',
        'l4_masked_image',
    ],
)
def test_cloud_mask_aux_bands_exist(masked_image: str, request: pytest.FixtureRequest):
    """Test the presence of auxiliary bands in cloud masked images."""
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    band_names = masked_image.ee_image.bandNames().getInfo()
    exp_band_names = {'FILL_MASK', 'CLOUDLESS_MASK', 'CLOUD_DIST'}
    assert exp_band_names.intersection(band_names) == exp_band_names


@pytest.mark.parametrize(
    'masked_image',
    [
        's2_sr_masked_image',
        's2_toa_masked_image',
        's2_sr_hm_masked_image',
        's2_toa_hm_masked_image',
        'l9_masked_image',
        'l8_masked_image',
        'l7_masked_image',
        'l5_masked_image',
        'l4_masked_image',
        'user_masked_image',
        'modis_nbar_masked_image',
        'gch_masked_image',
        's1_sar_masked_image',
        'gedi_agb_masked_image',
        'gedi_cth_masked_image',
        'landsat_ndvi_masked_image',
    ],
)
def test_set_region_stats(masked_image: str, region_100ha, request: pytest.FixtureRequest):
    """
    Test MaskedImage._set_region_stats() generates the expected properties and that these are in the valid range.
    """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    masked_image._set_region_stats(region_100ha, scale=masked_image._ee_proj.nominalScale())
    for stat_name in ['FILL_PORTION', 'CLOUDLESS_PORTION']:
        assert stat_name in masked_image.properties
        assert masked_image.properties[stat_name] >= 0 and masked_image.properties[stat_name] <= 100


@pytest.mark.parametrize(
    'masked_image, exp_scale',
    [
        ('s2_sr_hm_masked_image', 60),
        ('l9_masked_image', 30),
        ('l4_masked_image', 30),
        ('s1_sar_masked_image', 10),
        ('gedi_agb_masked_image', 1000),
        # include fixtures with bands that have no fixed projection
        ('s2_sr_hm_qa_zero_masked_image', 60),
        ('s2_sr_hm_nocp_masked_image', 60),
        ('s2_sr_hm_nocs_masked_image', 60),
    ],
)
def test_ee_proj(masked_image: str, exp_scale: float, request: pytest.FixtureRequest):
    """Test MaskedImage._ee_proj has the correct scale and CRS."""
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    proj = masked_image._ee_proj.getInfo()

    assert np.abs(proj['transform'][0]) == pytest.approx(exp_scale, rel=1e-3)
    assert proj.get('crs', 'wkt') != 'EPSG:4326'


@pytest.mark.parametrize(
    'image_id', ['l9_image_id', 'l8_image_id', 'l7_image_id', 'l5_image_id', 'l4_image_id']
)
def test_landsat_cloudless_portion(image_id: str, request: pytest.FixtureRequest):
    """Test `geedim` CLOUDLESS_PORTION for the whole image against related Landsat CLOUD_COVER property."""
    image_id: MaskedImage = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_shadows=False, mask_cirrus=False)
    masked_image._set_region_stats()

    # landsat provided cloudless portion
    landsat_cloudless_portion = 100 - float(masked_image.properties['CLOUD_COVER'])
    assert masked_image.properties['CLOUDLESS_PORTION'] == pytest.approx(
        landsat_cloudless_portion, abs=5
    )


@pytest.mark.parametrize(
    'image_id', ['s2_toa_image_id', 's2_sr_image_id', 's2_toa_hm_image_id', 's2_sr_hm_image_id']
)
def test_s2_cloudless_portion(image_id: str, request: pytest.FixtureRequest):
    """Test `geedim` CLOUDLESS_PORTION for the whole image against CLOUDY_PIXEL_PERCENTAGE Sentinel-2 property."""
    # Note that CLOUDY_PIXEL_PERCENTAGE does not use Cloud Score+ data and does not include shadows, which Cloud Score+
    # does.  So CLOUDLESS_PORTION (with cloud-score method) will only roughly match CLOUDY_PIXEL_PERCENTAGE.
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='cloud-score')
    masked_image._set_region_stats()

    # S2 provided cloudless portion
    s2_cloudless_portion = 100 - float(masked_image.properties['CLOUDY_PIXEL_PERCENTAGE'])
    assert masked_image.properties['CLOUDLESS_PORTION'] == pytest.approx(
        s2_cloudless_portion, abs=10
    )


@pytest.mark.parametrize('image_id', ['l9_image_id'])
def test_landsat_cloudmask_params(image_id: str, request: pytest.FixtureRequest):
    """Test Landsat cloud/shadow masking `mask_shadows` and `mask_cirrus` parameters."""
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_shadows=False, mask_cirrus=False)
    masked_image._set_region_stats()
    # cloud-free portion
    cloud_only_portion = masked_image.properties['CLOUDLESS_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_shadows=True, mask_cirrus=False)
    masked_image._set_region_stats()
    # cloud and shadow-free portion
    cloud_shadow_portion = masked_image.properties['CLOUDLESS_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_shadows=True, mask_cirrus=True)
    masked_image._set_region_stats()
    # cloud, cirrus and shadow-free portion
    cloudless_portion = masked_image.properties['CLOUDLESS_PORTION']

    # test `mask_shadows` and `mask_cirrus` affect CLOUDLESS_PORTION as expected
    assert cloud_only_portion > cloud_shadow_portion
    assert cloud_shadow_portion > cloudless_portion


@pytest.mark.parametrize('image_id', ['s2_sr_hm_image_id', 's2_toa_hm_image_id'])
def test_s2_cloudmask_mask_shadows(
    image_id: str, region_10000ha: dict, request: pytest.FixtureRequest
):
    """Test S2 cloud/shadow masking `mask_shadows` parameter."""
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='cloud-prob', mask_shadows=False)
    masked_image._set_region_stats(region_10000ha)
    # cloud-free portion
    cloud_only_portion = 100 * masked_image.properties['CLOUDLESS_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_method='cloud-prob', mask_shadows=True)
    masked_image._set_region_stats(region_10000ha)
    # cloud and shadow-free portion
    cloudless_portion = 100 * masked_image.properties['CLOUDLESS_PORTION']
    assert cloud_only_portion > cloudless_portion


@pytest.mark.parametrize('image_id', ['s2_sr_hm_image_id', 's2_toa_hm_image_id'])
def test_s2_cloudmask_prob(image_id: str, region_10000ha: dict, request: pytest.FixtureRequest):
    """Test S2 cloud/shadow masking `prob` parameter with the `cloud-prob` method."""
    image_id: str = request.getfixturevalue(image_id)
    cl_portions = []
    for prob in [80, 40]:
        masked_image = MaskedImage.from_id(
            image_id, mask_shadows=True, prob=prob, mask_method='cloud-prob'
        )
        masked_image._set_region_stats(region_10000ha)
        cl_portions.append(100 * masked_image.properties['CLOUDLESS_PORTION'])
    # test there is more cloud (less CLOUDLESS_PORTION) with prob=40 as compared to prob=80
    assert cl_portions[0] > cl_portions[1] > 0


@pytest.mark.parametrize('image_id', ['s2_sr_hm_image_id', 's2_toa_hm_image_id'])
def test_s2_cloudmask_score(image_id: str, region_10000ha: dict, request: pytest.FixtureRequest):
    """Test S2 cloud/shadow masking `score` parameter with the `cloud-score` method."""
    image_id: str = request.getfixturevalue(image_id)
    cl_portions = []
    for score in [0.6, 0.3]:
        masked_image = MaskedImage.from_id(
            image_id, mask_shadows=True, score=score, mask_method='cloud-score'
        )
        masked_image._set_region_stats(region_10000ha)
        cl_portions.append(100 * masked_image.properties['CLOUDLESS_PORTION'])
    # test there is more cloud (less CLOUDLESS_PORTION) with score=0.3 as compared to score=0.6
    assert cl_portions[0] < cl_portions[1] > 0


@pytest.mark.parametrize('image_id', ['s2_sr_hm_image_id', 's2_toa_hm_image_id'])
def test_s2_cloudmask_cs_band(image_id: str, region_10000ha: dict, request: pytest.FixtureRequest):
    """Test S2 cloud/shadow masking `cs_band` parameter with the `cloud-score` method."""
    image_id: str = request.getfixturevalue(image_id)
    cl_portions = []
    for cs_band in CloudScoreBand:
        masked_image = MaskedImage.from_id(image_id, mask_method='cloud-score', cs_band=cs_band)
        masked_image._set_region_stats(region_10000ha)
        cl_portions.append(100 * masked_image.properties['CLOUDLESS_PORTION'])

    # test `cs_band` changes CLOUDLESS_PORTION but not by much
    assert len(set(cl_portions)) == len(cl_portions)
    assert all([cl_portions[0] != pytest.approx(cp, abs=10) for cp in cl_portions[1:]])
    assert all([cp != pytest.approx(0, abs=1) for cp in cl_portions])


@pytest.mark.parametrize('image_id', ['s2_sr_hm_image_id', 's2_toa_hm_image_id'])
def test_s2_cloudmask_method(image_id: str, region_10000ha: dict, request: pytest.FixtureRequest):
    """Test S2 cloud/shadow masking `mask_method` parameter."""
    image_id: str = request.getfixturevalue(image_id)
    cl_portions = []
    for mask_method in CloudMaskMethod:
        masked_image = MaskedImage.from_id(image_id, mask_method=mask_method)
        masked_image._set_region_stats(region_10000ha)
        cl_portions.append(100 * masked_image.properties['CLOUDLESS_PORTION'])

    # test `mask_method` changes CLOUDLESS_PORTION but not by much
    assert len(set(cl_portions)) == len(cl_portions)
    assert all([cl_portions[0] != pytest.approx(cp, abs=10) for cp in cl_portions[1:]])
    assert all([cp != pytest.approx(0, abs=1) for cp in cl_portions])


@pytest.mark.parametrize('image_id', ['s2_sr_hm_image_id', 's2_toa_hm_image_id'])
def test_s2_cloudmask_mask_cirrus(
    image_id: str, region_10000ha: dict, request: pytest.FixtureRequest
):
    """Test S2 cloud/shadow masking `mask_cirrus` parameter with the `qa` method."""
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='qa', mask_cirrus=False)
    # cloud and shadow free portion
    masked_image._set_region_stats(region_10000ha)
    non_cirrus_portion = 100 * masked_image.properties['CLOUDLESS_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_method='qa', mask_cirrus=True)
    masked_image._set_region_stats(region_10000ha)
    # cloud, cirrus and shadow free portion
    cirrus_portion = 100 * masked_image.properties['CLOUDLESS_PORTION']
    assert non_cirrus_portion >= cirrus_portion


@pytest.mark.parametrize('image_id', ['s2_sr_hm_image_id', 's2_toa_hm_image_id'])
def test_s2_cloudmask_dark(image_id: str, region_10000ha: dict, request: pytest.FixtureRequest):
    """Test S2 cloud/shadow masking `dark` parameter."""
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='cloud-prob', dark=0.5)
    masked_image._set_region_stats(region_10000ha)
    dark_pt5_portion = 100 * masked_image.properties['CLOUDLESS_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_method='cloud-prob', dark=0.1)
    masked_image._set_region_stats(region_10000ha)
    dark_pt1_portion = 100 * masked_image.properties['CLOUDLESS_PORTION']
    # test that increasing `dark` results in an increase in detected shadow and corresponding decrease in
    # CLOUDLESS_PORTION
    assert dark_pt1_portion > dark_pt5_portion


@pytest.mark.parametrize('image_id', ['s2_sr_hm_image_id', 's2_toa_hm_image_id'])
def test_s2_cloudmask_shadow_dist(
    image_id: str, region_10000ha: dict, request: pytest.FixtureRequest
):
    """Test S2 cloud/shadow masking `shadow_dist` parameter."""
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='cloud-prob', shadow_dist=200)
    masked_image._set_region_stats(region_10000ha)
    sd200_portion = 100 * masked_image.properties['CLOUDLESS_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_method='cloud-prob', shadow_dist=1000)
    masked_image._set_region_stats(region_10000ha)
    sd1000_portion = 100 * masked_image.properties['CLOUDLESS_PORTION']
    # test that increasing `shadow_dist` results in an increase in detected shadow and corresponding decrease in
    # CLOUDLESS_PORTION
    assert sd200_portion > sd1000_portion


@pytest.mark.parametrize('image_id', ['s2_sr_hm_image_id', 's2_toa_hm_image_id'])
def test_s2_cloudmask_cdi_thresh(
    image_id: str, region_10000ha: dict, request: pytest.FixtureRequest
):
    """Test S2 cloud/shadow masking `cdi_thresh` parameter."""
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='cloud-prob', cdi_thresh=0.5)
    masked_image._set_region_stats(region_10000ha)
    cdi_pt5_portion = 100 * masked_image.properties['CLOUDLESS_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_method='cloud-prob', cdi_thresh=-0.5)
    masked_image._set_region_stats(region_10000ha)
    cdi_negpt5_portion = 100 * masked_image.properties['CLOUDLESS_PORTION']
    # test that increasing `cdi_thresh` results in an increase in detected cloud and corresponding decrease in
    # CLOUDLESS_PORTION
    assert cdi_negpt5_portion > cdi_pt5_portion


@pytest.mark.parametrize(
    'image_id, max_cloud_dist', [('s2_sr_hm_image_id', 100), ('s2_sr_hm_image_id', 400)]
)
def test_s2_cloud_dist_max(
    image_id: str, max_cloud_dist: int, region_10000ha: dict, request: pytest.FixtureRequest
):
    """Test S2 cloud distance `max_cloud_dist` parameter."""

    def get_max_cloud_dist(cloud_dist: ee.Image):
        """Get the maximum of `cloud_dist` over region_10000ha."""
        mcd = cloud_dist.reduceRegion(
            reducer='max', geometry=region_10000ha, bestEffort=True, maxPixels=1e4
        )
        return mcd.get('CLOUD_DIST').getInfo()

    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(
        image_id, max_cloud_dist=max_cloud_dist, mask_method='cloud-score'
    )
    cloud_dist = masked_image.ee_image.select('CLOUD_DIST')
    meas_max_cloud_dist = get_max_cloud_dist(cloud_dist)
    assert meas_max_cloud_dist == pytest.approx(max_cloud_dist, rel=0.1)


@pytest.mark.parametrize(
    'masked_image',
    ['s2_sr_hm_qa_zero_masked_image', 's2_sr_hm_nocp_masked_image', 's2_sr_hm_nocs_masked_image'],
)
def test_s2_region_stats_missing_data(
    masked_image: str, region_10000ha: dict, request: pytest.FixtureRequest
):
    """Test S2 region stats for unmasked images missing required cloud data."""
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    masked_image._set_region_stats(region_10000ha, scale=60)

    assert masked_image.properties is not None
    assert masked_image.properties['CLOUDLESS_PORTION'] == pytest.approx(0, abs=1)
    assert masked_image.properties['FILL_PORTION'] == pytest.approx(100, abs=1)


def _test_aux_stats(masked_image: MaskedImage, region: dict):
    """Sanity tests on cloud/shadow etc. aux bands for a given MaskedImage and region."""
    # create a pan image for testing brightnesses
    ee_image = masked_image.ee_image
    pan = ee_image.select([0, 1, 2]).reduce(ee.Reducer.mean())

    # mask the pan image with cloud, shadow & cloudless masks
    cloud_mask = ee_image.select('CLOUD_MASK')
    shadow_mask = ee_image.select('SHADOW_MASK')
    cloudless_mask = ee_image.select('CLOUDLESS_MASK')
    pan_cloud = pan.updateMask(cloud_mask).rename('PAN_CLOUD')
    pan_shadow = pan.updateMask(shadow_mask).rename('PAN_SHADOW')
    pan_cloudless = pan.updateMask(cloudless_mask).rename('PAN_CLOUDLESS')

    # mask the cloud distance image with cloud & cloudless masks
    cdist = ee_image.select('CLOUD_DIST')
    cdist_cloud = cdist.updateMask(cloud_mask).rename('CDIST_CLOUD')
    cdist_cloudless = cdist.updateMask(cloudless_mask).rename('CDIST_CLOUDLESS')

    # find mean stats of all masked images, and min of cloud distance where it is >0
    stats_image = ee.Image([pan_cloud, pan_shadow, pan_cloudless, cdist_cloud, cdist_cloudless])
    proj = masked_image._ee_proj
    stats = stats_image.reduceRegion(
        reducer='mean',
        geometry=region,
        crs=proj,
        scale=proj.nominalScale(),
        bestEffort=True,
        maxPixels=1e8,
    )
    cdist_min = cdist.updateMask(cdist).reduceRegion(
        reducer='min',
        geometry=region,
        crs=proj,
        scale=proj.nominalScale(),
        bestEffort=True,
        maxPixels=1e8,
    )
    stats = stats.set('CDIST_MIN', cdist_min.get('CLOUD_DIST'))
    stats = stats.getInfo()

    # test cloud is brighter than cloudless
    assert stats['PAN_CLOUD'] > stats['PAN_CLOUDLESS']
    # test cloudless is brighter than shadow
    assert stats['PAN_CLOUDLESS'] > stats['PAN_SHADOW']
    # test cloudless areas have greater distance to cloud than cloudy areas
    assert stats['CDIST_CLOUDLESS'] > stats['CDIST_CLOUD']
    # test min distance to cloud is pixel size
    assert stats['CDIST_MIN'] == ee_image.select('B1').projection().nominalScale().getInfo()


@pytest.mark.parametrize(
    'masked_image',
    ['l9_masked_image', 'l8_masked_image', 'l7_masked_image', 'l5_masked_image', 'l4_masked_image'],
)
def test_landsat_aux_bands(masked_image: str, region_10000ha: dict, request: pytest.FixtureRequest):
    """Test Landsat auxiliary band values for sanity."""
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    _test_aux_stats(masked_image, region_10000ha)


@pytest.mark.parametrize(
    'image_id, mask_methods',
    [
        ('s2_sr_image_id', ['cloud-prob', 'qa']),
        ('s2_toa_image_id', ['cloud-prob', 'qa']),
        ('s2_sr_hm_image_id', ['cloud-prob', 'qa']),
        ('s2_toa_hm_image_id', ['cloud-prob', 'qa']),
        # missing QA60 so do cloud-prob method only
        ('s2_sr_hm_qa_zero_image_id', ['cloud-prob']),
    ],
)
def test_s2_aux_bands(
    image_id: str, mask_methods: Iterable, region_10000ha: dict, request: pytest.FixtureRequest
):
    """Test Sentinel-2 auxiliary band values for sanity with all masking methods."""
    image_id: str = request.getfixturevalue(image_id)
    for mask_method in mask_methods:
        masked_image = MaskedImage.from_id(image_id, mask_method=mask_method)
        _test_aux_stats(masked_image, region_10000ha)


@pytest.mark.parametrize(
    'masked_image',
    ['s2_sr_hm_nocp_masked_image', 's2_sr_hm_qa_zero_masked_image', 's2_sr_hm_nocs_masked_image'],
)
def test_s2_aux_bands_missing_data(
    masked_image: str, region_10000ha: dict, request: pytest.FixtureRequest
):
    """Test Sentinel-2 auxiliary band masking / transparency for unmasked images missing required cloud data."""
    masked_image: MaskedImage = request.getfixturevalue(masked_image)

    # get region sums of the auxiliary masks
    proj = masked_image._ee_proj
    aux_bands = masked_image.ee_image.select('.*MASK|CLOUD_DIST')
    aux_mask = aux_bands.mask()
    stats = aux_mask.reduceRegion(
        reducer='sum',
        geometry=region_10000ha,
        crs=proj,
        scale=proj.nominalScale(),
        bestEffort=True,
        maxPixels=1e8,
    )
    stats = stats.getInfo()

    # test auxiliary masks are transparent
    assert stats['FILL_MASK'] > 0
    # s2_sr_hm_nocs_masked_image is missing CLOUD_MASK and SHADOW_MASK bands, so only include these when they exist
    band_names = ['CLOUDLESS_MASK', 'CLOUD_DIST'] + list(
        {'CLOUD_MASK', 'SHADOW_MASK'}.intersection(stats.keys())
    )
    for band_name in band_names:
        assert stats[band_name] == 0, band_name


@pytest.mark.parametrize(
    'masked_image',
    [
        'gedi_cth_masked_image',
        # use s2_sr_masked_image rather than s2_sr_hm_masked_image which complicates testing due
        # to fully masked MSK_CLASSI* bands
        's2_sr_masked_image',
        'l9_masked_image',
    ],
)
def test_mask_clouds(
    masked_image: str, region_100ha: dict, tmp_path, request: pytest.FixtureRequest
):
    """Test MaskedImage.mask_clouds() masks the fill or cloudless portion by downloading and
    examining dataset masks.
    """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    filename = tmp_path.joinpath('test_image.tif')
    masked_image.mask_clouds()
    proj_scale = masked_image._ee_proj.nominalScale()
    masked_image.download(filename, region=region_100ha, dtype='float32', scale=proj_scale)
    assert filename.exists()

    with rio.open(filename, 'r') as ds:
        coll_name, _ = utils.split_id(masked_image.id)
        mask_name = 'CLOUDLESS_MASK' if coll_name in schema.cloud_coll_names else 'FILL_MASK'
        mask = ds.read(ds.descriptions.index(mask_name) + 1, masked=True)

        # test areas outside CLOUDLESS_MASK / FILL_MASK are masked
        assert np.all(mask == 1)

        # test CLOUDLESS_MASK / FILL_MASK matches the nodata mask for each band
        mask = mask.filled(0).astype('bool')
        ds_masks = ds.read_masks().astype('bool')
        assert np.all(mask == ds_masks)

        # TODO: test CLOUDLESS_MASK < FILL_MASK < whole image


@pytest.mark.parametrize(
    'masked_image',
    ['s2_sr_hm_nocp_masked_image', 's2_sr_hm_qa_zero_masked_image', 's2_sr_hm_nocs_masked_image'],
)
def test_s2_mask_clouds_missing_data(
    masked_image: str, region_100ha: dict, tmp_path, request: pytest.FixtureRequest
):
    """Test Sentinel2SrClImage.mask_clouds() masks the entire image when it is missing required
    cloud data. Downloads and examines dataset masks.
    """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    filename = tmp_path.joinpath('test_image.tif')
    masked_image.mask_clouds()
    proj_scale = masked_image._ee_proj.nominalScale()
    masked_image.download(filename, region=region_100ha, dtype='float32', scale=proj_scale)
    assert filename.exists()

    # test all data is masked / nodata
    with rio.open(filename, 'r') as ds:
        ds_masks = ds.read_masks().astype('bool')
        assert not np.any(ds_masks)


def test_skysat_region_stats():
    """Test _set_region_stats() works on SKYSAT image with no region."""
    # TODO: make a fixture and add to test_set_region_stats - not sure why this test is here.  the
    #  skysat image does not have an epsg crs which makes it unusual and perhaps worth having as a
    #  fixture.  the gedi_agb_masked_image is also like this though.
    ee_image = ee.Image('SKYSAT/GEN-A/PUBLIC/ORTHO/RGB/s02_20141004T074858Z')
    masked_image = MaskedImage(ee_image)
    masked_image._set_region_stats()
    assert 'FILL_PORTION' in masked_image.properties
    assert masked_image.properties['FILL_PORTION'] > 80
