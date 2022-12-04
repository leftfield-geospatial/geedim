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
import numpy as np
import pytest
import rasterio as rio
from geedim.mask import MaskedImage, get_projection, class_from_id


def test_class_from_id(landsat_image_ids, s2_sr_image_id, s2_toa_hm_image_id, generic_image_ids):
    """  Test class_from_id(). """
    from geedim.mask import LandsatImage, Sentinel2SrClImage, Sentinel2ToaClImage

    assert all([class_from_id(im_id) == LandsatImage for im_id in landsat_image_ids])
    assert all([class_from_id(im_id) == MaskedImage for im_id in generic_image_ids])
    assert class_from_id(s2_sr_image_id) == Sentinel2SrClImage
    assert class_from_id(s2_toa_hm_image_id) == Sentinel2ToaClImage


def test_from_id():
    """ Test MaskedImage.from_id() sets _id. """
    ee_id = 'MODIS/006/MCD43A4/2022_01_01'
    gd_image = MaskedImage.from_id(ee_id)
    assert gd_image._id == ee_id


@pytest.mark.parametrize(
    'masked_image', [
        'user_masked_image', 'modis_nbar_masked_image', 'gch_masked_image', 's1_sar_masked_image',
        'gedi_agb_masked_image', 'gedi_cth_masked_image', 'landsat_ndvi_masked_image'
    ]
)
def test_gen_aux_bands_exist(masked_image: str, request: pytest.FixtureRequest):
    """ Test the presence of auxiliary band (i.e. FILL_MASK) in generic masked images. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    band_names = masked_image.ee_image.bandNames().getInfo()
    assert 'FILL_MASK' in band_names


@pytest.mark.parametrize(
    'masked_image', [
        's2_sr_masked_image', 's2_toa_masked_image', 's2_sr_hm_masked_image', 's2_toa_hm_masked_image',
        'l9_masked_image', 'l8_masked_image', 'l7_masked_image', 'l5_masked_image', 'l4_masked_image'
    ]
)
def test_cloud_mask_aux_bands_exist(masked_image: str, request: pytest.FixtureRequest):
    """ Test the presence of auxiliary bands in cloud masked images. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    band_names = masked_image.ee_image.bandNames().getInfo()
    exp_band_names = ['CLOUD_MASK', 'SHADOW_MASK', 'FILL_MASK', 'CLOUDLESS_MASK', 'CLOUD_DIST']
    for exp_band_name in exp_band_names:
        assert exp_band_name in band_names


@pytest.mark.parametrize(
    'masked_image', [
        's2_sr_masked_image', 's2_toa_masked_image', 's2_sr_hm_masked_image', 's2_toa_hm_masked_image',
        'l9_masked_image', 'l8_masked_image', 'l7_masked_image', 'l5_masked_image', 'l4_masked_image',
        'user_masked_image', 'modis_nbar_masked_image', 'gch_masked_image', 's1_sar_masked_image',
        'gedi_agb_masked_image', 'gedi_cth_masked_image', 'landsat_ndvi_masked_image'
    ]
)
def test_set_region_stats(masked_image: str, region_100ha, request: pytest.FixtureRequest):
    """
    Test MaskedImage._set_region_stats() generates the expected properties and that these are in the valid range.
    """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    masked_image._set_region_stats(region_100ha)
    for stat_name in ['FILL_PORTION', 'CLOUDLESS_PORTION']:
        assert stat_name in masked_image.properties
        assert masked_image.properties[stat_name] >= 0 and masked_image.properties[stat_name] <= 100


@pytest.mark.parametrize('image_id', ['l9_image_id', 'l8_image_id', 'l7_image_id', 'l5_image_id', 'l4_image_id'])
def test_landsat_cloudless_portion(image_id: str, request: pytest.FixtureRequest):
    """ Test `geedim` CLOUDLESS_PORTION for the whole image against related Landsat CLOUD_COVER property. """
    image_id: MaskedImage = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_shadows=False, mask_cirrus=False)
    masked_image._set_region_stats()
    # the `geedim` cloudless portion of the filled portion
    cloudless_portion = masked_image.properties['CLOUDLESS_PORTION']
    # landsat provided cloudless portion
    landsat_cloudless_portion = 100 - float(masked_image.properties['CLOUD_COVER'])
    assert cloudless_portion == pytest.approx(landsat_cloudless_portion, abs=5)


@pytest.mark.parametrize('image_id', ['s2_toa_image_id', 's2_sr_image_id', 's2_toa_hm_image_id', 's2_sr_hm_image_id'])
def test_s2_cloudless_portion(image_id: str, request: pytest.FixtureRequest):
    """ Test `geedim` CLOUDLESS_PORTION for the whole image against CLOUDY_PIXEL_PERCENTAGE Sentinel-2 property. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='qa', mask_shadows=False, mask_cirrus=False)
    masked_image._set_region_stats()
    # S2 provided cloudless portion
    s2_cloudless_portion = 100 - float(masked_image.properties['CLOUDY_PIXEL_PERCENTAGE'])
    # CLOUDLESS_MASK is eroded and dilated, so allow 10% difference to account for that
    assert masked_image.properties['CLOUDLESS_PORTION'] == pytest.approx(s2_cloudless_portion, abs=10)


@pytest.mark.parametrize('image_id', ['l9_image_id'])
def test_landsat_cloudmask_params(image_id: str, request: pytest.FixtureRequest):
    """ Test Landsat cloud/shadow masking `mask_shadows` and `mask_cirrus` parameters. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_shadows=False, mask_cirrus=False)
    masked_image._set_region_stats()
    # cloud-free portion
    cloud_only_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_shadows=True, mask_cirrus=False)
    masked_image._set_region_stats()
    # cloud and shadow-free portion
    cloud_shadow_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_shadows=True, mask_cirrus=True)
    masked_image._set_region_stats()
    # cloud, cirrus and shadow-free portion
    cloudless_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']

    # test `mask_shadows` and `mask_cirrus` affect CLOUDLESS_PORTION as expected
    assert cloud_only_portion > cloud_shadow_portion
    assert cloud_shadow_portion > cloudless_portion


@pytest.mark.parametrize('image_id', ['s2_sr_image_id', 's2_toa_image_id'])
def test_s2_cloudmask_mask_shadows(image_id: str, region_10000ha: Dict, request: pytest.FixtureRequest):
    """ Test S2 cloud/shadow masking `mask_shadows` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_shadows=False)
    masked_image._set_region_stats(region_10000ha)
    # cloud-free portion
    cloud_only_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_shadows=True)
    masked_image._set_region_stats(region_10000ha)
    # cloud and shadow-free portion
    cloudless_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    assert cloud_only_portion > cloudless_portion


@pytest.mark.parametrize('image_id', ['s2_sr_image_id', 's2_toa_image_id'])
def test_s2_cloudmask_prob(image_id: str, region_10000ha: Dict, request: pytest.FixtureRequest):
    """ Test S2 cloud/shadow masking `prob` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_shadows=True, prob=80)
    masked_image._set_region_stats(region_10000ha)
    prob80_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_shadows=True, prob=40)
    masked_image._set_region_stats(region_10000ha)
    prob40_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    # test there is more cloud (less CLOUDLESS_PORTION) with prob=40 as compared to prob=80
    assert prob80_portion > prob40_portion


@pytest.mark.parametrize('image_id', ['s2_sr_image_id', 's2_toa_image_id'])
def test_s2_cloudmask_method(image_id: str, region_10000ha: Dict, request: pytest.FixtureRequest):
    """ Test S2 cloud/shadow masking `mask_method` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='cloud-prob')
    masked_image._set_region_stats(region_10000ha)
    cloud_prob_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_method='qa')
    masked_image._set_region_stats(region_10000ha)
    qa_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    # test `mask_method` changes CLOUDLESS_PORTION but not by too much
    assert cloud_prob_portion != qa_portion
    assert cloud_prob_portion == pytest.approx(qa_portion, abs=10)


@pytest.mark.parametrize('image_id', ['s2_sr_image_id', 's2_toa_image_id'])
def test_s2_cloudmask_mask_cirrus(image_id: str, region_10000ha: Dict, request: pytest.FixtureRequest):
    """ Test S2 cloud/shadow masking `mask_cirrus` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='qa', mask_cirrus=False)
    # cloud and shadow free portion
    masked_image._set_region_stats(region_10000ha)
    non_cirrus_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_method='qa', mask_cirrus=True)
    masked_image._set_region_stats(region_10000ha)
    # cloud, cirrus and shadow free portion
    cirrus_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    assert non_cirrus_portion >= cirrus_portion


@pytest.mark.parametrize('image_id', ['s2_sr_image_id', 's2_toa_image_id'])
def test_s2_cloudmask_dark(image_id: str, region_10000ha: Dict, request: pytest.FixtureRequest):
    """ Test S2 cloud/shadow masking `dark` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, dark=0.5)
    masked_image._set_region_stats(region_10000ha)
    dark_pt5_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, dark=0.1)
    masked_image._set_region_stats(region_10000ha)
    datk_pt1_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    # test that increasing `dark` results in an increase in detected shadow and corresponding decrease in
    # CLOUDLESS_PORTION
    assert datk_pt1_portion > dark_pt5_portion


@pytest.mark.parametrize('image_id', ['s2_sr_image_id', 's2_toa_image_id'])
def test_s2_cloudmask_shadow_dist(image_id: str, region_10000ha: Dict, request: pytest.FixtureRequest):
    """ Test S2 cloud/shadow masking `shadow_dist` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, shadow_dist=200)
    masked_image._set_region_stats(region_10000ha)
    sd200_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, shadow_dist=1000)
    masked_image._set_region_stats(region_10000ha)
    sd1000_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    # test that increasing `shadow_dist` results in an increase in detected shadow and corresponding decrease in
    # CLOUDLESS_PORTION
    assert sd200_portion > sd1000_portion


@pytest.mark.parametrize('image_id', ['s2_sr_image_id', 's2_toa_image_id'])
def test_s2_cloudmask_cdi_thresh(image_id: str, region_10000ha: Dict, request: pytest.FixtureRequest):
    """ Test S2 cloud/shadow masking `cdi_thresh` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, cdi_thresh=.5)
    masked_image._set_region_stats(region_10000ha)
    cdi_pt5_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, cdi_thresh=-.5)
    masked_image._set_region_stats(region_10000ha)
    cdi_negpt5_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    # test that increasing `cdi_thresh` results in an increase in detected cloud and corresponding decrease in
    # CLOUDLESS_PORTION
    assert cdi_negpt5_portion > cdi_pt5_portion


@pytest.mark.parametrize('image_id, max_cloud_dist', [('s2_sr_image_id', 100), ('s2_sr_hm_image_id', 500)])
def test_s2_clouddist_max(image_id: str, max_cloud_dist: int, region_10000ha: Dict, request: pytest.FixtureRequest):
    """ Test S2 cloud distance `max_cloud_dist` parameter. """

    def get_max_cloud_dist(cloud_dist: ee.Image):
        """ Get the maximum of `cloud_dist` over region_10000ha. """
        mcd = cloud_dist.reduceRegion(reducer='max', geometry=region_10000ha, bestEffort=True, maxPixels=1e4)
        return mcd.get('CLOUD_DIST').getInfo() * 10

    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, max_cloud_dist=max_cloud_dist)
    cloud_dist = masked_image.ee_image.select('CLOUD_DIST')
    meas_max_cloud_dist = get_max_cloud_dist(cloud_dist)
    assert meas_max_cloud_dist == pytest.approx(max_cloud_dist, rel=0.1)


def get_mask_stats(masked_image: MaskedImage, region: Dict):
    """  Get cloud/shadow etc aux band statistics for a given MaskedImage and region. """
    ee_image = masked_image.ee_image
    pan = ee_image.select([0, 1, 2]).reduce(ee.Reducer.mean())
    cdist = ee_image.select('CLOUD_DIST')
    cloud_mask = ee_image.select('CLOUD_MASK')
    shadow_mask = ee_image.select('SHADOW_MASK')
    cloudless_mask = ee_image.select('CLOUDLESS_MASK')
    pan_cloud = pan.updateMask(cloud_mask).rename('PAN_CLOUD')
    pan_shadow = pan.mask(shadow_mask).rename('PAN_SHADOW')
    pan_cloudless = pan.updateMask(cloudless_mask).rename('PAN_CLOUDLESS')
    cdist_cloud = cdist.updateMask(cloud_mask).rename('CDIST_CLOUD')
    cdist_cloudless = cdist.updateMask(cloudless_mask).rename('CDIST_CLOUDLESS')
    stats_image = ee.Image([pan_cloud, pan_shadow, pan_cloudless, cdist_cloud, cdist_cloudless])
    proj = get_projection(ee_image, min_scale=False)
    means = stats_image.reduceRegion(
        reducer='mean', geometry=region, crs=proj.crs(), scale=proj.nominalScale(), bestEffort=True, maxPixels=1e8
    )
    cdist_min = cdist.updateMask(cdist).reduceRegion(
        reducer='min', geometry=region, crs=proj.crs(), scale=proj.nominalScale(), bestEffort=True, maxPixels=1e8
    )
    means = means.set('CDIST_MIN', cdist_min.get('CLOUD_DIST'))
    return means.getInfo()


@pytest.mark.parametrize(
    'masked_image', ['l9_masked_image', 'l8_masked_image', 'l7_masked_image', 'l5_masked_image', 'l4_masked_image']
)
def test_landsat_aux_bands(masked_image: str, region_10000ha: Dict, request: pytest.FixtureRequest):
    """ Test Landsat auxiliary band values. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    stats = get_mask_stats(masked_image, region_10000ha)
    assert stats['PAN_CLOUD'] > stats['PAN_CLOUDLESS']
    assert stats['PAN_CLOUDLESS'] > stats['PAN_SHADOW']
    assert stats['CDIST_CLOUDLESS'] > stats['CDIST_CLOUD']


@pytest.mark.parametrize('masked_image',
    ['s2_sr_masked_image', 's2_toa_masked_image', 's2_sr_hm_masked_image', 's2_toa_hm_masked_image']
)  # yapf: disable
def test_s2_aux_bands(masked_image: str, region_10000ha: Dict, request: pytest.FixtureRequest):
    """ Test Sentinel-2 auxiliary band values. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    stats = get_mask_stats(masked_image, region_10000ha)
    assert stats['PAN_CLOUD'] > stats['PAN_CLOUDLESS']
    assert stats['PAN_CLOUDLESS'] > stats['PAN_SHADOW']
    assert stats['CDIST_CLOUDLESS'] > stats['CDIST_CLOUD']
    proj_scale = get_projection(masked_image.ee_image, min_scale=False).nominalScale().getInfo()
    assert stats['CDIST_MIN'] * 10 == proj_scale


@pytest.mark.parametrize('masked_image', ['s2_sr_masked_image', 'l9_masked_image'])
def test_mask_clouds(masked_image: str, region_100ha: Dict, tmp_path, request: pytest.FixtureRequest):
    """ Test MaskedImage.mask_clouds() by downloading and examining dataset masks. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    filename = tmp_path.joinpath(f'test_image.tif')
    masked_image.mask_clouds()
    proj_scale = get_projection(masked_image.ee_image, min_scale=False).nominalScale()
    masked_image.download(filename, region=region_100ha, dtype='int32', scale=proj_scale)
    assert filename.exists()
    with rio.open(filename, 'r') as ds:
        ds: rio.DatasetReader = ds
        cloudless_mask = ds.read(ds.descriptions.index('CLOUDLESS_MASK') + 1, masked=True)
        assert np.all(cloudless_mask == 1)  # all cloud/shadow areas should be masked (0)
        cloudless_mask = cloudless_mask.filled(0).astype('bool')  # fill nodata with 0 and cast to bool
        # test that cloudless_mask is the same as the nodata/dataset mask for each bands
        ds_masks = ds.read_masks().astype('bool')
        assert np.all(cloudless_mask == ds_masks)


def test_skysat_region_stats():
    """ Test _set_region_stats() works on SKYSAT image with no region. """
    ee_image = ee.Image('SKYSAT/GEN-A/PUBLIC/ORTHO/RGB/s02_20141004T074858Z')
    masked_image = MaskedImage(ee_image)
    masked_image._set_region_stats()
    assert 'FILL_PORTION' in masked_image.properties
    assert masked_image.properties['FILL_PORTION'] > 0.8
##

