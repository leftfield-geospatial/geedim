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
import pathlib
from typing import Dict, Tuple, List

import ee
import numpy as np
import pytest
import rasterio as rio
from rasterio import Affine
from rasterio.coords import BoundingBox
from rasterio.features import bounds
from rasterio.warp import transform_geom
from rasterio.windows import union

from geedim.download import BaseImage, split_id
from geedim.enums import ResamplingMethod
from geedim.mask import MaskedImage


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
def test_gen_mask_aux_bands_exist(masked_image: str, request):
    """ Test the presence of auxiliary band (i.e. FILL_MASK) in generic masked images. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    band_names = masked_image.ee_image.bandNames().getInfo()
    assert 'FILL_MASK' in band_names


@pytest.mark.parametrize(
    'masked_image', [
        's2_sr_masked_image', 's2_toa_masked_image', 'l9_masked_image', 'l8_masked_image',
        'l7_masked_image', 'l5_masked_image', 'l4_masked_image'
    ]
)
def test_cloud_mask_aux_bands_exist(masked_image: str, request):
    """ Test the presence of auxiliary bands in cloud masked images. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    band_names = masked_image.ee_image.bandNames().getInfo()
    exp_band_names = ['CLOUD_MASK', 'SHADOW_MASK', 'FILL_MASK', 'CLOUDLESS_MASK', 'CLOUD_DIST']
    for exp_band_name in exp_band_names:
        assert exp_band_name in band_names


@pytest.mark.parametrize(
    'masked_image', [
        's2_sr_masked_image', 's2_toa_masked_image', 'l9_masked_image', 'l8_masked_image',
        'l7_masked_image', 'l5_masked_image', 'l4_masked_image',
        'user_masked_image', 'modis_nbar_masked_image', 'gch_masked_image', 's1_sar_masked_image',
        'gedi_agb_masked_image', 'gedi_cth_masked_image', 'landsat_ndvi_masked_image'
    ]
)
def test_set_region_stats(masked_image: str, region_100ha, request):
    """ Test MaskedImage.set_region_stats() generates the expected properties and that these are in the valid range. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    masked_image.set_region_stats(region_100ha)
    for stat_name in ['FILL_PORTION', 'CLOUDLESS_PORTION']:
        assert stat_name in masked_image.properties
        assert masked_image.properties[stat_name] >= 0 and masked_image.properties[stat_name] <= 100
    assert masked_image.properties['CLOUDLESS_PORTION'] <= masked_image.properties['FILL_PORTION']


@pytest.mark.parametrize(
    'image_id', ['l9_image_id', 'l8_image_id', 'l7_image_id', 'l5_image_id', 'l4_image_id']
)
def test_landsat_cloudless_portion(image_id: str, request):
    """ Test `geedim` CLOUDLESS_PORTION for the whole image against related Landsat CLOUD_COVER property. """
    image_id: MaskedImage = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_shadows=False, mask_cirrus=False)
    masked_image.set_region_stats()
    # the `geedim` cloudless portion inside the filled portion
    cloudless_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    # landsat provided cloudless portion
    landsat_cloudless_portion = 100 - float(masked_image.properties['CLOUD_COVER'])
    assert cloudless_portion == pytest.approx(landsat_cloudless_portion, abs=5)


@pytest.mark.parametrize(
    'image_id', ['s2_toa_image_id', 's2_sr_image_id']
)
def test_s2_cloudless_portion(image_id: str, request):
    """ Test `geedim` CLOUDLESS_PORTION for the whole image against related Sentinel-2 properties. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='qa', mask_shadows=False, mask_cirrus=False)
    # TODO: init images with method=qa, mask_shadow and mask_cirrus=False
    masked_image.set_region_stats()
    s2_cloudless_portion = 100 - float(masked_image.properties['CLOUDY_PIXEL_PERCENTAGE'])
    # CLOUDLESS_MASK is eroded and dilated, so allow 10% difference to account for that
    assert masked_image.properties['CLOUDLESS_PORTION'] == pytest.approx(s2_cloudless_portion, abs=10)


@pytest.mark.parametrize(
    'image_id', ['s2_sr_image_id', 's2_toa_image_id']
)
def test_s2_cloudmask_mask_shadows(image_id: str, region_10000ha, request):
    """ Test S2 cloud/shadow masking `mask_shadows` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_shadows=False)
    masked_image.set_region_stats(region_10000ha)
    cloud_only_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_shadows=True)
    masked_image.set_region_stats(region_10000ha)
    cloudless_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    assert cloud_only_portion > cloudless_portion

@pytest.mark.parametrize(
    'image_id', ['s2_sr_image_id', 's2_toa_image_id']
)
def test_s2_cloudmask_prob(image_id: str, region_10000ha, request):
    """ Test S2 cloud/shadow masking `prob` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_shadows=True, prob=80)
    masked_image.set_region_stats(region_10000ha)
    prob80_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_shadows=True, prob=40)
    masked_image.set_region_stats(region_10000ha)
    prob40_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    assert prob80_portion > prob40_portion

@pytest.mark.parametrize(
    'image_id', ['s2_sr_image_id', 's2_toa_image_id']
)
def test_s2_cloudmask_method(image_id: str, region_10000ha, request):
    """ Test S2 cloud/shadow masking `mask_method` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='cloud-prob')
    masked_image.set_region_stats(region_10000ha)
    cloud_prob_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_method='qa')
    masked_image.set_region_stats(region_10000ha)
    qa_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    assert cloud_prob_portion == pytest.approx(qa_portion, abs=5)

@pytest.mark.parametrize(
    'image_id', ['s2_sr_image_id', 's2_toa_image_id']
)
def test_s2_cloudmask_mask_cirrus(image_id: str, region_10000ha, request):
    """ Test S2 cloud/shadow masking `mask_cirrus` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, mask_method='qa', mask_cirrus=False)
    masked_image.set_region_stats(region_10000ha)
    non_cirrus_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, mask_method='qa', mask_cirrus=True)
    masked_image.set_region_stats(region_10000ha)
    cirrus_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    assert non_cirrus_portion > cirrus_portion

@pytest.mark.parametrize(
    'image_id', ['s2_sr_image_id', 's2_toa_image_id']
)
def test_s2_cloudmask_dark(image_id: str, region_10000ha, request):
    """ Test S2 cloud/shadow masking `dark` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, dark=0.5)
    masked_image.set_region_stats(region_10000ha)
    dark_pt5_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, dark=0.1)
    masked_image.set_region_stats(region_10000ha)
    datk_pt1_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    assert datk_pt1_portion > dark_pt5_portion

@pytest.mark.parametrize(
    'image_id', ['s2_sr_image_id', 's2_toa_image_id']
)
def test_s2_cloudmask_shadow_dist(image_id: str, region_10000ha, request):
    """ Test S2 cloud/shadow masking `shadow_dist` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, shadow_dist=200)
    masked_image.set_region_stats(region_10000ha)
    sd200_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, shadow_dist=1000)
    masked_image.set_region_stats(region_10000ha)
    sd1000_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    assert sd200_portion > sd1000_portion

@pytest.mark.parametrize(
    'image_id', ['s2_sr_image_id', 's2_toa_image_id']
)
def test_s2_cloudmask_cdi_thresh(image_id: str, region_10000ha, request):
    """ Test S2 cloud/shadow masking `cdi_thresh` parameter. """
    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, cdi_thresh=.5)
    masked_image.set_region_stats(region_10000ha)
    cdi_pt5_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    masked_image = MaskedImage.from_id(image_id, cdi_thresh=-.5)
    masked_image.set_region_stats(region_10000ha)
    cdi_negpt5_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    assert cdi_negpt5_portion > cdi_pt5_portion

@pytest.mark.parametrize(
    'image_id, max_cloud_dist', [('s2_sr_image_id', 100), ('s2_sr_image_id', 500)]
)
def test_s2_clouddist_max(image_id: str, max_cloud_dist:int, region_10000ha: Dict, request):
    """ Test S2 cloud distance `max_cloud_dist` parameter. """

    def get_max_cloud_dist(cloud_dist: ee.Image):
        """ Get the maximum of `cloud_dist` over region_10000ha. """
        mcd = cloud_dist.reduceRegion(reducer='max', geometry=region_10000ha, bestEffort=True, maxPixels=1e4)
        return mcd.get('CLOUD_DIST').getInfo()

    image_id: str = request.getfixturevalue(image_id)
    masked_image = MaskedImage.from_id(image_id, max_cloud_dist=max_cloud_dist)
    cloud_dist = masked_image.ee_image.select('CLOUD_DIST')
    _max_cloud_dist = get_max_cloud_dist(cloud_dist)
    assert _max_cloud_dist == pytest.approx(max_cloud_dist, rel=0.1)



@pytest.mark.parametrize(
    'masked_image', [
        'l9_masked_image', 'l8_masked_image', 'l7_masked_image', 'l5_masked_image'
    ]
)
def test_landsat_download_aux_bands(masked_image: str, region_100ha, tmp_path, request):
    """ Test the downloaded auxiliary bands on different cloud masked images. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    filename = tmp_path.joinpath(f'test_image.tif')
    aux_mask_names = ['CLOUD_MASK', 'SHADOW_MASK', 'FILL_MASK', 'CLOUDLESS_MASK']
    masked_image.set_region_stats(region_100ha)
    masked_image.download(filename, region=region_100ha)  # uint16 means nodata=0 wh
    assert filename.exists()

    with rio.open(filename, 'r') as ds:
        sr_band_idx = [
            ds.descriptions.index(band_name) + 1 for band_name in ds.descriptions if band_name.startswith('SR_B')
        ]
        ds_mask = np.all(ds.read_masks(sr_band_idx).astype('bool'), axis=0)
        masks = {}
        for mask_name in aux_mask_names:
            assert mask_name in ds.descriptions
            masks[mask_name] = ds_mask & ds.read(ds.descriptions.index(mask_name) + 1).astype('bool')

        assert np.all(ds_mask == masks['FILL_MASK'])
        cloud_dist = ds.read(ds.descriptions.index('CLOUD_DIST') + 1, masked=True)
        cloud_dist[~ds_mask] = 0
        pan = ds.read([1, 2, 3], masked=True).mean(axis=0)
        cloudless_mask = ~(masks['CLOUD_MASK'] | masks['SHADOW_MASK']) & masks['FILL_MASK']

        # some sanity checking on the masks
        assert np.all(cloudless_mask == masks['CLOUDLESS_MASK'])
        assert cloud_dist[masks['CLOUD_MASK']].mean() < cloud_dist[~masks['CLOUD_MASK']].mean()
        assert pan[masks['CLOUD_MASK']].mean() > pan[masks['CLOUDLESS_MASK']].mean()

@pytest.mark.parametrize(
    'masked_image', [
        's2_sr_masked_image', 's2_toa_masked_image'
    ]
)
def test_s2_download_aux_band(masked_image: str, region_100ha, tmp_path, request):
    """ Test the downloaded auxiliary bands on different cloud masked images. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    filename = tmp_path.joinpath(f'test_image.tif')
    aux_mask_names = ['CLOUD_MASK', 'SHADOW_MASK', 'FILL_MASK', 'CLOUDLESS_MASK']
    masked_image.set_region_stats(region_100ha)
    # download as int32 so that nodata does not overlap with CLOUD_DIST etc
    masked_image.download(filename, region=region_100ha, dtype='int32')
    assert filename.exists()

    with rio.open(filename, 'r') as ds:
        sr_band_idx = [
            ds.descriptions.index(band_name) + 1 for band_name in ds.descriptions if band_name.startswith('B')
        ]
        ds_mask = np.all(ds.read_masks(sr_band_idx).astype('bool'), axis=0)
        masks = {}
        for mask_name in aux_mask_names:
            assert mask_name in ds.descriptions
            masks[mask_name] = ds_mask & ds.read(ds.descriptions.index(mask_name) + 1).astype('bool')

        assert np.all(ds_mask == masks['FILL_MASK'])
        cloud_dist = ds.read(ds.descriptions.index('CLOUD_DIST') + 1, masked=True)
        cloud_dist[~ds_mask] = 0
        pan = ds.read([2, 3, 4], masked=True).mean(axis=0)

        # some sanity checking on the masks
        assert cloud_dist[~masks['CLOUDLESS_MASK']].mean() < cloud_dist[masks['CLOUDLESS_MASK']].mean()
        assert np.unique(cloud_dist[cloud_dist > 0])[0] == masked_image._proj_scale
        assert pan[masks['CLOUD_MASK']].mean() > pan[masks['CLOUDLESS_MASK']].mean()
        if np.sum(masks['SHADOW_MASK']) > 0:
            assert pan[masks['CLOUDLESS_MASK']].mean() > pan[masks['SHADOW_MASK']].mean()



# To test
# -----------
# - from_id sets _id
# - init for different supported images & some generic ones generates the expected masks (use bandNames or similar)
# - set_region_stats() might we be able to compare the image wide cloud properties with CLOUDLESS_PORTION and
# FILL_PORTION?  In any case check the range of these stats makes sense (at some point we should also compare against
# downloaded images).
# - mask_clouds() how can we test this? It would probably need to be in downloaded images....  We could do some rough
# checks like compare mean of cloud_mask area to mean of shadow_mask to mean of cloudless_mask.  And compare ee
# CLOUDLESS_PORTION, to a calc on downloaded image.
# - some sensor specific tests, like CLOUD_PROB for S2, and all the cloud mask parameters
#  - we can do things like compare CLOUDLESS_PORTION with shadow_mask=True/False, method=qa/cloud-prob etc
# - CLOUD_DIST:  Again we can test on downloaded images:  is CLOUD_DIST 0 under cloud/shadow mask?  do the range of
# values make sense i.e. if it is x pixels away from cloud, is the CLOUD_DIST ~ x * scale?  This would be a good
# check for the whole re-projection thing (S2 specific).
# - class_from_id...
# - get_projection() on S2, on non-fixed proj image.

# TO do
# -----------
# - How can we organise fixtures so that we might have >1 image for a particular sensor and region, and if possible
# 1 image covers a small and large region.  Also, not including water may help.
# - How can we organise fixtures so that we have a decent landsat4 image for the region

# Notes
# --------
# - keep session scope fixtures of masked images for all the supported collections & some generic.  this will save on
# getInfo's for any BaseImage property calls.
# - download images once only - so do all those tests from one function.  or even make the downloaded files fixtures?
# then we could test them in multiple different functions...   the thing is in terms of test design, it is the
# tests that should fail, not the fixtures. but perhaps it is much of a muchness...
