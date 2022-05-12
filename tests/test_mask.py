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
from typing import Dict, Tuple

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


@pytest.fixture(scope='session')
def user_masked_image() -> MaskedImage:
    """ A MaskedImage instance where the encapsulated image has no fixed projection or ID.  """
    return MaskedImage(ee.Image([1, 2, 3]))


@pytest.fixture(scope='session')
def s2_sr_masked_image() -> MaskedImage:
    """ A MaskedImage instance encapsulating a Sentinel-2 SR image.  Covers `small_region`.  """
    return MaskedImage.from_id('COPERNICUS/S2_SR/20220114T080159_20220114T082124_T35HKC')


@pytest.fixture(scope='session')
def s2_toa_masked_image() -> MaskedImage:
    """ A MaskedImage instance encapsulating a Sentinel-2 TOA image.  Covers `small_region`.  """
    return MaskedImage.from_id('COPERNICUS/S2/20220114T080159_20220114T082124_T35HKC')


@pytest.fixture(scope='session')
def l9_masked_image() -> MaskedImage:
    """ A MaskedImage instance encapsulating a Landsat-9 SR image.  Covers `small_region`.  """
    return MaskedImage.from_id('LANDSAT/LC09/C02/T1_L2/LC09_172083_20220213')


@pytest.fixture(scope='session')
def l8_masked_image() -> MaskedImage:
    """ A MaskedImage instance encapsulating a Landsat-8 SR image.  Covers `small_region`.  """
    return MaskedImage.from_id('LANDSAT/LC08/C02/T1_L2/LC08_171084_20211009')


@pytest.fixture(scope='session')
def l7_masked_image() -> MaskedImage:
    """ A MaskedImage instance encapsulating a Landsat-7 SR image.  Covers `small_region`.  """
    return MaskedImage.from_id('LANDSAT/LE07/C02/T1_L2/LE07_172083_20220128')


@pytest.fixture(scope='session')
def l5_masked_image() -> MaskedImage:
    """ A MaskedImage instance encapsulating a Landsat-5 SR image.  Covers `small_region`.  """
    return MaskedImage.from_id('LANDSAT/LT05/C02/T1_L2/LT05_171083_20070715')


@pytest.fixture(scope='session')
def l4_masked_image() -> MaskedImage:
    """ A MaskedImage instance encapsulating a Landsat-4 SR image.  Covers `small_region`.  """
    return MaskedImage.from_id('LANDSAT/LT04/C02/T1_L2/LT04_172083_19890306')


@pytest.fixture(scope='session')
def mnbar_masked_image(l9_masked_image) -> MaskedImage:
    """ A MaskedImage instance encapsulating a reprojected MODIS NBAR image.  Covers `small_region`.  """
    return MaskedImage(
        ee.Image('MODIS/006/MCD43A4/2022_01_01').clip(l9_masked_image.footprint).
            reproject(l9_masked_image.crs, scale=500)
    )


def test_from_id():
    """ Test MaskedImage.from_id() sets _id. """
    ee_id = 'MODIS/006/MCD43A4/2022_01_01'
    gd_image = MaskedImage.from_id(ee_id)
    assert gd_image._id == ee_id


@pytest.mark.parametrize(
    'masked_image', [
        'user_masked_image', 'mnbar_masked_image'
    ]
)
def test_mask_aux_bands(masked_image: str, request):
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
        'user_masked_image', 'mnbar_masked_image'
    ]
)
def test_set_region_stats(masked_image: str, small_region, request):
    """ Test MaskedImage.set_region_stats() generates the expected properties and that these are in the valid range. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    masked_image.set_region_stats(small_region)
    for stat_name in ['FILL_PORTION', 'CLOUDLESS_PORTION']:
        assert stat_name in masked_image.properties
        assert masked_image.properties[stat_name] >= 0 and masked_image.properties[stat_name] <= 100
    assert masked_image.properties['CLOUDLESS_PORTION'] <= masked_image.properties['FILL_PORTION']


@pytest.mark.parametrize(
    'masked_image', ['l9_masked_image', 'l8_masked_image', 'l7_masked_image', 'l5_masked_image']
)
def test_landsat_cloudless_portion(masked_image: str, request):
    """ Test `geedim` CLOUDLESS_PORTION for the whole image against related Landsat CLOUD_COVER property. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    masked_image.set_region_stats()
    # the `geedim` cloudless portion inside the filled portion
    cloudless_portion = 100 * masked_image.properties['CLOUDLESS_PORTION'] / masked_image.properties['FILL_PORTION']
    # landsat provided cloudless portion
    landsat_cloudless_portion = 100 - float(masked_image.properties['CLOUD_COVER'])
    # allow for 15% difference due to shadow (& cirrus?) not being included in CLOUD_COVER
    assert cloudless_portion == pytest.approx(landsat_cloudless_portion, abs=15)


@pytest.mark.parametrize(
    'masked_image', ['s2_sr_masked_image', 's2_toa_masked_image']
)
def test_s2_cloudless_portion(masked_image: str, request):
    """ Test `geedim` CLOUDLESS_PORTION for the whole image against related Sentinel-2 properties. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    masked_image.set_region_stats()
    # allow for 15% difference due to different algorithms etc
    assert masked_image.properties['CLOUDLESS_PORTION'] == pytest.approx(
        100 - float(masked_image.properties['CLOUDY_PIXEL_PERCENTAGE']), abs=15
    )


@pytest.mark.parametrize(
    'masked_image', [
        's2_sr_masked_image', 's2_toa_masked_image'
    ]
)
def test_s2_download_aux_bands(masked_image: str, small_region, tmp_path, request):
    """ Test the downloaded auxiliary bands on different cloud masked images. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    filename = tmp_path.joinpath('test_image.tif')
    aux_mask_names = ['CLOUD_MASK', 'SHADOW_MASK', 'FILL_MASK', 'CLOUDLESS_MASK']
    masked_image.set_region_stats(small_region)
    # download as int32 so that nodata does not overlap with CLOUD_DIST etc
    masked_image.download(filename, region=small_region, dtype='int32')
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
        pan = ds.read([1, 2, 3], masked=True).mean(axis=0)

        # some sanity checking on the masks
        assert cloud_dist[~masks['CLOUDLESS_MASK']].mean() < cloud_dist[masks['CLOUDLESS_MASK']].mean()
        assert np.unique(cloud_dist[cloud_dist > 0])[0] == masked_image._proj_scale
        assert pan[masks['CLOUD_MASK']].mean() > pan[masks['CLOUDLESS_MASK']].mean()
        assert pan[masks['CLOUDLESS_MASK']].mean() > pan[masks['SHADOW_MASK']].mean()


@pytest.mark.parametrize(
    'masked_image', [
        'l9_masked_image', 'l8_masked_image', 'l7_masked_image', 'l5_masked_image'
    ]
)
def test_landsat_download_aux_bands(masked_image: str, small_region, tmp_path, request):
    """ Test the downloaded auxiliary bands on different cloud masked images. """
    masked_image: MaskedImage = request.getfixturevalue(masked_image)
    filename = tmp_path.joinpath('test_image.tif')
    aux_mask_names = ['CLOUD_MASK', 'SHADOW_MASK', 'FILL_MASK', 'CLOUDLESS_MASK']
    masked_image.set_region_stats(small_region)
    masked_image.download(filename, region=small_region)  # uint16 means nodata=0 wh
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
        assert cloud_dist[~masks['CLOUDLESS_MASK']].mean() < cloud_dist[masks['CLOUDLESS_MASK']].mean()
        assert pan[masks['CLOUD_MASK']].mean() > pan[masks['CLOUDLESS_MASK']].mean()

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
