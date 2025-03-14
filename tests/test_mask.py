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

import ee
import numpy as np
import pytest

from geedim.mask import (
    _CloudlessImage,
    _get_class_for_id,
    _LandsatImage,
    _MaskedImage,
    _Sentinel2Image,
    _Sentinel2SrImage,
    _Sentinel2ToaImage,
)

# TODO:
#  - test _get_class_from_id
#  - _MaskedImage._get_mask_bands()
#   - dict contains FILL_MASK
#   - FILL_MASK is correct (e.g. reduceRegion on prepared_image)
#   - what images to test on?  not sure we need > 1
#  - _MaskedImage.add_mask_bands()
#   - test FILL_MASK exists in non-cloud/shadow image
#   [- test CS masks exist in CS image
#   - test new CS masks overwrite old ones for fixed proj]
#   - test no masks added on non-fixed proj
#  - _MaskedImage.mask_clouds()
#   - test masks with FILL_MASK on non CS image.  to test this, i think different bands will need
#   to be masked differently
#   [- test masks with CLOUDLESS_MASK on CS image.]
#   - don't think this needs >1 image of non-CS / CS each
#  - _MaskedImage.set_mask_portions()
#   - test FILL_PORTION and CLOUDLESS_PORTION on known non-CS image
#   [- test FILL_PORTION and CLOUDLESS_PORTION on a CS image]
#   - scale parameter makes a difference? e.g. with non-fixed image
#  - _CloudlessImage (see mask_clouds and set_mask_portions tests above)
#  -


@pytest.fixture(scope='session')
def l5_ee_image(l5_image_id: str) -> ee.Image:
    return ee.Image(l5_image_id)


@pytest.fixture(scope='session')
def l7_ee_image(l7_image_id: str) -> ee.Image:
    return ee.Image(l7_image_id)


@pytest.fixture(scope='session')
def l8_ee_image(l8_image_id: str) -> ee.Image:
    return ee.Image(l8_image_id)


@pytest.fixture(scope='session')
def l9_ee_image(l9_image_id: str) -> ee.Image:
    return ee.Image(l9_image_id)


@pytest.fixture(scope='session')
def landsat_ee_images(
    l5_ee_image: ee.Image, l7_ee_image: ee.Image, l8_ee_image: ee.Image, l9_ee_image: ee.Image
) -> list[ee.Image]:
    return [l5_ee_image, l7_ee_image, l8_ee_image, l9_ee_image]


@pytest.fixture(scope='session')
def s2_toa_ee_image(s2_toa_image_id: str) -> ee.Image:
    return ee.Image(s2_toa_image_id)


@pytest.fixture(scope='session')
def s2_sr_ee_image(s2_sr_image_id: str) -> ee.Image:
    return ee.Image(s2_sr_image_id)


@pytest.fixture(scope='session')
def s2_toa_hm_ee_image(s2_toa_hm_image_id: str) -> ee.Image:
    return ee.Image(s2_toa_hm_image_id)


@pytest.fixture(scope='session')
def s2_sr_hm_ee_image(s2_sr_hm_image_id: str) -> ee.Image:
    return ee.Image(s2_sr_hm_image_id)


@pytest.fixture(scope='session')
def s2_ee_images(
    s2_toa_ee_image: ee.Image,
    s2_sr_ee_image: ee.Image,
    s2_toa_hm_ee_image: ee.Image,
    s2_sr_hm_ee_image: ee.Image,
) -> list[ee.Image]:
    return [s2_toa_ee_image, s2_sr_ee_image, s2_toa_hm_ee_image, s2_sr_hm_ee_image]


@pytest.fixture(scope='session')
def s2_sr_hm_nocs_ee_image(s2_sr_hm_ee_image: ee.Image) -> ee.Image:
    return s2_sr_hm_ee_image.set('system:index', 'unknown')


@pytest.fixture(scope='session')
def modis_nbar_ee_image(modis_nbar_image_id: str) -> ee.Image:
    return ee.Image(modis_nbar_image_id)


@pytest.fixture(scope='session')
def landsat_ndvi_ee_image(landsat_ndvi_image_id: str) -> ee.Image:
    return ee.Image(landsat_ndvi_image_id)


@pytest.fixture(scope='session')
def const_ee_image(region_100ha: dict) -> ee.Image:
    coords = np.array(region_100ha['coordinates']).squeeze()
    bounds = [*coords.min(axis=0), *coords.max(axis=0)]
    bounds[0] += (bounds[2] - bounds[0]) / 2
    half_region = ee.Geometry.Rectangle(*bounds)
    im = ee.Image([1, 2, 3])
    band = im.select(2)
    band = band.updateMask(band.mask().clip(half_region))
    im = im.addBands(band, overwrite=True)
    return im


@pytest.fixture(scope='session')
def fixed_const_ee_image(const_ee_image: ee.Image, region_10000ha: dict) -> ee.Image:
    return const_ee_image.setDefaultProjection(crs='EPSG:3857', scale=10).clip(region_10000ha)


@pytest.fixture(scope='session')
def non_cs_ee_images(
    modis_nbar_ee_image: ee.Image, landsat_ndvi_ee_image: ee.Image, const_ee_image: ee.Image
) -> list[ee.Image]:
    return [modis_nbar_ee_image, landsat_ndvi_ee_image, const_ee_image, fixed_const_ee_image]


def test_get_class_for_id(
    landsat_image_ids: list[str],
    s2_sr_image_id: str,
    s2_sr_hm_image_id: str,
    s2_toa_image_id: str,
    s2_toa_hm_image_id: str,
    modis_nbar_image_id: str,
    landsat_ndvi_image_id: str,
):
    """Test _get_class_for_id()."""
    for im_id in landsat_image_ids:
        assert _get_class_for_id(im_id) == _LandsatImage
    for im_id in [s2_sr_image_id, s2_sr_hm_image_id]:
        assert _get_class_for_id(im_id) == _Sentinel2SrImage
    for im_id in [s2_toa_image_id, s2_toa_hm_image_id]:
        assert _get_class_for_id(im_id) == _Sentinel2ToaImage
    for im_id in [modis_nbar_image_id, landsat_ndvi_image_id, None, '']:
        assert _get_class_for_id(im_id) == _MaskedImage


def test_masked_image_get_mask_bands(fixed_const_ee_image: ee.Image, region_100ha: dict):
    """Test _MaskedImage._get_mask_bands()."""
    mask_bands = _MaskedImage._get_mask_bands(fixed_const_ee_image)
    assert list(mask_bands.keys()) == ['fill']

    # test fill mask is named correctly and covers half of region_100ha
    fill_portion = mask_bands['fill'].reduceRegion('mean', geometry=region_100ha, scale=10)
    fill_portion = fill_portion.getInfo()
    assert 'FILL_MASK' in fill_portion
    assert fill_portion['FILL_MASK'] == pytest.approx(0.5, abs=0.01)


def test_masked_image_add_mask_bands(
    fixed_const_ee_image: ee.Image, const_ee_image: ee.Image, region_100ha: dict
):
    """Test _MaskedImage.add_mask_bands()."""
    # create test fixed projection images with mask band added
    masked_image = _MaskedImage.add_mask_bands(fixed_const_ee_image)
    # zero the mask of the first band, and re-add mask band (should overwrite)
    band = masked_image.select(0).updateMask(0)
    remasked_image = masked_image.addBands(band, overwrite=True)
    remasked_image = _MaskedImage.add_mask_bands(remasked_image)

    # find portion of FILL_MASK that covers region_100ha in masked_image & remasked_image
    fill_portions = [
        im.select('FILL_MASK').reduceRegion('mean', geometry=region_100ha, scale=10)
        for im in [masked_image, remasked_image]
    ]

    # create a test non-fixed projection image (should have no mask band)
    comp_image = _MaskedImage.add_mask_bands(const_ee_image)

    # find band names of all test images
    band_names = [im.bandNames() for im in [masked_image, remasked_image, comp_image]]

    # combine all getInfo() calls into one
    info = ee.Dictionary(dict(fill_portions=ee.List(fill_portions), band_names=ee.List(band_names)))
    info = info.getInfo()

    # test FILL_MASK exists in fixed projection images and not in non-fixed projection image
    for band_names in [info['band_names'][0], info['band_names'][1]]:
        assert len(band_names) == 4
        assert 'FILL_MASK' in band_names
    assert len(info['band_names'][2]) == 3
    assert 'FILL_MASK' not in info['band_names'][2]

    # test FILL_MASK was overwritten in second fixed projection image
    assert info['fill_portions'][0]['FILL_MASK'] == pytest.approx(0.5, abs=0.01)
    assert info['fill_portions'][1]['FILL_MASK'] == pytest.approx(0.0, abs=0.01)


def test_masked_image_mask_clouds(fixed_const_ee_image: ee.Image, region_100ha: dict):
    """Test _MaskedImage.mask_clouds()."""
    image = _MaskedImage.add_mask_bands(fixed_const_ee_image)
    image = _MaskedImage.mask_clouds(image)
    coverages = image.mask().reduceRegion('mean', geometry=region_100ha, scale=10)
    coverages = coverages.getInfo()
    assert len(coverages) == 4
    assert all([v == pytest.approx(0.5, abs=0.01) for v in coverages.values()])


def test_masked_image_set_mask_portions(fixed_const_ee_image: ee.Image, region_100ha: dict):
    """Test _MaskedImage.set_mask_portions()."""
    image = _MaskedImage.add_mask_bands(fixed_const_ee_image)
    image = _MaskedImage.set_mask_portions(image, region=region_100ha)
    # coarser scale portions
    image_scale100 = _MaskedImage.set_mask_portions(image, region=region_100ha, scale=100)

    # combine all getInfo() calls into one
    portions = ee.List(
        [im.toDictionary(['FILL_PORTION', 'CLOUDLESS_PORTION']) for im in [image, image_scale100]]
    ).getInfo()

    assert portions[0]['FILL_PORTION'] == pytest.approx(50, abs=1)
    assert portions[0]['CLOUDLESS_PORTION'] == pytest.approx(100, abs=1)

    # test scale param affects portion accuracy
    assert portions[1]['FILL_PORTION'] == pytest.approx(50, abs=10)
    assert abs(portions[1]['FILL_PORTION'] - 50) > abs(portions[0]['FILL_PORTION'] - 50)


def test_cloudless_image_mask_clouds(s2_sr_hm_ee_image: ee.Image, region_100ha: dict):
    """Test _CloudlessImage.mask_clouds()."""
    image = _Sentinel2SrImage.add_mask_bands(s2_sr_hm_ee_image)
    image = _CloudlessImage.mask_clouds(image)
    # exclude MSK_CLASSI_* bands as they are fully masked
    coverages = (
        image.select('B.*|.*MASK').mask().reduceRegion('mean', geometry=region_100ha, scale=10)
    )
    coverages = coverages.getInfo()
    assert 'FILL_MASK' in coverages and 'CLOUDLESS_MASK' in coverages
    assert len(set(coverages.values())) == 1  # test all equal
    assert all([0 < v < 100 for v in coverages.values()])


def test_cloudless_image_set_mask_portions(s2_sr_hm_ee_image: ee.Image, region_100ha: dict):
    """Test _CloudlessImage.set_mask_portions()"""
    image = _Sentinel2SrImage.add_mask_bands(s2_sr_hm_ee_image)
    image = _CloudlessImage.set_mask_portions(image, region=region_100ha)
    # coarser scale portions
    image_scale100 = _CloudlessImage.set_mask_portions(image, region=region_100ha, scale=100)

    # combine all getInfo() calls into one
    portions = ee.List(
        [im.toDictionary(['FILL_PORTION', 'CLOUDLESS_PORTION']) for im in [image, image_scale100]]
    ).getInfo()

    for portion in portions:
        assert portion['FILL_PORTION'] == 100
        assert 0 < portion['CLOUDLESS_PORTION'] < 100

    # test scale param affects portion
    assert portions[1]['CLOUDLESS_PORTION'] != pytest.approx(
        portions[0]['CLOUDLESS_PORTION'], abs=0.1
    )


def test_landsat_image_get_mask_bands(landsat_ee_images: list[ee.Image], region_10000ha: dict):
    """Test _LandsatImage._get_mask_bands() collection support."""
    # find mask bands for each landsat image
    mask_bands_list = [_LandsatImage._get_mask_bands(im) for im in landsat_ee_images]
    assert set(mask_bands_list[0].keys()) == {'shadow', 'cloud', 'cloudless', 'fill', 'dist'}

    # find means of mask bands over region_10000ha, combining all getInfo() calls into one
    means = [
        ee.Image(list(mb.values())).reduceRegion('mean', geometry=region_10000ha, scale=30)
        for mb in mask_bands_list
    ]
    means = ee.List(means).getInfo()

    # test mean values / ranges
    for mean in means:
        assert all([0 < mean[bn] < 1 for bn in ['CLOUD_MASK', 'SHADOW_MASK', 'CLOUDLESS_MASK']])
        # this assumes that for the most part, pixels are either cloud or shadow, not both
        assert (mean['CLOUD_MASK'] + mean['SHADOW_MASK']) == pytest.approx(
            1 - mean['CLOUDLESS_MASK'], abs=0.05
        )
        assert mean['FILL_MASK'] == pytest.approx(1, abs=0.01)
        assert 0 < mean['CLOUD_DIST'] < 5000


def test_landsat_image_get_mask_bands_params(l9_ee_image: ee.Image):
    """Test _LandsatImage._get_mask_bands() parameters."""
    # create test stats for each parameter (there is no cirrus in any of region_*ha, so stats are
    # found over the whole image for the mask_cirrus test)
    # reference
    reduce_kwargs = dict(scale=100, bestEffort=True)
    mask_bands = _LandsatImage._get_mask_bands(
        l9_ee_image, mask_shadows=True, mask_cirrus=True, max_cloud_dist=5000
    )
    mask_image = ee.Image([mask_bands[k] for k in ['shadow', 'cloudless', 'cloud']])
    stats = dict(ref=mask_image.reduceRegion('mean', **reduce_kwargs))

    # mask_shadows
    mask_bands = _LandsatImage._get_mask_bands(l9_ee_image, mask_shadows=False)
    assert 'shadow' not in mask_bands
    stats['mask_shadows'] = mask_bands['cloudless'].reduceRegion('mean', **reduce_kwargs)

    # mask_cirrus
    mask_bands = _LandsatImage._get_mask_bands(l9_ee_image, mask_cirrus=False)
    stats['mask_cirrus'] = mask_bands['cloud'].reduceRegion('mean', **reduce_kwargs)

    # max_cloud_dist
    max_cloud_dist = 100
    mask_bands = _LandsatImage._get_mask_bands(l9_ee_image, max_cloud_dist=max_cloud_dist)
    stats['max_cloud_dist'] = mask_bands['dist'].reduceRegion('max', **reduce_kwargs)

    # fetch stats, combining all getInfo() calls into one
    stats = ee.Dictionary(stats).getInfo()

    # test mask_shadows
    assert stats['mask_shadows']['CLOUDLESS_MASK'] > stats['ref']['CLOUDLESS_MASK']
    # this assumes that for the most part, pixels are either cloud or shadow, not both
    assert stats['mask_shadows']['CLOUDLESS_MASK'] == pytest.approx(
        stats['ref']['CLOUDLESS_MASK'] + stats['ref']['SHADOW_MASK'], abs=0.05
    )

    # test mask_cirrus
    assert stats['mask_cirrus']['CLOUD_MASK'] < stats['ref']['CLOUD_MASK']

    # test max_cloud_dist
    assert stats['max_cloud_dist']['CLOUD_DIST'] == 100


def test_s2_image_get_cloud_dist(s2_sr_hm_ee_image: ee.Image, region_100ha: dict):
    """Test _Sentinel2Image._get_cloud_dist()."""
    # find min & max of cloud distance for different max_cloud_dist vals, combining all getInfo()
    # calls into one
    mask_bands = _Sentinel2SrImage()._get_mask_bands(s2_sr_hm_ee_image)
    min_maxs = {}
    for max_cloud_dist in [100, 200]:
        cloud_dist = _Sentinel2Image._get_cloud_dist(
            mask_bands['cloudless'],
            proj=s2_sr_hm_ee_image.select('B1').projection(),
            max_cloud_dist=max_cloud_dist,
        )
        min_maxs[max_cloud_dist] = cloud_dist.reduceRegion(
            ee.Reducer.minMax(), geometry=region_100ha, scale=10
        )
    min_maxs = ee.Dictionary(min_maxs).getInfo()

    for max_cloud_dist, min_max in min_maxs.items():
        assert min_max['CLOUD_DIST_min'] == 0
        assert min_max['CLOUD_DIST_max'] == float(max_cloud_dist)


def test_s2_image_get_mask_bands(s2_ee_images: list[ee.Image], region_100ha: dict):
    """Test _Sentinel2Image._get_mask_bands() collection support."""
    # find mask bands for each s2 image (using the _Sentinel2Image base class, leaving s2_toa=False
    # as it has no effect on the 'cloud-score' method)
    mask_bands_list = [_Sentinel2Image._get_mask_bands(im) for im in s2_ee_images]
    assert set(mask_bands_list[0].keys()) == {'cloudless', 'fill', 'dist', 'score'}

    # find means of mask bands over region_100ha, combining all getInfo() calls into one
    means = [
        ee.Image(list(mb.values())).reduceRegion('mean', geometry=region_100ha, scale=10)
        for mb in mask_bands_list
    ]
    means = ee.List(means).getInfo()

    # test mean values / ranges
    for mean in means:
        assert 0 < mean['CLOUDLESS_MASK'] < 1
        assert mean['FILL_MASK'] == pytest.approx(1, abs=0.01)
        assert 0 < mean['CLOUD_DIST'] < 5000
        assert 0 < mean['CLOUD_SCORE'] < 1


def test_s2_image_get_mask_bands_params(s2_sr_hm_ee_image: ee.Image, region_100ha: dict):
    """Test _Sentinel2Image._get_mask_bands() parameters with the 'cloud-score' method."""

    def image_from_mask_bands(mask_bands: dict[str, ee.Image]) -> ee.Image:
        """Return an image of the cloudless mask and score bands."""
        return ee.Image([mask_bands[k] for k in ['cloudless', 'score']])

    # create test stats for each parameter
    # reference
    reduce_kwargs = dict(geometry=region_100ha, scale=10)
    mask_bands = _Sentinel2Image._get_mask_bands(
        s2_sr_hm_ee_image, mask_method='cloud-score', score=0.6, cs_band='cs'
    )
    stats = dict(ref=image_from_mask_bands(mask_bands).reduceRegion('mean', **reduce_kwargs))

    # score
    mask_bands = _Sentinel2Image._get_mask_bands(
        s2_sr_hm_ee_image, mask_method='cloud-score', score=0.5, cs_band='cs'
    )
    stats['score'] = image_from_mask_bands(mask_bands).reduceRegion('mean', **reduce_kwargs)

    # cs_band
    mask_bands = _Sentinel2Image._get_mask_bands(
        s2_sr_hm_ee_image, mask_method='cloud-score', score=0.6, cs_band='cs_cdf'
    )
    mask_bands = ee.Image([mask_bands[k] for k in ['cloudless', 'score']])
    stats['cs_band'] = mask_bands.reduceRegion('mean', **reduce_kwargs)

    # fetch stats, combining all getInfo() calls into one
    stats = ee.Dictionary(stats).getInfo()

    # test ranges
    for stat in stats.values():
        assert 0 < stat['CLOUDLESS_MASK'] < 1
        assert 0 < stat['CLOUD_SCORE'] < 1

    # test score
    assert stats['score']['CLOUDLESS_MASK'] > stats['ref']['CLOUDLESS_MASK']
    assert stats['score']['CLOUD_SCORE'] == stats['ref']['CLOUD_SCORE']

    # test cs_band
    assert stats['cs_band']['CLOUDLESS_MASK'] != stats['ref']['CLOUDLESS_MASK']
    assert stats['cs_band']['CLOUD_SCORE'] != stats['ref']['CLOUD_SCORE']


def test_s2_image_get_mask_bands_no_cloud_score(
    s2_sr_hm_nocs_ee_image: ee.Image, region_100ha: dict
):
    """Test _Sentinel2Image._get_mask_bands() fully masks cloud score dependent bands when no
    cloud score image exists.
    """
    # find region sums of cloud score dependent bands, and region mean of fill mask, combining
    # all getInfo() calls into one (sums rather than means are used to avoid division by zero)
    mask_bands = _Sentinel2Image._get_mask_bands(s2_sr_hm_nocs_ee_image, mask_method='cloud-score')
    sum_image = ee.Image([mask_bands[k] for k in ['cloudless', 'score', 'dist']])
    sums = sum_image.reduceRegion('sum', geometry=region_100ha, scale=10)
    fill_mean = mask_bands['fill'].reduceRegion('mean', geometry=region_100ha, scale=10)
    stats = ee.Dictionary(dict(sums=sums, fill=fill_mean)).getInfo()

    # test cloud score dependent bands are fully masked, and fill mask is fully valid
    assert all([sum == 0 for sum in stats['sums'].values()])
    assert stats['fill']['FILL_MASK'] == pytest.approx(1, abs=0.01)


def test_s2_image_get_mask_bands_deprecated(s2_sr_hm_ee_image: ee.Image, region_100ha: dict):
    """Test _Sentinel2Image._get_mask_bands() with the 'qa' and 'cloud-prob' methods."""
    # find mask bands for each method
    mask_bands_list = [
        _Sentinel2SrImage._get_mask_bands(s2_sr_hm_ee_image, mask_method=mm)
        for mm in ['qa', 'cloud-prob']
    ]

    # find means of mask bands over region_100ha, combining all getInfo() calls into one
    means = [
        ee.Image(list(mb.values())).reduceRegion('mean', geometry=region_100ha, scale=10)
        for mb in mask_bands_list
    ]
    means = ee.List(means).getInfo()

    # test mean values / ranges
    for mean in means:
        assert 0 < mean['CLOUDLESS_MASK'] < 1
        assert mean['FILL_MASK'] == pytest.approx(1, abs=0.01)
        assert 0 < mean['CLOUD_DIST'] < 5000

    assert 0 < means[1]['CLOUD_PROB'] < 100
