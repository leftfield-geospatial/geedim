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
from geedim.mask import (
    MaskedImage,
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


# OLD TESTS
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
    assert stats['CDIST_MIN'] == ee_image.select(0).projection().nominalScale().getInfo()


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
