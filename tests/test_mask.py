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

from functools import partial
from itertools import product

import ee
import pytest

from geedim import mask
from geedim.enums import CloudMaskMethod


@pytest.fixture(scope='session')
def constant_ee_image() -> ee.Image:
    """Constant image with a known mask."""
    # half of the last band is masked
    crs = 'EPSG:3857'
    image_bounds = ee.Geometry.Rectangle((-1.0, -1.0, 1.0, 1.0), proj=crs)
    mask_bounds = ee.Geometry.Rectangle((0.0, -1.0, 1.0, 1.0), proj=crs)
    image = ee.Image([1, 2, 3]).rename(['B1', 'B2', 'B3']).toUint8()
    image = image.setDefaultProjection(crs, scale=0.1)
    b3 = image.select('B3')
    b3 = b3.updateMask(b3.mask().multiply(0).paint(mask_bounds, 1))
    return image.addBands(b3, overwrite=True).clip(image_bounds)


@pytest.fixture(scope='session')
def fill_ee_image(constant_ee_image: ee.Image) -> ee.Image:
    """Mock non cloud/shadow image with a known FILL_MASK band."""
    # FILL_MASK covers half the image
    fill_mask = constant_ee_image.mask().reduce(ee.Reducer.allNonZero()).rename('FILL_MASK')
    return constant_ee_image.addBands(fill_mask)


@pytest.fixture(scope='session')
def cloudless_ee_image(fill_ee_image: ee.Image) -> ee.Image:
    """Mock cloud/shadow image with known FILL_MASK and CLOUDLESS_MASK bands."""
    # CLOUDLESS_MASK covers half of FILL_MASK (i.e. CLOUDLESS_MASK and FILL_MASK image portions
    # are 0.25 and 0.5 resp.).
    cloudless_bounds = ee.Geometry.Rectangle((0.5, -1.0, 1.0, 1.0), proj='EPSG:3857')
    cloudless_mask = fill_ee_image.select('FILL_MASK').rename('CLOUDLESS_MASK')
    cloudless_mask = cloudless_mask.multiply(0).paint(cloudless_bounds, 1)
    return fill_ee_image.addBands(cloudless_mask)


@pytest.fixture(scope='session')
def mock_ls_toa_raw_ee_image() -> ee.Image:
    """Mock Landsat TOA reflectance / at sensor radiance image with known mask, QA_PIXEL
    and QA_RADSAT bands.
    """
    # The image mask and QA_PIXEL give cloud, shadow and non-filled portions of 0.4, 0.3,
    # 0.2 & 0.1 resp..  The cloud portion is divided into dilated cloud, cloud and cirrus
    # portions of 0.1 each.  QA_RADSAT gives a saturated portion of 0.1.
    crs = 'EPSG:3857'
    # TODO: if proj= uses an ee.Projection() rather than a CRS, coordinates can be given in
    #  pixels, which may be clearer while allowing a bigger scale to be used
    image_bounds = ee.Geometry.Rectangle((-1.0, -1.0, 1.0, 1.0), proj=crs)
    # initial cloudless image (no mask, and QA_PIXEL all valid)
    image = (
        ee.Image([10000, 20000, 30000, 40000, 0, 0])
        .rename(['B1', 'B2', 'B3', 'B4', 'QA_PIXEL', 'QA_RADSAT'])
        .toUint16()
    )
    image = image.setDefaultProjection(crs, scale=0.1)

    # EE mask:
    # set masked portion (divided equally between SR_B1 and SR_B3 EE masks)
    b1_mask_bounds = ee.Geometry.Rectangle((-1.0, -1.0, -0.9, 1.0), proj=crs)
    b1 = image.select('B1')
    b1 = b1.updateMask(b1.mask().paint(b1_mask_bounds, 0))
    b3_mask_bounds = ee.Geometry.Rectangle((-0.9, -1.0, -0.8, 1.0), proj=crs)
    b3 = image.select('B3')
    b3 = b3.updateMask(b3.mask().paint(b3_mask_bounds, 0))

    # QA_PIXEL:
    # set non-cirrus cloud portion (divided equally cloud and dilated cloud, with cloud divided
    # equally between medium confidence and high confidence cloud)
    qa_pixel = image.select('QA_PIXEL')
    mid_cloud_bounds = ee.Geometry.Rectangle((0.0, -1.0, 0.1, 1.0), proj=crs)
    high_cloud_bounds = ee.Geometry.Rectangle((0.1, -1.0, 0.2, 1.0), proj=crs)
    dilated_cloud_bounds = ee.Geometry.Rectangle((0.2, -1.0, 0.4, 1.0), proj=crs)
    qa_pixel = qa_pixel.paint(mid_cloud_bounds, 1 << 9 | 1 << 3)
    qa_pixel = qa_pixel.paint(high_cloud_bounds, 3 << 8 | 1 << 3)
    qa_pixel = qa_pixel.paint(dilated_cloud_bounds, 1 << 1)

    # set shadow portion
    shadow_bounds = ee.Geometry.Rectangle((0.4, -1.0, 0.8, 1.0), proj=crs)
    qa_pixel = qa_pixel.paint(shadow_bounds, 3 << 10 | 1 << 4)

    # set cirrus portion
    cirrus_bounds = ee.Geometry.Rectangle((0.8, -1.0, 1.0, 1.0), proj=crs)
    qa_pixel = qa_pixel.paint(cirrus_bounds, 3 << 14 | 1 << 2)

    # set saturated portion in QA_RADSAT
    qa_radsat = image.select('QA_RADSAT')
    sat_bounds = ee.Geometry.Rectangle((-0.8, -1.0, -0.6, 1.0), proj=crs)
    qa_radsat = qa_radsat.paint(sat_bounds, 1)

    image = image.addBands([b1, b3, qa_pixel, qa_radsat], overwrite=True)
    return image.clipToBoundsAndScale(image_bounds)


@pytest.fixture(scope='session')
def mock_ls_sr_ee_image(mock_ls_toa_raw_ee_image: ee.Image) -> ee.Image:
    """Mock Landsat surface reflectance image with known nonphysical reflectance portion"""
    crs = 'EPSG:3857'
    image = mock_ls_toa_raw_ee_image.rename(
        ['SR_B1', 'SR_B2', 'SR_B3', 'ST_B4', 'QA_PIXEL', 'QA_RADSAT']
    )

    # set a 0.1 nonphysical portion split equally between SR_B1 and SR_B3
    b1 = image.select('SR_B1')
    nonphys_bounds1 = ee.Geometry.Rectangle((-0.6, -1.0, -0.5, 1.0), proj=crs)
    b1 = b1.paint(nonphys_bounds1, 1000)
    nonphys_bounds2 = ee.Geometry.Rectangle((-0.5, -1.0, -0.4, 1.0), proj=crs)
    b3 = image.select('SR_B3')
    b3 = b3.paint(nonphys_bounds2, 50000)

    return image.addBands([b1, b3], overwrite=True).clipToBoundsAndScale(image.geometry())


@pytest.fixture(scope='session')
def mock_ls_sr_aerosol_ee_image(mock_ls_sr_ee_image: ee.Image) -> ee.Image:
    """Mock Landsat surface reflectance image with known SR_QA_AEROSOL band."""
    # create a mock SR_QA_AEROSOL band with a 0.1 portion of high aerosol
    proj = mock_ls_sr_ee_image.projection()
    sr_qa_aerosol = ee.Image(0).toUint8().setDefaultProjection(proj).rename('SR_QA_AEROSOL')
    aerosol_bounds = ee.Geometry.Rectangle((-0.4, -1.0, -0.2, 1.0), proj=proj.crs())
    sr_qa_aerosol = sr_qa_aerosol.paint(aerosol_bounds, 3 << 6 | 1 << 1)
    image = mock_ls_sr_ee_image.addBands([sr_qa_aerosol])
    return image.clipToBoundsAndScale(mock_ls_sr_ee_image.geometry())


@pytest.fixture(scope='session')
def mock_s2_ee_image() -> ee.Image:
    """Mock Sentinel-2 image with known mask and nonphysical reflectance portion."""
    # set masked portion of 0.1 in the B3 band
    crs = 'EPSG:3857'
    image_bounds = ee.Geometry.Rectangle((-1.0, -1.0, 1.0, 1.0), proj=crs)
    image = ee.Image([1, 2, 3, 0]).rename(['B1', 'B2', 'B3', 'QA60'])
    image = image.toUint16().setDefaultProjection(crs, scale=0.1)
    image = image.set('system:index', 'mock')

    mask_bounds = ee.Geometry.Rectangle((-1.0, -1.0, -0.8, 1.0), proj=crs)
    b3 = image.select('B3')
    b3 = b3.updateMask(b3.mask().paint(mask_bounds, 0))

    # set nonphysical portion of 0.1 in the B3 band
    nonphys_bounds = ee.Geometry.Rectangle((-0.8, -1.0, -0.6, 1.0), proj=crs)
    b3 = b3.paint(nonphys_bounds, 11000)

    return image.addBands([b3], overwrite=True).clip(image_bounds)


@pytest.fixture(scope='session')
def mock_s2_cloud_score_ee_image(mock_s2_ee_image: ee.Image) -> ee.Image:
    """Mock Sentinel-2 Cloud Score+ image matching 'mock_s2_ee_image', with known scores."""
    # initial cloudless scores image
    proj = mock_s2_ee_image.projection()
    image = ee.Image([1.0, 1.0]).rename(['cs', 'cs_cdf'])
    image = image.setDefaultProjection(proj)
    # set system:index to match mock_s2_ee_image
    image = image.set('system:index', mock_s2_ee_image.get('system:index'))
    cs = image.select('cs')
    cs_cdf = image.select('cs_cdf')

    # set a 0.3 portion of scores to cs=0.7 and cs_cdf=0.5
    cloud_bounds = ee.Geometry.Rectangle((0.0, -1.0, 0.6, 1.0), proj=proj.crs())
    cs = cs.paint(cloud_bounds, 0.7)
    cs_cdf = cs_cdf.paint(cloud_bounds, 0.5)

    # set a 0.2 portion of scores to cs=0.5 and cs_cdf=0.7
    cloud_bounds = ee.Geometry.Rectangle((0.6, -1.0, 1.0, 1.0), proj=proj.crs())
    cs = cs.paint(cloud_bounds, 0.5)
    cs_cdf = cs_cdf.paint(cloud_bounds, 0.7)

    # possible EE issue with masking the image as mock_s2_ee_image is masked, so it is left
    # unmasked
    return image.addBands([cs, cs_cdf], overwrite=True).clip(mock_s2_ee_image.geometry())


def test_get_class_for_id(
    ls_raw_image_ids: list[str],
    ls_toa_image_ids: list[str],
    ls_sr_image_ids: list[str],
    l9_sr_image_id: str,
    l8_sr_image_id: str,
    s2_sr_image_id: str,
    s2_sr_hm_image_id: str,
    s2_toa_image_id: str,
    s2_toa_hm_image_id: str,
    modis_nbar_image_id: str,
    landsat_ndvi_image_id: str,
):
    """Test _get_class_for_id()."""
    for im_id in [*ls_toa_image_ids, *ls_raw_image_ids]:
        assert mask._get_class_for_id(im_id) is mask._LandsatToaRawImage
    for im_id in ls_sr_image_ids:
        assert issubclass(mask._get_class_for_id(im_id), mask._LandsatSrImage)
    for im_id in [l9_sr_image_id, l8_sr_image_id]:
        assert mask._get_class_for_id(im_id) is mask._LandsatSrAerosolImage
    for im_id in [s2_sr_image_id, s2_sr_hm_image_id]:
        assert mask._get_class_for_id(im_id) is mask._Sentinel2SrImage
    for im_id in [s2_toa_image_id, s2_toa_hm_image_id]:
        assert mask._get_class_for_id(im_id) is mask._Sentinel2ToaImage
    for im_id in [modis_nbar_image_id, landsat_ndvi_image_id, None, '']:
        assert mask._get_class_for_id(im_id) is mask._MaskedImage


def test_masked_image_get_mask_bands(constant_ee_image: ee.Image):
    """Test _MaskedImage._get_mask_bands()."""
    mask_bands = mask._MaskedImage._get_mask_bands(constant_ee_image)
    assert list(mask_bands.keys()) == ['fill']

    # test fill mask naming and coverage
    fill_portion = mask_bands['fill'].reduceRegion('mean')
    fill_portion = fill_portion.getInfo()
    assert 'FILL_MASK' in fill_portion
    assert fill_portion['FILL_MASK'] == 0.5


def test_masked_image_add_mask_bands(constant_ee_image: ee.Image):
    """Test _MaskedImage.add_mask_bands()."""
    masked_image = mask._MaskedImage.add_mask_bands(constant_ee_image)
    # zero the mask of the first band, and re-add mask band (should overwrite)
    band = masked_image.select(0).updateMask(0)
    remasked_image = masked_image.addBands(band, overwrite=True)
    remasked_image = mask._MaskedImage.add_mask_bands(remasked_image)

    # find FILL_MASK portion in masked_image & remasked_image
    fill_portions = [
        im.select('FILL_MASK').reduceRegion('mean') for im in [masked_image, remasked_image]
    ]

    # image to test mask bands are not added to a non-fixed projection image
    comp_image = mask._MaskedImage.add_mask_bands(ee.Image([1, 2, 3]))

    # band names of all test images
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
    assert info['fill_portions'][0]['FILL_MASK'] == 0.5
    assert info['fill_portions'][1]['FILL_MASK'] == 0.0


def test_masked_image_mask_clouds(fill_ee_image: ee.Image):
    """Test _MaskedImage.mask_clouds() returns the source image unaltered."""
    image = mask._MaskedImage.mask_clouds(fill_ee_image)
    assert image == fill_ee_image


def test_masked_image_set_mask_portions(fill_ee_image: ee.Image):
    """Test _MaskedImage.set_mask_portions() with different parameters."""
    images = {}
    images['default'] = mask._MaskedImage.set_mask_portions(fill_ee_image)
    images['region'] = mask._MaskedImage.set_mask_portions(
        fill_ee_image.translate(0, 1), region=fill_ee_image.geometry()
    )
    images['scale'] = mask._MaskedImage.set_mask_portions(fill_ee_image, scale=0.11)

    # combine all getInfo() calls into one
    portions = {k: v.toDictionary(['FILL_PORTION', 'CLOUDLESS_PORTION']) for k, v in images.items()}
    portions = ee.Dictionary(portions).getInfo()

    # test known / exact portions for the default and region parameter cases
    assert portions['default']['FILL_PORTION'] == 50
    assert portions['default']['CLOUDLESS_PORTION'] == 100

    assert portions['region']['FILL_PORTION'] == 25
    assert portions['region']['CLOUDLESS_PORTION'] == 100

    # test scale param affects portion accuracy
    assert portions['scale']['FILL_PORTION'] == pytest.approx(50, abs=10)
    assert abs(portions['scale']['FILL_PORTION'] - 50) > abs(
        portions['default']['FILL_PORTION'] - 50
    )


def test_cloudless_image_get_cloud_dist():
    """Test _CloudlessImage._get_cloud_dist()."""
    # Create test cloudless mask. (Bottom left pixel is cloud & max distance to it is 50m. Image
    # scale needs to be integer.)
    crs = 'EPSG:3857'
    cloudless_mask = ee.Image(1).setDefaultProjection(crs, scale=1).toUint8()
    image_bounds = ee.Geometry.Rectangle((0, 0, 41, 31), proj=crs)
    bl_pixel_bounds = ee.Geometry.Rectangle((0, 0, 1, 1), proj=crs)
    cloudless_mask = cloudless_mask.paint(bl_pixel_bounds, 0).clip(image_bounds)

    # find min & max of cloud distance for different max_cloud_dist vals, combining all getInfo()
    # calls into one
    max_cloud_dists = [100, 10]
    min_maxs = [
        mask._Sentinel2Image._get_cloud_dist(
            cloudless_mask, proj=cloudless_mask.projection(), max_cloud_dist=max_cloud_dist
        ).reduceRegion(ee.Reducer.minMax())
        for max_cloud_dist in max_cloud_dists
    ]
    min_maxs = ee.List(min_maxs).getInfo()

    # test first case where max distance < max_cloud_dist
    assert min_maxs[0]['CLOUD_DIST_min'] == 0
    assert min_maxs[0]['CLOUD_DIST_max'] == 50

    # test second case where max distance is clamped to max_cloud_dist
    assert min_maxs[1]['CLOUD_DIST_min'] == 0
    assert min_maxs[1]['CLOUD_DIST_max'] == max_cloud_dists[1]


def test_cloudless_image_mask_clouds(cloudless_ee_image: ee.Image):
    """Test _CloudlessImage.mask_clouds()."""
    image = mask._CloudlessImage.mask_clouds(cloudless_ee_image)
    coverages = image.mask().reduceRegion('mean')
    coverages = coverages.getInfo()
    assert len(coverages) == 5
    assert all([v == 0.25 for v in coverages.values()])


def test_cloudless_image_set_mask_portions(cloudless_ee_image: ee.Image):
    """Test _CloudlessImage.set_mask_portions() with different parameters."""
    images = {}
    images['default'] = mask._CloudlessImage.set_mask_portions(cloudless_ee_image)
    images['region'] = mask._CloudlessImage.set_mask_portions(
        cloudless_ee_image.translate(0, 1), region=cloudless_ee_image.geometry()
    )
    images['scale'] = mask._CloudlessImage.set_mask_portions(cloudless_ee_image, scale=0.11)

    # combine all getInfo() calls into one
    portions = {k: v.toDictionary(['FILL_PORTION', 'CLOUDLESS_PORTION']) for k, v in images.items()}
    portions = ee.Dictionary(portions).getInfo()

    # test known / exact portions for the default and region parameter cases
    assert portions['default']['FILL_PORTION'] == 50
    assert portions['default']['CLOUDLESS_PORTION'] == 50

    assert portions['region']['FILL_PORTION'] == 25
    assert portions['region']['CLOUDLESS_PORTION'] == 50

    # test scale param affects portion accuracy
    for key, val in zip(['FILL_PORTION', 'CLOUDLESS_PORTION'], [50, 50], strict=False):
        assert portions['scale'][key] == pytest.approx(val, abs=10)
        assert abs(portions['scale'][key] - val) > abs(portions['default'][key] - val)


def test_landsat_toa_raw_image_get_mask_bands_support(
    ls_raw_image_ids: list[str],
    ls_toa_image_ids: list[str],
    ls_sr_image_ids: list[str],
    region_10000ha: dict,
):
    """Test _LandsatToaRawImage._get_mask_bands() works on all supported Landsat collections."""
    # find mask bands for each landsat image
    im_ids = [*ls_sr_image_ids, *ls_toa_image_ids, *ls_raw_image_ids]
    mask_bands_list = [
        mask._LandsatToaRawImage._get_mask_bands(
            ee.Image(im_id), mask_shadows=True, mask_cirrus=True, mask_saturation=True
        )
        for im_id in im_ids
    ]
    exp_keys = {'shadow', 'cloud', 'saturation', 'cloudless', 'fill', 'dist'}
    for mb in mask_bands_list:
        assert set(mb.keys()) == exp_keys

    # find means of mask bands over region_10000ha, combining all getInfo() calls into one
    means = [
        ee.Image(list(mb.values())).reduceRegion('mean', geometry=region_10000ha, scale=30)
        for mb in mask_bands_list
    ]
    means = ee.List(means).getInfo()

    # test mean values / ranges
    for mean in means:
        assert all([0 < mean[bn] < 1 for bn in ['CLOUD_MASK', 'SHADOW_MASK', 'CLOUDLESS_MASK']])
        assert 0.5 < mean['FILL_MASK'] <= 1  # landsat7 has scanline error
        assert 0 < mean['CLOUD_DIST'] < 5000
        assert 0 <= mean['SATURATION_MASK'] < 1


def test_landsat_sr_image_get_mask_bands_support(ls_sr_image_ids: list[str], region_10000ha: dict):
    """Test _LandsatSrImage._get_mask_bands() nonphysical masking works on supported Landsat
    collections.
    """
    # find mask bands for each landsat image
    nonphysical_masks = [
        mask._LandsatSrImage._get_mask_bands(ee.Image(im_id), mask_nonphysical=True)['nonphysical']
        for im_id in ls_sr_image_ids
    ]

    # find & test means of aerosol masks over region_10000ha, combining all getInfo() calls into one
    means = [am.reduceRegion('mean', geometry=region_10000ha, scale=30) for am in nonphysical_masks]
    means = ee.List(means).getInfo()
    assert all(0 <= mean['NONPHYSICAL_MASK'] < 1 for mean in means)


def test_landsat_sr_aerosol_image_get_mask_bands_support(
    l9_sr_image_id: str, l8_sr_image_id: str, region_10000ha: dict
):
    """Test _LandsatSrAerosolImage._get_mask_bands() aerosol masking works on supported Landsat
    collections.
    """
    # find mask bands for each landsat image
    aerosol_masks = [
        mask._LandsatSrAerosolImage._get_mask_bands(ee.Image(im_id), mask_aerosols=True)['aerosol']
        for im_id in [l9_sr_image_id, l8_sr_image_id]
    ]

    # find & test means of aerosol masks over region_10000ha, combining all getInfo() calls into one
    means = [am.reduceRegion('mean', geometry=region_10000ha, scale=30) for am in aerosol_masks]
    means = ee.List(means).getInfo()
    assert all(0 < mean['AEROSOL_MASK'] < 1 for mean in means)


def test_landsat_toa_raw_image_get_mask_bands(mock_ls_toa_raw_ee_image: ee.Image):
    """Test _LandsatToaRawImage._get_mask_bands() masking and parameters."""
    get_mask_bands = partial(
        mask._LandsatToaRawImage._get_mask_bands, mock_ls_toa_raw_ee_image, max_cloud_dist=2
    )
    # create test stats for each parameter:
    # reference
    stats = {}
    mask_bands = get_mask_bands(mask_shadows=True, mask_cirrus=True)
    mask_image = ee.Image(list(mask_bands.values()))
    stats['ref'] = mask_image.reduceRegion('mean')

    # mask_shadows
    mask_bands = get_mask_bands(mask_shadows=False)
    assert 'shadow' not in mask_bands
    stats['mask_shadows'] = mask_bands['cloudless'].reduceRegion('mean')

    # mask_cirrus
    mask_bands = get_mask_bands(mask_cirrus=False)
    stats['mask_cirrus'] = mask_bands['cloud'].reduceRegion('mean')

    # mask_saturation
    mask_bands = get_mask_bands(mask_saturation=True)
    mask_image = ee.Image([mask_bands[k] for k in ['saturation', 'cloudless']])
    stats['mask_saturation'] = mask_image.reduceRegion('mean')

    # fetch stats, combining all getInfo() calls into one
    stats = ee.Dictionary(stats).getInfo()

    # test masking against known portions in the reference image
    assert stats['ref']['FILL_MASK'] == 0.9
    assert stats['ref']['CLOUDLESS_MASK'] == 0.4
    assert stats['ref']['CLOUD_MASK'] == 0.3
    assert stats['ref']['SHADOW_MASK'] == 0.2

    # test mask_shadows
    assert stats['mask_shadows']['CLOUDLESS_MASK'] == 0.6

    # test mask_cirrus
    assert stats['mask_cirrus']['CLOUD_MASK'] == 0.2

    # test mask_saturation
    assert stats['mask_saturation']['SATURATION_MASK'] == 0.1
    assert stats['mask_saturation']['CLOUDLESS_MASK'] == 0.3


def test_landsat_sr_image_get_mask_bands(mock_ls_sr_ee_image: ee.Image):
    """Test _LandsatSrImage._get_mask_bands() nonphysical masking."""
    mask_bands = mask._LandsatSrAerosolImage._get_mask_bands(
        mock_ls_sr_ee_image,
        mask_shadows=True,
        mask_cirrus=True,
        mask_saturation=True,
        mask_nonphysical=True,
        max_cloud_dist=2,
    )
    mask_image = ee.Image([mask_bands[k] for k in ['nonphysical', 'cloudless']])
    stats = mask_image.reduceRegion('mean', geometry=mock_ls_sr_ee_image.geometry()).getInfo()

    assert stats['NONPHYSICAL_MASK'] == 0.1
    assert stats['CLOUDLESS_MASK'] == 0.2


def test_landsat_sr_aerosol_image_get_mask_bands(mock_ls_sr_aerosol_ee_image: ee.Image):
    """Test _LandsatSrAerosolImage._get_mask_bands() aerosol masking."""
    mask_bands = mask._LandsatSrAerosolImage._get_mask_bands(
        mock_ls_sr_aerosol_ee_image,
        mask_shadows=True,
        mask_cirrus=True,
        mask_saturation=True,
        mask_nonphysical=True,
        mask_aerosols=True,
        max_cloud_dist=2,
    )
    mask_image = ee.Image([mask_bands[k] for k in ['aerosol', 'cloudless']])
    stats = mask_image.reduceRegion('mean').getInfo()

    assert stats['AEROSOL_MASK'] == 0.1
    assert stats['CLOUDLESS_MASK'] == 0.1


def test_s2_image_get_mask_bands_support(s2_image_ids: list[str], region_100ha: dict):
    """Test _Sentinel2Image._get_mask_bands() works on all supported Sentinel-2 collections with
    all mask methods.
    """
    # find mask bands for each s2 image & method
    mask_bands_list = [
        mask._get_class_for_id(im_id)._get_mask_bands(ee.Image(im_id), mask_method=meth)
        for im_id, meth in product(s2_image_ids, CloudMaskMethod)
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


def test_s2_image_get_mask_bands(
    mock_s2_ee_image: ee.Image,
    mock_s2_cloud_score_ee_image: ee.Image,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test _Sentinel2Image._get_mask_bands() masking and parameters with the 'cloud-score'
    method.
    """
    get_mask_bands = partial(
        mask._Sentinel2Image._get_mask_bands,
        mock_s2_ee_image,
        mask_method='cloud-score',
        max_cloud_dist=2,
    )
    # patch mask._Sentinel2Image so that it searches for a matching cloud score image in a
    # collection containing mock_s2_cloud_score_ee_image
    monkeypatch.setattr(
        mask._Sentinel2Image, '_cloud_score_coll_id', [mock_s2_cloud_score_ee_image]
    )

    # create test stats for each parameter:
    # reference
    stats = {}
    mask_bands = get_mask_bands(score=0.6, cs_band='cs')
    mask_image = ee.Image(list(mask_bands.values()))
    stats['ref'] = mask_image.reduceRegion('mean')

    # score
    mask_bands = get_mask_bands(score=0.8, cs_band='cs')
    stats['score'] = mask_bands['cloudless'].reduceRegion('mean')

    # cs_band
    mask_bands = get_mask_bands(score=0.6, cs_band='cs_cdf')
    stats['cs_band'] = mask_bands['cloudless'].reduceRegion('mean')

    # mask_nonphysical
    mask_bands = get_mask_bands(score=0.6, cs_band='cs', mask_nonphysical=True)
    mask_image = ee.Image([mask_bands[k] for k in ['cloudless', 'nonphysical']])
    stats['mask_nonphysical'] = mask_image.reduceRegion(
        'mean', geometry=mock_s2_ee_image.geometry()
    )

    # fetch stats, combining all getInfo() calls into one
    stats = ee.Dictionary(stats).getInfo()

    # test masking against known portions (see the mock_s2_cloud_score_ee_image() definition to
    # understand the values)
    assert stats['ref']['FILL_MASK'] == 0.9
    assert stats['ref']['CLOUDLESS_MASK'] == 0.7
    assert 0 < stats['ref']['CLOUD_SCORE'] < 1

    # test score
    assert stats['score']['CLOUDLESS_MASK'] == 0.4

    # test cs_band
    assert stats['cs_band']['CLOUDLESS_MASK'] == 0.6

    # test mask_nonphysical
    assert stats['mask_nonphysical']['CLOUDLESS_MASK'] == 0.6
    assert stats['mask_nonphysical']['NONPHYSICAL_MASK'] == 0.1


def test_s2_image_get_mask_bands_no_cloud_score(mock_s2_ee_image: ee.Image):
    """Test _Sentinel2Image._get_mask_bands() fully masks cloud score dependent bands when no
    cloud score image exists.
    """
    # find region sums of cloud score dependent bands, and region mean of fill mask, combining
    # all getInfo() calls into one (sums rather than means are used to avoid division by zero)
    reduce_kwargs = dict(
        geometry=mock_s2_ee_image.geometry(), scale=mock_s2_ee_image.projection().nominalScale()
    )
    mask_bands = mask._Sentinel2Image._get_mask_bands(
        mock_s2_ee_image, mask_method='cloud-score', max_cloud_dist=2
    )
    sum_image = ee.Image([mask_bands[k] for k in ['cloudless', 'score', 'dist']])
    sums = sum_image.reduceRegion('sum', **reduce_kwargs)
    fill_mean = mask_bands['fill'].reduceRegion('mean', **reduce_kwargs)
    stats = ee.Dictionary(dict(sums=sums, fill=fill_mean)).getInfo()

    # test cloud score dependent bands are fully masked, and fill mask is as defined in
    # mock_s2_ee_image()
    assert all([sum == 0 for sum in stats['sums'].values()])
    assert stats['fill']['FILL_MASK'] == 0.9


def test_masked_image_init_from_id(l9_sr_image_id: str, region_100ha: dict):
    """Test MaskedImage.__init__() and MaskedImage.from_id() without and with parameters."""
    # create images and get their info dicts
    kwargs_list = [dict(), dict(region=region_100ha, mask_shadows=False)]
    init_images = [
        mask.MaskedImage(ee.Image(l9_sr_image_id), **kwargs)._ee_image for kwargs in kwargs_list
    ]
    from_id_images = [
        mask.MaskedImage.from_id(l9_sr_image_id, **kwargs)._ee_image for kwargs in kwargs_list
    ]
    infos_list = ee.List([ee.List(init_images), ee.List(from_id_images)]).getInfo()

    for infos in infos_list:
        for info, kwargs in zip(infos, kwargs_list, strict=False):
            # test mask bands exist
            band_names = {b['id'] for b in info['bands']}
            assert set(band_names).issuperset(
                {'CLOUD_MASK', 'CLOUDLESS_MASK', 'FILL_MASK', 'CLOUD_DIST'}
            )
            # test the effect of the mask_shadows parameter
            assert (
                'SHADOW_MASK' in band_names
                if kwargs.get('mask_shadows', True)
                else 'SHADOW_MASK' not in band_names
            )

        # test passing region set the fill and cloudless portions in the second image
        props = infos[1]['properties']
        assert 'FILL_PORTION' in props and 'CLOUDLESS_PORTION' in props


def test_masked_image_mask_clouds_(l9_sr_image_id: str, region_100ha: dict):
    """Test MaskedImage.mask_clouds()."""
    image = mask.MaskedImage(ee.Image(l9_sr_image_id))
    image.mask_clouds()
    coverages = image._ee_image.mask().reduceRegion('mean', geometry=region_100ha, scale=30)
    coverages = coverages.getInfo()
    assert 'FILL_MASK' in coverages and 'CLOUDLESS_MASK' in coverages
    assert len(set(coverages.values())) == 1  # test all equal
    assert all([0 < v < 100 for v in coverages.values()])
