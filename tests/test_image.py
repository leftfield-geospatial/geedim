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

import json
from datetime import datetime, timezone
from pathlib import Path

import ee
import fsspec
import numpy as np
import pytest
import rasterio as rio
from rasterio.enums import Compression
from rasterio.features import bounds
from rasterio.warp import transform_geom

from geedim import utils
from geedim.enums import ExportType
from geedim.image import ImageAccessor, _nodata_vals, _open_raster

# TODO: probably move these fixtures to conftest to replace *_masked_image


def accessors_from_images(ee_images: list[ee.Image]) -> list[ImageAccessor]:
    """Return a list of ImageAccessor objects, with cached info properties, for the given list of
    ee.Image objects, combining all getInfo() calls into one.
    """

    def band_scale(band_name: ee.ComputedObject, ee_image: ee.Image):
        """Return scale in meters for band_name."""
        return ee_image.select(ee.String(band_name)).projection().nominalScale()

    infos = []
    for ee_image in ee_images:
        scales = ee_image.bandNames().map(lambda bn: band_scale(bn, ee_image))
        infos.append(ee.List([scales, ee_image]))

    infos = ee.List(infos).getInfo()
    images = []
    for ee_image, (scales, info) in zip(ee_images, infos):
        for scale, bdict in zip(scales, info.get('bands', [])):
            bdict['scale'] = scale
        image = ImageAccessor(ee_image)
        image.info = info
        images.append(image)

    return images


def transform_bounds(geometry: dict, crs: str = 'EPSG:4326') -> tuple[float, ...]:
    """Return the bounds of the given GeoJSON geometry in crs coordinates."""
    src_crs = geometry['crs']['properties']['name'] if 'crs' in geometry else 'EPSG:4326'
    geometry = geometry if crs == src_crs else transform_geom(src_crs, crs, geometry)
    return bounds(geometry)


def test_ee_props(l9_sr_image: ImageAccessor, modis_nbar_image: ImageAccessor):
    """Test all properties excluding dtype, stac, stac-related properties & cloudShadowSupport."""
    images = [l9_sr_image, modis_nbar_image]
    # combine getInfo() calls into one
    infos = ee.List([ee.Image(l9_sr_image.id), ee.Image(modis_nbar_image.id)]).getInfo()

    for image, info in zip(images, infos):
        props = info['properties']
        bands = info['bands']

        assert image.id == info['id']
        assert image.index == info['id'].split('/')[-1]
        assert image.date == datetime.fromtimestamp(
            props['system:time_start'] / 1000, tz=timezone.utc
        )
        assert image.properties == props

        assert image.crs == bands[0]['crs']
        assert image.transform == tuple(bands[0]['crs_transform'])
        assert image.shape == tuple(bands[0]['dimensions'][::-1])
        assert image.count == len(bands)
        assert image.nodata == _nodata_vals[image.dtype]
        assert image.size == np.dtype(image.dtype).itemsize * np.prod(image.shape) * image.count
        assert image.profile is not None

        assert image.scale == pytest.approx(
            np.sqrt(abs(image.transform[0]) * abs(image.transform[4]))
        )
        assert image.geometry == ee.Geometry(props['system:footprint']).toGeoJSON()
        assert image.bandNames == [bi['id'] for bi in bands]


@pytest.mark.parametrize(
    'ee_dtypes, exp_dtype',
    [
        ([('int', 10, 11), ('int', 100, 101)], 'uint8'),
        ([('int', -128, -100), ('int', 0, 127)], 'int8'),
        ([('int', 256, 257)], 'uint16'),
        ([('int', -32768, 32767)], 'int16'),
        ([('int', 2**15, 2**32 - 1)], 'uint32'),
        ([('int', -(2**31), 2**31 - 1)], 'int32'),
        ([('float', 0.0, 1.0e9), ('float', 0.0, 1.0)], 'float32'),
        ([('int', 0.0, 2**31 - 1), ('float', 0.0, 1.0)], 'float64'),
        ([('int', 0, 255), ('double', -1e100, 1e100)], 'float64'),
    ],
)
def test_dtype(ee_dtypes: list, exp_dtype: str):
    """Test the dtype property with different band combinations."""
    # patch image.info with a mock EE info dict that simulates different band data types
    image = ImageAccessor(ee.Image())
    ee_dtypes = [dict(zip(['precision', 'min', 'max'], dt)) for dt in ee_dtypes]
    image.info = dict(bands=[dict(data_type=dt) for dt in ee_dtypes])

    assert image.dtype == exp_dtype


def test_profile(modis_nbar_image: ImageAccessor):
    """Test the profile property."""
    # tests the workaround for the MODIS CRS bug
    assert modis_nbar_image.profile['crs'] == utils.rio_crs(modis_nbar_image.crs)
    assert modis_nbar_image.profile['width'] == modis_nbar_image.shape[1]
    assert modis_nbar_image.profile['height'] == modis_nbar_image.shape[0]
    for attr in ['transform', 'count', 'dtype']:
        assert modis_nbar_image.profile[attr] == getattr(modis_nbar_image, attr)


@pytest.mark.parametrize(
    'image, exp_scale',
    [
        ('l9_sr_image', 30),
        ('s2_sr_hm_image', 10),
        ('landsat_ndvi_image', 111319.5),  # composite with 1deg scale
    ],
)
def test_scale(image: str, exp_scale: float, request: pytest.FixtureRequest):
    """Test the scale property matches the minimum scale in meters."""
    image: ImageAccessor = request.getfixturevalue(image)
    assert image.scale == pytest.approx(exp_scale, abs=0.1)


@pytest.mark.parametrize(
    'image, exp_bounded',
    [
        ('l9_sr_image', True),
        ('s2_sr_hm_image', True),
        ('modis_nbar_image', False),
        ('landsat_ndvi_image', False),
        ('const_image', False),
    ],
)
def test_bounded(image: str, exp_bounded: bool, request: pytest.FixtureRequest):
    """Test the bounded property."""
    image: ImageAccessor = request.getfixturevalue(image)
    assert image.bounded == exp_bounded


@pytest.mark.parametrize('image', ['landsat_ndvi_image', 'const_image'])
def test_non_fixed_props(image: str, request: pytest.FixtureRequest):
    """Test that composites / images without fixed projections have no geometry, shape,
    size or profile.
    """
    image: ImageAccessor = request.getfixturevalue(image)
    assert image.geometry is None
    assert image.shape is None
    assert image.size is None
    assert image.profile is None


def test_stac_props(s2_sr_hm_image: ImageAccessor):
    """Test the stac and stac-related properties."""
    assert s2_sr_hm_image.stac is not None
    assert len(s2_sr_hm_image.bandProps) == len(s2_sr_hm_image.bandNames)
    assert all(['name' in bp for bp in s2_sr_hm_image.bandProps])
    assert len(s2_sr_hm_image.specBands) > 0
    spec_bands = [bp['name'] for bp in s2_sr_hm_image.bandProps if 'center_wavelength' in bp]
    assert s2_sr_hm_image.specBands == spec_bands


@pytest.mark.parametrize(
    'image, exp_support',
    [('l9_sr_image', True), ('s2_sr_hm_image', True), ('modis_nbar_image', False)],
)
def test_cs_support(image: str, exp_support: bool, request: pytest.FixtureRequest):
    """Test the cloudShadowSupport property."""
    image: ImageAccessor = request.getfixturevalue(image)
    assert image.cloudShadowSupport == exp_support


def test_projection(s2_sr_hm_image: ImageAccessor):
    """Test projection() on an image with different band projections."""
    projs = ee.List(
        [s2_sr_hm_image.projection(min_scale=True), s2_sr_hm_image.projection(min_scale=False)]
    )
    projs = projs.getInfo()
    assert projs[0]['crs'] == s2_sr_hm_image.crs
    assert tuple(projs[0]['transform']) == s2_sr_hm_image.transform
    assert projs[1]['crs'] == 'EPSG:4326'
    assert projs[1]['transform'] == [1, 0, 0, 0, 1, 0]


def test_fixed(
    s2_sr_hm_image: ImageAccessor, landsat_ndvi_image: ImageAccessor, const_image: ImageAccessor
):
    """Test fixed()."""
    # s2_sr_hm_image has a combination of fixed and non-fixed bands, landsat_ndvi_image has
    # one non-fixed band, and const_image has all non-fixed bands
    fixeds = ee.List([s2_sr_hm_image.fixed(), landsat_ndvi_image.fixed(), const_image.fixed()])
    fixeds = fixeds.getInfo()
    assert fixeds == [True, False]


def test_resample(
    l9_sr_image: ImageAccessor, landsat_ndvi_image: ImageAccessor, region_100ha: dict
):
    """Test resample() on fixed and non-fixed projection images with up- and downsampling."""

    def get_reproj_std(image: ee.Image, **reproj_kwargs) -> ee.Number:
        """Return the average standard deviation of the reprojected image."""
        reproj_im = image.reproject(**reproj_kwargs)
        return reproj_im.reduceRegion('stdDev', geometry=region_100ha).values().reduce('mean')

    # find standard deviations before and after resampling for different method and image
    # combinations, combining all getInfo() calls into one
    up_kwargs = dict(crs=l9_sr_image.crs, scale=15)
    down_kwargs = dict(crs=l9_sr_image.crs, scale=60)
    std_tests = []
    std_infos = []
    for src_im, method, reproj_kwargs in zip(
        [l9_sr_image, l9_sr_image, l9_sr_image, landsat_ndvi_image],
        ['bilinear', 'bicubic', 'average', 'average'],
        [up_kwargs, up_kwargs, down_kwargs, down_kwargs],
    ):
        resample_im = src_im.resample(method)
        stds = [get_reproj_std(im, **reproj_kwargs) for im in [src_im._ee_image, resample_im]]
        std_tests.append(ee.List(stds))
        std_infos.append(dict(image=src_im.id, method=method, fixed=src_im.shape is not None))

    std_tests = ee.List(std_tests).getInfo()

    # test standard deviations are reduced by resampling when the image has a fixed projection
    for std_test, std_info in zip(std_tests, std_infos):
        if std_info['fixed']:
            assert std_test[0] > std_test[1]
        else:
            assert std_test[0] == std_test[1]


def test_to_dtype(landsat_ndvi_image):
    """Test toDType()."""
    # convert to all possible dtypes
    dtypes = list(_nodata_vals.keys())
    converted_images = [landsat_ndvi_image.toDType(dtype) for dtype in dtypes]
    # combine getInfo() of converted images into one
    converted_images = accessors_from_images(converted_images)

    # test dtype of converted images
    for converted_image, dtype in zip(converted_images, dtypes):
        # patch image.info with the converted EE info dict
        assert converted_image.dtype == dtype, dtype


def test_scale_offset(
    s2_sr_hm_image: ImageAccessor,
    l9_sr_image: ImageAccessor,
    modis_nbar_image: ImageAccessor,
    region_100ha: dict,
):
    """Test scaleOffset()."""
    # find min / max of reflectance bands after they have been scaled & offset
    src_ims = [s2_sr_hm_image, l9_sr_image, modis_nbar_image]
    scale_offset_ims = [im.scaleOffset() for im in src_ims]
    min_max_tests = []
    min_max_infos = []
    for src_im, scale_offset_im in zip(src_ims, scale_offset_ims):
        # spectral bands without units (i.e. reflectance and not temperature)
        refl_bands = [
            bp['name']
            for bp in src_im.bandProps
            if 'center_wavelength' in bp and 'gee:units' not in bp
        ]
        min_max = scale_offset_im.select(refl_bands).reduceRegion(
            reducer=ee.Reducer.minMax(), geometry=region_100ha, bestEffort=True
        )
        min_max_tests.append(min_max)
        min_max_infos.append(dict(image=src_im.id, refl_bands=refl_bands))

    # combine getInfo() of min / max values & scaled / offset images into one
    results = ee.Dictionary(
        dict(min_max=ee.List(min_max_tests), image=ee.List(scale_offset_ims))
    ).getInfo()

    # test min / max values are ~0-1 reflectance
    for min_max_test, min_max_info in zip(results['min_max'], min_max_infos):
        assert len(min_max_test) == len(2 * min_max_info['refl_bands'])
        for bn in min_max_info['refl_bands']:
            bmin, bmax = min_max_test[bn + '_min'], min_max_test[bn + '_max']
            assert bmin >= -0.5
            assert bmax <= 1.5

    # test scaled and offset images have the same bands and properties as source images
    for src_im, scaled_offset_info in zip(src_ims, results['image']):
        band_names = [bp['id'] for bp in scaled_offset_info['bands']]
        assert band_names == src_im.bandNames
        assert scaled_offset_info['properties'] == src_im.properties


def test_region_coverage(prepared_image: ImageAccessor):
    """Test regionCoverage()."""
    # find & test coverages for regions spanning image and mask bounds, combining all getInfo()'s
    # into one (see prepared_image() to understand regions & exp_coverages)
    mask = ImageAccessor(prepared_image._ee_image.mask())
    regions = [
        ee.Geometry.Rectangle((0.0, 0.0, 2.0, 2.0), proj=prepared_image.crs),
        ee.Geometry.Rectangle((0.0, 0.0, 1.0, 1.0), proj=prepared_image.crs),
        ee.Geometry.Rectangle((-0.5, -0.5, 0.5, 0.5), proj=prepared_image.crs),
    ]
    exp_coverages = [6.25, 25, 100]
    coverages = [
        mask.regionCoverage(region=region, scale=prepared_image.scale) for region in regions
    ]
    coverages = ee.List(coverages).getInfo()

    for coverage, exp_coverage in zip(coverages, exp_coverages):
        for v in coverage.values():
            assert v == pytest.approx(exp_coverage, abs=0.01)


def test_add_mask_bands(l9_sr_image: ImageAccessor):
    """Test addMaskBands()"""
    # This just tests bands exist and kwargs were passed. Detailed mask testing is done in
    # test_mask.py.
    cs_image = l9_sr_image.addMaskBands(mask_shadows=False)
    cs_image = ImageAccessor(cs_image)
    for bn in ['CLOUDLESS_MASK', 'CLOUD_DIST', 'FILL_MASK']:
        assert bn in cs_image.bandNames
    assert 'SHADOW_MASK' not in cs_image.bandNames


def test_mask_clouds(s2_sr_hm_image: ImageAccessor, region_100ha: dict):
    """Test maskClouds()."""
    # This just tests the masked area increases. Detailed mask testing is done in test_mask.py.
    masked_image = ImageAccessor(s2_sr_hm_image.addMaskBands()).maskClouds()
    sums = [
        im.mask().reduceRegion('sum', geometry=region_100ha, scale=30).values().reduce('mean')
        for im in [s2_sr_hm_image._ee_image, masked_image]
    ]
    sums = ee.List(sums).getInfo()
    assert sums[0] > sums[1]


def test_prepare_for_export(s2_sr_hm_image: ImageAccessor, region_100ha: dict):
    """Test prepareForExport()."""
    crs = 'EPSG:3857'
    prep_kwargs_list = [
        dict(crs=crs, region=region_100ha, scale=60),
        dict(crs=crs, region=region_100ha, shape=(300, 400)),
        dict(crs=crs, crs_transform=(60.0, 0.0, 500000.0, 0.0, -30.0, 6400000.0), shape=(600, 400)),
        dict(region=region_100ha),
        dict(crs=crs, region=region_100ha, scale=60, dtype='int16', bands=['B4', 'B3', 'B2']),
        # maintain pixel grid
        dict(region=region_100ha, scale=10),
        dict(region=region_100ha),
        dict(),
    ]
    prep_ims = [s2_sr_hm_image.prepareForExport(**prep_kwargs) for prep_kwargs in prep_kwargs_list]
    prep_ims = accessors_from_images(prep_ims)

    # test prepared image properties
    arg_to_attr_names = dict(
        crs='crs',
        crs_transform='transform',
        shape='shape',
        scale='scale',
        dtype='dtype',
        bands='bandNames',
    )
    for prep_im, prep_kwargs in zip(prep_ims, prep_kwargs_list):
        for arg_name, attr_name in arg_to_attr_names.items():
            if attr_name in prep_kwargs:
                assert getattr(prep_im, attr_name) == prep_kwargs[arg_name]

        # region is a special case that is approximate & needs transformation between CRSs
        if 'region' in prep_kwargs:
            region_bounds = transform_bounds(ee.Geometry(prep_kwargs['region']).toGeoJSON())
            image_bounds = transform_bounds(prep_im.geometry)
            assert image_bounds == pytest.approx(region_bounds, abs=1e-3)

    # test pixel grid is maintained when arguments allow
    src_transform = rio.Affine(*s2_sr_hm_image.transform)
    for prep_im in prep_ims[-3:]:
        prep_transform = rio.Affine(*prep_im.transform)
        assert (prep_transform[0], prep_transform[4]) == (src_transform[0], src_transform[4])
        pixel_offset = ~src_transform * (prep_transform[2], prep_transform[5])
        assert pixel_offset == (int(pixel_offset[0]), int(pixel_offset[1]))


def test_prepare_for_export_errors(
    s2_sr_hm_image: ImageAccessor, landsat_ndvi_image: ImageAccessor
):
    """Test prepareForExport() error conditions."""
    # composite image without spatial params
    with pytest.raises(ValueError) as ex:
        landsat_ndvi_image.prepareForExport()
    assert 'fixed projection' in str(ex.value)

    # scale and shape together
    with pytest.raises(ValueError) as ex:
        s2_sr_hm_image.prepareForExport(scale=10, shape=(400, 300))
    assert 'shape' in str(ex.value) and 'scale' in str(ex.value)


@pytest.mark.parametrize('type', ExportType)
def test_to_google_cloud(s2_sr_hm_image: ImageAccessor, region_25ha: dict, type: ExportType):
    """Test toGoogleCloud() starts the task."""
    # Just test the export starts - completion is tested in integration.py.  Note that the asset
    # and cloud options should start but will ultimately fail (for asset, an existing asset
    # cannot overwritten, and for cloud, there is no 'geedim' bucket).
    prep_im = ImageAccessor(s2_sr_hm_image.prepareForExport(region=region_25ha, scale=30))
    prep_im.toGoogleCloud('test_export', type=type, folder='geedim', wait=False)


def test_open_raster(tmp_path: Path):
    """Test _open_raster() with local and 'remote' files."""
    file = tmp_path.joinpath('test.tif')
    shape = (1, 1, 1)
    array = np.ones(shape, dtype='uint8')
    profile = dict(
        driver='gtiff', width=shape[2], height=shape[1], count=shape[0], dtype=array.dtype
    )
    # simulate a remote file with memory://
    ofiles = [fsspec.open(file, 'wb'), fsspec.open('memory://test.tif', 'wb')]
    try:
        for ofile in ofiles:
            with _open_raster(ofile, **profile) as ds:
                ds.write(array)
            assert ofile.fs.exists(ofile.path)
    finally:
        [ofile.fs.rm(ofile.path) for ofile in ofiles]


@pytest.mark.parametrize(
    'kwargs', [dict(driver='gtiff'), dict(driver='cog', nodata=False), dict(nodata=1)]
)
def test_to_geotiff(prepared_image: ImageAccessor, tmp_path: Path, kwargs: dict):
    """Test toGeoTIFF() with different driver / nodata parameters."""
    file = tmp_path.joinpath('test.tif')
    image = prepared_image
    prepared_image.toGeoTIFF(file, **kwargs)

    assert file.exists()
    nodata = kwargs.get('nodata', True)
    with rio.open(file) as ds:
        # format
        assert ds.crs == image.crs
        assert ds.transform[:6] == image.transform
        assert ds.shape == image.shape
        assert ds.count == image.count
        assert ds.dtypes[0] == image.dtype
        assert ds.compression == Compression.deflate
        if nodata is True:
            assert ds.nodata == _nodata_vals[image.dtype]
        elif nodata is False:
            assert ds.nodata is None
        else:
            assert ds.nodata == nodata

        # contents
        array = ds.read()
        # masked pixels will always == _nodata_vals[image.dtype], irrespective of the nodata value
        mask = array != _nodata_vals[image.dtype]
        assert not np.all(mask)
        assert np.all((array.T == range(1, image.count + 1)) == mask.T)


def test_to_geotiff_overwrite(prepared_image: ImageAccessor, tmp_path: Path):
    """Test toGeoTIFF() overwrite behaviour."""
    file = tmp_path.joinpath('overwrite.tif')
    file.touch()
    with pytest.raises(FileExistsError):
        prepared_image.toGeoTIFF(file)
    prepared_image.toGeoTIFF(file, overwrite=True)
    # test file is a valid geotiff
    with rio.open(file) as ds:
        ds.read()


@pytest.mark.parametrize('masked', [False, True])
def test_to_xarray(prepared_image: ImageAccessor, masked: bool):
    """Test toXarray()."""
    image = prepared_image
    array = image.toXarray(masked=masked)

    # coordinates
    assert all(array.coords['band'] == image.bandNames)
    y = np.arange(0.5, image.shape[1] + 0.5) * image.transform[4] + image.transform[5]
    x = np.arange(0.5, image.shape[0] + 0.5) * image.transform[0] + image.transform[2]
    assert all(array.coords['x'] == x)
    assert all(array.coords['y'] == y)

    # dtype & nodata
    if masked:
        assert array.dtype == np.promote_types(image.dtype, 'float32')
        assert np.isnan(array.attrs['nodata'])
    else:
        assert array.dtype == image.dtype
        assert array.attrs['nodata'] == image.nodata

    # attributes
    for attr in ['id', 'crs', 'transform']:
        assert array.attrs[attr] == getattr(image, attr), attr
    assert array.attrs.get('ee') == json.dumps(image.properties)
    assert array.attrs.get('stac') == json.dumps(image.stac)

    # contents
    mask = ~array.isnull() if masked else array != image.nodata
    assert not mask.all()
    assert ((array == range(1, image.count + 1)) == mask).all()


@pytest.mark.parametrize('masked, structured', [(False, False), (True, False), (True, True)])
def test_to_numpy(prepared_image: ImageAccessor, masked: bool, structured: bool):
    """Test toNumpy()."""
    image = prepared_image
    array = image.toNumPy(masked=masked, structured=structured)

    # dimensions and dtype
    if structured:
        assert array.shape == image.shape
        assert len(array.dtype) == image.count
        dtype = np.dtype(dict(names=image.bandNames, formats=[image.dtype] * len(image.bandNames)))
        assert array.dtype == dtype
    else:
        assert array.shape == (*image.shape, image.count)
        assert array.dtype == image.dtype

    # masking
    if masked:
        assert isinstance(array, np.ma.MaskedArray)
        assert array.fill_value == np.array(image.nodata, array.dtype)
    else:
        assert not isinstance(array, np.ma.MaskedArray)

    # contents
    array_ = array.view(image.dtype).reshape(*image.shape, image.count) if structured else array
    mask = ~array_.mask if masked else array_ != image.nodata
    assert not np.all(mask)
    assert np.all((array_ == range(1, image.count + 1)) == mask)
