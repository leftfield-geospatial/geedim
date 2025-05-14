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
from dataclasses import make_dataclass
from datetime import datetime, timezone
from pathlib import Path

import ee
import fsspec
import numpy as np
import pytest
import rasterio as rio
from rasterio.enums import Compression

from geedim.enums import ExportType
from geedim.image import ImageAccessor, _nodata_vals, _open_raster, _rio_crs
from tests.conftest import accessors_from_images, transform_bounds


@pytest.mark.parametrize(
    'image, exp_min_band, exp_support',
    [('s2_sr_hm_image', 1, True), ('l9_sr_image', 0, True), ('modis_nbar_image', 0, False)],
)
def test_properties(
    image: str, exp_min_band: int, exp_support: bool, request: pytest.FixtureRequest
):
    """Test properties excluding dtype and bounded."""
    image: ImageAccessor = request.getfixturevalue(image)
    props = image.info['properties']
    min_band = image.info['bands'][exp_min_band]

    # EE info dependent properties
    assert image.id == image.info['id']
    assert image.index == props['system:index']
    assert image.date == datetime.fromtimestamp(props['system:time_start'] / 1000, tz=timezone.utc)
    assert image.properties == props
    assert image.scale == np.sqrt(abs(image.transform[0] * image.transform[4]))
    assert image.geometry == ee.Geometry(props['system:footprint']).toGeoJSON()
    assert image.bandNames == [bi['id'] for bi in image.info['bands']]
    assert image.size == np.dtype(image.dtype).itemsize * np.prod(image.shape) * image.count

    # RasterIO-like properties
    assert image.crs == min_band['crs']
    assert image.transform == tuple(min_band['crs_transform'])
    assert image.shape == (tuple(min_band['dimensions'][::-1]))
    assert image.count == len(image.info['bands'])
    assert image.nodata == _nodata_vals[image.dtype]
    assert image.profile == dict(
        crs=_rio_crs(image.crs),  # test conversion for MODIS CRS
        transform=image.transform,
        width=image.shape[1],
        height=image.shape[0],
        count=image.count,
        dtype=image.dtype,
    )

    # STAC dependent properties
    assert image.stac is not None
    assert len(image.bandProps) == image.count
    assert all([{'name', 'description'}.issubset(bp) for bp in image.bandProps])
    spec_bands = [bp['name'] for bp in image.bandProps if 'center_wavelength' in bp]
    if exp_support:
        assert len(spec_bands) > 0
    assert image.specBands == spec_bands

    # cloud/shadow support
    assert image.cloudShadowSupport == exp_support


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
def test_properties_non_fixed(image: str, request: pytest.FixtureRequest):
    """Test that composites / images without fixed projections have no geometry, shape,
    size or profile.
    """
    image: ImageAccessor = request.getfixturevalue(image)
    assert image.geometry is None
    assert image.shape is None
    assert image.size is None
    assert image.profile is None


@pytest.mark.parametrize('patch_export_task', ['export_task_success_sequence'], indirect=True)
def test_monitor_task(patch_export_task, capsys: pytest.CaptureFixture):
    """Test monitorTask()."""
    # mock ee.batch.Task object
    task = make_dataclass('Task', [('name', str, 'task')])
    ImageAccessor.monitorTask(task)
    assert '100%' in capsys.readouterr().err


@pytest.mark.parametrize('patch_export_task', ['export_task_fail'], indirect=True)
def test_monitor_task_error(patch_export_task, export_task_fail, capsys: pytest.CaptureFixture):
    """Test monitorTask() error condition."""
    # mock ee.batch.Task object
    task = make_dataclass('Task', [('name', str, 'task')])
    with pytest.raises(OSError, match=export_task_fail[0]['error']['message']):
        ImageAccessor.monitorTask(task)


def test_projection(s2_sr_hm_image: ImageAccessor):
    """Test projection() on an image with different band projections."""
    projs = ee.List(
        [s2_sr_hm_image.projection(min_scale=True), s2_sr_hm_image.projection(min_scale=False)]
    )
    projs = projs.getInfo()
    min_scale_band = s2_sr_hm_image.info['bands'][1]
    assert projs[0]['crs'] == min_scale_band['crs']
    assert projs[0]['transform'] == min_scale_band['crs_transform']
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
    assert fixeds == [True, False, False]


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
    fixeds = []
    for src_im, method, reproj_kwargs in zip(
        [l9_sr_image, l9_sr_image, l9_sr_image, landsat_ndvi_image],
        ['bilinear', 'bicubic', 'average', 'average'],
        [up_kwargs, up_kwargs, down_kwargs, down_kwargs],
    ):
        resample_im = src_im.resample(method)
        stds = [get_reproj_std(im, **reproj_kwargs) for im in [src_im._ee_image, resample_im]]
        std_tests.append(ee.List(stds))
        fixeds.append(src_im.shape is not None)

    std_tests = ee.List(std_tests).getInfo()

    # test standard deviations are reduced by resampling when the image has a fixed projection
    for std_test, fixed in zip(std_tests, fixeds):
        if fixed:
            assert std_test[0] > std_test[1]
        else:
            assert std_test[0] == std_test[1]


def test_to_dtype(landsat_ndvi_image: ImageAccessor):
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


def test_to_dtype_error(landsat_ndvi_image: ImageAccessor):
    """Test toDType() raises an error with an unsupported dtype."""
    with pytest.raises(ValueError, match='Unsupported dtype'):
        landsat_ndvi_image.toDType('int64')


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
    refl_bands_list = []
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
        refl_bands_list.append(refl_bands)

    # combine getInfo() of min / max values & scaled / offset images into one
    results = ee.Dictionary(
        dict(min_max=ee.List(min_max_tests), image=ee.List(scale_offset_ims))
    ).getInfo()

    # test min / max values are ~0-1 reflectance
    for min_max_test, refl_bands in zip(results['min_max'], refl_bands_list):
        assert len(min_max_test) == len(2 * refl_bands)
        for bn in refl_bands:
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


def test_mask_clouds(l9_sr_image: ImageAccessor, region_100ha: dict):
    """Test maskClouds()."""
    # This just tests the masked area increases. Detailed mask testing is done in test_mask.py.
    masked_image = ImageAccessor(l9_sr_image.addMaskBands()).maskClouds()
    sums = [
        im.mask().reduceRegion('sum', geometry=region_100ha, scale=30).values().reduce('mean')
        for im in [l9_sr_image._ee_image, masked_image]
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
        dict(crs=s2_sr_hm_image.crs, region=region_100ha),
        dict(region=region_100ha, scale=s2_sr_hm_image.scale),
        dict(region=region_100ha),
        dict(),
    ]
    prep_ims = [s2_sr_hm_image.prepareForExport(**prep_kwargs) for prep_kwargs in prep_kwargs_list]
    prep_ims = accessors_from_images(prep_ims)

    # test prepared image properties
    for prep_im, prep_kwargs in zip(prep_ims, prep_kwargs_list):
        assert prep_im.crs == prep_kwargs.get('crs', s2_sr_hm_image.crs)
        assert prep_im.dtype == prep_kwargs.get('dtype', s2_sr_hm_image.dtype)
        assert prep_im.bandNames == prep_kwargs.get('bands', s2_sr_hm_image.bandNames)
        if 'shape' in prep_kwargs:
            assert prep_im.shape == prep_kwargs['shape']
        if 'scale' in prep_kwargs:
            assert prep_im.scale == prep_kwargs['scale']
        if 'crs_transform' in prep_kwargs:
            assert prep_im.transform == prep_kwargs['crs_transform']

        # region is a special case that is approximate & needs transformation between CRSs
        if 'region' in prep_kwargs:
            region_bounds = transform_bounds(ee.Geometry(prep_kwargs['region']).toGeoJSON(), crs)
            image_bounds = transform_bounds(prep_im.geometry, crs)
            assert image_bounds == pytest.approx(region_bounds, abs=60)

    # test pixel grid is maintained when arguments allow
    src_transform = rio.Affine(*s2_sr_hm_image.transform)
    for prep_im in prep_ims[-4:]:
        prep_transform = rio.Affine(*prep_im.transform)
        assert (prep_transform[0], prep_transform[4]) == (src_transform[0], src_transform[4])
        pixel_offset = ~src_transform * (prep_transform[2], prep_transform[5])
        assert pixel_offset == (int(pixel_offset[0]), int(pixel_offset[1]))


def test_prepare_for_export_scale_offset(s2_sr_hm_image: ImageAccessor, region_100ha: dict):
    """Test the prepareForExport() scale_offset parameter."""
    # This just tests the scale_offset parameter was acted on. Detailed scaleOffset() testing is
    # done in test_scale_offset().
    prep_ims = [
        s2_sr_hm_image.prepareForExport(region=region_100ha, scale_offset=scale_offset)
        for scale_offset in [False, True]
    ]
    maxs = [
        prep_im.select('B.*')
        .reduceRegion(reducer='max', geometry=region_100ha, bestEffort=True)
        .values()
        .reduce('max')
        for prep_im in prep_ims
    ]
    maxs = ee.List(maxs).getInfo()
    assert maxs[1] < maxs[0]


def test_prepare_for_export_errors(
    s2_sr_hm_image: ImageAccessor, landsat_ndvi_image: ImageAccessor
):
    """Test prepareForExport() error conditions."""
    # composite image without sufficient spatial params
    with pytest.raises(ValueError, match='fixed projection'):
        landsat_ndvi_image.prepareForExport()
    with pytest.raises(ValueError, match='fixed projection'):
        landsat_ndvi_image.prepareForExport(crs='EPSG:3857', scale=500)

    # scale and shape together
    with pytest.raises(ValueError, match="'scale' or 'shape'"):
        s2_sr_hm_image.prepareForExport(scale=10, shape=(400, 300))


@pytest.mark.parametrize(
    'etype, patch_export_task',
    itertools.product(ExportType, ['export_task_success']),
    indirect=['patch_export_task'],
)
def test_to_google_cloud(
    prepared_image: ImageAccessor,
    etype: ExportType,
    patch_export_task,
    capsys: pytest.CaptureFixture,
):
    """Test toGoogleCloud()."""
    kwargs = dict(type=etype, folder='geedim')
    _ = prepared_image.toGoogleCloud('test_export', wait=False, **kwargs)
    # test monitorTask is not called with wait=False
    assert capsys.readouterr().err == ''

    _ = prepared_image.toGoogleCloud('test_export', wait=True, **kwargs)
    # test monitorTask is called with wait=True
    assert '100%' in capsys.readouterr().err


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
def test_to_geotiff(
    prepared_image: ImageAccessor, prepared_image_array: np.ndarray, tmp_path: Path, kwargs: dict
):
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
        array = np.moveaxis(array, 0, -1)
        # masked pixels will always == _nodata_vals[image.dtype], irrespective of the nodata value
        mask = array != _nodata_vals[image.dtype]
        assert (mask == ~prepared_image_array.mask).all()
        assert (array == prepared_image_array).all()

        # metadata
        metadata = ds.tags()
        props = {k.replace(':', '-'): str(v) for k, v in image.properties.items()}
        assert all([metadata.get(k) == v for k, v in props.items()])
        assert metadata.get('LICENSE') is not None
        assert ds.descriptions == tuple([bp['name'] for bp in image.bandProps])
        for bi in range(image.count):
            band_props = {k.replace(':', '-'): str(v) for k, v in image.bandProps[bi].items()}
            assert ds.tags(bi + 1) == band_props


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


@pytest.mark.parametrize('masked, structured', [(False, False), (True, False), (True, True)])
def test_to_numpy(
    prepared_image: ImageAccessor, prepared_image_array: np.ndarray, masked: bool, structured: bool
):
    """Test toNumpy()."""
    image = prepared_image
    array = image.toNumPy(masked=masked, structured=structured)

    # dimensions and dtype
    if structured:
        assert array.shape == image.shape
        assert len(array.dtype) == image.count
        dtype = np.dtype(dict(names=image.bandNames, formats=[image.dtype] * image.count))
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
    assert (mask == ~prepared_image_array.mask).all()
    assert (array_ == prepared_image_array).all()


@pytest.mark.parametrize('masked', [False, True])
def test_to_xarray(prepared_image: ImageAccessor, prepared_image_array: np.ndarray, masked: bool):
    """Test toXarray()."""
    image = prepared_image
    array = image.toXarray(masked=masked)

    # coordinates
    assert (array.coords['band'] == image.bandNames).all()
    y = np.arange(0.5, image.shape[1] + 0.5) * image.transform[4] + image.transform[5]
    x = np.arange(0.5, image.shape[0] + 0.5) * image.transform[0] + image.transform[2]
    assert (array.coords['x'] == x).all()
    assert (array.coords['y'] == y).all()

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
    assert array.attrs['ee'] == json.dumps(image.properties)
    assert array.attrs['stac'] == json.dumps(image.stac)

    # contents
    mask = ~array.isnull() if masked else array != image.nodata
    assert (mask == ~prepared_image_array.mask).all()
    assert (array == prepared_image_array).all()


def test_export_error(landsat_ndvi_image: ImageAccessor, tmp_path: Path):
    """Test to*() error conditions."""
    # mock an image with no bands
    image = ImageAccessor(None)
    image.info = dict(bands=[])

    with pytest.raises(ValueError, match='no bands'):
        image.toGeoTIFF(tmp_path.joinpath('test.tif'))
    with pytest.raises(ValueError, match='no bands'):
        image.toXarray()
    with pytest.raises(ValueError, match='no bands'):
        image.toNumPy()

    # just in case, prevent this test running large exports if landsat_ndvi_image does have a
    # fixed projection
    assert landsat_ndvi_image.shape is None

    with pytest.raises(ValueError, match='fixed projection'):
        landsat_ndvi_image.toGeoTIFF(tmp_path.joinpath('test.tif'))
    with pytest.raises(ValueError, match='fixed projection'):
        landsat_ndvi_image.toXarray()
    with pytest.raises(ValueError, match='fixed projection'):
        landsat_ndvi_image.toNumPy()
