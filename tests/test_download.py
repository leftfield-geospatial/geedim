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

import pathlib
from datetime import datetime, timezone
from itertools import product
from typing import Dict, List

import ee
import numpy as np
import pytest
import rasterio as rio
from rasterio import Affine, features, warp, windows

from geedim import utils
from geedim.download import _nodata_vals, BaseImage, BaseImageAccessor
from geedim.enums import ExportType, ResamplingMethod
from tests.conftest import region_25ha


class BaseImageLike:
    """Mock ``BaseImage`` for ``_get_tile_shape()`` and ``tiles()``."""

    def __init__(
        self,
        shape: tuple[int, int],
        count: int = 10,
        dtype: str = 'uint16',
        transform: Affine = Affine.identity(),
    ):
        self.shape = shape
        self.count = count
        self.dtype = dtype
        self.transform = transform
        dtype_size = np.dtype(dtype).itemsize
        self.size = shape[0] * shape[1] * count * dtype_size

    _tiles = BaseImage._tiles
    _get_tile_shape = BaseImageAccessor._get_tile_shape


def _bounds(geom: dict, dst_crs: str | rio.CRS = 'EPSG:4326') -> tuple[float, ...]:
    """Return the bounds of GeoJSON polygon ``geom`` in the ``dst_crs`` CRS."""
    # transform geom, then find bounds (if geom is an ee footprint, finding its bounds then transforming expands
    # bounds beyond their true location)
    src_crs = utils.rio_crs(geom['crs']['properties']['name']) if 'crs' in geom else 'EPSG:4326'
    dst_crs = utils.rio_crs(dst_crs) or 'EPSG:4326'
    geom = geom if dst_crs == src_crs else warp.transform_geom(src_crs, dst_crs, geom)
    return features.bounds(geom)


def _intersect_bounds(
    bounds1: tuple[float, ...], bounds2: tuple[float, ...]
) -> tuple[float, ...] | None:
    """Return the intersection of ``bounds1`` and ``bounds2``, or ``None`` when there is no intersection."""
    bounds = np.array([*np.fmax(bounds1[:2], bounds2[:2]), *np.fmin(bounds1[2:], bounds2[2:])])
    return tuple(bounds.tolist()) if np.all((bounds[2:] - bounds[:2]) > 0) else None


@pytest.fixture(scope='session')
def user_base_image() -> BaseImage:
    """A BaseImage instance where the encapsulated image has no fixed projection or ID."""
    return BaseImage(ee.Image([1, 2, 3]))


@pytest.fixture(scope='session')
def user_fix_base_image() -> BaseImage:
    """A BaseImage instance where the encapsulated image has a fixed projection (EPSG:4326), a scale in degrees,
    is unbounded and has no ID.
    """
    return BaseImage(ee.Image([1, 2, 3]).setDefaultProjection(crs='EPSG:4326', scale=30))


@pytest.fixture(scope='session')
def user_fix_bnd_base_image(region_10000ha) -> BaseImage:
    """A BaseImage instance where the encapsulated image has a fixed projection (EPSG:4326), a scale in degrees,
    is bounded, and has no ID.
    """
    return BaseImage(
        ee.Image([1, 2, 3])
        .setDefaultProjection(crs='EPSG:4326', scale=30)
        .clipToBoundsAndScale(region_10000ha)
    )


@pytest.fixture(scope='session')
def s2_sr_hm_base_image(s2_sr_hm_image_id: str) -> BaseImage:
    """A BaseImage instance encapsulating a Sentinel-2 image.  Covers `region_*ha`."""
    return BaseImage.from_id(s2_sr_hm_image_id)


@pytest.fixture(scope='session')
def l9_base_image(l9_image_id: str) -> BaseImage:
    """A BaseImage instance encapsulating a Landsat-9 image.  Covers `region_*ha`."""
    return BaseImage.from_id(l9_image_id)


@pytest.fixture(scope='session')
def l8_base_image(l8_image_id: str) -> BaseImage:
    """A BaseImage instance encapsulating a Landsat-8 image.  Covers `region_*ha`."""
    return BaseImage.from_id(l8_image_id)


@pytest.fixture(scope='session')
def landsat_ndvi_base_image(landsat_ndvi_image_id: str) -> BaseImage:
    """A BaseImage instance encapsulating a Landsat NDVI composite image.  Covers `region_*ha`."""
    return BaseImage.from_id(landsat_ndvi_image_id)


@pytest.fixture(scope='session')
def modis_nbar_base_image_unbnd(modis_nbar_image_id: str) -> BaseImage:
    """A BaseImage instance encapsulating an unbounded MODIS NBAR image in its native CRS.  Covers `region_*ha`."""
    return BaseImage(ee.Image(modis_nbar_image_id))


@pytest.fixture(scope='session')
def modis_nbar_base_image(modis_nbar_image_id: str, region_100ha: Dict) -> BaseImage:
    """A BaseImage instance encapsulating a MODIS NBAR image.  Covers `region_*ha`."""
    return BaseImage(ee.Image(modis_nbar_image_id).clipToBoundsAndScale(region_100ha))


@pytest.fixture(scope='session')
def google_dyn_world_base_image(google_dyn_world_image_id: str) -> BaseImage:
    """A BaseImage instance encapsulating a Google Dynamic World image with positive y-axis transform.  Covers
    `region_*ha`.
    """
    return BaseImage.from_id(google_dyn_world_image_id)


def bounds_polygon(left: float, bottom: float, right: float, top: float, crs: str = None):
    """Return a geojson polygon of the given bounds."""
    coordinates = [
        [
            [left, bottom],
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ]
    ]
    poly = dict(type='Polygon', coordinates=coordinates)
    if crs and crs != 'EPSG:4326':
        poly.update(crs=dict(type='name', properties=dict(name=crs)))
    return poly


def _test_export_image(exp_image: BaseImageAccessor, ref_image: BaseImageAccessor, **exp_kwargs):
    """Test ``exp_image`` against ``ref_image``, and optional crs, region & scale ``exp_kwargs``
    arguments to ``BaseImage._prepare_for_export``.
    """
    # remove None value items from exp_kwargs
    exp_kwargs = {k: v for k, v in exp_kwargs.items() if v is not None}

    assert exp_image.dtype == exp_kwargs.get('dtype', ref_image.dtype)
    assert exp_image.band_properties == ref_image.band_properties

    # test crs and scale
    crs = exp_kwargs.get('crs', ref_image.crs)
    assert exp_image.crs == crs

    if 'shape' in exp_kwargs:
        assert exp_image.shape == exp_kwargs['shape']
    else:
        ref_scale = exp_kwargs.get('scale', ref_image.scale)
        assert exp_image.scale == ref_scale

    # test export bounds contain reference bounds
    region = exp_kwargs.get('region', ref_image.geometry)
    ref_bounds = _bounds(region, exp_image.crs)
    exp_bounds = _bounds(exp_image.geometry, exp_image.crs)
    tol = 1e-9 if exp_image.crs == 'EPSG:4326' else 1e-6
    assert _intersect_bounds(exp_bounds, ref_bounds) == pytest.approx(ref_bounds, abs=tol)

    # test export transform is on the reference grid
    if {'crs', 'scale', 'shape'}.isdisjoint(exp_kwargs.keys()):
        ji = ~rio.Affine(*exp_image.transform) * (ref_image.transform[2], ref_image.transform[5])
        assert ji == pytest.approx(np.round(ji), abs=1e-6)


def test_id_name(user_base_image: BaseImage, s2_sr_hm_base_image: BaseImage):
    """Test ``id`` and ``name`` properties for different scenarios."""
    assert user_base_image.id is None
    assert user_base_image.name is None
    # check that BaseImage.from_id() sets id without a getInfo
    assert s2_sr_hm_base_image._id is not None
    assert s2_sr_hm_base_image.name == s2_sr_hm_base_image.id.replace('/', '-')


def test_user_props(user_base_image: BaseImage):
    """Test non fixed projection image properties (other than ``id`` and ``has_fixed_projection``)."""
    assert user_base_image.crs is None
    assert user_base_image.scale is None
    assert user_base_image.transform is None
    assert user_base_image.shape is None
    assert user_base_image.date is None
    assert user_base_image.size is None
    assert user_base_image.footprint is None
    assert user_base_image.dtype == 'uint8'
    assert user_base_image.count == 3


def test_fix_user_props(user_fix_base_image: BaseImage):
    """Test fixed projection image properties (other than ``id`` and ``has_fixed_projection``)."""
    assert user_fix_base_image.crs == 'EPSG:4326'
    assert user_fix_base_image.scale == pytest.approx(30, abs=1e-6)
    assert user_fix_base_image.transform is not None
    assert user_fix_base_image.shape is None
    assert user_fix_base_image.date is None
    assert user_fix_base_image.size is None
    assert user_fix_base_image.footprint is None
    assert user_fix_base_image.dtype == 'uint8'
    assert user_fix_base_image.count == 3


def test_s2_props(s2_sr_hm_base_image: BaseImage):
    """Test fixed projection S2 image properties (other than ``id`` and ``has_fixed_projection``)."""
    min_band_info = s2_sr_hm_base_image.info['bands'][1]
    assert s2_sr_hm_base_image.crs == min_band_info['crs']
    assert s2_sr_hm_base_image.scale == min_band_info['crs_transform'][0]
    assert s2_sr_hm_base_image.transform == Affine(*min_band_info['crs_transform'])
    assert s2_sr_hm_base_image.shape == tuple(min_band_info['dimensions'][::-1])
    assert s2_sr_hm_base_image.date == datetime.fromtimestamp(
        s2_sr_hm_base_image.properties['system:time_start'] / 1000, timezone.utc
    )
    assert s2_sr_hm_base_image.size is not None
    assert s2_sr_hm_base_image.geometry is not None
    assert s2_sr_hm_base_image.geometry['type'] in ['Polygon', 'LinearRing']
    assert s2_sr_hm_base_image.dtype == 'uint32'
    assert s2_sr_hm_base_image.count == len(s2_sr_hm_base_image.info['bands'])


@pytest.mark.parametrize(
    'base_image',
    ['landsat_ndvi_base_image', 's2_sr_hm_base_image', 'l9_base_image', 'modis_nbar_base_image'],
)
def test_band_props(base_image: str, request: pytest.FixtureRequest):
    """Test ``band_properties`` completeness for generic / user / reflectance images."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    assert base_image.band_properties is not None
    assert [bd['name'] for bd in base_image.band_properties] == [
        bd['id'] for bd in base_image.info['bands']
    ]
    for key in ['gsd', 'description']:
        has_key = [key in bd for bd in base_image.band_properties]
        assert all(has_key)


def test_has_fixed_projection(
    user_base_image: BaseImage, user_fix_base_image: BaseImage, s2_sr_hm_base_image
):
    """Test the ``has_fixed_projection`` property."""
    assert not user_base_image.has_fixed_projection
    assert user_fix_base_image.has_fixed_projection
    assert s2_sr_hm_base_image.has_fixed_projection


@pytest.mark.parametrize(
    'data_types, exp_dtype',
    [
        (
            [
                {'precision': 'int', 'min': 10, 'max': 11},
                {'precision': 'int', 'min': 100, 'max': 101},
            ],
            'uint8',
        ),
        (
            [
                {'precision': 'int', 'min': -128, 'max': -100},
                {'precision': 'int', 'min': 0, 'max': 127},
            ],
            'int8',
        ),
        ([{'precision': 'int', 'min': 256, 'max': 257}], 'uint16'),
        ([{'precision': 'int', 'min': -32768, 'max': 32767}], 'int16'),
        ([{'precision': 'int', 'min': 2**15, 'max': 2**32 - 1}], 'uint32'),
        ([{'precision': 'int', 'min': -(2**31), 'max': 2**31 - 1}], 'int32'),
        (
            [
                {'precision': 'float', 'min': 0.0, 'max': 1.0e9},
                {'precision': 'float', 'min': 0.0, 'max': 1.0},
            ],
            'float32',
        ),
        (
            [
                {'precision': 'int', 'min': 0.0, 'max': 2**31 - 1},
                {'precision': 'float', 'min': 0.0, 'max': 1.0},
            ],
            'float64',
        ),
        (
            [
                {'precision': 'int', 'min': 0, 'max': 255},
                {'precision': 'double', 'min': -1e100, 'max': 1e100},
            ],
            'float64',
        ),
    ],
)
def test_min_dtype(data_types: List, exp_dtype: str):
    """Test ``BaseImage.dtype`` with mocked EE info dicts."""
    # mock a BaseImage with data_types in its EE info
    info = dict(bands=[dict(data_type=data_type) for data_type in data_types])
    im = BaseImage(None)
    im.info = info

    assert im.dtype == exp_dtype


def _test_convert_dtype_error():
    """Test ``BaseImage.test_convert_dtype()`` raises an error with incorrect dtype."""
    # TODO: replace with BaseImageAccessor test
    with pytest.raises(TypeError):
        BaseImage._convert_dtype(ee.Image(1), dtype='unknown')


@pytest.mark.parametrize(
    'params',
    [
        dict(),
        dict(crs='EPSG:4326'),
        dict(crs='EPSG:4326', region='region_25ha'),
        dict(crs='EPSG:4326', scale=100),
        dict(crs='EPSG:4326', crs_transform=Affine.identity()),
        dict(crs='EPSG:4326', shape=(100, 100)),
        dict(region='region_25ha', scale=100),
        dict(crs_transform=Affine.identity(), shape=(100, 100)),
    ],
)
def test_prepare_no_fixed_projection(
    user_base_image: BaseImage, params: Dict, request: pytest.FixtureRequest
):
    """Test ``BaseImage._prepare_for_export()`` raises an exception when the image has no fixed projection,
    and insufficient CRS & region defining arguments.
    """
    if 'region' in params:
        params['region'] = request.getfixturevalue(params['region'])
    with pytest.raises(ValueError) as ex:
        user_base_image.prepareForExport(**params)
    assert 'does not have a fixed projection' in str(ex)


@pytest.mark.parametrize(
    'base_image, params',
    [
        ('user_fix_base_image', dict()),
        ('modis_nbar_base_image_unbnd', dict(crs='EPSG:4326')),
        ('modis_nbar_base_image_unbnd', dict(crs='EPSG:4326', scale=100)),
        ('modis_nbar_base_image_unbnd', dict(crs='EPSG:4326', crs_transform=Affine.identity())),
        ('modis_nbar_base_image_unbnd', dict(crs='EPSG:4326', shape=(100, 100))),
    ],
)
def test_prepare_unbounded(base_image: BaseImage, params: Dict, request: pytest.FixtureRequest):
    """Test ``BaseImage._prepare_for_export()`` raises an exception when the image is unbounded, and insufficient
    bounds defining parameters are specified.
    """
    base_image: BaseImage = request.getfixturevalue(base_image)
    with pytest.raises(ValueError) as ex:
        base_image.prepareForExport(**params)
    assert 'This image is unbounded' in str(ex)


def test_prepare_exceptions(
    user_base_image: BaseImage, user_fix_base_image: BaseImage, region_25ha: Dict
):
    """Test remaining ``BaseImage._prepare_for_export()`` error cases."""
    with pytest.raises(ValueError):
        # no fixed projection and resample
        user_base_image.prepareForExport(
            region=region_25ha, crs='EPSG:3857', scale=30, resampling=ResamplingMethod.bilinear
        )


@pytest.mark.parametrize(
    'base_image',
    [
        'user_fix_bnd_base_image',
        's2_sr_hm_base_image',
        'l9_base_image',
        'modis_nbar_base_image',
        'google_dyn_world_base_image',
    ],
)
def test_prepare_defaults(base_image: str, request: pytest.FixtureRequest):
    """Test ``BaseImage._prepare_for_export()`` with no (i.e. default) arguments."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    exp_image = BaseImageAccessor(base_image.prepareForExport())

    _test_export_image(exp_image, base_image)


@pytest.mark.parametrize(
    'base_image, crs, crs_transform, shape',
    [
        ('user_base_image', 'EPSG:4326', (-1e-3, 0, -180.1, 0, 2e-3, 0.1), (300, 400)),
        ('user_fix_base_image', 'EPSG:32734', (10, 0, 100, 0, 10, 200), (300, 400)),
        ('s2_sr_hm_base_image', 'EPSG:3857', (10, 0, 100, 0, -10, 200), (300, 400)),
    ],
)
def test_prepare_transform_shape(
    base_image: str,
    crs: str,
    crs_transform: tuple[float, ...],
    shape: tuple[int, int],
    request: pytest.FixtureRequest,
):
    """Test ``BaseImage._prepare_for_export()`` with ``crs_transform`` and ``shape`` parameters."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    exp_image = BaseImageAccessor(
        base_image.prepareForExport(crs=crs, crs_transform=crs_transform, shape=shape)
    )

    assert exp_image.dtype == base_image.dtype
    assert exp_image.band_properties == base_image.band_properties

    assert exp_image.crs == crs or base_image.crs
    # assert exp_image.scale == np.abs((crs_transform[0], crs_transform[4])).mean()
    assert exp_image.shape == shape
    assert exp_image.transform[:6] == crs_transform


@pytest.mark.parametrize(
    'base_image, crs, region, scale',
    [
        ('user_base_image', 'EPSG:3857', 'region_25ha', 0.1),
        ('user_fix_base_image', 'EPSG:32734', 'region_25ha', 30),
        ('l9_base_image', 'EPSG:3857', 'region_100ha', 10),
        ('s2_sr_hm_base_image', 'EPSG:32734', 'region_100ha', 30),
        ('modis_nbar_base_image', 'EPSG:3857', 'region_10000ha', 1000),
        ('google_dyn_world_base_image', 'EPSG:4326', 'region_10000ha', 1.0e-5),
    ],
)
def test_prepare_region_scale(
    base_image: str, crs: str, region: str, scale: float, request: pytest.FixtureRequest
):
    """Test ``BaseImage._prepare_for_export()`` with ``region`` and ``scale`` parameters."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    region: dict = request.getfixturevalue(region) if region else None

    exp_kwargs = dict(crs=crs, region=region, scale=scale)
    exp_image = BaseImageAccessor(base_image.prepareForExport(**exp_kwargs))

    _test_export_image(exp_image, base_image, **exp_kwargs)


@pytest.mark.parametrize(
    'base_image, crs, region, shape',
    [
        ('user_base_image', 'EPSG:3857', 'region_25ha', (30, 40)),
        ('user_fix_base_image', 'EPSG:32734', 'region_100ha', (300, 400)),
        ('s2_sr_hm_base_image', 'EPSG:4326', 'region_10000ha', (300, 400)),
    ],
)
def test_prepare_region_shape(
    base_image: str, crs: str, region: str, shape: tuple[int, int], request: pytest.FixtureRequest
):
    """Test ``BaseImage._prepare_for_export()`` with ``region`` and ``shape`` parameters."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    region: dict = request.getfixturevalue(region) if region else None

    exp_kwargs = dict(crs=crs, region=region, shape=shape)
    exp_image = BaseImageAccessor(base_image.prepareForExport(**exp_kwargs))

    _test_export_image(exp_image, base_image, **exp_kwargs)


@pytest.mark.parametrize(
    'base_image, region',
    [
        ('user_fix_bnd_base_image', 'region_25ha'),
        ('l8_base_image', 'region_100ha'),
        ('l9_base_image', 'region_10000ha'),
        ('s2_sr_hm_base_image', 'region_25ha'),
        ('modis_nbar_base_image', 'region_100ha'),
        ('google_dyn_world_base_image', 'region_10000ha'),
    ],
)
def test_prepare_src_grid(base_image: str, region: str, request: pytest.FixtureRequest):
    """Test ``BaseImage._prepare_for_export()`` maintains the source pixel grid, with default value ``crs`` and
    ``scale`` arguments.
    """
    base_image: BaseImage = request.getfixturevalue(base_image)
    region: dict = request.getfixturevalue(region)

    exp_kwargs = dict(region=region)
    exp_image = BaseImageAccessor(base_image.prepareForExport(**exp_kwargs))

    _test_export_image(exp_image, base_image, **exp_kwargs)


@pytest.mark.parametrize(
    'base_image, bands',
    [('s2_sr_hm_base_image', ['B1', 'B5']), ('l9_base_image', ['SR_B4', 'SR_B3', 'SR_B2'])],
)
def test_prepare_bands(
    base_image: str, bands: List[str], region_25ha: dict, request: pytest.FixtureRequest
):
    """Test ``BaseImage._prepare_for_export()`` with ``bands`` parameter."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    ref_image = BaseImageAccessor(base_image.ee_image.select(bands))
    exp_image = BaseImageAccessor(base_image.prepareForExport(bands=bands))

    _test_export_image(exp_image, ref_image)


def test_prepare_bands_error(s2_sr_hm_base_image):
    """Test ``BaseImage._prepare_for_export()`` raises an error with incorrect bands."""
    with pytest.raises(ee.EEException) as ex:
        s2_sr_hm_base_image.prepareForExport(bands=['unknown'])
    assert 'band' in str(ex.value)


def test_prepared_profile(s2_sr_hm_base_image: BaseImage):
    """Test the profile on image returned by ``BaseImage._prepare_for_export()``."""
    exp_image = BaseImageAccessor(s2_sr_hm_base_image.prepareForExport())
    _test_export_image(exp_image, s2_sr_hm_base_image)

    # test dynamic values
    for key in ['dtype', 'count', 'crs', 'transform']:
        assert exp_image.profile[key] == getattr(exp_image, key), key

    assert exp_image.profile['width'] == exp_image.shape[1]
    assert exp_image.profile['height'] == exp_image.shape[0]
    assert exp_image.profile['nodata'] == _nodata_vals[exp_image.profile['dtype']]


@pytest.mark.parametrize('dtype, exp_nodata', _nodata_vals.items())
def test_prepare_nodata(user_fix_bnd_base_image: BaseImage, dtype: str, exp_nodata: float):
    """Test ``BaseImage._prepare_for_export()`` profile sets the ``nodata`` value correctly for
    different dtypes.
    """
    exp_image = BaseImageAccessor(user_fix_bnd_base_image.prepareForExport(dtype=dtype))
    assert exp_image.dtype == dtype
    assert exp_image.profile['nodata'] == exp_nodata


@pytest.mark.parametrize(
    'src_image, dtype',
    [('s2_sr_hm_base_image', 'float32'), ('l9_base_image', None), ('modis_nbar_base_image', None)],
)
def test_scale_offset(
    src_image: str, dtype: str, region_100ha: Dict, request: pytest.FixtureRequest
):
    """Test ``BaseImage._prepare_for_export(scale_offset=True)`` gives expected properties and reflectance ranges."""
    src_image: BaseImage = request.getfixturevalue(src_image)
    exp_image = BaseImageAccessor(src_image.prepareForExport(scale_offset=True))

    assert exp_image.crs == src_image.crs
    assert exp_image.scale == src_image.scale
    assert exp_image.band_properties == src_image.band_properties
    assert exp_image.dtype == dtype or 'float64'

    def get_min_max_refl(base_image: BaseImage) -> tuple[dict, dict]:
        """Return the min & max of each reflectance band of ``base_image``."""
        band_props = base_image.band_properties
        refl_bands = [
            bp['name']
            for bp in band_props
            if ('center_wavelength' in bp) and (bp['center_wavelength'] < 1)
        ]
        ee_image = base_image._ee_image.select(refl_bands)
        min_max_dict = ee_image.reduceRegion(
            reducer=ee.Reducer.minMax(), geometry=region_100ha, bestEffort=True
        ).getInfo()
        min_dict = {k: v for k, v in min_max_dict.items() if 'min' in k}
        max_dict = {k: v for k, v in min_max_dict.items() if 'max' in k}
        return min_dict, max_dict

    # test the scaled and offset reflectance values lie between -0.5 and 1.5
    exp_min, exp_max = get_min_max_refl(exp_image)
    assert all(np.array(list(exp_min.values())) >= -0.5)
    assert all(np.array(list(exp_max.values())) <= 1.5)


def test_tile_shape():
    """Test ``BaseImage._get_tile_shape()`` satisfies the tile size limit for different image shapes."""
    max_tile_dim = 10000
    for max_tile_size, count, height, width in product(
        range(16, 32, 16), range(1, 11000, 2000), range(1, 11000, 2000), range(1, 11000, 2000)
    ):
        im_shape = (count, height, width)
        exp_image = BaseImageLike(shape=(height, width), count=count)  # mock a BaseImage
        tile_shape = exp_image._get_tile_shape(
            max_tile_size=max_tile_size, max_tile_dim=max_tile_dim
        )
        tile_size = np.prod(tile_shape) * np.dtype(exp_image.dtype).itemsize

        assert all(np.array(tile_shape) <= np.array(im_shape)), (max_tile_size, im_shape)
        assert all(np.array(tile_shape) <= max_tile_dim), (max_tile_size, im_shape)
        assert tile_size <= (max_tile_size << 20), (max_tile_size, im_shape)
    pass


@pytest.mark.parametrize(
    'image_shape, image_count, max_tile_size, image_transform',
    [
        ((2000, 500), 10, 1, Affine.identity()),
        ((10, 1000), 2000, 1, Affine.scale(1.23)),
        ((150, 1002), 150, 1, Affine.scale(1.23) * Affine.translation(12, 34)),
    ],
)
def test_tiles(
    image_shape: tuple[int, int], image_count: int, max_tile_size: int, image_transform: Affine
):
    """Test continuity and coverage of tiles."""
    exp_image = BaseImageLike(shape=image_shape, count=image_count, transform=image_transform)
    tile_shape = exp_image._get_tile_shape(max_tile_size=max_tile_size)
    tiles = [tile for tile in exp_image._tiles(tile_shape)]

    # test tile continuity
    prev_tile = tiles[0]
    accum_window = prev_tile.window
    accum_indexes = {*prev_tile.indexes}
    for tile in tiles[1:]:
        tile_size = np.prod((tile.count, *tile.shape)) * np.dtype(exp_image.dtype).itemsize
        assert tile_size <= (max_tile_size << 20)
        accum_window = windows.union(accum_window, tile.window)
        accum_indexes |= {*tile.indexes}

        prev_transform = rio.Affine(*prev_tile.tile_transform)
        if tile.row_off == prev_tile.row_off and tile.band_off == prev_tile.band_off:
            assert tile.col_off == (prev_tile.col_off + prev_tile.width)
            ref_transform = prev_transform * Affine.translation(prev_tile.width, 0)
        elif tile.band_off == prev_tile.band_off:
            assert tile.row_off == (prev_tile.row_off + prev_tile.height)
            ref_transform = prev_transform * Affine.translation(
                -prev_tile.col_off, prev_tile.height
            )
        else:
            assert tile.band_off == (prev_tile.band_off + prev_tile.count)
            ref_transform = image_transform

        assert tile.tile_transform == pytest.approx(ref_transform[:6], abs=1e-9)
        prev_tile = tile

    # test exp_image is fully covered by tiles
    assert (accum_window.height, accum_window.width) == exp_image.shape
    assert accum_indexes == {*range(1, exp_image.count + 1)}


def test_download_transform_shape(user_base_image: BaseImage, tmp_path: pathlib.Path):
    """Test ``BaseImage.download()`` file properties and contents with ``crs_transform`` and ``shape`` arguments."""
    # reference profile to test against
    ref_profile = dict(
        crs='EPSG:3857',
        transform=Affine(1, 0, 0, 0, -1, 0),
        width=10,
        height=10,
        dtype='uint16',
        count=3,
    )

    # form export kwargs from ref_profile
    shape = (ref_profile['height'], ref_profile['width'])
    exp_kwargs = dict(
        crs=ref_profile['crs'],
        crs_transform=ref_profile['transform'],
        shape=shape,
        dtype=ref_profile['dtype'],
    )

    # download
    filename = tmp_path.joinpath('test.tif')
    user_base_image.download(filename, **exp_kwargs)
    assert filename.exists()

    # test file format and contents
    with rio.open(filename, 'r') as ds:
        for key in ref_profile:
            assert ds.profile[key] == ref_profile[key]

        array = ds.read()
        for i in range(ds.count):
            assert np.all(array[i] == i + 1)


def test_download_region_scale(user_base_image: BaseImage, tmp_path: pathlib.Path):
    """Test ``BaseImage.download()`` file properties and contents with ``region`` and ``scale`` arguments."""
    # reference profile to test against
    ref_profile = dict(
        crs='EPSG:3857',
        transform=Affine(1, 0, 0, 0, -1, 0),
        width=10,
        height=10,
        dtype='uint16',
        count=3,
    )

    # form export kwargs from ref_profile
    shape = (ref_profile['height'], ref_profile['width'])
    bounds = windows.bounds(windows.Window(0, 0, *shape[::-1]), ref_profile['transform'])
    region = bounds_polygon(*bounds, crs=ref_profile['crs'])
    exp_kwargs = dict(
        crs=ref_profile['crs'],
        region=region,
        scale=ref_profile['transform'][0],
        dtype=ref_profile['dtype'],
    )

    # download
    filename = tmp_path.joinpath('test.tif')
    user_base_image.download(filename, **exp_kwargs)
    assert filename.exists()

    # test file format and contents
    with rio.open(filename, 'r') as ds:
        for key in ref_profile:
            assert ds.profile[key] == ref_profile[key]
        array = ds.read()
        for i in range(ds.count):
            assert np.all(array[i] == i + 1)


def test_overviews(user_base_image: BaseImage, region_25ha: Dict, tmp_path: pathlib.Path):
    """Test overviews get built by ``BaseImage.download()``."""
    filename = tmp_path.joinpath('test_user_download.tif')
    user_base_image.download(filename, region=region_25ha, crs='EPSG:3857', scale=1)
    assert filename.exists()
    with rio.open(filename, 'r') as ds:
        for bi in ds.indexes:
            assert len(ds.overviews(bi)) > 0
            assert ds.overviews(bi)[0] == 2


def test_metadata(s2_sr_hm_base_image: BaseImage, region_25ha: Dict, tmp_path: pathlib.Path):
    """Test metadata is written by ``BaseImage.download()``."""
    filename = tmp_path.joinpath('test_s2_band_subset_download.tif')
    s2_sr_hm_base_image.download(
        filename, region=region_25ha, crs='EPSG:3857', scale=60, bands=['B9']
    )
    assert filename.exists()
    with rio.open(filename, 'r') as ds:
        assert 'LICENSE' in ds.tags()
        assert len(ds.tags()['LICENSE']) > 0
        assert 'B9' in ds.descriptions
        band_dict = ds.tags(1)
        for key in ['gsd', 'name', 'description']:
            assert key in band_dict


@pytest.mark.parametrize('type', ExportType)
def test_start_export(type: ExportType, user_fix_base_image: BaseImage, region_25ha: Dict):
    """Test ``BaseImage.export()`` starts a small export."""
    # Note that this export should start successfully, but will ultimately fail for the asset and cloud options.  For
    # the asset option, there will be an overwrite issue.  For cloud storage, there is no 'geedim' bucket.
    task = user_fix_base_image.export(
        'test_export', type=type, folder='geedim', scale=30, region=region_25ha, wait=False
    )
    assert task.active()
    assert task.status()['state'] == 'READY' or task.status()['state'] == 'RUNNING'


def __test_export_asset(user_fix_base_image: BaseImage, region_25ha: Dict):
    """Test start of a small export to EE asset."""
    # TODO: consider removing this slow test in favour of the equivalent integration test
    filename = f'test_export_{np.random.randint(1<<31)}'
    folder = 'geedim'
    asset_id = utils.asset_id(filename, folder)
    # Note: to allow parallel tests exporting to assets, we use random asset names to prevent conflicts.  The test
    # must wait for export to complete and clean up after itself, otherwise lots of test assets will accumulate in my
    # geedim cloud project.

    try:
        # export and test asset exists
        task = user_fix_base_image.export(
            filename, type=ExportType.asset, folder=folder, scale=30, region=region_25ha, wait=True
        )
        assert task.status()['state'] == 'COMPLETED'
        assert ee.data.getAsset(asset_id) is not None
    finally:
        # delete asset
        try:
            ee.data.deleteAsset(asset_id)
        except ee.ee_exception.EEException:
            pass


def test_prepare_ee_geom(l9_base_image: BaseImage, tmp_path: pathlib.Path):
    """Test that ``BaseImage._prepare_for_export()`` works with an ee.Geometry region at native CRS and scale (Issue
    #6).
    """
    region = l9_base_image.ee_image.geometry()
    exp_image = BaseImageAccessor(l9_base_image.prepareForExport(region=region))
    assert exp_image.scale == l9_base_image.scale


@pytest.mark.parametrize(
    'base_image, exp_value',
    [
        ('s2_sr_hm_base_image', True),
        ('l9_base_image', True),
        ('modis_nbar_base_image_unbnd', False),
        ('user_base_image', False),
        ('user_fix_base_image', False),
    ],
)
def test_bounded(base_image: str, exp_value: bool, request: pytest.FixtureRequest):
    """Test ``BaseImage.bounded`` for different images."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    assert base_image.bounded == exp_value


# TODO:
#  - test float mask/nodata in downloaded image
