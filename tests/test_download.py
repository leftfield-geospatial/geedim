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

import copy
import pathlib
from datetime import datetime
from typing import Dict, Tuple, List

import ee
import numpy as np
import pytest
import rasterio as rio
from rasterio import Affine, windows
from rasterio import features, warp
from rasterio.crs import CRS

from geedim import utils
from geedim.download import BaseImage
from geedim.enums import ResamplingMethod, ExportType


class BaseImageLike:
    """Emulate BaseImage for _get_tile_shape() and tiles()."""

    def __init__(
        self, shape: Tuple[int, int], count: int = 10, dtype: str = 'uint16', transform: Affine = Affine.identity()
    ):
        self.shape = shape
        self.count = count
        self.dtype = dtype
        self.transform = transform
        dtype_size = np.dtype(dtype).itemsize
        self.size = shape[0] * shape[1] * count * dtype_size

    _tiles = BaseImage._tiles
    _get_tile_shape = BaseImage._get_tile_shape


@pytest.fixture(scope='session')
def user_base_image() -> BaseImage:
    """A BaseImage instance where the encapsulated image has no fixed projection or ID."""
    return BaseImage(ee.Image([1, 2, 3]))


@pytest.fixture(scope='session')
def user_fix_base_image() -> BaseImage:
    """
    A BaseImage instance where the encapsulated image has a fixed projection (EPSG:4326), a scale in degrees,
    and no footprint or ID.
    """
    return BaseImage(ee.Image([1, 2, 3]).reproject(crs='EPSG:4326', scale=30))


@pytest.fixture(scope='session')
def s2_sr_base_image(s2_sr_image_id: str) -> BaseImage:
    """A BaseImage instance encapsulating a Sentinel-2 image.  Covers `region_*ha`."""
    return BaseImage.from_id(s2_sr_image_id)


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
    return BaseImage(ee.Image(modis_nbar_image_id).clip(region_100ha))


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


def _test_export_profile(exp_profile, tgt_profile, transform_shape=False):
    """
    Test an export/download rasterio profile against a target rasterio profile.
    ``crs_transform`` specifies whether the export/download profile was generated by specifying crs_transform.
    """

    for key in ['dtype', 'count']:
        assert exp_profile[key] == tgt_profile[key]

    assert utils.rio_crs(exp_profile['crs']) == utils.rio_crs(tgt_profile['crs'])

    assert exp_profile['transform'][0] == tgt_profile['transform'][0]

    if transform_shape:
        for key in ['width', 'height']:
            assert exp_profile[key] == tgt_profile[key]
        assert exp_profile['transform'][:6] == pytest.approx(tgt_profile['transform'][:6], rel=1e-9)
    else:
        assert exp_profile['transform'][:6] == pytest.approx(tgt_profile['transform'][:6], rel=0.05)

    tgt_bounds = windows.bounds(
        windows.Window(0, 0, tgt_profile['width'], tgt_profile['height']), tgt_profile['transform']
    )
    exp_bounds = windows.bounds(
        windows.Window(0, 0, exp_profile['width'], exp_profile['height']), exp_profile['transform']
    )
    if transform_shape:
        assert exp_bounds == pytest.approx(tgt_bounds, rel=1e-9)
    else:
        assert exp_bounds == pytest.approx(tgt_bounds, rel=0.05)


def test_id_name(user_base_image: BaseImage, s2_sr_base_image: BaseImage):
    """Test `id` and `name` properties for different scenarios."""
    assert user_base_image.id is None
    assert user_base_image.name is None
    # check that BaseImage.from_id() sets id without a getInfo
    assert s2_sr_base_image._id is not None
    assert s2_sr_base_image.name == s2_sr_base_image.id.replace('/', '-')


def test_user_props(user_base_image: BaseImage):
    """Test non fixed projection image properties (other than id and has_fixed_projection)."""
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
    """Test fixed projection image properties (other than id and has_fixed_projection)."""
    assert user_fix_base_image.crs == 'EPSG:4326'
    assert user_fix_base_image.scale < 1
    assert user_fix_base_image.transform is not None
    assert user_fix_base_image.shape is None
    assert user_fix_base_image.date is None
    assert user_fix_base_image.size is None
    assert user_fix_base_image.footprint is None
    assert user_fix_base_image.dtype == 'uint8'
    assert user_fix_base_image.count == 3


def test_s2_props(s2_sr_base_image: BaseImage):
    """Test fixed projection S2 image properties (other than id and has_fixed_projection)."""
    min_band_info = s2_sr_base_image._ee_info['bands'][1]
    assert s2_sr_base_image.crs == min_band_info['crs']
    assert s2_sr_base_image.scale == min_band_info['crs_transform'][0]
    assert s2_sr_base_image.transform == Affine(*min_band_info['crs_transform'])
    assert s2_sr_base_image.shape == min_band_info['dimensions'][::-1]
    assert s2_sr_base_image.date == datetime.utcfromtimestamp(s2_sr_base_image.properties['system:time_start'] / 1000)
    assert s2_sr_base_image.size is not None
    assert s2_sr_base_image.footprint is not None
    assert s2_sr_base_image.footprint['type'] == 'Polygon'
    assert s2_sr_base_image.dtype == 'uint32'
    assert s2_sr_base_image.count == len(s2_sr_base_image._ee_info['bands'])


@pytest.mark.parametrize(
    'base_image', ['landsat_ndvi_base_image', 's2_sr_base_image', 'l9_base_image', 'modis_nbar_base_image']
)
def test_band_props(base_image: str, request: pytest.FixtureRequest):
    """Test `band_properties` completeness for generic/user/reflectance images."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    assert base_image.band_properties is not None
    assert [bd['name'] for bd in base_image.band_properties] == [bd['id'] for bd in base_image._ee_info['bands']]
    for key in ['gsd', 'description']:
        has_key = [key in bd for bd in base_image.band_properties]
        assert all(has_key)


def test_has_fixed_projection(user_base_image: BaseImage, user_fix_base_image: BaseImage, s2_sr_base_image):
    """Test the `has_fixed_projection` property."""
    assert not user_base_image.has_fixed_projection
    assert user_fix_base_image.has_fixed_projection
    assert s2_sr_base_image.has_fixed_projection


# yapf: disable
@pytest.mark.parametrize(
    'ee_data_type_list, exp_dtype', [
        ([{'precision': 'int', 'min': 10, 'max': 11}, {'precision': 'int', 'min': 100, 'max': 101}], 'uint8'),
        ([{'precision': 'int', 'min': -128, 'max': -100}, {'precision': 'int', 'min': 0, 'max': 127}], 'int8'),
        ([{'precision': 'int', 'min': 256, 'max': 257}], 'uint16'),
        ([{'precision': 'int', 'min': -32768, 'max': 32767}], 'int16'),
        ([{'precision': 'int', 'min': 2**15, 'max': 2**32 - 1}], 'uint32'),
        ([{'precision': 'int', 'min': -2**31, 'max': 2**31 - 1}], 'int32'),
        ([{'precision': 'float', 'min': 0., 'max': 1.e9}, {'precision': 'float', 'min': 0., 'max': 1.}], 'float32'),
        ([{'precision': 'int', 'min': 0., 'max': 2**31 - 1}, {'precision': 'float', 'min': 0., 'max': 1.}], 'float64'),
        ([{'precision': 'int', 'min': 0, 'max': 255}, {'precision': 'double', 'min': -1e100, 'max': 1e100}], 'float64'),
    ]
)
# yapf: enable
def test_min_dtype(ee_data_type_list: List, exp_dtype: str):
    """Test BasicImage.__get_min_dtype() with emulated EE info dicts."""
    ee_info = dict(bands=[])
    for ee_data_type in ee_data_type_list:
        ee_info['bands'].append(dict(data_type=ee_data_type))
    assert BaseImage._get_min_dtype(ee_info) == exp_dtype


def test_convert_dtype_error():
    """Test BaseImage.test_convert_dtype() raises an error with incorrect dtype."""
    with pytest.raises(TypeError):
        BaseImage._convert_dtype(ee.Image(1), dtype='unknown')


@pytest.mark.parametrize('size, exp_str', [(1024, '1.02 KB'), (234.56e6, '234.56 MB'), (1e9, '1.00 GB')])
def test_str_format_size(size: int, exp_str: str):
    """Test formatting of byte sizes as human readable strings."""
    assert BaseImage._str_format_size(size) == exp_str


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
)  # yapf: disable
def test_prepare_no_fixed_projection(user_base_image: BaseImage, params: Dict, request: pytest.FixtureRequest):
    """
    Test BaseImage._prepare_for_export raises an exception when the image has no fixed projection, and insufficient
    crs & region defining parameters are specified.
    """
    if 'region' in params:
        params['region'] = request.getfixturevalue(params['region'])
    with pytest.raises(ValueError) as ex:
        user_base_image._prepare_for_export(**params)
    assert 'does not have a fixed projection' in str(ex)


@pytest.mark.parametrize(
    'base_image, params',
    [
        ('user_fix_base_image', dict()),
        ('modis_nbar_base_image_unbnd', dict(crs='EPSG:4326')),
        ('modis_nbar_base_image_unbnd', dict(crs='EPSG:4326', scale=100)),
        ('modis_nbar_base_image_unbnd', dict(crs='EPSG:4326', crs_transform=Affine.identity())),
        ('modis_nbar_base_image_unbnd', dict(crs='EPSG:4326', shape=(100, 100))),
        ('modis_nbar_base_image_unbnd', dict(crs_transform=Affine.identity(), shape=(100, 100))),
    ],
)  # yapf: disable
def test_prepare_unbounded(base_image: BaseImage, params: Dict, request: pytest.FixtureRequest):
    """
    Test BaseImage._prepare_for_export raises an exception when the image is unbounded, and insufficient bounds
    defining parameters are specified.
    """
    base_image: BaseImage = request.getfixturevalue(base_image)
    with pytest.raises(ValueError) as ex:
        base_image._prepare_for_export(**params)
    assert 'This image is unbounded' in str(ex)


def test_prepare_exceptions(user_base_image: BaseImage, user_fix_base_image: BaseImage, region_25ha: Dict):
    """Test remaining BaseImage._prepare_for_export() error cases."""
    with pytest.raises(ValueError):
        # EPSG:4326 and no scale
        user_fix_base_image._prepare_for_export(region=region_25ha)
    with pytest.raises(ValueError):
        # no fixed projection and resample
        user_base_image._prepare_for_export(
            region=region_25ha, crs='EPSG:3857', scale=30, resampling=ResamplingMethod.bilinear
        )


@pytest.mark.parametrize(
    'base_image',
    ['s2_sr_base_image', 'l9_base_image', 'modis_nbar_base_image'],
)
def test_prepare_defaults(base_image: str, request: pytest.FixtureRequest):
    """Test BaseImage._prepare_for_export() with no (i.e. default) arguments with bounded images."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    exp_image = base_image._prepare_for_export()

    tgt_profile = base_image.profile
    _test_export_profile(exp_image.profile, tgt_profile, True)
    assert exp_image.scale == base_image.scale


@pytest.mark.parametrize(
    'base_image, param_image',
    [
        ('s2_sr_base_image', 's2_sr_base_image'),
        ('l9_base_image', 's2_sr_base_image'),
        ('modis_nbar_base_image', 's2_sr_base_image'),
        ('user_base_image', 's2_sr_base_image'),
        ('user_fix_base_image', 's2_sr_base_image'),
        ('l9_base_image', 'l9_base_image'),
        ('s2_sr_base_image', 'l9_base_image'),
        ('modis_nbar_base_image', 'l9_base_image'),
        ('user_base_image', 'l9_base_image'),
        ('user_fix_base_image', 'l9_base_image'),
        ('l9_base_image', 'modis_nbar_base_image'),
        ('s2_sr_base_image', 'modis_nbar_base_image'),
        ('modis_nbar_base_image', 'modis_nbar_base_image'),
        ('user_base_image', 'modis_nbar_base_image'),
        ('user_fix_base_image', 'modis_nbar_base_image'),
    ],  # yapf: disable
)
def test_prepare_transform_shape(base_image: str, param_image: str, request: pytest.FixtureRequest):
    """Test BaseImage._prepare_for_export() with crs_transform and shape parameters."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    param_image: BaseImage = request.getfixturevalue(param_image)

    exp_params = dict(
        crs=param_image.crs, crs_transform=param_image.transform, shape=param_image.shape, dtype=param_image.dtype
    )
    exp_image = base_image._prepare_for_export(**exp_params)

    tgt_profile = param_image.profile
    tgt_profile.update(count=base_image.count)

    _test_export_profile(exp_image.profile, tgt_profile, True)
    assert exp_image.scale == param_image.scale


@pytest.mark.parametrize(
    'base_image, param_image',
    [
        ('s2_sr_base_image', 's2_sr_base_image'),
        ('l9_base_image', 's2_sr_base_image'),
        ('modis_nbar_base_image', 's2_sr_base_image'),
        ('user_base_image', 's2_sr_base_image'),
        ('user_fix_base_image', 's2_sr_base_image'),
        ('l9_base_image', 'l9_base_image'),
        ('s2_sr_base_image', 'l9_base_image'),
        ('modis_nbar_base_image', 'l9_base_image'),
        ('user_base_image', 'l9_base_image'),
        ('user_fix_base_image', 'l9_base_image'),
        ('l9_base_image', 'modis_nbar_base_image'),
        ('s2_sr_base_image', 'modis_nbar_base_image'),
        ('modis_nbar_base_image', 'modis_nbar_base_image'),
        ('user_base_image', 'modis_nbar_base_image'),
        ('user_fix_base_image', 'modis_nbar_base_image'),
    ],
)  # yapf: disable
def test_prepare_region_scale(base_image: str, param_image: str, region_25ha: dict, request: pytest.FixtureRequest):
    """Test BaseImage._prepare_for_export() with region and scale parameters."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    _param_image: BaseImage = request.getfixturevalue(param_image)
    param_image = copy.deepcopy(_param_image)  # avoid changing session fixture
    # clip param_image, so that param_image.transform can be tested against below
    param_image.ee_image = param_image.ee_image.clip(region_25ha)

    exp_params = dict(
        crs=param_image.crs, region=param_image.footprint, scale=param_image.scale, dtype=param_image.dtype
    )
    exp_image = base_image._prepare_for_export(**exp_params)

    tgt_profile = param_image.profile
    tgt_profile.update(count=base_image.count)

    _test_export_profile(exp_image.profile, tgt_profile, base_image.id == param_image.id)
    assert exp_image.scale == param_image.scale

    # test export bounds contain target bounds
    exp_bounds = features.bounds(exp_image.footprint)
    tgt_bounds = features.bounds(param_image.footprint)
    tgt_footprint_crs = (
        param_image.footprint['crs']['properties']['name'] if 'crs' in param_image.footprint else 'EPSG:4326'
    )
    exp_crs = CRS.from_string(utils.rio_crs(exp_image.crs))
    tgt_bounds = warp.transform_bounds(tgt_footprint_crs, exp_crs, *tgt_bounds)
    assert (
        (exp_bounds[0] <= tgt_bounds[0])
        and (exp_bounds[1] <= tgt_bounds[1])
        and (exp_bounds[2] >= tgt_bounds[2])
        and (exp_bounds[3] >= tgt_bounds[3])
    )


@pytest.mark.parametrize(
    'base_image, bands',
    [
        ('s2_sr_base_image', ['B1', 'B5']),
        ('l9_base_image', ['SR_B4', 'SR_B3', 'SR_B2']),
    ],
)  # yapf: disable
def test_prepare_bands(base_image: str, bands: List[str], region_25ha: dict, request: pytest.FixtureRequest):
    """Test BaseImage._prepare_for_export() with bands parameter."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    param_image = BaseImage(base_image.ee_image.select(bands))

    exp_image = base_image._prepare_for_export(bands=bands)

    assert exp_image.count == len(bands)
    for attr in ['crs', 'transform', 'scale', 'shape', 'band_properties']:
        assert exp_image.__getattribute__(attr) == param_image.__getattribute__(attr)


def test_prepare_bands_error(s2_sr_base_image):
    """Test BaseImage._prepare_for_export() raises an error with incorrect bands."""
    with pytest.raises(ValueError):
        s2_sr_base_image._prepare_for_export(bands=['unknown'])


@pytest.mark.parametrize(
    'base_image',
    ['s2_sr_base_image', 'l9_base_image', 'modis_nbar_base_image'],
)
def test_prepare_for_download(base_image: str, request: pytest.FixtureRequest):
    """Test BaseImage._prepare_for_download() sets rasterio profile as expected."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    exp_image, exp_profile = base_image._prepare_for_download()
    tgt_profile = base_image.profile
    _test_export_profile(exp_profile, tgt_profile, True)
    assert exp_profile['nodata'] is not None


@pytest.mark.parametrize(
    'dtype, exp_nodata',
    [
        ('uint8', 0),
        ('int8', -(2**7)),
        ('uint16', 0),
        ('int16', -(2**15)),
        ('uint32', 0),
        ('int32', -(2**31)),
        ('float32', float('nan')),
        ('float64', float('nan')),
    ],
)
def test_prepare_nodata(user_fix_base_image: BaseImage, region_25ha: Dict, dtype: str, exp_nodata: float):
    """Test BaseImage._prepare_for_download() sets rasterio profile nodata correctly for different dtypes."""
    exp_image, exp_profile = user_fix_base_image._prepare_for_download(region=region_25ha, scale=30, dtype=dtype)
    assert exp_image.dtype == dtype
    if np.isnan(exp_profile['nodata']):
        assert np.isnan(exp_nodata)
    else:
        assert exp_profile['nodata'] == exp_nodata


@pytest.mark.parametrize(
    'src_image, dtype',
    [
        ('s2_sr_base_image', 'float32'),
        ('l9_base_image', None),
        ('modis_nbar_base_image', None),
    ],
)  # yapf: disable
def test_scale_offset(src_image: str, dtype: str, region_100ha: Dict, request: pytest.FixtureRequest):
    """Test BaseImage._prepare_for_export(scale_offset=True) gives expected properties and reflectance ranges."""

    src_image: BaseImage = request.getfixturevalue(src_image)
    exp_image = src_image._prepare_for_export(scale_offset=True)
    assert exp_image.crs == src_image.crs
    assert exp_image.scale == src_image.scale
    assert exp_image.band_properties == src_image.band_properties
    assert exp_image.dtype == dtype or 'float64'

    def get_min_max_refl(base_image: BaseImage) -> Dict:
        """Get the min & max of each reflectance band of base_image."""
        band_props = base_image.band_properties
        refl_bands = [
            bp['name'] for bp in band_props if ('center_wavelength' in bp) and (bp['center_wavelength'] < 1)
        ]  # yapf: disable
        ee_image = base_image.ee_image.select(refl_bands)
        min_max_dict = ee_image.reduceRegion(
            reducer=ee.Reducer.minMax(), geometry=region_100ha, bestEffort=True
        ).getInfo()  # yapf: disable
        min_dict = {k: v for k, v in min_max_dict.items() if 'min' in k}
        max_dict = {k: v for k, v in min_max_dict.items() if 'max' in k}
        return min_dict, max_dict

    # test the scaled and offset reflectance values lie between -0.5 and 1.5
    exp_min, exp_max = get_min_max_refl(exp_image)
    assert all(np.array(list(exp_min.values())) >= -0.5)
    assert all(np.array(list(exp_max.values())) <= 1.5)


def test_tile_shape():
    """Test BaseImage._get_tile_shape() satisfies the tile size limit for different image shapes."""
    max_tile_dim = 10000

    for max_tile_size in range(4, 32, 4):
        for height in range(1, 11000, 100):
            for width in range(1, 11000, 100):
                exp_shape = (height, width)
                exp_image = BaseImageLike(shape=exp_shape)  # emulate a BaseImage
                tile_shape, num_tiles = exp_image._get_tile_shape(
                    max_tile_size=max_tile_size, max_tile_dim=max_tile_dim
                )
                assert all(np.array(tile_shape) <= np.array(exp_shape))
                assert all(np.array(tile_shape) <= max_tile_dim)
                tile_image = BaseImageLike(shape=tile_shape)
                assert tile_image.size <= (max_tile_size << 20)


@pytest.mark.parametrize(
    'image_shape, tile_shape, image_transform',
    [
        ((1000, 500), (101, 101), Affine.identity()),
        ((1000, 100), (101, 101), Affine.scale(1.23)),
        ((1000, 102), (101, 101), Affine.scale(1.23) * Affine.translation(12, 34)),
    ],
)  # yapf: disable
def test_tiles(image_shape: Tuple, tile_shape: Tuple, image_transform: Affine):
    """Test continuity and coverage of tiles."""
    exp_image = BaseImageLike(shape=image_shape, transform=image_transform)
    tiles = [tile for tile in exp_image._tiles(tile_shape=tile_shape)]

    # test window coverage, and window & transform continuity
    prev_tile = tiles[0]
    accum_window = prev_tile.window
    for tile in tiles[1:]:
        accum_window = windows.union(accum_window, tile.window)
        assert all(np.array(tile._shape) <= np.array(tile_shape))

        if tile.window.row_off == prev_tile.window.row_off:
            assert tile.window.col_off == (prev_tile.window.col_off + prev_tile.window.width)
            assert tile._transform == pytest.approx(
                (prev_tile._transform * Affine.translation(prev_tile.window.width, 0)), rel=0.001
            )
        else:
            assert tile.window.row_off == (prev_tile.window.row_off + prev_tile.window.height)
            assert tile._transform == pytest.approx(
                prev_tile._transform * Affine.translation(-prev_tile.window.col_off, prev_tile.window.height), rel=0.001
            )
        prev_tile = tile
    assert (accum_window.height, accum_window.width) == exp_image.shape


def test_download_transform_shape(user_base_image: str, tmp_path: pathlib.Path, request: pytest.FixtureRequest):
    """Test download file properties and pixel data with crs_transform and shape arguments."""
    tgt_prof = dict(crs='EPSG:3857', transform=Affine(1, 0, 0, 0, -1, 0), width=10, height=10, dtype='uint16', count=3)

    # form download parameters from tgt_prof
    shape = (tgt_prof['height'], tgt_prof['width'])
    tgt_bounds = windows.bounds(windows.Window(0, 0, *shape[::-1]), tgt_prof['transform'])
    download_params = dict(
        crs=tgt_prof['crs'], crs_transform=tgt_prof['transform'], shape=shape, dtype=tgt_prof['dtype']
    )
    filename = tmp_path.joinpath('test.tif')
    user_base_image.download(filename, **download_params)
    assert filename.exists()
    with rio.open(filename, 'r') as ds:
        _test_export_profile(ds.profile, tgt_prof, 'crs_transform' in download_params)
        array = ds.read()
        for i in range(ds.count):
            assert np.all(array[i] == i + 1)


def test_download_region_scale(user_base_image: str, tmp_path: pathlib.Path, request: pytest.FixtureRequest):
    """Test download file properties and pixel data with region and scale arguments."""
    tgt_prof = dict(crs='EPSG:3857', transform=Affine(1, 0, 0, 0, -1, 0), width=10, height=10, dtype='uint16', count=3)

    # form download parameters from tgt_prof
    shape = (tgt_prof['height'], tgt_prof['width'])
    tgt_bounds = windows.bounds(windows.Window(0, 0, *shape[::-1]), tgt_prof['transform'])
    region = bounds_polygon(*tgt_bounds, crs=tgt_prof['crs'])
    download_params = dict(crs=tgt_prof['crs'], region=region, scale=tgt_prof['transform'][0], dtype=tgt_prof['dtype'])
    filename = tmp_path.joinpath('test.tif')
    user_base_image.download(filename, **download_params)
    assert filename.exists()
    with rio.open(filename, 'r') as ds:
        _test_export_profile(ds.profile, tgt_prof, 'crs_transform' in download_params)
        array = ds.read()
        for i in range(ds.count):
            assert np.all(array[i] == i + 1)


def test_overviews(user_base_image: BaseImage, region_25ha: Dict, tmp_path: pathlib.Path):
    """Test overviews get built on download."""
    filename = tmp_path.joinpath('test_user_download.tif')
    user_base_image.download(filename, region=region_25ha, crs='EPSG:3857', scale=1)
    assert filename.exists()
    with rio.open(filename, 'r') as ds:
        for band_i in range(ds.count):
            assert len(ds.overviews(band_i + 1)) > 0
            assert ds.overviews(band_i + 1)[0] == 2


def test_metadata(s2_sr_base_image: BaseImage, region_25ha: Dict, tmp_path: pathlib.Path):
    """Test metadata is written to a downloaded file."""
    filename = tmp_path.joinpath('test_s2_band_subset_download.tif')
    s2_sr_base_image.download(filename, region=region_25ha, crs='EPSG:3857', scale=60, bands=['B9'])
    assert filename.exists()
    with rio.open(filename, 'r') as ds:
        assert 'LICENSE' in ds.tags()
        assert len(ds.tags()['LICENSE']) > 0
        assert 'B9' in ds.descriptions
        band_dict = ds.tags(1)
        for key in ['gsd', 'name', 'description']:
            assert key in band_dict


@pytest.mark.parametrize('type', [ExportType.drive, ExportType.asset, ExportType.cloud])  # yapf: disable
def test_start_export(type, user_fix_base_image: BaseImage, region_25ha: Dict):
    """Test start of a small export."""
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


def test_download_bigtiff(s2_sr_base_image: BaseImage, tmp_path: pathlib.Path):
    """Test that BIGTIFF gets set in the profile of images larger than 4GB."""
    exp_image, profile = s2_sr_base_image._prepare_for_download()
    assert exp_image.size >= 4e9
    assert 'bigtiff' in profile
    assert profile['bigtiff']


def test_prepare_ee_geom(l9_base_image: BaseImage, tmp_path: pathlib.Path):
    """Test that _prepare_for_export works with an ee.Geometry region at native crs and scale (Issue #6)."""
    region = l9_base_image.ee_image.geometry()
    exp_image = l9_base_image._prepare_for_export(region=region)
    assert exp_image.scale == l9_base_image.scale


@pytest.mark.parametrize(
    'base_image, exp_value',
    [
        ('s2_sr_base_image', True),
        ('l9_base_image', True),
        ('modis_nbar_base_image_unbnd', False),
        ('user_base_image', False),
        ('user_fix_base_image', False),
    ],
)  # yapf: disable
def test_bounded(base_image: str, exp_value: bool, request: pytest.FixtureRequest):
    """Test BaseImage.bounded has correct value for different images."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    assert base_image.bounded == exp_value


# TODO:
# - export(): test an export of small file
# - different generic collection images are downloaded ok (perhaps this goes with MaskedImage more than BaseImage)
# - test float mask/nodata in downloaded image
# - test mult tile download has no discontinuities

##
