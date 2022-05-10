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


class BaseImageLike:
    """ Emulate BaseImage for _get_tile_shape() and tiles(). """

    def __init__(
        self, shape: Tuple[int, int], count: int = 10, dtype: str = 'uint16', transform: Affine = Affine.identity()
    ):
        self.shape = shape
        self.count = count
        self.dtype = dtype
        self.transform = transform
        dtype_size = np.dtype(dtype).itemsize
        self.size_in_bytes = shape[0] * shape[1] * count * dtype_size


@pytest.fixture(scope='session')
def user_base_image() -> BaseImage:
    """ A BaseImage instance where the encapsulated image has no fixed projection or ID.  """
    return BaseImage(ee.Image([1, 2, 3]))


@pytest.fixture(scope='session')
def user_fix_base_image() -> BaseImage:
    """
    A BaseImage instance where the encapsulated image has a fixed projection (EPSG:4326), a scale in degrees,
    and no footprint or ID.
    """
    return BaseImage(ee.Image([1, 2, 3]).reproject(crs='EPSG:4326', scale=30))


@pytest.fixture(scope='session')
def s2_base_image() -> BaseImage:
    """ A BaseImage instance encapsulating a Sentinel-2 image.  Covers `small_region`.  """
    return BaseImage.from_id('COPERNICUS/S2_SR/20220114T080159_20220114T082124_T35HKC')


@pytest.fixture(scope='session')
def l9_base_image() -> BaseImage:
    """ A BaseImage instance encapsulating a Landsat-9 image.  Covers `small_region`.  """
    return BaseImage.from_id('LANDSAT/LC09/C02/T1_L2/LC09_171084_20220427')

@pytest.fixture(scope='session')
def mnbar_base_image(small_region) -> BaseImage:
    """ A BaseImage instance encapsulating a MODIS NBAR image.  Covers `small_region`.  """
    return BaseImage(
        ee.Image('MODIS/006/MCD43A4/2022_01_01').clip(small_region).reproject(crs='EPSG:3857', scale=500)
    )



def test_properties(synth_fixed_ee_image: ee.Image, synth_fixed_ee_info: Dict, small_region: Dict):
    base_image = BaseImage(synth_fixed_ee_image)
    assert base_image.ee_info == synth_fixed_ee_info
    assert base_image.crs == 'EPSG:3857'
    assert base_image.scale == 30
    assert bounds(base_image.footprint) == bounds(small_region)
    assert base_image.id is None
    assert base_image.name is None
    assert base_image.dtype == 'uint8'
    assert base_image.has_fixed_projection
    band_info = synth_fixed_ee_info['bands'][0]
    assert base_image.shape == band_info['dimensions'][::-1]
    assert base_image.count == 3
    transform = Affine(*band_info['crs_transform']) * Affine.translation(*band_info['origin'])
    assert base_image.transform == transform


def test_download(synth_unfixed_ee_image: ee.Image, small_region: Dict, tmp_path: pathlib.Path):
    base_image = BaseImage(synth_unfixed_ee_image.reproject(crs='EPSG:3857', scale=30))
    filename = tmp_path.joinpath('synth.tif')
    download_args = dict(region=small_region, crs='EPSG:3857', scale=30)
    exp_image, profile = base_image._prepare_for_download(**download_args)

    base_image.download(filename, overwrite=True, region=small_region)
    assert filename.exists()
    with rio.open(filename, 'r') as ds:
        array = ds.read()
    assert array.shape == (exp_image.count, *exp_image.shape)
    assert array.dtype == np.dtype(exp_image.dtype)
    for i in range(3):
        assert np.all(array[i] == i + 1)


@pytest.mark.parametrize('id, exp_split', [('A/B/C', ('A/B', 'C')), ('ABC', ('', 'ABC')), (None, (None, None))])
def test_split_id(id, exp_split):
    """ Test split_id(). """
    assert split_id(id) == exp_split


def test_id_name(user_base_image: BaseImage, s2_base_image: BaseImage):
    """ Test `id` and `name` properties for different scenarios. """
    assert user_base_image.id is None
    assert user_base_image.name is None
    # check that BaseImage.from_id() sets id without a getInfo
    assert (s2_base_image.id is not None) and (s2_base_image._ee_info is None)
    assert s2_base_image.name == s2_base_image.id.replace('/', '-')


def test_user_props(user_base_image: BaseImage):
    """ Test non fixed projection image properties (other than id and has_fixed_projection). """
    assert user_base_image.crs is None
    assert user_base_image.scale is None
    assert user_base_image.transform is None
    assert user_base_image.shape is None
    assert user_base_image.size_in_bytes is None
    assert user_base_image.footprint is None
    assert user_base_image.dtype == 'uint8'
    assert user_base_image.count == 3


def test_fix_user_props(user_fix_base_image: BaseImage):
    """ Test fixed projection image properties (other than id and has_fixed_projection). """
    assert user_fix_base_image.crs == 'EPSG:4326'
    assert user_fix_base_image.scale < 1
    assert user_fix_base_image.transform is not None
    assert user_fix_base_image.shape is None
    assert user_fix_base_image.size_in_bytes is None
    assert user_fix_base_image.footprint is None
    assert user_fix_base_image.dtype == 'uint8'
    assert user_fix_base_image.count == 3


def test_s2_props(s2_base_image: BaseImage):
    """ Test fixed projection S2 image properties (other than id and has_fixed_projection). """
    min_band_info = s2_base_image.ee_info['bands'][1]
    assert s2_base_image.crs == min_band_info['crs']
    assert s2_base_image.scale == min_band_info['crs_transform'][0]
    assert s2_base_image.transform == Affine(*min_band_info['crs_transform'])
    assert s2_base_image.shape == min_band_info['dimensions'][::-1]
    assert s2_base_image.size_in_bytes is not None
    assert s2_base_image.footprint is not None
    assert s2_base_image.dtype == 'uint32'
    assert s2_base_image.count == len(s2_base_image.ee_info['bands'])


def test_has_fixed_projection(user_base_image: BaseImage, user_fix_base_image: BaseImage, s2_base_image: BaseImage):
    """ Test the `has_fixed_projection` property.  """
    assert not user_base_image.has_fixed_projection
    assert user_fix_base_image.has_fixed_projection
    assert s2_base_image.has_fixed_projection


@pytest.mark.parametrize(
    'ee_data_type_list, exp_dtype', [
        ([{'precision': 'int', 'min': 1, 'max': 100},
          {'precision': 'int', 'min': 0, 'max': 255}], 'uint8'),
        ([{'precision': 'int', 'min': 1, 'max': 255},
          {'precision': 'int', 'min': -255, 'max': -1}], 'int16'),
        ([{'precision': 'int', 'min': 0, 'max': 1},
          {'precision': 'int', 'min': 1, 'max': 2 << 32}], 'uint32'),
        ([{'precision': 'float', 'min': 0, 'max': 2 << 32},
          {'precision': 'float', 'min': 0, 'max': 1}], 'float32'),
        ([{'precision': 'int', 'min': 0, 'max': 2 << 32},
          {'precision': 'double', 'min': 0, 'max': 1}], 'float64'),
    ]
)
def test_min_dtype(ee_data_type_list, exp_dtype):
    """ Test BasicImage.__get_min_dtype() with emulated EE info dicts.  """
    ee_info = dict(bands=[])
    for ee_data_type in ee_data_type_list:
        ee_info['bands'].append(dict(data_type=ee_data_type))
    assert BaseImage._get_min_dtype(ee_info) == exp_dtype


@pytest.mark.parametrize('size, exp_str', [(1024, '1.02 KB'), (234.56e6, '234.56 MB'), (1e9, '1.00 GB')])
def test_str_format_size(size, exp_str):
    """ Test formatting of byte sizes as human readable strings. """
    assert BaseImage._str_format_size(size) == exp_str


def test_prepare_exceptions(user_base_image: BaseImage, user_fix_base_image: BaseImage, small_region: Dict):
    """ Test BaseImage._prepare_for_export() error cases. """
    with pytest.raises(ValueError):
        # no fixed projection and no region / crs / scale
        user_base_image._prepare_for_export()
    with pytest.raises(ValueError):
        # no footprint or region
        user_fix_base_image._prepare_for_export()
    with pytest.raises(ValueError):
        # EPSG:4326 and no scale
        user_fix_base_image._prepare_for_export(region=small_region)
    with pytest.raises(ValueError):
        # export in 'SR-ORG:6974'
        user_fix_base_image._prepare_for_export(region=small_region, crs='SR-ORG:6974', scale=30)
    with pytest.raises(ValueError):
        # no fixed projection and resample
        user_base_image._prepare_for_export(
            region=small_region, crs='EPSG:3857', scale=30, resampling=ResamplingMethod.bilinear
        )


@pytest.mark.parametrize(
    'src_image, tgt_image', [
        ('s2_base_image', 's2_base_image'),
        ('l9_base_image', 's2_base_image'),
        ('mnbar_base_image', 's2_base_image'),
        ('user_base_image', 's2_base_image'),
        ('user_fix_base_image', 's2_base_image'),
        ('l9_base_image', 'l9_base_image'),
        ('s2_base_image', 'l9_base_image'),
        ('mnbar_base_image', 'l9_base_image'),
        ('user_base_image', 'l9_base_image'),
        ('user_fix_base_image', 'l9_base_image'),
        ('l9_base_image', 'mnbar_base_image'),
        ('s2_base_image', 'mnbar_base_image'),
        ('mnbar_base_image', 'mnbar_base_image'),
        ('user_base_image', 'mnbar_base_image'),
        ('user_fix_base_image', 'mnbar_base_image'),
    ]
)
def test_prepare_for_export(src_image: str, tgt_image: str, request):
    """ Test BaseImage._prepare_for_export() sets properties of export image as expected.  """
    src_image: BaseImage = request.getfixturevalue(src_image)
    tgt_image: BaseImage = request.getfixturevalue(tgt_image)
    if tgt_image == src_image:
        exp_image = src_image._prepare_for_export()
    else:
        exp_image = src_image._prepare_for_export(
            crs=tgt_image.crs, scale=tgt_image.scale, region=tgt_image.footprint, dtype=tgt_image.dtype
        )

    assert exp_image.crs == tgt_image.crs
    assert exp_image.scale == tgt_image.scale
    assert exp_image.count == src_image.count
    assert exp_image.dtype == tgt_image.dtype

    # Note that exp_image = ee.Image.prepare_for_export(<tgt_image properties>) resamples and can give an exp_image
    # with different grid and shape compared to tgt_image.  So just test the region here, as that is what is passed
    # to EE.
    exp_region = transform_geom(exp_image.footprint['crs']['properties']['name'], 'EPSG:4326', exp_image.footprint)
    exp_bounds = bounds(exp_region)
    tgt_bounds = bounds(tgt_image.footprint)
    assert exp_bounds == pytest.approx(tgt_bounds, rel=.05)
    # test exp_bounds contain tgt_bounds
    assert (
        (exp_bounds[0] <= tgt_bounds[0]) and (exp_bounds[1] <= tgt_bounds[1]) and
        (exp_bounds[2] >= tgt_bounds[2]) and (exp_bounds[3] >= tgt_bounds[3])
    )


@pytest.mark.parametrize(
    'src_image, tgt_image', [
        ('s2_base_image', 's2_base_image'),
        ('user_base_image', 's2_base_image'),
    ]
)
def test_prepare_for_download(src_image: str, tgt_image: str, small_region, request):
    """ Test BaseImage._prepare_for_download() sets rasterio profile as expected.  """
    # TODO: I suspect this will be duplicated elsewhere and can be removed
    src_image: BaseImage = request.getfixturevalue(src_image)
    tgt_image: BaseImage = request.getfixturevalue(tgt_image)
    if tgt_image == src_image:
        exp_image, exp_profile = src_image._prepare_for_download()
    else:
        exp_image, exp_profile = src_image._prepare_for_download(
            crs=tgt_image.crs, scale=tgt_image.scale, region=small_region, dtype=tgt_image.dtype
        )
    assert f'EPSG:{exp_profile["crs"].to_epsg()}' == tgt_image.crs
    assert exp_profile['count'] == src_image.count
    assert exp_profile['dtype'] == tgt_image.dtype
    assert exp_profile['nodata'] is not None


@pytest.mark.parametrize(
    'dtype, exp_nodata', [
        ('uint8', 0), ('int8', -2 ** 7), ('uint16', 0), ('int16', -2 ** 15), ('uint32', 0), ('int32', -2 ** 31),
        ('float32', float('nan')), ('float64', float('nan'))
    ]
)
def test_prepare_nodata(user_fix_base_image, small_region, dtype, exp_nodata, request):
    """ Test BaseImage._prepare_for_download() sets rasterio profile nodata correctly for different dtypes.  """
    exp_image, exp_profile = user_fix_base_image._prepare_for_download(region=small_region, scale=30, dtype=dtype)
    assert exp_image.dtype == dtype
    if np.isnan(exp_profile['nodata']):
        assert np.isnan(exp_nodata)
    else:
        assert exp_profile['nodata'] == exp_nodata


def test_tile_shape():
    """ Test BaseImage._get_tile_shape() statisfies the EE download limit for different image shapes. """
    max_download_size = 32 << 20
    max_grid_dimension = 10000

    for height in range(1, 11000, 100):
        for width in range(1, 11000, 100):
            exp_shape = (height, width)
            exp_image = BaseImageLike(shape=exp_shape)  # emulate a BaseImage
            tile_shape, num_tiles = BaseImage._get_tile_shape(exp_image)
            assert all(np.array(tile_shape) <= np.array(exp_shape))
            assert all(np.array(tile_shape) <= max_grid_dimension)
            tile_image = BaseImageLike(shape=tile_shape)
            assert tile_image.size_in_bytes <= max_download_size


@pytest.mark.parametrize(
    'image_shape, tile_shape, image_transform', [
        ((1000, 500), (101, 101), Affine.identity()),
        ((1000, 100), (101, 101), Affine.scale(1.23)),
        ((1000, 102), (101, 101), Affine.scale(1.23) * Affine.translation(12, 34)),
    ]
)
def test_tiles(image_shape, tile_shape, image_transform):
    """ Test continuity and coverage of tiles. """
    exp_image = BaseImageLike(shape=image_shape, transform=image_transform)
    tiles = [tile for tile in BaseImage.tiles(exp_image, tile_shape=tile_shape)]

    # test window coverage, and window & transform continuity
    prev_tile = tiles[0]
    accum_window = prev_tile.window
    for tile in tiles[1:]:
        accum_window = union(accum_window, tile.window)
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


@pytest.mark.parametrize(
    'base_image, region', [
        ('user_base_image', 'small_region'),
        ('user_fix_base_image', 'small_region'),
        # ('s2_base_image', 'small_region'),
        # ('l9_base_image', 'small_region'),
        # ('mnbar_base_image', 'small_region'),
    ]
)
def test_download(base_image, region, tmp_path, request):
    base_image = request.getfixturevalue(base_image)
    region = request.getfixturevalue(region)
    filename = tmp_path.joinpath('test_user_download.tif')
    crs = 'EPSG:3857'
    dtype = 'uint16'
    scale = 30
    base_image.download(filename, region=region, crs=crs, scale=scale, dtype=dtype)
    exp_region = transform_geom('EPSG:4326', crs, region)
    exp_bounds = BoundingBox(*bounds(exp_region))
    assert filename.exists()
    with rio.open(filename, 'r') as ds:
        assert ds.count == base_image.count
        assert ds.dtypes[0] == dtype
        assert ds.nodata == 0
        assert abs(ds.transform[0]) == scale
        assert ds.transform.xoff <= exp_bounds.left
        assert ds.transform.yoff >= exp_bounds.top  # 'EPSG:3857' has y -ve
        ds_bounds = ds.bounds
        assert (
            (ds_bounds[0] <= exp_bounds[0]) and (ds_bounds[1] <= exp_bounds[1]) and
            (ds_bounds[2] >= exp_bounds[2]) and (ds_bounds[3] >= exp_bounds[3])
        )
        if ds.count < 4:
            array = ds.read()
            for i in range(ds.count):
                assert np.all(array[i] == i + 1)

# TO test
# --------
# split_id()
# get_bounds() # test this on downloaded image against download region
# from_id has _id set
# user image has no id, ee image does & name
# unfixed & fixed image properties i.e.:  has_fixed_projection (scale, crs, shape, transform ?) effectively,
# this is testing _get_projection
# scale, shape, transform  on s2 image match band 1 info from ee_info
# other properties (incl in one of the above?): count, x size_in_bytes
# footprint: fixed and unfixed & clipped images.  can be combined with other fixed / unfixed above?

# dtype & _get_min_dtype on mock-up / known ee_info dicts
# _str_format_size on a few sizes
# _convert_dtype raises TypeError (we should test in some other place the dtype of a downloaded image,
# and/or of exp_image)
# _get_band_metadata: leave for now
# _prepare_for_export exceptions:
#   - raises ValueError with non fixed projection image and no crs / scale / region
#   - raises ValueError with no footprint and no region (can this occur in a non non fixed proj situation?)
#   - raise ValueError crs = EPSG:4326 and no scale
#   - raise ValueError export MODIS with no crs, or export to 'SR-ORG:6974'
#   - raise ValueError resampling and non fixed proj image
# _prepare_for_export return BaseImage: check relevant props of returned image match args to _prepare_for_export both
# default and non-default cases
# _prepare_for_download: test profile matches params similar to above
# _get_tile_shape: test with mockup BaseImages that steadily / randomly increase image shape and/or count,
# and tests that the tile size does not increase past max_size
# _build_overviews: test downloaded file has overviews.  as part of another test.
# _write_metadata: test downloaded bands have ids / descriptions
# tiles: use mockup BaseImage to test tiles are contiguous and cover the image properly / exactly.  both in terms of
#  pixel and proj co-ords.
# export(): test an export of small file (with wait ? - it kind of has to be to test monitor_export_task() )
# download(): test a download of small file(s) but with >1 tiles.  perhaps make it a synthetic file so we can test
#   tiling easily i.e. no blank lines between tiles, and tiles align correctly.  or somehow test download of same image
#   but as A one tile, and B many tiles, and then compare the 2 images.  Test the shape, transform, dtype etc of the
#   downloaded image against the exp_image.  Perhaps we have to make the exp_image ourselves with prepare_for_export

# Other to test:
# - nodata for different data types (in profile?)
# - resampling smooths things out
# - different generic collection images are downloaded ok (perhaps this goes with MaskedImage more than BaseImage)
# - test float mask/nodata in downloaded image

# (a thought - if we make per session BaseImage objects, perhaps we can use them in above tests and avoid repeat
# getInfo calls)
##
