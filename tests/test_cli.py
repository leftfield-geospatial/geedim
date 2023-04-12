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
import json
import pathlib
from datetime import datetime
from glob import glob
from typing import List, Dict, Tuple

import ee
import numpy as np
import pytest
import rasterio as rio
from click.testing import CliRunner
from geedim.cli import cli
from geedim.utils import root_path, asset_id
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.features import bounds
from rasterio.warp import transform_geom
from rasterio.transform import Affine


@pytest.fixture()
def l4_5_image_id_list(l4_image_id, l5_image_id) -> List[str]:
    """ A list of landsat 4 & 5 image ID's. """
    return [l4_image_id, l5_image_id]


@pytest.fixture()
def l8_9_image_id_list(l8_image_id, l9_image_id) -> List[str]:
    """ A list of landsat 8 & 9 image ID's. """
    return [l8_image_id, l9_image_id]


@pytest.fixture()
def s2_sr_image_id_list() -> List[str]:
    """ A list of Sentinel-2 SR image IDs. """
    return [
        'COPERNICUS/S2_SR/20211004T080801_20211004T083709_T34HEJ',
        'COPERNICUS/S2_SR/20211123T081241_20211123T083704_T34HEJ',
        'COPERNICUS/S2_SR/20220107T081229_20220107T083059_T34HEJ'
    ]


@pytest.fixture()
def gedi_image_id_list() -> List[str]:
    """ A list of GEDI canopy top height ID's. """
    return [
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202008_018E_036S', 'LARSE/GEDI/GEDI02_A_002_MONTHLY/202009_018E_036S',
        'LARSE/GEDI/GEDI02_A_002_MONTHLY/202005_018E_036S'
    ]


def _test_downloaded_file(
    filename: pathlib.Path, region: Dict = None, crs: str = None, scale: float = None, dtype: str = None,
    bands: List[str] = None, scale_offset: bool = None, transform: Affine = None, shape: Tuple[int, int] = None
):
    """ Helper function to test image file format against given parameters. """
    with rio.open(filename, 'r') as ds:
        ds: rio.DatasetReader = ds
        assert ds.nodata is not None
        array = ds.read(masked=True)
        am = array.mean()
        assert np.isfinite(am) and (am != 0)
        if region:
            exp_region = transform_geom('EPSG:4326', ds.crs, region)
            exp_bounds = BoundingBox(*bounds(exp_region))
            assert (
                (ds.bounds[0] <= exp_bounds[0]) and (ds.bounds[1] <= exp_bounds[1]) and
                (ds.bounds[2] >= exp_bounds[2]) and (ds.bounds[3] >= exp_bounds[3])
            )
        if crs:
            assert CRS(ds.crs) == CRS.from_string(crs)
        if scale:
            assert abs(ds.transform[0]) == scale
        if dtype:
            assert ds.dtypes[0] == dtype
        if scale_offset:
            refl_bands = [
                i for i in range(1, ds.count + 1)
                if ('center_wavelength' in ds.tags(i)) and (float(ds.tags(i)['center_wavelength']) < 1)
            ]
            array = ds.read(refl_bands, masked=True)
            assert all(array.min(axis=(1, 2)) >= -0.5)
            assert all(array.max(axis=(1, 2)) <= 1.5)
        if transform:
            assert ds.transform[:6] == transform[:6]
        if shape:
            assert ds.shape == tuple(shape)
        if bands:
            assert set(bands) == set(ds.descriptions)
            assert set(bands) ==  set([ds.tags(bi)['name'] for bi in range(1, ds.count + 1)])


@pytest.mark.parametrize(
    'name, start_date, end_date, region, fill_portion, cloudless_portion, is_csmask', [
        ('LANDSAT/LC09/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha_file', 10, 50, True),
        ('LANDSAT/LE07/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha_file', 0, 0, True),
        ('LANDSAT/LT05/C02/T1_L2', '2005-01-01', '2006-02-01', 'region_100ha_file', 40, 50, True),
        ('COPERNICUS/S2_SR', '2022-01-01', '2022-01-15', 'region_100ha_file', 0, 50, True),
        ('COPERNICUS/S2', '2022-01-01', '2022-01-15', 'region_100ha_file', 50, 40, True),
        ('COPERNICUS/S2_SR_HARMONIZED', '2022-01-01', '2022-01-15', 'region_100ha_file', 0, 50, True),
        ('COPERNICUS/S2_HARMONIZED', '2022-01-01', '2022-01-15', 'region_100ha_file', 50, 40, True),
        ('LARSE/GEDI/GEDI02_A_002_MONTHLY', '2021-11-01', '2022-01-01', 'region_100ha_file', 1, 0, False)
    ]
)
def test_search(
    name, start_date: str, end_date: str, region: str, fill_portion: float, cloudless_portion: float, is_csmask: bool,
    tmp_path: pathlib.Path, runner: CliRunner, request: pytest.FixtureRequest
):
    """
    Test search command gives valid results for different cloud/shadow maskable, and generic collections.
    """
    region_file: Dict = request.getfixturevalue(region)
    results_file = tmp_path.joinpath('search_results.json')
    cli_str = (
        f'search -c {name} -s {start_date} -e {end_date} -r {region_file} -fp {fill_portion} '
        f'-cp {cloudless_portion} -op {results_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (results_file.exists())
    with open(results_file, 'r') as f:
        properties = json.load(f)

    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    im_dates = np.array(
        [datetime.utcfromtimestamp(im_props['system:time_start'] / 1000) for im_props in properties.values()]
    )
    # test FILL_PORTION in expected range
    im_fill_portions = np.array([im_props['FILL_PORTION'] for im_props in properties.values()])
    assert np.all(im_fill_portions >= fill_portion) and np.all(im_fill_portions <= 100)
    if is_csmask:  # is a cloud/shadow masked collection
        # test CLOUDLESS_PORTION in expected range
        im_cl_portions = np.array([im_props['CLOUDLESS_PORTION'] for im_props in properties.values()])
        assert np.all(im_cl_portions >= cloudless_portion) and np.all(im_cl_portions <= 100)
    # test search result image dates lie between `start_date` and `end_date`
    assert np.all(im_dates >= start_date) and np.all(im_dates < end_date)
    # test search result image dates are sorted
    assert np.all(sorted(im_dates) == im_dates)


def test_config_search_s2(region_10000ha_file: pathlib.Path, runner: CliRunner, tmp_path: pathlib.Path):
    """ Test `config` sub-command chained with `search` of Sentinel-2 affects CLOUDLESS_PORTION as expected. """
    results_file = tmp_path.joinpath('search_results.json')
    name = 'COPERNICUS/S2_SR'
    cl_portion_list = []
    for prob in [40, 80]:
        cli_str = (
            f'config --prob {prob} search -c {name} -s 2022-01-01 -e 2022-02-01 -r {region_10000ha_file} -fp -op '
            f'{results_file}'
        )
        result = runner.invoke(cli, cli_str.split())
        assert (result.exit_code == 0)
        assert (results_file.exists())
        with open(results_file, 'r') as f:
            properties = json.load(f)
        cl_portion_list.append(np.array([prop_dict['CLOUDLESS_PORTION'] for prop_dict in properties.values()]))

    assert np.any(cl_portion_list[0] < cl_portion_list[1])
    assert not np.any(cl_portion_list[0] > cl_portion_list[1])


def test_config_search_l9(region_10000ha_file: pathlib.Path, runner: CliRunner, tmp_path: pathlib.Path):
    """ Test `config` sub-command chained with `search` of Landsat-9 affects CLOUDLESS_PORTION as expected. """
    results_file = tmp_path.joinpath('search_results.json')
    name = 'LANDSAT/LC09/C02/T1_L2'
    cl_portion_list = []
    for param in ['--mask-shadows', '--no-mask-shadows']:
        cli_str = (
            f'config {param} search -c {name} -s 2022-02-15 -e 2022-04-01 -r {region_10000ha_file} -fp -op'
            f' {results_file}'
        )
        result = runner.invoke(cli, cli_str.split())
        assert (result.exit_code == 0)
        assert (results_file.exists())
        with open(results_file, 'r') as f:
            properties = json.load(f)
        cl_portion_list.append(np.array([prop_dict['CLOUDLESS_PORTION'] for prop_dict in properties.values()]))

    assert np.any(cl_portion_list[0] < cl_portion_list[1])
    assert not np.any(cl_portion_list[0] > cl_portion_list[1])


def test_region_bbox_search(region_100ha_file: pathlib.Path, runner: CliRunner, tmp_path: pathlib.Path):
    """ Test --bbox gives same search results as --region <geojson file>. """

    results_file = tmp_path.joinpath('search_results.json')
    with open(region_100ha_file, 'r') as f:
        region = json.load(f)
    bbox = bounds(region)
    bbox_str = ' '.join([str(b) for b in bbox])
    cli_strs = [
        f'search -c LANDSAT/LC09/C02/T1_L2 -s 2022-01-01 -e 2022-02-01 -r {region_100ha_file} -op {results_file}',
        f'search -c LANDSAT/LC09/C02/T1_L2 -s 2022-01-01 -e 2022-02-01 -b {bbox_str} -op {results_file}'
    ]

    props_list = []
    for cli_str in cli_strs:
        result = runner.invoke(cli, cli_str.split())
        assert (result.exit_code == 0)
        assert (results_file.exists())
        with open(results_file, 'r') as f:
            properties = json.load(f)
        props_list.append(properties)

    assert props_list[0] == props_list[1]


def test_raster_region_search(const_image_25ha_file, region_25ha_file, runner: CliRunner, tmp_path: pathlib.Path):
    """ Test --region works with a raster file. """

    results_file = tmp_path.joinpath('search_results.json')
    cli_strs = [
        f'search -c LANDSAT/LC09/C02/T1_L2 -s 2022-01-01 -e 2022-02-01 -r {region_25ha_file} -op {results_file}',
        f'search -c LANDSAT/LC09/C02/T1_L2 -s 2022-01-01 -e 2022-02-01 -r {const_image_25ha_file} -op {results_file}'
    ]

    props_list = []
    for cli_str in cli_strs:
        result = runner.invoke(cli, cli_str.split())
        assert (result.exit_code == 0)
        assert (results_file.exists())
        with open(results_file, 'r') as f:
            properties = json.load(f)
        props_list.append(properties)

    assert props_list[0].keys() == props_list[1].keys()


def test_search_add_props_l9(region_25ha_file: pathlib.Path, runner: CliRunner, tmp_path: pathlib.Path):
    """ Test --add-property generates results with the additional property keys. """
    results_file = tmp_path.joinpath('search_results.json')
    name = 'LANDSAT/LC09/C02/T1_L2'
    add_props = ['CLOUD_COVER', 'GEOMETRIC_RMSE_VERIFY']
    add_props_str = ''.join([f' -ap {add_prop} ' for add_prop in add_props])
    cli_str = (
        f'search -c {name} -s 2022-01-01 -e 2022-02-01 -r {region_25ha_file} {add_props_str} -op {results_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (results_file.exists())
    with open(results_file, 'r') as f:
        properties = json.load(f)
    prop_keys = list(properties.values())[0].keys()
    assert all([add_prop in prop_keys for add_prop in add_props])
    assert all([add_prop in result.output for add_prop in add_props])


def test_search_custom_filter_l9(region_25ha_file: pathlib.Path, runner: CliRunner, tmp_path: pathlib.Path):
    """ Test --custom-filter filters the search results as specified. """
    results_file = tmp_path.joinpath('search_results.json')
    name = 'LANDSAT/LC09/C02/T1_L2'
    cc_thresh = 30
    add_prop = 'CLOUD_COVER'
    cli_str = (
        f'search -c {name} -s 2022-01-01 -e 2022-04-01 -r {region_25ha_file} -ap {add_prop} '
        f'-cf {add_prop}>{cc_thresh} -op {results_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (results_file.exists())
    with open(results_file, 'r') as f:
        properties = json.load(f)
    prop_keys = list(properties.values())[0].keys()
    assert add_prop in prop_keys
    assert all([prop[add_prop] > cc_thresh for prop in properties.values()])


def test_search_cloudless_portion_no_value(
    region_25ha_file: pathlib.Path, runner: CliRunner, tmp_path: pathlib.Path
):
    """ Test `search --cloudless-portion` gives the same results as `search --cloudless-portion 0`. """
    results_file = tmp_path.joinpath(f'search_results.json')
    name = 'LANDSAT/LC09/C02/T1_L2'
    clp_list = []
    for post_fix, cp_spec in zip(['no_val', 'zero_val'], ['-cp', '-cp 0']):
        cli_str = (
            f'search -c {name} -s 2022-01-01 -e 2022-04-01 -r {region_25ha_file} {cp_spec} -op {results_file}'
        )
        result = runner.invoke(cli, cli_str.split())
        assert (result.exit_code == 0)
        assert (results_file.exists())
        with open(results_file, 'r') as f:
            props = json.load(f)
        prop_keys = list(props.values())[0].keys()
        assert 'FILL_PORTION' in prop_keys
        assert 'CLOUDLESS_PORTION' in prop_keys
        clp = np.array([prop['CLOUDLESS_PORTION'] for prop in props.values()])
        clp_list.append(clp)
    assert np.all(clp_list[0] == clp_list[1])


@pytest.mark.parametrize(
    'image_id, region_file', [
        ('l8_image_id', 'region_25ha_file'),
        ('s2_sr_hm_image_id', 'region_25ha_file'),
        ('gedi_cth_image_id', 'region_25ha_file'),
        ('modis_nbar_image_id', 'region_25ha_file'),
    ]
)  # yapf: disable
def test_download_region_defaults(
    image_id: str, region_file: pathlib.Path, tmp_path: pathlib.Path, runner: CliRunner, request
):
    """ Test image download with default crs, scale, dtype etc.  """
    image_id = request.getfixturevalue(image_id)
    region_file = request.getfixturevalue(region_file)
    out_file = tmp_path.joinpath(image_id.replace('/', '-') + '.tif')

    cli_str = f'download -i {image_id} -r {region_file} -dd {tmp_path}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (out_file.exists())

    # test downloaded file readability and format
    with open(region_file) as f:
        region = json.load(f)
    _test_downloaded_file(out_file, region=region)


@pytest.mark.parametrize(
    'image_id, region_file', [
        ('l8_image_id', 'region_25ha_file'),
        ('s2_sr_hm_image_id', 'region_25ha_file'),
        ('gedi_cth_image_id', 'region_25ha_file'),
    ]
)  # yapf: disable
def test_download_crs_transform(
    image_id: str, region_file: pathlib.Path, tmp_path: pathlib.Path, runner: CliRunner, request
):
    """ Test image download with crs, crs_transform, & shape specified. """
    image_id = request.getfixturevalue(image_id)
    region_file = request.getfixturevalue(region_file)
    out_file = tmp_path.joinpath(image_id.replace('/', '-') + '.tif')

    # find a transform and shape for region_file
    with open(region_file) as f:
        region = json.load(f)
    region_bounds = bounds(region)
    crs = 'EPSG:4326'
    shape = (11, 12)
    shape_str = ' '.join(map(str, shape))
    crs_transform = rio.transform.from_bounds(*region_bounds, *shape[::-1])
    crs_transform_str = ' '.join(map(str, crs_transform[:6]))

    # run the download
    cli_str = f'download -i {image_id} -c {crs} -ct {crs_transform_str} -sh {shape_str} -dd {tmp_path}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (out_file.exists())

    # test downloaded file readability and format
    _test_downloaded_file(out_file, crs=crs, region=region, transform=crs_transform, shape=shape)


def test_download_like(
    l8_image_id: str, s2_sr_image_id: str, region_25ha_file: pathlib.Path, tmp_path: pathlib.Path, runner: CliRunner
):
    """ Test image download using --like. """
    l8_file = tmp_path.joinpath(l8_image_id.replace('/', '-') + '.tif')
    s2_file = tmp_path.joinpath(s2_sr_image_id.replace('/', '-') + '.tif')

    # download the landsat 8 image to be the template
    l8_cli_str = f'download -i {l8_image_id} -r {region_25ha_file} -dd {tmp_path}'
    result = runner.invoke(cli, l8_cli_str.split())
    assert (result.exit_code == 0)
    assert (l8_file.exists())

    # download the sentinel 2 image like the landsat 8 image
    s2_cli_str = f'download -i {s2_sr_image_id} --like {l8_file} -dd {tmp_path}'
    result = runner.invoke(cli, s2_cli_str.split())
    assert (result.exit_code == 0)
    assert (s2_file.exists())

    # test the landsat 8 image is 'like' the sentinel 2 image
    with rio.open(l8_file) as l8_im, rio.open(s2_file, 'r') as s2_im:
        assert l8_im.crs == s2_im.crs
        assert l8_im.shape == s2_im.shape
        assert l8_im.transform[:6] == s2_im.transform[:6]


@pytest.mark.parametrize(
    'image_id, region_file, crs, scale, dtype, bands, mask, resampling, scale_offset, max_tile_size, max_tile_dim', [
        ('l5_image_id', 'region_25ha_file', 'EPSG:3857', 30, 'uint16', None, False, 'near', False, 16, 10000),
        ('l9_image_id', 'region_25ha_file', 'EPSG:3857', 30, 'float32', None, False, 'near', True, 32, 10000),
        (
            's2_toa_image_id', 'region_25ha_file', 'EPSG:3857', 10, 'float64', ['B5', 'B9'], True, 'bilinear', True, 32,
            10000
        ),
        ('modis_nbar_image_id', 'region_100ha_file', 'EPSG:3857', 500, 'int32', None, False, 'bicubic', False, 4, 100),
        (
            'gedi_cth_image_id', 'region_25ha_file', 'EPSG:3857', 10, 'float32', ['rh99'], True, 'bilinear', False, 32,
            10000
        ),
        ('landsat_ndvi_image_id', 'region_25ha_file', 'EPSG:3857', 30, 'float64', None, True, 'near', False, 32, 10000),
    ]
) # yapf: disable
def test_download_params(
    image_id: str, region_file: str, crs: str, scale: float, dtype: str, bands: List[str], mask: bool, resampling: str,
    scale_offset: bool, max_tile_size: float, max_tile_dim: int, tmp_path: pathlib.Path, runner: CliRunner,
    request: pytest.FixtureRequest
):
    """ Test image download, specifying all cli params except crs_transform and shape. """
    image_id = request.getfixturevalue(image_id)
    region_file = request.getfixturevalue(region_file)
    out_file = tmp_path.joinpath(image_id.replace('/', '-') + '.tif')

    cli_str = (
        f'download -i {image_id} -r {region_file} -dd {tmp_path} --crs {crs} --scale {scale} --dtype {dtype} '
        f'--resampling {resampling} -mts {max_tile_size} -mtd {max_tile_dim}'
    )
    cli_str += ' --mask' if mask else ' --no-mask'
    cli_str += ' --scale-offset' if scale_offset else ' --no-scale-offset'
    cli_str += ''.join([f' --band-name {bn}' for bn in bands]) if bands else ''
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (out_file.exists())

    with open(region_file) as f:
        region = json.load(f)
    # test downloaded file readability and format
    _test_downloaded_file(
        out_file, region=region, crs=crs, scale=scale, dtype=dtype, bands=bands, scale_offset=scale_offset
    )


def test_max_tile_size_error(
    s2_sr_image_id: str, region_100ha_file: pathlib.Path, tmp_path: pathlib.Path, runner: CliRunner, request
):
    """ Test image download with max_tile_size > EE limit raises an EE error.  """
    cli_str = f'download -i {s2_sr_image_id} -r {region_100ha_file} -dd {tmp_path} -mts 100'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert isinstance(result.exception, ValueError)
    assert 'download size limit' in str(result.exception)


def test_max_tile_dim_error(
    s2_sr_image_id: str, region_100ha_file: pathlib.Path, tmp_path: pathlib.Path, runner: CliRunner, request
):
    """ Test image download with max_tile_dim > EE limit raises an EE error.  """
    cli_str = f'download -i {s2_sr_image_id} -r {region_100ha_file} -dd {tmp_path} -mtd 100000'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert isinstance(result.exception, ValueError)
    assert 'download limit' in str(result.exception)


def test_export_drive_params(l8_image_id: str, region_25ha_file: pathlib.Path, runner: CliRunner):
    """ Test export to google drive starts ok, specifying all cli params"""
    cli_str = (
        f'export -i {l8_image_id} -r {region_25ha_file} -df geedim/test --crs EPSG:3857 --scale 30 --dtype uint16 '
        f'--mask --resampling bilinear --no-wait --band-name SR_B4 --band-name SR_B3 --band-name SR_B2'
    )
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)


def test_export_asset_params(l8_image_id: str, region_25ha_file: pathlib.Path, runner: CliRunner):
    """ Test export to asset starts ok, specifying all cli params"""
    # Note when e.g. github runs this test in parallel, it could run into problems trying to overwrite an existing
    # asset.  The overwrite error won't be raised with --no-wait though.  So this test serves at least to check the
    # CLI export options work, and won't fail if run in parallel, even if it runs into overwrite problems.
    folder = f'geedim'
    test_asset_id = asset_id(l8_image_id, folder)
    try:
        ee.data.deleteAsset(test_asset_id)
    except ee.ee_exception.EEException:
        pass

    cli_str = (
        f'export -i {l8_image_id} -r {region_25ha_file} -f {folder} --crs EPSG:3857 --scale 30 --dtype uint16 '
        f'--mask --resampling bilinear --no-wait --type asset --band-name SR_B4 --band-name SR_B3 --band-name SR_B2'
    )
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)


def test_export_asset_no_folder_error(l8_image_id: str, region_25ha_file: pathlib.Path, runner: CliRunner):
    """ Test export to asset raises an error when no folder is specified. """
    cli_str = (
        f'export -i {l8_image_id} -r {region_25ha_file} --crs EPSG:3857 --scale 30 '
        f'--dtype uint16 --mask --resampling bilinear --no-wait --type asset'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert '--folder' in result.output


@pytest.mark.parametrize('image_list, scale', [('s2_sr_image_id_list', 10), ('l8_9_image_id_list', 30)])
def test_composite_defaults(
    image_list: str, scale: float, region_25ha_file: pathlib.Path, runner: CliRunner, tmp_path: pathlib.Path,
    request: pytest.FixtureRequest
):
    """ Test composite with default CLI parameters.  """
    image_list = request.getfixturevalue(image_list)
    image_ids_str = ' -i '.join(image_list)
    cli_str = f'composite -i {image_ids_str} download --crs EPSG:3857 --scale {scale} -r {region_25ha_file} -dd' \
              f' {tmp_path}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)

    # test downloaded file exists
    out_files = glob(str(tmp_path.joinpath(f'*COMP*.tif')))
    assert len(out_files) == 1

    # test downloaded file readability and format
    with open(region_25ha_file) as f:
        region = json.load(f)
    _test_downloaded_file(out_files[0], region)


@pytest.mark.parametrize(
    'image_list, method, region_file, date, mask, resampling, download_scale', [
        ('s2_sr_image_id_list', 'mosaic', None, '2021-10-01', True, 'near', 10),
        ('l8_9_image_id_list', 'q-mosaic', 'region_25ha_file', None, True, 'bilinear', 30),
        ('l8_9_image_id_list', 'medoid', 'region_25ha_file', None, True, 'near', 30),
        ('gedi_image_id_list', 'medoid', None, None, True, 'bilinear', 25),
    ]
)
def test_composite_params(
    image_list: str, method: str, region_file: str, date: str, mask: bool, resampling: str, download_scale: float,
    region_25ha_file, runner: CliRunner, tmp_path: pathlib.Path, request: pytest.FixtureRequest
):
    """ Test composite with default CLI parameters. """
    image_list = request.getfixturevalue(image_list)
    region_file = request.getfixturevalue(region_file) if region_file else None
    image_ids_str = ' -i '.join(image_list)
    cli_comp_str = f'composite -i {image_ids_str} -cm {method} --resampling {resampling}'
    cli_comp_str += f' -r {region_file}' if region_file else ''
    cli_comp_str += f' -d {date}' if date else ''
    cli_comp_str += ' --mask' if mask else ' --no-mask'
    cli_download_str = f'download -r {region_25ha_file} --crs EPSG:3857 --scale {download_scale} -dd {tmp_path}'
    cli_str = cli_comp_str + ' ' + cli_download_str
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)

    # test downloaded file exists
    out_files = glob(str(tmp_path.joinpath(f'*COMP*.tif')))
    assert len(out_files) == 1

    # test downloaded file readability and format
    with open(region_25ha_file) as f:
        region = json.load(f)
    _test_downloaded_file(out_files[0], region=region, crs='EPSG:3857', scale=download_scale)


def test_search_composite_download(region_25ha_file, runner: CliRunner, tmp_path: pathlib.Path):
    """ Test chaining of `search`, `composite` and `download`. """

    cli_search_str = f'search -c COPERNICUS/S1_GRD -s 2022-01-01 -e 2022-02-01 -r {region_25ha_file}'
    cli_comp_str = f'composite --mask'
    cli_download_str = f'download --crs EPSG:3857 --scale 10 -dd {tmp_path}'
    cli_str = cli_search_str + ' ' + cli_comp_str + ' ' + cli_download_str
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)

    # test downloaded file exists
    out_files = glob(str(tmp_path.joinpath(f'*COMP*.tif')))
    assert len(out_files) == 1

    # test downloaded file readability and format
    with open(region_25ha_file) as f:
        region = json.load(f)
    _test_downloaded_file(out_files[0], region=region, crs='EPSG:3857', scale=10)


def test_search_composite_x2_download(region_25ha_file, runner: CliRunner, tmp_path: pathlib.Path):
    """
    Test chaining of `search`, `composite`, `composite` and `download` i.e. the first composite is included as a
    component image in the second composite.
    """

    cli_search_str = f'search -c l7-c2-l2 -s 2022-01-15 -e 2022-04-01 -r {region_25ha_file} -cp 20'
    cli_comp1_str = f'composite --mask'
    cli_comp2_str = f'composite -i LANDSAT/LE07/C02/T1_L2/LE07_173083_20220103 -cm mosaic --date 2022-04-01 --mask'
    cli_download_str = f'download --crs EPSG:3857 --scale 30 -dd {tmp_path}'
    cli_str = cli_search_str + ' ' + cli_comp1_str + ' ' + cli_comp2_str + ' ' + cli_download_str
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)

    # test downloaded file exists
    out_files = glob(str(tmp_path.joinpath(f'*COMP*.tif')))
    assert len(out_files) == 1

    # test downloaded file readability and format
    with open(region_25ha_file) as f:
        region = json.load(f)
    _test_downloaded_file(out_files[0], region=region, crs='EPSG:3857', scale=30)
