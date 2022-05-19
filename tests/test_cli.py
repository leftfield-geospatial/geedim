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

import numpy as np
import pytest
from click.testing import CliRunner
from rasterio.features import bounds

from geedim import root_path
from geedim.cli import cli


# TODO: some way of avoiding multiple calls to ee_init?
@pytest.fixture
def runner():
    """ click runner for command line execution. """
    return CliRunner()


@pytest.fixture
def region_25ha_file():
    """ Path to region_25ha geojson file. """
    return root_path.joinpath('data/inputs/tests/region_25ha.geojson')


@pytest.fixture
def region_100ha_file():
    """ Path to region_100ha geojson file. """
    return root_path.joinpath('data/inputs/tests/region_100ha.geojson')


@pytest.fixture
def region_10000ha_file():
    """ Path to region_10000ha geojson file. """
    return root_path.joinpath('data/inputs/tests/region_10000ha.geojson')


@pytest.mark.parametrize(
    'name, start_date, end_date, region, cloudless_portion, is_csmask', [
        ('LANDSAT/LC09/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha_file', 50, True),
        ('LANDSAT/LE07/C02/T1_L2', '2022-01-01', '2022-02-01', 'region_100ha_file', 0, True),
        ('LANDSAT/LT05/C02/T1_L2', '2005-01-01', '2006-02-01', 'region_100ha_file', 50, True),
        ('COPERNICUS/S2_SR', '2022-01-01', '2022-01-15', 'region_100ha_file', 50, True),
        ('COPERNICUS/S2', '2022-01-01', '2022-01-15', 'region_100ha_file', 50, True),
        ('LARSE/GEDI/GEDI02_A_002_MONTHLY', '2021-11-01', '2022-01-01', 'region_100ha_file', 1, False)
    ]
)
def test_search(
    name, start_date: str, end_date: str, region: str, cloudless_portion: float, is_csmask: bool,
    tmp_path: pathlib.Path, runner: CliRunner, request
):
    """
    Test search command gives valid results for different cloud/shadow maskable, and generic collections.
    """
    region_file: dict = request.getfixturevalue(region)
    results_file = tmp_path.joinpath('search_results.json')
    cli_str = (
        f'search -c {name} -s {start_date} -e {end_date} -r {region_file} -cp {cloudless_portion} -o {results_file}'
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
    assert np.all(im_fill_portions >= cloudless_portion) and np.all(im_fill_portions <= 100)
    if is_csmask:  # is a cloud/shadow masked collection
        # test CLOUDLESS_PORTION in expected range
        im_cl_portions = np.array([im_props['CLOUDLESS_PORTION'] for im_props in properties.values()])
        assert np.all(im_cl_portions >= cloudless_portion) and np.all(im_cl_portions <= 100)
        assert np.all(im_cl_portions <= im_fill_portions)
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
            f'config --prob {prob} search -c {name} -s 2022-01-01 -e 2022-02-01 -r {region_10000ha_file} -o '
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
            f'config {param} search -c {name} -s 2022-02-15 -e 2022-04-01 -r {region_10000ha_file} -o'
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
    """ Test --bbox gives same search results as --region. """

    results_file = tmp_path.joinpath('search_results.json')
    with open(region_100ha_file, 'r') as f:
        region = json.load(f)
    bbox = bounds(region)
    bbox_str = ' '.join([str(b) for b in bbox])
    cli_strs = [
        f'search -c LANDSAT/LC09/C02/T1_L2 -s 2022-01-01 -e 2022-02-01 -r {region_100ha_file} -o {results_file}',
        f'search -c LANDSAT/LC09/C02/T1_L2 -s 2022-01-01 -e 2022-02-01 -b {bbox_str} -o {results_file}'
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
