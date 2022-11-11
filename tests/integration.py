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

import ee
from pathlib import Path
from httplib2 import Http
import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.features import bounds
from rasterio.warp import transform_geom
from rasterio.coords import BoundingBox
from click.testing import CliRunner
import pytest
import json
import geedim as gd
from geedim import utils, cli


@pytest.fixture(scope='session', autouse=True)
def ee_init():
    """ Override the ee_init fixture, so that we only initialise as geemap does, below. """
    return


def test_geemap_integration(tmp_path: Path):
    """ Simulate the geemap download example. """
    gd.Initialize(opt_url=None, http_transport=Http())    # a replica of geemap Initialize
    ee_image = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").first()
    gd_image = gd.download.BaseImage(ee_image)
    out_file = tmp_path.joinpath('landsat.tif')
    gd_image.download(out_file, scale=100)
    assert out_file.exists()
    assert out_file.stat().st_size > 100e6


def test_geeml_integration(tmp_path: Path):
    """ Test the geeml `user memory limit exceeded` example. """
    gd.Initialize()
    region = {
        'geodesic': False,
        'crs': {'type': 'name', 'properties': {'name': 'EPSG:4326'}},
        'type': 'Polygon',
        'coordinates': [[
            [6.030749828407996, 53.66867883985145],
            [6.114742307473171, 53.66867883985145],
            [6.114742307473171, 53.76381042843971],
            [6.030749828407996, 53.76381042843971],
            [6.030749828407996, 53.66867883985145]
        ]]
    }  # yapf: disable

    ee_image = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').
        filterDate('2019-01-01', '2020-01-01').
        filterBounds(region).
        select(['B4', 'B3', 'B2', 'B8']).
        reduce(ee.Reducer.percentile([35]))
    )  # yapf: disable

    gd_image = gd.download.BaseImage(ee_image)
    out_file = tmp_path.joinpath('test.tif')
    # test we get user memory limit exceeded error with default max_tile_size
    # (EE does not always raise this - memory limit is dynamic?, or percentile implementation changed?)
    # with pytest.raises(IOError) as ex:
    #     gd_image.download(
    #         out_file, crs='EPSG:4326', region=region, scale=10, num_threads=1, dtype='float64', overwrite=True
    #     )
    # assert 'user memory limit exceeded' in str(ex).lower()

    # test we can download the image with a max_tile_size of 16 MB
    gd_image.download(
        out_file, crs='EPSG:4326', region=region, scale=10, dtype='float64',overwrite=True, max_tile_size=16,
    )
    assert out_file.exists()
    with rio.open(out_file, 'r') as ds:
        assert ds.count == 4
        assert ds.dtypes[0] == 'float64'
        assert np.isnan(ds.nodata)
        assert ds.transform.xoff == pytest.approx(region['coordinates'][0][0][0])
        assert ds.transform.yoff == pytest.approx(region['coordinates'][0][2][1])


def test_cli_asset_export(l8_image_id, region_25ha_file: Path, runner: CliRunner, tmp_path: Path):
    """  Export a test image to an asset using the CLI. """
    # create a randomly named folder to allow parallel tests without overwriting the same asset
    gd.Initialize()
    folder = f'geedim/int_test_asset_export_{np.random.randint(1 << 31)}'
    asset_folder = f'projects/{Path(folder).parts[0]}/assets/{Path(folder).parts[1]}'
    crs = 'EPSG:3857'
    scale = 30

    try:
        # export image to asset via CLI
        test_asset_id = utils.asset_id(l8_image_id, folder)
        ee.data.createAsset(dict(type='Folder'), asset_folder)
        cli_str = (
            f'export -i {l8_image_id} -r {region_25ha_file} -f {folder} --crs {crs} --scale {scale} '
            f'--dtype uint16 --mask --resampling bilinear --wait --type asset'
        )
        result = runner.invoke(cli.cli, cli_str.split())
        assert (result.exit_code == 0)
        assert ee.data.getAsset(test_asset_id) is not None

        # download the asset image
        asset_image = gd.download.BaseImage.from_id(test_asset_id)
        download_filename = tmp_path.joinpath(f'integration_test.tif')
        asset_image.download(download_filename)
        assert download_filename.exists()

    finally:
        # clean up the asset and its folder
        try:
            ee.data.deleteAsset(test_asset_id)
            ee.data.deleteAsset(asset_folder)
        except ee.ee_exception.EEException:
            pass

    # test downloaded asset image
    with open(region_25ha_file) as f:
        region = json.load(f)
    with rio.open(download_filename, 'r') as im:
        im : rio.DatasetReader
        exp_region = transform_geom('EPSG:4326', im.crs, region)
        exp_bounds = BoundingBox(*bounds(exp_region))
        assert im.crs == CRS.from_string(crs)
        assert im.transform[0] == scale
        assert im.count > 1
        assert (
            (im.bounds[0] <= exp_bounds[0]) and (im.bounds[1] <= exp_bounds[1]) and
            (im.bounds[2] >= exp_bounds[2]) and (im.bounds[3] >= exp_bounds[3])
        )
