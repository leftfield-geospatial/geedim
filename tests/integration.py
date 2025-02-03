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
from pathlib import Path

import ee
import numpy as np
import pytest
import rasterio as rio
from click.testing import CliRunner
from httplib2 import Http
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.features import bounds
from rasterio.warp import transform_geom

import geedim as gd
from geedim import cli, utils
from geedim.download import BaseImage


@pytest.fixture(scope='session', autouse=True)
def ee_init():
    """Override the ee_init fixture, so that we only initialise as geemap does, below."""
    return


def test_geemap_integration(tmp_path: Path):
    """Simulate the geemap download example."""
    gd.Initialize(opt_url=None, http_transport=Http())  # a replica of geemap Initialize
    ee_image = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").first()
    gd_image = gd.download.BaseImage(ee_image)
    out_file = tmp_path.joinpath('landsat.tif')
    gd_image.download(out_file, scale=100)
    assert out_file.exists()
    assert out_file.stat().st_size > 100e6


def test_geeml_integration(tmp_path: Path):
    """Test the geeml `user memory limit exceeded` example."""
    gd.Initialize()
    region = {
        'geodesic': False,
        'crs': {'type': 'name', 'properties': {'name': 'EPSG:4326'}},
        'type': 'Polygon',
        'coordinates': [
            [
                [6.030749828407996, 53.66867883985145],
                [6.114742307473171, 53.66867883985145],
                [6.114742307473171, 53.76381042843971],
                [6.030749828407996, 53.76381042843971],
                [6.030749828407996, 53.66867883985145],
            ]
        ],
    }  # yapf: disable

    ee_image = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate('2019-01-01', '2020-01-01')
        .filterBounds(region)
        .select(['B4', 'B3', 'B2', 'B8'])
        .reduce(ee.Reducer.percentile([35]))
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
        out_file,
        crs='EPSG:4326',
        region=region,
        scale=10,
        dtype='float64',
        overwrite=True,
        max_tile_size=16,
    )
    assert out_file.exists()
    with rio.open(out_file, 'r') as ds:
        assert ds.count == 4
        assert ds.dtypes[0] == 'float64'
        assert np.isinf(ds.nodata)
        region_cnrs = np.array(region['coordinates'][0])
        region_bounds = rio.coords.BoundingBox(*region_cnrs.min(axis=0), *region_cnrs.max(axis=0))
        # sometimes the top/bottom bounds of the dataset are swapped, so extract and compare UL and BR corners
        print(f'region_bounds: {region_bounds}')
        print(f'ds.bounds: {ds.bounds}')
        ds_ul = np.array(
            [min(ds.bounds.left, ds.bounds.right), min(ds.bounds.top, ds.bounds.bottom)]
        )
        ds_lr = np.array(
            [max(ds.bounds.left, ds.bounds.right), max(ds.bounds.top, ds.bounds.bottom)]
        )
        assert region_cnrs.min(axis=0) == pytest.approx(ds_ul, abs=1e-3)
        assert region_cnrs.max(axis=0) == pytest.approx(ds_lr, abs=1e-3)


def test_cli_asset_export(l8_image_id, region_25ha_file: Path, runner: CliRunner, tmp_path: Path):
    """Export a test image to an asset using the CLI."""
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
        assert result.exit_code == 0
        assert ee.data.getAsset(test_asset_id) is not None

        # download the asset image
        asset_image = gd.download.BaseImage.from_id(test_asset_id)
        download_filename = tmp_path.joinpath('integration_test.tif')
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
        im: rio.DatasetReader
        exp_region = transform_geom('EPSG:4326', im.crs, region)
        exp_bounds = BoundingBox(*bounds(exp_region))
        assert im.crs == CRS.from_string(crs)
        assert im.transform[0] == scale
        assert im.count > 1
        assert (
            (im.bounds[0] <= exp_bounds[0])
            and (im.bounds[1] <= exp_bounds[1])
            and (im.bounds[2] >= exp_bounds[2])
            and (im.bounds[3] >= exp_bounds[3])
        )


@pytest.mark.parametrize(
    'dtype', ['float32', 'float64', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32']
)
def test_ee_geotiff_nodata(dtype: str, l9_image_id: str):
    """Test the nodata value of the Earth Engine GeoTIFF returned by ``ee.data.computePixels()`` or
    ``ee.Image.getDownloadUrl()`` equals the geedim expected value (see
    https://issuetracker.google.com/issues/350528377 for context).
    """
    # use geedim to prepare an image for downloading as dtype
    gd.Initialize()
    masked_image = gd.MaskedImage.from_id(l9_image_id)
    shape = (10, 10)
    exp_image = BaseImage(masked_image.prepareForExport(shape=shape, dtype=dtype))

    # download a small tile with ee.data.computePixels
    request = {
        'expression': exp_image._ee_image,
        'bandIds': ['SR_B3'],
        'grid': {'dimensions': {'width': shape[1], 'height': shape[0]}},
        'fileFormat': 'GEO_TIFF',
    }
    im_bytes = ee.data.computePixels(request)

    # test nodata with rasterio
    with rio.MemoryFile(im_bytes) as mf, mf.open() as ds:
        assert ds.nodata == exp_image.nodata
        # test the EE dtype is not lower precision compared to expected dtype
        assert np.promote_types(exp_image.profile['dtype'], ds.dtypes[0]) == ds.dtypes[0]
