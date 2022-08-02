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
import ee
from pathlib import Path
from httplib2 import Http
import numpy as np
import rasterio as rio
import geedim as gd
import pytest


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
    with pytest.raises(IOError) as ex:
        gd_image.download(
            out_file, crs='EPSG:4326', region=region, scale=10, num_threads=1, dtype='float64', overwrite=True
        )
    assert 'user memory limit exceeded' in str(ex).lower()

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
