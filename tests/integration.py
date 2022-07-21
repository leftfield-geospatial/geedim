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
import geedim as gd
import pytest


@pytest.fixture(scope='session', autouse=True)
def ee_init():
    """ Override the ee_init fixture, so that we only initialise through geemap below. """
    return


@pytest.mark.no_ee_init
def test_geemap_integration(tmp_path: Path):
    """ Simulate the geemap download example. """
    gd.Initialize(http_transport=Http())    # a replica of geemap Initialize
    ee_image = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").first()
    gd_image = gd.download.BaseImage(ee_image)
    out_file = tmp_path.joinpath('landsat.tif')
    gd_image.download(out_file, scale=100)
    assert out_file.exists()
