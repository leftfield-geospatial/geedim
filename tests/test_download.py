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
from typing import Dict

import ee
import numpy as np
import rasterio as rio
from rasterio import Affine
from rasterio.features import bounds

from geedim.download import BaseImage


def test_init(synth_fixed_ee_image: ee.Image, synth_fixed_ee_info:Dict, small_region:Dict):
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

def test_download(synth_unfixed_ee_image: ee.Image, small_region:Dict, tmp_path: pathlib.Path):
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


