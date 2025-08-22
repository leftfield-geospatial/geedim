# Copyright The Geedim Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from __future__ import annotations

import pathlib

import ee
import numpy as np
import pytest
import rasterio as rio

from geedim.download import BaseImage
from geedim.image import ImageAccessor


@pytest.fixture(scope='session')
def const_base_image() -> BaseImage:
    """Constant BaseImage with no fixed projection."""
    return BaseImage(ee.Image([1, 2, 3]))


@pytest.fixture(scope='session')
def s2_sr_hm_base_image(s2_sr_hm_image_id: str) -> BaseImage:
    """Harmonised Sentinel-2 SR BaseImage with partial cloud in region_*ha."""
    return BaseImage.from_id(s2_sr_hm_image_id)


def test_init_deprecation():
    """Test __init__() issues a deprecation warning."""
    with pytest.warns(FutureWarning, match='deprecated'):
        _ = BaseImage(ee.Image())


def test_from_id(s2_sr_hm_image_id: str):
    """Test from_id()."""
    base_image = BaseImage.from_id(s2_sr_hm_image_id)
    assert base_image.id == s2_sr_hm_image_id


@pytest.mark.parametrize(
    'base_image, accessor',
    [('s2_sr_hm_base_image', 's2_sr_hm_image'), ('const_base_image', 'const_image')],
)
def test_properties(
    base_image: BaseImage, accessor: ImageAccessor, request: pytest.FixtureRequest
):
    """Test BaseImage specific properties against a matching ImageAccessor."""
    base_image: BaseImage = request.getfixturevalue(base_image)
    accessor: ImageAccessor = request.getfixturevalue(accessor)
    assert base_image.ee_image == accessor._ee_image
    assert base_image.name == (accessor.id.replace('/', '-') if accessor.id else None)
    assert base_image.transform == rio.Affine(*accessor.transform)
    assert base_image.footprint == accessor.geometry
    assert base_image.has_fixed_projection == (accessor.shape is not None)
    assert base_image.refl_bands == accessor.specBands
    assert base_image.band_properties == accessor.bandProps


def test_ee_image_setter(s2_sr_hm_image_id: str, s2_sr_hm_image: ImageAccessor):
    """Test setting ee_image updates cached properties."""
    base_image = BaseImage(ee.Image())
    _ = base_image.info
    base_image.ee_image = ee.Image(s2_sr_hm_image_id)
    for attr in ['_mi', '_min_projection', 'info', 'dtype']:
        assert getattr(base_image, attr) == getattr(s2_sr_hm_image, attr)


@pytest.mark.parametrize('patch_export_task', ['export_task_success'], indirect=True)
def test_export(
    const_base_image: BaseImage, patch_export_task, capsys: pytest.CaptureFixture
):
    """Test export()."""
    # adapted from test_image.test_to_google_cloud()
    kwargs = dict(
        type='drive',
        folder='geedim',
        crs='EPSG:3857',
        crs_transform=(0.1, 0, -1.0, 0, 0.1, -1.0),
        shape=(20, 20),
        dtype='uint8',
    )
    _ = const_base_image.export('test_export', wait=False, **kwargs)
    # test monitorTask is not called with wait=False
    assert capsys.readouterr().err == ''

    _ = const_base_image.export('test_export', wait=True, **kwargs)
    # test monitorTask is called with wait=True
    assert '100%' in capsys.readouterr().err


def test_download(const_base_image: BaseImage, tmp_path: pathlib.Path):
    """Test download()."""
    file = tmp_path.joinpath('test.tif')
    crs = 'EPSG:3857'
    transform = (0.1, 0, -1.0, 0, 0.1, -1.0)
    shape = (20, 20)
    dtype = 'uint8'
    nodata = 100

    # download and test a deprecation warning is issued for the num_threads param
    with pytest.warns(FutureWarning, match='deprecated'):
        const_base_image.download(
            file,
            num_threads=1,
            nodata=nodata,
            crs=crs,
            crs_transform=transform,
            shape=shape,
            dtype=dtype,
        )

    # test file was downloaded and is formatted correctly
    assert file.exists()
    with rio.open(file) as ds:
        assert ds.crs == crs
        assert ds.transform[:6] == transform
        assert ds.shape == shape
        assert ds.count == const_base_image.count
        assert ds.dtypes[0] == dtype
        assert ds.nodata == nodata

        array = ds.read()
        assert np.all(array.T == range(1, const_base_image.count + 1))
