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

import ee
import numpy as np
import pytest
import rasterio as rio

from geedim import enums, utils
from geedim.image import ImageAccessor


def test_asset_export(prepared_image: ImageAccessor, prepared_image_array: np.ndarray):
    """Test exporting an image to an Earth Engine asset."""
    # create asset ID and folder
    folder = f'geedim/test_{np.random.randint(0, np.iinfo("int32").max)}'
    asset_id = utils.asset_id('test_export', folder)
    asset_folder = '/'.join(asset_id.split('/')[:-1])
    ee.data.createFolder(asset_folder)
    try:
        # export
        task = prepared_image.toGoogleCloud(
            asset_id, type=enums.ExportType.asset, wait=True
        )
        status = task.status()
        assert status['state'] == 'COMPLETED'

        # export the asset image to a NumPy array to inspect it
        asset_im = ImageAccessor(ee.Image(asset_id))
        asset_array = asset_im.toNumPy()
    finally:
        ee.data.deleteAsset(asset_id)
        ee.data.deleteAsset(asset_folder)

    assert asset_im.crs == prepared_image.crs
    assert asset_im.transform == prepared_image.transform
    assert asset_im.shape == prepared_image.shape
    assert asset_im.dtype == prepared_image.dtype

    asset_mask = asset_array != prepared_image.nodata
    assert (asset_mask == ~prepared_image_array.mask).all()
    assert (asset_array == prepared_image_array).all()


@pytest.mark.parametrize(
    'dtype',
    ['float32', 'float64', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32'],
)
def test_ee_geotiff_nodata(dtype: str, l9_sr_image_id: str):
    """Test the nodata value of the Earth Engine GeoTIFF returned by
    ``ee.data.computePixels()`` or ``ee.Image.getDownloadUrl()`` equals the geedim
    expected value (see https://issuetracker.google.com/issues/350528377 for context).
    """
    # prepare an image for downloading as dtype
    image = ImageAccessor(ee.Image(l9_sr_image_id))
    shape = (10, 10)
    prep_image = image.prepareForExport(shape=shape, dtype=dtype)

    # download a small tile with ee.data.computePixels
    request = {
        'expression': prep_image,
        'bandIds': ['SR_B3'],
        'grid': {'dimensions': {'width': shape[1], 'height': shape[0]}},
        'fileFormat': 'GEO_TIFF',
    }
    im_bytes = ee.data.computePixels(request)

    # test nodata with rasterio
    prep_image = ImageAccessor(prep_image)
    with rio.MemoryFile(im_bytes) as mf, mf.open() as ds:
        assert ds.nodata == prep_image.nodata
        # test the EE dtype is not lower precision compared to expected dtype
        assert np.promote_types(prep_image.dtype, ds.dtypes[0]) == ds.dtypes[0]
