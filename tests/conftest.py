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

import itertools
import pathlib

import ee
import numpy as np
import pytest
from click.testing import CliRunner
from rasterio.features import bounds
from rasterio.warp import transform_geom

from geedim import utils
from geedim.collection import ImageCollectionAccessor
from geedim.image import ImageAccessor

tests_path = pathlib.Path(__file__).absolute().parents[0]


def accessors_from_images(ee_images: list[ee.Image]) -> list[ImageAccessor]:
    """Return a list of ImageAccessor objects, with cached info properties, for the
    given list of ee.Image objects, using a single getInfo() call.
    """
    infos = ee.List(ee_images).getInfo()
    return [
        ImageAccessor._with_info(ee_image, info)
        for ee_image, info in zip(ee_images, infos, strict=True)
    ]


def accessors_from_collections(
    ee_colls: list[ee.ImageCollection],
) -> list[ImageCollectionAccessor]:
    """Return a list of ImageCollectionAccessor objects, with cached info, for the given
    list of ee.ImageCollection objects, using a single getInfo() call.
    """
    coll_images = [
        ee_coll.toList(ImageCollectionAccessor._max_export_images)
        for ee_coll in ee_colls
    ]
    infos = ee.List([ee_colls, coll_images]).getInfo()
    return [
        ImageCollectionAccessor._with_info(
            ee_coll, dict(**coll_info, features=image_infos)
        )
        for ee_coll, coll_info, image_infos in zip(ee_colls, *infos, strict=True)
    ]


def transform_bounds(geometry: dict, crs: str = 'EPSG:4326') -> tuple[float, ...]:
    """Return the bounds of the given GeoJSON geometry in crs coordinates."""
    src_crs = (
        geometry['crs']['properties']['name'] if 'crs' in geometry else 'EPSG:4326'
    )
    geometry = geometry if crs == src_crs else transform_geom(src_crs, crs, geometry)
    return bounds(geometry)


@pytest.fixture(scope='session', autouse=True)
def ee_initialize() -> None:
    """Initialise EE and prevent it being initialised again."""
    utils_initialize = utils.Initialize

    def initialize(**kwargs):
        if not ee.data._initialized:
            utils_initialize(**kwargs)

    with pytest.MonkeyPatch.context() as mp:
        # patch utils.Initialize() so that EE is initialized once only (prevents
        # delays when testing e.g. CLI commands that call utils.Initialize() in their
        # invoke())
        mp.setattr(utils, 'Initialize', initialize)
        utils.Initialize()
        yield


@pytest.fixture
def runner():
    """Click runner for command line execution."""
    return CliRunner()


@pytest.fixture(scope='session')
def region_25ha() -> dict:
    """A geojson polygon defining a 500x500m region."""
    return {
        'type': 'Polygon',
        'coordinates': [
            [
                [21.6389, -33.4474],
                [21.6389, -33.452],
                [21.6442, -33.452],
                [21.6442, -33.4474],
            ]
        ],
        'evenOdd': True,
    }


@pytest.fixture(scope='session')
def region_100ha() -> dict:
    """A GeoJSON polygon defining a 1x1km region."""
    return {
        'type': 'Polygon',
        'coordinates': [
            [
                [21.6374, -33.4455],
                [21.6374, -33.4547],
                [21.648, -33.4547],
                [21.648, -33.4455],
            ]
        ],
        'evenOdd': True,
    }


@pytest.fixture(scope='session')
def region_10000ha() -> dict:
    """A geojson polygon defining a 10x10km region."""
    return {
        'type': 'Polygon',
        'coordinates': [
            [
                [21.5893, -33.4038],
                [21.5893, -33.4964],
                [21.696, -33.4964],
                [21.696, -33.4038],
            ]
        ],
        'evenOdd': True,
    }


@pytest.fixture(scope='session')
def l4_sr_image_id() -> str:
    """Landsat-4 surface reflectance EE ID for image that covering `region_*ha`, with
    partial cloud/shadow for `region10000ha` only.
    """
    return 'LANDSAT/LT04/C02/T1_L2/LT04_173083_19880310'


@pytest.fixture(scope='session')
def l4_toa_image_id(l4_sr_image_id: str) -> str:
    """Landsat-4 TOA reflectance EE ID for image that covering `region_*ha`, with
    partial cloud/shadow for `region10000ha` only.
    """
    return 'LANDSAT/LT04/C02/T1_TOA/' + l4_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def l4_raw_image_id(l4_sr_image_id: str) -> str:
    """Landsat-4 at sensor radiance EE ID for image that covering `region_*ha`, with
    partial cloud/shadow for `region10000ha` only.
    """
    return 'LANDSAT/LT04/C02/T1/' + l4_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def l5_sr_image_id() -> str:
    """Landsat-5 surface reflectance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LT05/C02/T1_L2/LT05_173083_20051112'


@pytest.fixture(scope='session')
def l5_toa_image_id(l5_sr_image_id: str) -> str:
    """Landsat-5 TOA reflectance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LT05/C02/T1_TOA/' + l5_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def l5_raw_image_id(l5_sr_image_id: str) -> str:
    """Landsat-5 at sensor radiance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LT05/C02/T1/' + l5_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def l7_sr_image_id() -> str:
    """Landsat-7 surface reflectance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LE07/C02/T1_L2/LE07_173083_20220119'


@pytest.fixture(scope='session')
def l7_toa_image_id(l7_sr_image_id: str) -> str:
    """Landsat-7 TOA reflectance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LE07/C02/T1_TOA/' + l7_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def l7_raw_image_id(l7_sr_image_id: str) -> str:
    """Landsat-7 at sensor radiance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LE07/C02/T1/' + l7_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def l8_sr_image_id() -> str:
    """Landsat-8 surface reflectance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LC08/C02/T1_L2/LC08_173083_20180217'


@pytest.fixture(scope='session')
def l8_toa_image_id(l8_sr_image_id: str) -> str:
    """Landsat-8 TOA reflectance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LC08/C02/T1_TOA/' + l8_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def l8_raw_image_id(l8_sr_image_id: str) -> str:
    """Landsat-8 at sensor radiance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LC08/C02/T1/' + l8_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def l9_sr_image_id() -> str:
    """Landsat-9 surface reflectance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LC09/C02/T1_L2/LC09_173083_20220308'


@pytest.fixture(scope='session')
def l9_toa_image_id(l9_sr_image_id: str) -> str:
    """Landsat-9 TOA reflectance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LC09/C02/T1_TOA/' + l9_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def l9_raw_image_id(l9_sr_image_id: str) -> str:
    """Landsat-9 at sensor radiance EE ID for image covering `region_*ha` with partial
    cloud/shadow.
    """
    return 'LANDSAT/LC09/C02/T1/' + l9_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def ls_sr_image_ids(
    l4_sr_image_id, l5_sr_image_id, l7_sr_image_id, l8_sr_image_id, l9_sr_image_id
) -> list[str]:
    """Landsat4-9 surface reflectance EE IDs for images covering `region_*ha` with
    partial cloud/shadow.
    """
    return [
        l4_sr_image_id,
        l5_sr_image_id,
        l7_sr_image_id,
        l8_sr_image_id,
        l9_sr_image_id,
    ]


@pytest.fixture(scope='session')
def ls_toa_image_ids(
    l4_toa_image_id, l5_toa_image_id, l7_toa_image_id, l8_toa_image_id, l9_toa_image_id
) -> list[str]:
    """Landsat4-9 TOA reflectance EE IDs for images covering `region_*ha` with partial
    cloud/shadow.
    """
    return [
        l4_toa_image_id,
        l5_toa_image_id,
        l7_toa_image_id,
        l8_toa_image_id,
        l9_toa_image_id,
    ]


@pytest.fixture(scope='session')
def ls_raw_image_ids(
    l4_raw_image_id, l5_raw_image_id, l7_raw_image_id, l8_raw_image_id, l9_raw_image_id
) -> list[str]:
    """Landsat4-9 at sensor radiance EE IDs for images covering `region_*ha` with
    partial cloud/shadow.
    """
    return [
        l4_raw_image_id,
        l5_raw_image_id,
        l7_raw_image_id,
        l8_raw_image_id,
        l9_raw_image_id,
    ]


@pytest.fixture(scope='session')
def s2_sr_image_id() -> str:
    """Sentinel-2 surface reflectance EE ID for image with QA* data, covering
    `region_*ha` with partial cloud/shadow.
    """
    return 'COPERNICUS/S2_SR/20200929T080731_20200929T083634_T34HEJ'


@pytest.fixture(scope='session')
def s2_toa_image_id() -> str:
    """Sentinel-2 TOA reflectance EE ID for image with QA* data, covering `region_*ha`
    with partial cloud/shadow.
    """
    return 'COPERNICUS/S2/20210216T081011_20210216T083703_T34HEJ'


@pytest.fixture(scope='session')
def s2_sr_hm_image_id(s2_sr_image_id: str) -> str:
    """Harmonised Sentinel-2 surface reflectance EE ID for image with QA* data, covering
    `region_*ha` with partial cloud/shadow.
    """
    return 'COPERNICUS/S2_SR_HARMONIZED/' + s2_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def s2_toa_hm_image_id(s2_toa_image_id: str) -> str:
    """Harmonised Sentinel-2 TOA reflectance EE ID for image with QA* data, covering
    `region_*ha` with partial cloud/shadow.
    """
    return 'COPERNICUS/S2_HARMONIZED/' + s2_toa_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def s2_image_ids(
    s2_sr_image_id, s2_toa_image_id, s2_sr_hm_image_id, s2_toa_hm_image_id
) -> list[str]:
    """Sentinel-2 EE IDs for images covering `region_*ha` with partial cloud/shadow."""
    return [s2_sr_image_id, s2_toa_image_id, s2_sr_hm_image_id, s2_toa_hm_image_id]


@pytest.fixture(scope='session')
def modis_nbar_image_id() -> str:
    """Global MODIS NBAR image ID."""
    return 'MODIS/061/MCD43A4/2022_01_01'


@pytest.fixture(scope='session')
def gedi_cth_image_id() -> str:
    """GEDI canopy top height EE image ID. 25m."""
    return 'LARSE/GEDI/GEDI02_A_002_MONTHLY/202112_018E_036S'


@pytest.fixture(scope='session')
def landsat_ndvi_image_id() -> str:
    """Landsat 8-day NDVI composite EE image iD.  Composite in WGS84 with underlying
    30m scale.
    """
    return 'LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI/20211211'


@pytest.fixture(scope='session')
def l9_sr_image_ids(l9_sr_image_id: str) -> list[str]:
    """A list of Landsat-9 C2 SR image IDs, covering `region_*ha` with partial
    cloud/shadow..
    """
    return [
        l9_sr_image_id,
        'LANDSAT/LC09/C02/T1_L2/LC09_173083_20221205',
        'LANDSAT/LC09/C02/T1_L2/LC09_173083_20230106',
    ]


@pytest.fixture(scope='session')
def s2_sr_hm_image_ids(s2_sr_image_id: str, s2_toa_image_id: str) -> list[str]:
    """A list of harmonised Sentinel-2 SR image IDs, covering `region_*ha` with partial
    cloud/shadow.
    """
    return [
        'COPERNICUS/S2_SR_HARMONIZED/' + s2_sr_image_id.split('/')[-1],
        'COPERNICUS/S2_SR_HARMONIZED/' + s2_toa_image_id.split('/')[-1],
        'COPERNICUS/S2_SR_HARMONIZED/20191229T081239_20191229T083040_T34HEJ',
    ]


@pytest.fixture(scope='session')
def l9_sr_image(l9_sr_image_id: str) -> ImageAccessor:
    """Landsat-9 SR image with partial cloud/shadow in region_*ha."""
    return ImageAccessor(ee.Image(l9_sr_image_id))


@pytest.fixture(scope='session')
def s2_sr_hm_image(s2_sr_hm_image_id: str) -> ImageAccessor:
    """Harmonised Sentinel-2 SR image with partial cloud/shadow in region_*ha."""
    return ImageAccessor(ee.Image(s2_sr_hm_image_id))


@pytest.fixture(scope='session')
def modis_nbar_image(modis_nbar_image_id: str) -> ImageAccessor:
    """MODIS NBAR image."""
    return ImageAccessor(ee.Image(modis_nbar_image_id))


@pytest.fixture(scope='session')
def landsat_ndvi_image(landsat_ndvi_image_id: str) -> ImageAccessor:
    """Landsat 8-day NDVI composite image."""
    return ImageAccessor(ee.Image(landsat_ndvi_image_id))


@pytest.fixture(scope='session')
def const_image() -> ImageAccessor:
    """Constant image with no fixed projection."""
    return ImageAccessor(ee.Image([1, 2, 3]))


@pytest.fixture(scope='session')
def prepared_image(const_image: ImageAccessor) -> ImageAccessor:
    """Constant image with a masked border, prepared for exporting."""
    crs = 'EPSG:3857'
    image_bounds = ee.Geometry.Rectangle((-1.0, -1.0, 1.0, 1.0), proj=crs)
    mask_bounds = ee.Geometry.Rectangle((-0.5, -0.5, 0.5, 0.5), proj=crs)

    image = const_image._ee_image.toUint8()
    image = image.setDefaultProjection(crs, scale=0.1).clipToBoundsAndScale(
        image_bounds
    )
    # mask outside of mask_bounds
    image = image.updateMask(image.clip(mask_bounds).mask())
    # set ID and band names to those of a known collection so that STAC properties
    # are populated
    image = image.set(
        {
            'system:id': 'COPERNICUS/S2_SR_HARMONIZED/prepared_image1',
            'system:index': 'prepared_image1',
            'system:time_start': 0,
        }
    )
    image = image.rename(['B2', 'B3', 'B4'])
    return ImageAccessor(image)


@pytest.fixture(scope='session')
def prepared_image_array(prepared_image: ImageAccessor) -> np.ndarray:
    """NumPy array corresponding to the contents of prepared_image, with (row, column,
    band) dimensions.
    """
    array = np.ma.ones(
        (*prepared_image.shape, prepared_image.count), dtype=prepared_image.dtype
    )
    array = array * np.arange(1, prepared_image.count + 1)
    pad = 5
    array[:pad] = array[-pad:] = array[:, :pad] = array[:, -pad:] = 0
    array.mask = array == 0
    return array


@pytest.fixture(scope='session')
def prepared_coll(prepared_image: ImageAccessor) -> ImageCollectionAccessor:
    """Collection of two constant images with masked borders, prepared for exporting."""
    image1 = prepared_image._ee_image
    image2 = image1.add(3).toUint8()
    image2 = image2.set(
        {
            'system:id': 'COPERNICUS/S2_SR_HARMONIZED/prepared_image2',
            'system:index': 'prepared_image2',
            'system:time_start': 24 * 60 * 60e3,
        }
    )
    coll = ee.ImageCollection([image1, image2])
    # set ID to known collection so that STAC properties are populated
    coll = coll.set(
        {
            'system:id': 'COPERNICUS/S2_SR_HARMONIZED',
            'period': 0,
            'type_name': 'ImageCollection',
        }
    )
    return ImageCollectionAccessor(coll)


@pytest.fixture(scope='session')
def prepared_coll_array(
    prepared_coll: ImageCollectionAccessor, prepared_image_array: np.ndarray
) -> np.ndarray:
    """NumPy array corresponding to the contents of prepared_coll, with (row, column,
    band, image) dimensions.
    """
    # create an image array matching the second collection image
    im_array2 = prepared_image_array + prepared_image_array.shape[2]

    # create collection array
    array = np.ma.ones(
        (*prepared_image_array.shape, len(prepared_coll.properties)),
        dtype=prepared_image_array.dtype,
    )
    array[:, :, :, 0] = prepared_image_array
    array[:, :, :, 1] = im_array2
    return array


@pytest.fixture()
def export_task_success() -> list[dict]:
    """Mock ee.data.getOperation() success status."""
    return [
        {
            'metadata': {'state': 'SUCCEEDED', 'description': 'export', 'progress': 1},
            'done': True,
        }
    ]


@pytest.fixture()
def export_task_success_sequence() -> list[dict]:
    """Mock ee.data.getOperation() status sequence ending in success."""
    return [
        {'metadata': {'state': 'PENDING', 'description': 'export'}},
        {'metadata': {'state': 'RUNNING', 'description': 'export', 'progress': 0.5}},
        {
            'metadata': {'state': 'SUCCEEDED', 'description': 'export', 'progress': 1},
            'done': True,
        },
    ]


@pytest.fixture()
def export_task_fail() -> list[dict]:
    """Mock ee.data.getOperation() failure status."""
    return [
        {
            'metadata': {'state': 'FAILED', 'description': 'export'},
            'done': True,
            'error': {'message': 'error message'},
        }
    ]


@pytest.fixture()
def patch_export_task(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> dict[str, int]:
    """Patches to the EE API for testing export tasks."""
    # get list of ee.data.getOperation() statuses from test function indirect
    # parameter (expects it to refer to an 'export_task_*' fixture)
    statuses = request.getfixturevalue(request.param)
    statuses = itertools.cycle(statuses)

    def start(self: ee.batch.Task):
        """Mock ee.batch.Task.start() method that just sets task name without running
        the task.
        """
        self.name = 'test-task'

    monkeypatch.setattr(ee.batch.Task, 'start', start)
    monkeypatch.setattr(ee.data, 'getOperation', lambda name: next(statuses))
