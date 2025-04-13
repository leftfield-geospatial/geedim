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

from __future__ import annotations

import itertools
import pathlib

import ee
import numpy as np
import pytest
from click.testing import CliRunner
from rasterio.features import bounds
from rasterio.warp import transform_geom

from geedim import Initialize, MaskedImage
from geedim.collection import ImageCollectionAccessor
from geedim.image import ImageAccessor
from geedim.utils import root_path


def accessors_from_images(ee_images: list[ee.Image]) -> list[ImageAccessor]:
    """Return a list of ImageAccessor objects, with cached info properties, for the given list of
    ee.Image objects, combining all getInfo() calls into one.
    """
    infos = ee.List(ee_images).getInfo()
    return [ImageAccessor._with_info(ee_image, info) for ee_image, info in zip(ee_images, infos)]


def transform_bounds(geometry: dict, crs: str = 'EPSG:4326') -> tuple[float, ...]:
    """Return the bounds of the given GeoJSON geometry in crs coordinates."""
    src_crs = geometry['crs']['properties']['name'] if 'crs' in geometry else 'EPSG:4326'
    geometry = geometry if crs == src_crs else transform_geom(src_crs, crs, geometry)
    return bounds(geometry)


@pytest.fixture(scope='session', autouse=True)
def ee_init():
    Initialize()
    return


@pytest.fixture(scope='session')
def region_25ha() -> dict:
    """A geojson polygon defining a 500x500m region."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [21.6389, -33.4520],
                [21.6389, -33.4474],
                [21.6442, -33.4474],
                [21.6442, -33.4520],
                [21.6389, -33.4520],
            ]
        ],
    }


@pytest.fixture(scope='session')
def region_100ha() -> dict:
    """A geojson polygon defining a 1x1km region."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [21.6374, -33.4547],
                [21.6374, -33.4455],
                [21.6480, -33.4455],
                [21.6480, -33.4547],
                [21.6374, -33.4547],
            ]
        ],
    }


@pytest.fixture(scope='session')
def region_10000ha() -> dict:
    """A geojson polygon defining a 10x10km region."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [21.5893, -33.4964],
                [21.5893, -33.4038],
                [21.6960, -33.4038],
                [21.6960, -33.4964],
                [21.5893, -33.4964],
            ]
        ],
    }


@pytest.fixture
def const_image_25ha_file() -> pathlib.Path:
    return root_path.joinpath('tests/data/const_image_25ha.tif')


@pytest.fixture(scope='session')
def l4_image_id() -> str:
    """Landsat-4 EE ID for image that covering `region_*ha`, with partial cloud/shadow for `region10000ha` only."""
    return 'LANDSAT/LT04/C02/T1_L2/LT04_173083_19880310'


@pytest.fixture(scope='session')
def l5_image_id() -> str:
    """Landsat-5 EE ID for image covering `region_*ha` with partial cloud/shadow."""
    return 'LANDSAT/LT05/C02/T1_L2/LT05_173083_20051112'


@pytest.fixture(scope='session')
def l7_image_id() -> str:
    """Landsat-7 EE ID for image covering `region_*ha` with partial cloud/shadow."""
    return 'LANDSAT/LE07/C02/T1_L2/LE07_173083_20220119'


@pytest.fixture(scope='session')
def l8_image_id() -> str:
    """Landsat-8 EE ID for image covering `region_*ha` with partial cloud/shadow."""
    return 'LANDSAT/LC08/C02/T1_L2/LC08_173083_20180217'


@pytest.fixture(scope='session')
def l9_image_id() -> str:
    """Landsat-9 EE ID for image covering `region_*ha` with partial cloud/shadow."""
    return 'LANDSAT/LC09/C02/T1_L2/LC09_173083_20220308'


@pytest.fixture(scope='session')
def landsat_image_ids(l4_image_id, l5_image_id, l7_image_id, l8_image_id, l9_image_id) -> list[str]:
    """Landsat4-9 EE IDs for images covering `region_*ha` with partial cloud/shadow."""
    return [l4_image_id, l5_image_id, l7_image_id, l8_image_id, l9_image_id]


@pytest.fixture(scope='session')
def s2_sr_image_id() -> str:
    """Sentinel-2 SR EE ID for image with QA* data, covering `region_*ha` with partial cloud/shadow."""
    return 'COPERNICUS/S2_SR/20200929T080731_20200929T083634_T34HEJ'


@pytest.fixture(scope='session')
def s2_toa_image_id() -> str:
    """Sentinel-2 TOA EE ID for image with QA* data, covering `region_*ha` with partial cloud/shadow."""
    return 'COPERNICUS/S2/20210216T081011_20210216T083703_T34HEJ'


@pytest.fixture(scope='session')
def s2_sr_hm_image_id(s2_sr_image_id: str) -> str:
    """Harmonised Sentinel-2 SR EE ID for image with QA* data, covering `region_*ha` with partial cloud/shadow."""
    return 'COPERNICUS/S2_SR_HARMONIZED/' + s2_sr_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def s2_sr_hm_qa_zero_image_id() -> str:
    """Harmonised Sentinel-2 SR EE ID for image with zero QA* data, covering `region_*ha` with partial cloud/shadow."""
    return 'COPERNICUS/S2_SR_HARMONIZED/20230721T080609_20230721T083101_T34HEJ'


@pytest.fixture(scope='session')
def s2_toa_hm_image_id(s2_toa_image_id: str) -> str:
    """Harmonised Sentinel-2 TOA EE ID for image with QA* data, covering `region_*ha` with partial cloud/shadow."""
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
def gch_image_id() -> str:
    """Global Canopy Height (10m) image derived from Sentinel-2 and GEDI.  WGS84 @ 10m.
    https://nlang.users.earthengine.app/view/global-canopy-height-2020.
    """
    return 'users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1'


@pytest.fixture(scope='session')
def s1_sar_image_id() -> str:
    """Sentinel-1 SAR GRD EE image ID.  10m."""
    return 'COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20220112T171750_20220112T171815_041430_04ED28_0A04'


@pytest.fixture(scope='session')
def gedi_agb_image_id() -> str:
    """GEDI aboveground biomass density EE image ID.  1km."""
    return 'LARSE/GEDI/GEDI04_B_002'


@pytest.fixture(scope='session')
def gedi_cth_image_id() -> str:
    """GEDI canopy top height EE image ID.  25m."""
    return 'LARSE/GEDI/GEDI02_A_002_MONTHLY/202112_018E_036S'


@pytest.fixture(scope='session')
def landsat_ndvi_image_id() -> str:
    """Landsat 8-day NDVI composite EE image iD.  Composite in WGS84 with underlying 30m scale."""
    return 'LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI/20211211'


@pytest.fixture(scope='session')
def google_dyn_world_image_id(s2_sr_hm_image_id) -> str:
    """Google Dynamic World EE ID.  10m with positive y-axis transform."""
    return 'GOOGLE/DYNAMICWORLD/V1/' + s2_sr_hm_image_id.split('/')[-1]


@pytest.fixture(scope='session')
def s2_sr_hm_image_ids(s2_sr_image_id: str, s2_toa_image_id: str) -> list[str]:
    """A list of harmonised Sentinel-2 SR image IDs, covering `region_*ha` with partial cloud/shadow.."""
    return [
        'COPERNICUS/S2_SR_HARMONIZED/' + s2_sr_image_id.split('/')[-1],
        'COPERNICUS/S2_SR_HARMONIZED/' + s2_toa_image_id.split('/')[-1],
        'COPERNICUS/S2_SR_HARMONIZED/20191229T081239_20191229T083040_T34HEJ',
    ]


@pytest.fixture(scope='session')
def generic_image_ids(
    modis_nbar_image_id,
    gch_image_id,
    s1_sar_image_id,
    gedi_agb_image_id,
    gedi_cth_image_id,
    landsat_ndvi_image_id,
) -> list[str]:
    """A list of various EE image IDs for non-cloud/shadow masked images."""
    return [
        modis_nbar_image_id,
        gch_image_id,
        s1_sar_image_id,
        gedi_agb_image_id,
        gedi_cth_image_id,
        landsat_ndvi_image_id,
    ]


@pytest.fixture(scope='session')
def l9_sr_image(l9_image_id: str) -> ImageAccessor:
    """Landsat-9 SR image with partial cloud/shadow in region_*ha."""
    return ImageAccessor(ee.Image(l9_image_id))


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
    image = image.setDefaultProjection(crs, scale=0.1).clipToBoundsAndScale(image_bounds)
    # mask outside of mask_bounds
    image = image.updateMask(image.clip(mask_bounds).mask())
    # set ID and band names to those of a known collection so that STAC properties are populated
    image = image.set('system:id', 'COPERNICUS/S2_SR_HARMONIZED/prepared_image')
    image = image.rename(['B2', 'B3', 'B4'])
    return ImageAccessor(image)


@pytest.fixture(scope='session')
def prepared_image_array(prepared_image: ImageAccessor) -> np.ndarray:
    """NumPy array corresponding to the contents of prepared_image, with (row, column, band)
    dimensions.
    """
    array = np.ma.ones((*prepared_image.shape, prepared_image.count), dtype=prepared_image.dtype)
    array = array * np.arange(1, prepared_image.count + 1)
    pad = 5
    array[:pad] = array[-pad:] = array[:, :pad] = array[:, -pad:] = 0
    array.mask = array == 0
    return array


@pytest.fixture(scope='session')
def prepared_coll(prepared_image: ImageAccessor) -> ImageCollectionAccessor:
    """Constant image with a masked border, prepared for exporting."""
    # TODO: change masking of second image?
    image1 = prepared_image._ee_image
    image1 = image1.set({'system:index': 'prepared_image1', 'system:time_start': 0})
    image2 = image1.add(3).toUint8()
    image2 = image2.set({'system:index': 'prepared_image2', 'system:time_start': 24 * 60 * 60e3})

    coll = ee.ImageCollection([image1, image2])
    # set ID to known collection so that STAC properties are populated
    coll = coll.set(
        {'system:id': 'COPERNICUS/S2_SR_HARMONIZED', 'period': 0, 'type_name': 'ImageCollection'}
    )

    return ImageCollectionAccessor(coll)


@pytest.fixture(scope='session')
def prepared_coll_array(
    prepared_coll: ImageCollectionAccessor, prepared_image_array: np.ndarray
) -> np.ndarray:
    """NumPy array corresponding to the contents of prepared_coll, with (row, column, band, image)
    dimensions.
    """
    array = np.ma.ones(
        (*prepared_image_array.shape, len(prepared_coll.properties)),
        dtype=prepared_image_array.dtype,
    )
    array[:, :, :, 0] = prepared_image_array
    array[:, :, :, 1] = prepared_image_array + prepared_image_array.shape[2]
    pad = 5
    array[:pad] = array[-pad:] = array[:, :pad] = array[:, -pad:] = 0
    array.mask = array == 0
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
    # get list of ee.data.getOperation() statuses from test function indirect parameter (expects
    # it to refer to an 'export_task_*' fixture)
    statuses = request.getfixturevalue(request.param)
    statuses = itertools.cycle(statuses)

    def start(self: ee.batch.Task):
        """Mock ee.batch.Task.start() method that just sets task name without running the task."""
        self.name = 'test-task'

    monkeypatch.setattr(ee.batch.Task, 'start', start)
    monkeypatch.setattr(ee.data, 'getOperation', lambda name: next(statuses))


@pytest.fixture(scope='session')
def l4_masked_image(l4_image_id) -> MaskedImage:
    """Landsat-4 MaskedImage covering `region_*ha`, with partial cloud for `region10000ha` only."""
    return MaskedImage.from_id(l4_image_id)


@pytest.fixture(scope='session')
def l5_masked_image(l5_image_id) -> MaskedImage:
    """Landsat-5 MaskedImage covering `region_*ha` with partial cloud/shadow."""
    return MaskedImage.from_id(l5_image_id)


@pytest.fixture(scope='session')
def l7_masked_image(l7_image_id) -> MaskedImage:
    """Landsat-7 MaskedImage covering `region_*ha` with partial cloud/shadow."""
    return MaskedImage.from_id(l7_image_id)


@pytest.fixture(scope='session')
def l8_masked_image(l8_image_id) -> MaskedImage:
    """Landsat-8 MaskedImage that cover `region_*ha` with partial cloud cover."""
    return MaskedImage.from_id(l8_image_id)


@pytest.fixture(scope='session')
def l9_masked_image(l9_image_id) -> MaskedImage:
    """Landsat-9 MaskedImage covering `region_*ha` with partial cloud/shadow."""
    return MaskedImage.from_id(l9_image_id)


@pytest.fixture(scope='session')
def s2_sr_masked_image(s2_sr_image_id) -> MaskedImage:
    """Sentinel-2 SR MaskedImage with QA* data, covering `region_*ha` with partial cloud/shadow."""
    return MaskedImage.from_id(s2_sr_image_id)


@pytest.fixture(scope='session')
def s2_toa_masked_image(s2_toa_image_id) -> MaskedImage:
    """Sentinel-2 TOA MaskedImage with QA* data, covering `region_*ha` with partial cloud/shadow."""
    return MaskedImage.from_id(s2_toa_image_id)


@pytest.fixture(scope='session')
def s2_sr_hm_masked_image(s2_sr_hm_image_id) -> MaskedImage:
    """Harmonised Sentinel-2 SR MaskedImage with QA* data, covering `region_*ha` with partial cloud/shadow."""
    return MaskedImage.from_id(s2_sr_hm_image_id)


@pytest.fixture(scope='session')
def s2_sr_hm_nocp_masked_image(s2_sr_hm_image_id) -> MaskedImage:
    """Harmonised Sentinel-2 SR MaskedImage with no corresponding cloud probability, covering `region_*ha` with partial
    cloud/shadow.
    """
    # create an image with unknown id to prevent linking to cloud probability
    ee_image = ee.Image(s2_sr_hm_image_id)
    ee_image = ee_image.set('system:index', 'unknown')
    return MaskedImage(ee_image, mask_method='cloud-prob')


@pytest.fixture(scope='session')
def s2_sr_hm_nocs_masked_image(s2_sr_hm_image_id) -> MaskedImage:
    """Harmonised Sentinel-2 SR MaskedImage with no corresponding cloud score, covering `region_*ha` with partial
    cloud/shadow.
    """
    # create an image with unknown id to prevent linking to cloud score
    ee_image = ee.Image(s2_sr_hm_image_id)
    ee_image = ee_image.set('system:index', 'unknown')
    return MaskedImage(ee_image, mask_method='cloud-score')


@pytest.fixture(scope='session')
def s2_sr_hm_qa_zero_masked_image(s2_sr_hm_qa_zero_image_id: str) -> MaskedImage:
    """Harmonised Sentinel-2 SR MaskedImage with zero QA* bands, covering `region_*ha` with partial cloud/shadow."""
    return MaskedImage.from_id(s2_sr_hm_qa_zero_image_id, mask_method='qa')


@pytest.fixture(scope='session')
def s2_toa_hm_masked_image(s2_toa_hm_image_id) -> MaskedImage:
    """Harmonised Sentinel-2 TOA MaskedImage with QA* data, covering `region_*ha` with partial cloud/shadow."""
    return MaskedImage.from_id(s2_toa_hm_image_id)


@pytest.fixture(scope='session')
def user_masked_image() -> MaskedImage:
    """A MaskedImage instance where the encapsulated image has no fixed projection or ID."""
    return MaskedImage(ee.Image([1, 2, 3]))


@pytest.fixture(scope='session')
def modis_nbar_masked_image(modis_nbar_image_id) -> MaskedImage:
    """MODIS NBAR MaskedImage with global coverage."""
    return MaskedImage.from_id(modis_nbar_image_id)


@pytest.fixture(scope='session')
def gch_masked_image(gch_image_id) -> MaskedImage:
    """Global Canopy Height (10m) MaskedImage."""
    return MaskedImage.from_id(gch_image_id)


@pytest.fixture(scope='session')
def s1_sar_masked_image(s1_sar_image_id) -> MaskedImage:
    """Sentinel-1 SAR GRD MaskedImage.  10m."""
    return MaskedImage.from_id(s1_sar_image_id)


@pytest.fixture(scope='session')
def gedi_agb_masked_image(gedi_agb_image_id) -> MaskedImage:
    """GEDI aboveground biomass density MaskedImage.  1km."""
    return MaskedImage.from_id(gedi_agb_image_id)


@pytest.fixture(scope='session')
def gedi_cth_masked_image(gedi_cth_image_id) -> MaskedImage:
    """GEDI canopy top height MaskedImage.  25m."""
    return MaskedImage.from_id(gedi_cth_image_id)


@pytest.fixture(scope='session')
def landsat_ndvi_masked_image(landsat_ndvi_image_id) -> MaskedImage:
    """Landsat 8-day NDVI composite MaskedImage.  Composite in WGS84 with underlying 30m scale."""
    return MaskedImage.from_id(landsat_ndvi_image_id)


@pytest.fixture
def runner():
    """click runner for command line execution."""
    return CliRunner()


@pytest.fixture
def region_25ha_file() -> pathlib.Path:
    """Path to region_25ha geojson file."""
    return root_path.joinpath('tests/data/region_25ha.geojson')


@pytest.fixture
def region_100ha_file() -> pathlib.Path:
    """Path to region_100ha geojson file."""
    return root_path.joinpath('tests/data/region_100ha.geojson')


@pytest.fixture
def region_10000ha_file() -> pathlib.Path:
    """Path to region_10000ha geojson file."""
    return root_path.joinpath('tests/data/region_10000ha.geojson')
