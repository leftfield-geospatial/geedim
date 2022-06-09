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
from typing import Dict, List

import ee
import pytest

from geedim import Initialize, MaskedImage
from geedim.utils import root_path


@pytest.fixture(scope='session', autouse=True)
def ee_init() -> None:
    Initialize()
    return None


@pytest.fixture(scope='session')
def region_25ha() -> Dict:
    """ A geojson polygon defining a 500x500m region. """
    return {
        "type": "Polygon",
        "coordinates":
        [[[21.6389, -33.4520], [21.6389, -33.4474], [21.6442, -33.4474], [21.6442, -33.4520], [21.6389, -33.4520]]]
    }


@pytest.fixture(scope='session')
def region_100ha() -> Dict:
    """ A geojson polygon defining a 1x1km region. """
    return {
        "type": "Polygon",
        "coordinates":
        [[[21.6374, -33.4547], [21.6374, -33.4455], [21.6480, -33.4455], [21.6480, -33.4547], [21.6374, -33.4547]]]
    }


@pytest.fixture(scope='session')
def region_10000ha() -> Dict:
    """ A geojson polygon defining a 10x10km region. """
    return {
        "type": "Polygon",
        "coordinates":
        [[[21.5893, -33.4964], [21.5893, -33.4038], [21.6960, -33.4038], [21.6960, -33.4964], [21.5893, -33.4964]]]
    }


@pytest.fixture
def const_image_25ha_file() -> pathlib.Path:
    return root_path.joinpath('tests/data/const_image_25ha.tif')


@pytest.fixture(scope='session')
def l4_image_id() -> str:
    """ Landsat-4 EE ID for image that covers `region_*ha`, with partial cloud cover only for `region10000ha`.  """
    return 'LANDSAT/LT04/C02/T1_L2/LT04_173083_19880310'


@pytest.fixture(scope='session')
def l5_image_id() -> str:
    """ Landsat-5 EE ID for image that covers `region_*ha` with partial cloud cover.  """
    return 'LANDSAT/LT05/C02/T1_L2/LT05_173083_20051112'  # 'LANDSAT/LT05/C02/T1_L2/LT05_173083_20070307'


@pytest.fixture(scope='session')
def l7_image_id() -> str:
    """ Landsat-7 EE ID for image that covers `region_*ha` with partial cloud cover.  """
    return 'LANDSAT/LE07/C02/T1_L2/LE07_173083_20220119'  # 'LANDSAT/LE07/C02/T1_L2/LE07_173083_20200521'


@pytest.fixture(scope='session')
def l8_image_id() -> str:
    """ Landsat-8 EE ID for image that covers `region_*ha` with partial cloud cover.  """
    return 'LANDSAT/LC08/C02/T1_L2/LC08_173083_20180217'  # 'LANDSAT/LC08/C02/T1_L2/LC08_173083_20171113'


@pytest.fixture(scope='session')
def l9_image_id() -> str:
    """ Landsat-9 EE ID for image that covers `region_*ha` with partial cloud cover. """
    return 'LANDSAT/LC09/C02/T1_L2/LC09_173083_20220308'  # 'LANDSAT/LC09/C02/T1_L2/LC09_173083_20220103'


@pytest.fixture(scope='session')
def landsat_image_ids(l4_image_id, l5_image_id, l7_image_id, l8_image_id, l9_image_id) -> List[str]:
    """ Landsat4-9 EE IDs for images that covers `region_*ha` with partial cloud cover. """
    return [l4_image_id, l5_image_id, l7_image_id, l8_image_id, l9_image_id]


@pytest.fixture(scope='session')
def s2_sr_image_id() -> str:
    """ Sentinel-2 SR EE ID for image that covers `region_*ha` with partial cloud cover. """
    # 'COPERNICUS/S2/20220107T081229_20220107T083059_T34HEJ'
    return 'COPERNICUS/S2_SR/20211004T080801_20211004T083709_T34HEJ'


@pytest.fixture(scope='session')
def s2_toa_image_id() -> str:
    """ Sentinel-2 TOA EE ID for image that covers `region_*ha` with partial cloud cover. """
    return 'COPERNICUS/S2/20220107T081229_20220107T083059_T34HEJ'


@pytest.fixture(scope='session')
def s2_image_ids(s2_sr_image_id, s2_toa_image_id) -> List[str]:
    """ Sentinel-2 TOA/SR EE IDs for images that covers `region_*ha` with partial cloud cover. """
    return [s2_sr_image_id, s2_toa_image_id]


@pytest.fixture(scope='session')
def modis_nbar_image_id() -> str:
    """ Global MODIS NBAR image ID.  WGS84 @ 500m. """
    return 'MODIS/006/MCD43A4/2022_01_01'


@pytest.fixture(scope='session')
def gch_image_id() -> str:
    """
    Global Canopy Height (10m) image derived from Sentinel-2 and GEDI.  WGS84 @ 10m.
    https://nlang.users.earthengine.app/view/global-canopy-height-2020.
    """
    return 'users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1'


@pytest.fixture(scope='session')
def s1_sar_image_id() -> str:
    """ Sentinel-1 SAR GRD EE image ID.  10m. """
    return 'COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20220112T171750_20220112T171815_041430_04ED28_0A04'


@pytest.fixture(scope='session')
def gedi_agb_image_id() -> str:
    """ GEDI aboveground biomass density EE image ID.  1km."""
    return 'LARSE/GEDI/GEDI04_B_002'


@pytest.fixture(scope='session')
def gedi_cth_image_id() -> str:
    """ GEDI canopy top height EE image ID.  25m."""
    return 'LARSE/GEDI/GEDI02_A_002_MONTHLY/202010_018E_036S'


@pytest.fixture(scope='session')
def landsat_ndvi_image_id() -> str:
    """ Landsat 8-day NDVI composite EE image iD.  Composite in WGS84 with underlying 30m scale."""
    return 'LANDSAT/LC08/C01/T1_8DAY_NDVI/20211219'


@pytest.fixture(scope='session')
def generic_image_ids(
    modis_nbar_image_id, gch_image_id, s1_sar_image_id, gedi_agb_image_id, gedi_cth_image_id, landsat_ndvi_image_id
) -> List[str]:
    """ A list of various EE image IDs for non-cloud/shadow masked images. """
    return [
        modis_nbar_image_id, gch_image_id, s1_sar_image_id, gedi_agb_image_id, gedi_cth_image_id, landsat_ndvi_image_id
    ]


@pytest.fixture(scope='session')
def l4_masked_image(l4_image_id) -> MaskedImage:
    """ Landsat-4 MaskedImage that covers `region_*ha`, with partial cloud cover only for `region10000ha`. """
    return MaskedImage.from_id(l4_image_id)


@pytest.fixture(scope='session')
def l5_masked_image(l5_image_id) -> MaskedImage:
    """ Landsat-5 MaskedImage that covers `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(l5_image_id)


@pytest.fixture(scope='session')
def l7_masked_image(l7_image_id) -> MaskedImage:
    """ Landsat-7 MaskedImage that covers `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(l7_image_id)


@pytest.fixture(scope='session')
def l8_masked_image(l8_image_id) -> MaskedImage:
    """ Landsat-8 MaskedImage that cover `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(l8_image_id)


@pytest.fixture(scope='session')
def l9_masked_image(l9_image_id) -> MaskedImage:
    """ Landsat-9 MaskedImage that covers `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(l9_image_id)


@pytest.fixture(scope='session')
def landsat_masked_images(l4_masked_image, l5_masked_image, l7_masked_image, l8_masked_image,
    l9_masked_image) -> List[MaskedImage]:
    """ Landsat4-9 MaskedImage's that cover `region_*ha` with partial cloud cover. """
    return [l4_masked_image, l5_masked_image, l7_masked_image, l8_masked_image, l9_masked_image]


@pytest.fixture(scope='session')
def s2_sr_masked_image(s2_sr_image_id) -> MaskedImage:
    """ Sentinel-2 SR MaskedImage that covers `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(s2_sr_image_id)


@pytest.fixture(scope='session')
def s2_toa_masked_image(s2_toa_image_id) -> MaskedImage:
    """ Sentinel-2 TOA MaskedImage that covers `region_*ha` with partial cloud cover. """
    return MaskedImage.from_id(s2_toa_image_id)


@pytest.fixture(scope='session')
def s2_masked_images(s2_sr_masked_image, s2_toa_masked_image) -> List[MaskedImage]:
    """ Sentinel-2 TOA and SRR MaskedImage's that cover `region_*ha` with partial cloud cover. """
    return [s2_sr_masked_image, s2_toa_masked_image]


@pytest.fixture(scope='session')
def user_masked_image() -> MaskedImage:
    """ A MaskedImage instance where the encapsulated image has no fixed projection or ID.  """
    return MaskedImage(ee.Image([1, 2, 3]))


@pytest.fixture(scope='session')
def modis_nbar_masked_image(modis_nbar_image_id) -> MaskedImage:
    """ MODIS NBAR MaskedImage with global coverage.  """
    return MaskedImage.from_id(modis_nbar_image_id)


@pytest.fixture(scope='session')
def gch_masked_image(gch_image_id) -> MaskedImage:
    """ Global Canopy Height (10m) MaskedImage. """
    return MaskedImage.from_id(gch_image_id)


@pytest.fixture(scope='session')
def s1_sar_masked_image(s1_sar_image_id) -> MaskedImage:
    """ Sentinel-1 SAR GRD MaskedImage.  10m. """
    return MaskedImage.from_id(s1_sar_image_id)


@pytest.fixture(scope='session')
def gedi_agb_masked_image(gedi_agb_image_id) -> MaskedImage:
    """ GEDI aboveground biomass density MaskedImage.  1km."""
    return MaskedImage.from_id(gedi_agb_image_id)


@pytest.fixture(scope='session')
def gedi_cth_masked_image(gedi_cth_image_id) -> MaskedImage:
    """ GEDI canopy top height MaskedImage.  25m."""
    return MaskedImage.from_id(gedi_cth_image_id)


@pytest.fixture(scope='session')
def landsat_ndvi_masked_image(landsat_ndvi_image_id) -> MaskedImage:
    """ Landsat 8-day NDVI composite MaskedImage.  Composite in WGS84 with underlying 30m scale."""
    return MaskedImage.from_id(landsat_ndvi_image_id)


@pytest.fixture(scope='session')
def generic_masked_images(
    modis_nbar_masked_image, gch_masked_image, s1_sar_masked_image, gedi_agb_masked_image, gedi_cth_masked_image,
    landsat_ndvi_masked_image
) -> List[MaskedImage]:
    """ A list of various non-cloud/shadow MaskedImage's. """
    return [
        modis_nbar_masked_image, gch_masked_image, s1_sar_masked_image, gedi_agb_masked_image, gedi_cth_masked_image,
        landsat_ndvi_masked_image
    ]


def get_image_std(ee_image: ee.Image, region: Dict, std_scale: float):
    """
    Helper function to return the mean of the local/neighbourhood image std. dev., over a region.  This serves as a
    measure of image smoothness.
    """
    # Note that for Sentinel-2 images, only the 20m and 60m bands get resampled by EE (and hence smoothed), so
    # here B1 @ 60m is used for testing.
    test_image = ee_image.select(0)
    proj = test_image.projection()
    std_image = test_image.reduceNeighborhood(reducer='stdDev', kernel=ee.Kernel.square(2)).rename('TEST')
    mean_std_image = std_image.reduceRegion(
        reducer='mean', geometry=region, crs=proj.crs(), scale=std_scale, bestEffort=True, maxPixels=1e6
    )
    return mean_std_image.get('TEST').getInfo()
