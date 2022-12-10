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
# schema definitions for MaskedImage.from_id(), geedim <-> EE collection names, and search properties
from tabulate import tabulate
from textwrap import wrap
import geedim.mask

# yapf: disable
default_prop_schema = {
    'system:id': {'abbrev': 'ID', 'description': 'Earth Engine image id'},
    'system:time_start': {'abbrev': 'DATE', 'description': 'Image capture date/time (UTC)'},
    'FILL_PORTION': {'abbrev': 'FILL', 'description': 'Portion of region pixels that are valid (%)'},
}

landsat_prop_schema = {
    'system:id': {'abbrev': 'ID', 'description': 'Earth Engine image id'},
    'system:time_start': {'abbrev': 'DATE', 'description': 'Image capture date/time (UTC)'},
    'FILL_PORTION': {'abbrev': 'FILL', 'description': 'Portion of region pixels that are valid (%)'},
    'CLOUDLESS_PORTION': {'abbrev': 'CLOUDLESS', 'description': 'Portion of filled pixels that are cloud/shadow free (%)'},
    'GEOMETRIC_RMSE_MODEL': {'abbrev': 'GRMSE', 'description': 'Orthorectification RMSE (m)'},
    'SUN_AZIMUTH': {'abbrev': 'SAA', 'description': 'Solar azimuth angle (deg)'},
    'SUN_ELEVATION': {'abbrev': 'SEA', 'description': 'Solar elevation angle (deg)'}
}

s2_prop_schema = {
    'system:id': {'abbrev': 'ID', 'description': 'Earth Engine image id'},
    'system:time_start': {'abbrev': 'DATE', 'description': 'Image capture date/time (UTC)'},
    'FILL_PORTION': {'abbrev': 'FILL', 'description': 'Portion of region pixels that are valid (%)'},
    'CLOUDLESS_PORTION': {'abbrev': 'CLOUDLESS', 'description': 'Portion of filled pixels that are cloud/shadow free (%)'},
    'RADIOMETRIC_QUALITY': {'abbrev': 'RADQ', 'description': 'Radiometric quality check'},
    'GEOMETRIC_QUALITY': {'abbrev': 'GEOMQ', 'description': 'Geometric quality check'},
    'MEAN_SOLAR_AZIMUTH_ANGLE': {'abbrev': 'SAA', 'description': 'Solar azimuth angle (deg)'},
    'MEAN_SOLAR_ZENITH_ANGLE': {'abbrev': 'SZA', 'description': 'Solar zenith angle (deg)'},
    'MEAN_INCIDENCE_AZIMUTH_ANGLE_B1': {'abbrev': 'VAA', 'description': 'View (B1) azimuth angle (deg)'},
    'MEAN_INCIDENCE_ZENITH_ANGLE_B1': {'abbrev': 'VZA', 'description': 'View (B1) zenith angle (deg)'}
}

collection_schema = {
    'LANDSAT/LT04/C02/T1_L2': {
        'gd_coll_name': 'l4-c2-l2',
        'prop_schema': landsat_prop_schema,
        'image_type': geedim.mask.LandsatImage,
        'ee_url': 'https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT04_C02_T1_L2',
        'description': 'Landsat 4, collection 2, tier 1, level 2 surface reflectance.'
    },
    'LANDSAT/LT05/C02/T1_L2': {
        'gd_coll_name': 'l5-c2-l2',
        'prop_schema': landsat_prop_schema,
        'image_type': geedim.mask.LandsatImage,
        'ee_url': 'https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2',
        'description': 'Landsat 5, collection 2, tier 1, level 2 surface reflectance.'
    },
    'LANDSAT/LE07/C02/T1_L2': {
        'gd_coll_name': 'l7-c2-l2',
        'prop_schema': landsat_prop_schema,
        'image_type': geedim.mask.LandsatImage,
        'ee_url': 'https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2',
        'description': 'Landsat 7, collection 2, tier 1, level 2 surface reflectance.'
    },
    'LANDSAT/LC08/C02/T1_L2': {
        'gd_coll_name': 'l8-c2-l2',
        'prop_schema': landsat_prop_schema,
        'image_type': geedim.mask.LandsatImage,
        'ee_url': 'https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2',
        'description': 'Landsat 8, collection 2, tier 1, level 2 surface reflectance.'
    },
    'LANDSAT/LC09/C02/T1_L2': {
        'gd_coll_name': 'l9-c2-l2',
        'prop_schema': landsat_prop_schema,
        'image_type': geedim.mask.LandsatImage,
        'ee_url': 'https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2',
        'description': 'Landsat 9, collection 2, tier 1, level 2 surface reflectance.'
    },
    'COPERNICUS/S2': {
        'gd_coll_name': 's2-toa',
        'prop_schema': s2_prop_schema,
        'image_type': geedim.mask.Sentinel2ToaClImage,
        'ee_url': 'https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2',
        'description': 'Sentinel-2, level 1C, top of atmosphere reflectance.'
    },
    'COPERNICUS/S2_SR': {
        'gd_coll_name': 's2-sr',
        'prop_schema': s2_prop_schema,
        'image_type': geedim.mask.Sentinel2SrClImage,
        'ee_url': 'https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR',
        'description': 'Sentinel-2, level 2A, surface reflectance.'
    },
    'COPERNICUS/S2_HARMONIZED': {
        'gd_coll_name': 's2-toa-hm',
        'prop_schema': s2_prop_schema,
        'image_type': geedim.mask.Sentinel2ToaClImage,
        'ee_url': 'https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED',
        'description': 'Harmonised Sentinel-2, level 1C, top of atmosphere reflectance.'
    },
    'COPERNICUS/S2_SR_HARMONIZED': {
        'gd_coll_name': 's2-sr-hm',
        'prop_schema': s2_prop_schema,
        'image_type': geedim.mask.Sentinel2SrClImage,
        'ee_url': 'https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED',
        'description': 'Harmonised Sentinel-2, level 2A, surface reflectance.'
    },
    'MODIS/006/MCD43A4': {
        'gd_coll_name': 'modis-nbar',
        'prop_schema': default_prop_schema,
        'image_type': geedim.mask.MaskedImage,
        'ee_url': 'https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD43A4',
        'description': 'MODIS nadir BRDF adjusted daily reflectance.'
    }
}
# yapf: enable

# Dict to convert from geedim to Earth Engine collection names
ee_to_gd = dict([(k, v['gd_coll_name']) for k, v in collection_schema.items()])

# Dict to convert from Earth Engine to geedim collection names
gd_to_ee = dict([(v['gd_coll_name'], k) for k, v in collection_schema.items()])

# "Two way" dict to convert Earth Engine to/from geedim collection names
coll_names = dict(**gd_to_ee, **ee_to_gd)

# A list of cloud/shadow mask supported EE collection names
cloud_coll_names = [k for k, v in collection_schema.items() if v['image_type'] != geedim.mask.MaskedImage]


def cli_cloud_coll_table() -> str:
    """ Return a table of cloud/shadow mask supported collections for use in CLI help strings. """
    headers = dict(gd_coll_name='geedim name', ee_coll_name='EE name')
    data = []
    for key, val in collection_schema.items():
        if val['image_type'] != geedim.mask.MaskedImage:
            data.append(dict(gd_coll_name=val['gd_coll_name'], ee_coll_name=key))
    return tabulate(data, headers=headers, tablefmt='rst')


def cloud_coll_table(descr_join='\n') -> str:
    """
    Return a table of cloud/shadow mask supported collections.
    * Use descr_join='\n' for github README friendly formatting.
    * Use descr_join='\n\n' for RTD/Sphinx friendly formatting.

    Instructions for adding cloud/shadow supported collections to CLI help and github README:
    * print(cli_cloud_coll_table()) and paste into cli.search() and  cli.config() command docstrings.
    * print(cloud_coll_table()) and paste into the README.
    * The equivalent RTD table is auto-generated in docs/conf.py.
    """
    headers = dict(ee_coll_name='EE name', descr='Description')
    data = []
    for key, val in collection_schema.items():
        if val['image_type'] != geedim.mask.MaskedImage:
            ee_coll_name = '\n'.join(wrap(f'`{key} <{val["ee_url"]}>`_', width=40))
            descr = descr_join.join(wrap(val['description'], width=60))   # for RTD multiline table
            data.append(dict(ee_coll_name=ee_coll_name, descr=descr))

    return tabulate(data, headers=headers, tablefmt='grid')
