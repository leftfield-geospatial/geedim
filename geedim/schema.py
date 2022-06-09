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
import geedim.mask
# yapf: disable
default_prop_schema = {
    'system:id': {'abbrev': 'ID', 'description': 'Earth Engine image id'},
    'system:time_start': {'abbrev': 'DATE', 'description': 'Image capture date/time (UTC)'},
    'FILL_PORTION': {'abbrev': 'FILL', 'description': 'Portion of valid pixels (%)'},
}

landsat_prop_schema = {
    'system:id': {'abbrev': 'ID', 'description': 'Earth Engine image id'},
    'system:time_start': {'abbrev': 'DATE', 'description': 'Image capture date/time (UTC)'},
    'FILL_PORTION': {'abbrev': 'FILL', 'description': 'Portion of valid pixels (%)'},
    'CLOUDLESS_PORTION': {'abbrev': 'CLOUDLESS', 'description': 'Portion of cloud/shadow free pixels (%)'},
    'GEOMETRIC_RMSE_MODEL': {'abbrev': 'GRMSE', 'description': 'Orthorectification RMSE (m)'},
    'SUN_AZIMUTH': {'abbrev': 'SAA', 'description': 'Solar azimuth angle (deg)'},
    'SUN_ELEVATION': {'abbrev': 'SEA', 'description': 'Solar elevation angle (deg)'}
}

s2_prop_schema = {
    'system:id': {'abbrev': 'ID', 'description': 'Earth Engine image id'},
    'system:time_start': {'abbrev': 'DATE', 'description': 'Image capture date/time (UTC)'},
    'FILL_PORTION': {'abbrev': 'FILL', 'description': 'Portion of valid pixels (%)'},
    'CLOUDLESS_PORTION': {'abbrev': 'CLOUDLESS', 'description': 'Portion of cloud/shadow free pixels (%)'},
    'RADIOMETRIC_QUALITY': {'abbrev': 'RADQ', 'description': 'Radiometric quality check'},
    'GEOMETRIC_QUALITY': {'abbrev': 'GEOMQ', 'description': 'Geometric quality check'},
    'MEAN_SOLAR_AZIMUTH_ANGLE': {'abbrev': 'SAA', 'description': 'Solar azimuth angle (deg)'},
    'MEAN_SOLAR_ZENITH_ANGLE': {'abbrev': 'SZA', 'description': 'Solar zenith angle (deg)'},
    'MEAN_INCIDENCE_AZIMUTH_ANGLE_B1': {'abbrev': 'VAA', 'description': 'View (B1) azimuth angle (deg)'},
    'MEAN_INCIDENCE_ZENITH_ANGLE_B1': {'abbrev': 'VZA', 'description': 'View (B1) zenith angle (deg)'}
}
# yapf: enable

collection_schema = {
    'LANDSAT/LT04/C02/T1_L2': {
        'gd_coll_name': 'landsat4-c2-l2',
        'prop_schema': landsat_prop_schema,
        'image_type': geedim.mask.LandsatImage,
    },
    'LANDSAT/LT05/C02/T1_L2': {
        'gd_coll_name': 'landsat5-c2-l2',
        'prop_schema': landsat_prop_schema,
        'image_type': geedim.mask.LandsatImage,
    },
    'LANDSAT/LE07/C02/T1_L2': {
        'gd_coll_name': 'landsat7-c2-l2',
        'prop_schema': landsat_prop_schema,
        'image_type': geedim.mask.LandsatImage,
    },
    'LANDSAT/LC08/C02/T1_L2': {
        'gd_coll_name': 'landsat8-c2-l2',
        'prop_schema': landsat_prop_schema,
        'image_type': geedim.mask.LandsatImage,
    },
    'LANDSAT/LC09/C02/T1_L2': {
        'gd_coll_name': 'landsat9-c2-l2',
        'prop_schema': landsat_prop_schema,
        'image_type': geedim.mask.LandsatImage,
    },
    'COPERNICUS/S2': {
        'gd_coll_name': 'sentinel2-toa',
        'prop_schema': s2_prop_schema,
        'image_type': geedim.mask.Sentinel2ToaClImage,
    },
    'COPERNICUS/S2_SR': {
        'gd_coll_name': 'sentinel2-sr',
        'prop_schema': s2_prop_schema,
        'image_type': geedim.mask.Sentinel2SrClImage,
    },
    'MODIS/006/MCD43A4': {
        'gd_coll_name': 'modis-nbar',
        'prop_schema': default_prop_schema,
        'image_type': geedim.mask.MaskedImage,
    }
}

# Dict to convert from geedim to Earth Engine collection names
ee_to_gd = dict([(k, v['gd_coll_name']) for k, v in collection_schema.items()])

# Dict to convert from Earth Engine to geedim collection names
gd_to_ee = dict([(v['gd_coll_name'], k) for k, v in collection_schema.items()])

# "Two way" dict to convert Earth Engine to/from geedim collection names
coll_names = dict(**gd_to_ee, **ee_to_gd)

# A list of cloud/shadow mask supported EE collection names
cloud_coll_names = [k for k, v in collection_schema.items() if v['image_type'] != geedim.mask.MaskedImage]
