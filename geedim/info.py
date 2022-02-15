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
# Metadata for geedim supported Earth Engine collections
collection_info = {
"landsat4_c2_l2": {
    "start_date": "1982-08-22",
    "end_date": "1993-06-24",
    "ee_coll_name": "LANDSAT/LT04/C02/T1_L2",
    "bands": [
        {"id": "SR_B1", "name": "blue", "abbrev": "B", "bw_start": 0.45, "bw_end": 0.52, "res": 30},
        {"id": "SR_B2", "name": "green", "abbrev": "G", "bw_start": 0.52, "bw_end": 0.60, "res": 30},
        {"id": "SR_B3", "name": "red", "abbrev": "R", "bw_start": 0.63, "bw_end": 0.69, "res": 30},
        {"id": "SR_B4", "name": "near infrared", "abbrev": "NIR", "bw_start": 0.77, "bw_end": 0.90, "res": 30},
        {"id": "SR_B5", "name": "shortwave infrared 1", "abbrev": "SWIR1", "bw_start": 1.55, "bw_end": 1.75, "res": 30},
        {"id": "SR_B7", "name": "shortwave infrared 2", "abbrev": "SWIR2", "bw_start": 2.08, "bw_end": 2.35, "res": 30},
    ],
    "properties": [
        {"PROPERTY": "system:id", "ABBREV": "ID", "DESCRIPTION": "Earth Engine image id"},
        {"PROPERTY": "system:time_start", "ABBREV": "DATE", "DESCRIPTION": "Image capture date/time (UTC)"},
        {"PROPERTY": "VALID_PORTION", "ABBREV": "VALID", "DESCRIPTION": "Portion of cloud and shadow free pixels (%)"},
        {"PROPERTY": "AVG_SCORE", "ABBREV": "SCORE", "DESCRIPTION": "Average distance to cloud/shadow (m)"},
        {"PROPERTY": "GEOMETRIC_RMSE_MODEL", "ABBREV": "GRMSE", "DESCRIPTION": "Orthorectification RMSE (m)"},
        {"PROPERTY": "SUN_AZIMUTH", "ABBREV": "SAA", "DESCRIPTION": "Solar azimuth angle (deg)"},
        {"PROPERTY": "SUN_ELEVATION", "ABBREV": "SEA", "DESCRIPTION": "Solar elevation angle (deg)"}
    ]},
"landsat5_c2_l2": {
    "start_date": "1984-03-16",
    "end_date": "2012-05-05",
    "ee_coll_name": "LANDSAT/LT05/C02/T1_L2",
    "bands": [
        {"id": "SR_B1", "name": "blue", "abbrev": "B", "bw_start": 0.45, "bw_end": 0.52, "res": 30},
        {"id": "SR_B2", "name": "green", "abbrev": "G", "bw_start": 0.52, "bw_end": 0.60, "res": 30},
        {"id": "SR_B3", "name": "red", "abbrev": "R", "bw_start": 0.63, "bw_end": 0.69, "res": 30},
        {"id": "SR_B4", "name": "near infrared", "abbrev": "NIR", "bw_start": 0.77, "bw_end": 0.90, "res": 30},
        {"id": "SR_B5", "name": "shortwave infrared 1", "abbrev": "SWIR1", "bw_start": 1.55, "bw_end": 1.75, "res": 30},
        {"id": "SR_B7", "name": "shortwave infrared 2", "abbrev": "SWIR2", "bw_start": 2.08, "bw_end": 2.35, "res": 30},
    ],
    "properties": [
        {"PROPERTY": "system:id", "ABBREV": "ID", "DESCRIPTION": "Earth Engine image id"},
        {"PROPERTY": "system:time_start", "ABBREV": "DATE", "DESCRIPTION": "Image capture date/time (UTC)"},
        {"PROPERTY": "VALID_PORTION", "ABBREV": "VALID", "DESCRIPTION": "Portion of cloud and shadow free pixels (%)"},
        {"PROPERTY": "AVG_SCORE", "ABBREV": "SCORE", "DESCRIPTION": "Average distance to cloud/shadow (m)"},
        {"PROPERTY": "GEOMETRIC_RMSE_MODEL", "ABBREV": "GRMSE", "DESCRIPTION": "Orthorectification RMSE (m)"},
        {"PROPERTY": "SUN_AZIMUTH", "ABBREV": "SAA", "DESCRIPTION": "Solar azimuth angle (deg)"},
        {"PROPERTY": "SUN_ELEVATION", "ABBREV": "SEA", "DESCRIPTION": "Solar elevation angle (deg)"}
    ]},
"landsat7_c2_l2": {
    "start_date": "1999-01-01",
    "end_date": None,
    "ee_coll_name": "LANDSAT/LE07/C02/T1_L2",
    "bands": [
        {"id": "SR_B1", "name": "blue", "abbrev": "B", "bw_start": 0.45, "bw_end": 0.52, "res": 30},
        {"id": "SR_B2", "name": "green", "abbrev": "G", "bw_start": 0.52, "bw_end": 0.60, "res": 30},
        {"id": "SR_B3", "name": "red", "abbrev": "R", "bw_start": 0.63, "bw_end": 0.69, "res": 30},
        {"id": "SR_B4", "name": "near infrared", "abbrev": "NIR", "bw_start": 0.77, "bw_end": 0.90, "res": 30},
        {"id": "SR_B5", "name": "shortwave infrared 1", "abbrev": "SWIR1", "bw_start": 1.55, "bw_end": 1.75, "res": 30},
        {"id": "SR_B7", "name": "shortwave infrared 2", "abbrev": "SWIR2", "bw_start": 2.08, "bw_end": 2.35, "res": 30},
        {"id": "ST_B6", "name": "brightness temperature", "abbrev": "BT", "bw_start": 10.40, "bw_end": 12.50, "res": 30},
    ],
    "properties": [
        {"PROPERTY": "system:id", "ABBREV": "ID", "DESCRIPTION": "Earth Engine image id"},
        {"PROPERTY": "system:time_start", "ABBREV": "DATE", "DESCRIPTION": "Image capture date/time (UTC)"},
        {"PROPERTY": "VALID_PORTION", "ABBREV": "VALID", "DESCRIPTION": "Portion of cloud and shadow free pixels (%)"},
        {"PROPERTY": "AVG_SCORE", "ABBREV": "SCORE", "DESCRIPTION": "Average distance to cloud/shadow (m)"},
        {"PROPERTY": "GEOMETRIC_RMSE_MODEL", "ABBREV": "GRMSE", "DESCRIPTION": "Orthorectification RMSE (m)"},
        {"PROPERTY": "SUN_AZIMUTH", "ABBREV": "SAA", "DESCRIPTION": "Solar azimuth angle (deg)"},
        {"PROPERTY": "SUN_ELEVATION", "ABBREV": "SEA", "DESCRIPTION": "Solar elevation angle (deg)"}
    ]},
"landsat8_c2_l2": {
    "start_date": "2013-04-11",
    "end_date": None,
    "ee_coll_name": "LANDSAT/LC08/C02/T1_L2",
    "bands": [
        {"id": "SR_B1", "name": "ultra blue", "abbrev": "UB", "bw_start": 0.435, "bw_end": 0.451, "res": 30},
        {"id": "SR_B2", "name": "blue", "abbrev": "B", "bw_start": 0.452, "bw_end": 0.512, "res": 30},
        {"id": "SR_B3", "name": "green", "abbrev": "G", "bw_start": 0.533, "bw_end": 0.590, "res": 30},
        {"id": "SR_B4", "name": "red", "abbrev": "R", "bw_start": 0.636, "bw_end": 0.673, "res": 30},
        {"id": "SR_B5", "name": "near infrared", "abbrev": "NIR", "bw_start": 0.851, "bw_end": 0.879, "res": 30},
        {"id": "SR_B6", "name": "shortwave infrared 1", "abbrev": "SWIR1", "bw_start": 1.566, "bw_end": 1.651, "res": 30},
        {"id": "SR_B7", "name": "shortwave infrared 2", "abbrev": "SWIR2", "bw_start": 2.107, "bw_end": 2.294, "res": 30},
        {"id": "ST_B10", "name": "brightness temperature", "abbrev": "BT", "bw_start": 10.60, "bw_end": 11.19, "res": 30}
    ],
    "properties": [
        {"PROPERTY": "system:id", "ABBREV": "ID", "DESCRIPTION": "Earth Engine image id"},
        {"PROPERTY": "system:time_start", "ABBREV": "DATE", "DESCRIPTION": "Image capture date/time (UTC)"},
        {"PROPERTY": "VALID_PORTION", "ABBREV": "VALID", "DESCRIPTION": "Portion of cloud and shadow free pixels (%)"},
        {"PROPERTY": "AVG_SCORE", "ABBREV": "SCORE", "DESCRIPTION": "Average distance to cloud/shadow (m)"},
        {"PROPERTY": "GEOMETRIC_RMSE_MODEL", "ABBREV": "GRMSE", "DESCRIPTION": "Orthorectification RMSE (m)"},
        {"PROPERTY": "SUN_AZIMUTH", "ABBREV": "SAA", "DESCRIPTION": "Solar azimuth angle (deg)"},
        {"PROPERTY": "SUN_ELEVATION", "ABBREV": "SEA", "DESCRIPTION": "Solar elevation angle (deg)"}
    ]},
"sentinel2_toa": {
    "start_date": "2015-06-23",
    "end_date": None,
    "ee_coll_name": "COPERNICUS/S2",
    "bands": [
        {"id": "B1", "name": "aerosols", "abbrev": "UB", "bw_start": 0.421, "bw_end": 0.447, "res": 60},
        {"id": "B2", "name": "blue", "abbrev": "B", "bw_start": 0.439, "bw_end": 0.535, "res": 10},
        {"id": "B3", "name": "green", "abbrev": "G", "bw_start": 0.537, "bw_end": 0.582, "res": 10},
        {"id": "B4", "name": "red", "abbrev": "R", "bw_start": 0.646, "bw_end": 0.685, "res": 10},
        {"id": "B5", "name": "red edge 1", "abbrev": "RE1", "bw_start": 0.694, "bw_end": 0.714, "res": 20},
        {"id": "B6", "name": "red edge 2", "abbrev": "RE2", "bw_start": 0.731, "bw_end": 0.749, "res": 20},
        {"id": "B7", "name": "red edge 3", "abbrev": "RE3", "bw_start": 0.768, "bw_end": 0.796, "res": 20},
        {"id": "B8", "name": "near infrared 1", "abbrev": "NIR1", "bw_start": 0.767, "bw_end": 0.908, "res": 10},
        {"id": "B8A", "name": "red edge 4", "abbrev": "RE4", "bw_start": 0.848, "bw_end": 0.881, "res": 20},
        {"id": "B9", "name": "water vapour", "abbrev": "WV", "bw_start": 0.931, "bw_end": 0.958, "res": 60},
        {"id": "B11", "name": "shortwave infrared 1", "abbrev": "SWIR1", "bw_start": 1.539, "bw_end": 1.681, "res": 20},
        {"id": "B12", "name": "shortwave infrared 2", "abbrev": "SWIR2", "bw_start": 2.072, "bw_end": 2.312, "res": 20}
    ],
    "properties": [
        {"PROPERTY": "system:id", "ABBREV": "ID", "DESCRIPTION": "Earth Engine image id"},
        {"PROPERTY": "system:time_start", "ABBREV": "DATE", "DESCRIPTION": "Image capture date/time (UTC)"},
        {"PROPERTY": "VALID_PORTION", "ABBREV": "VALID", "DESCRIPTION": "Portion of cloud and shadow free pixels (%)"},
        {"PROPERTY": "AVG_SCORE", "ABBREV": "SCORE", "DESCRIPTION": "Average distance to cloud/shadow (m)"},
        {"PROPERTY": "RADIOMETRIC_QUALITY", "ABBREV": "RADQ", "DESCRIPTION": "Radiometric quality check"},
        {"PROPERTY": "GEOMETRIC_QUALITY", "ABBREV": "GEOMQ", "DESCRIPTION": "Geometric quality check"},
        {"PROPERTY": "MEAN_SOLAR_AZIMUTH_ANGLE", "ABBREV": "SAA", "DESCRIPTION": "Solar azimuth angle (deg)"},
        {"PROPERTY": "MEAN_SOLAR_ZENITH_ANGLE", "ABBREV": "SZA", "DESCRIPTION": "Solar zenith angle (deg)"},
        {"PROPERTY": "MEAN_INCIDENCE_AZIMUTH_ANGLE_B1", "ABBREV": "VAA", "DESCRIPTION": "View (B1) azimuth angle (deg)"},
        {"PROPERTY": "MEAN_INCIDENCE_ZENITH_ANGLE_B1", "ABBREV": "VZA", "DESCRIPTION": "View (B1) zenith angle (deg)"}
    ]},
"sentinel2_sr": {
    "start_date": "2017-03-28",
    "end_date": None,
    "ee_coll_name": "COPERNICUS/S2_SR",
    "bands": [
        {"id": "B1", "name": "aerosols", "abbrev": "UB", "bw_start": 0.421, "bw_end": 0.447, "res": 60},
        {"id": "B2", "name": "blue", "abbrev": "B", "bw_start": 0.439, "bw_end": 0.535, "res": 10},
        {"id": "B3", "name": "green", "abbrev": "G", "bw_start": 0.537, "bw_end": 0.582, "res": 10},
        {"id": "B4", "name": "red", "abbrev": "R", "bw_start": 0.646, "bw_end": 0.685, "res": 10},
        {"id": "B5", "name": "red edge 1", "abbrev": "RE1", "bw_start": 0.694, "bw_end": 0.714, "res": 20},
        {"id": "B6", "name": "red edge 2", "abbrev": "RE2", "bw_start": 0.731, "bw_end": 0.749, "res": 20},
        {"id": "B7", "name": "red edge 3", "abbrev": "RE3", "bw_start": 0.768, "bw_end": 0.796, "res": 20},
        {"id": "B8", "name": "near infrared 1", "abbrev": "NIR1", "bw_start": 0.767, "bw_end": 0.908, "res": 10},
        {"id": "B8A", "name": "red edge 4", "abbrev": "RE4", "bw_start": 0.848, "bw_end": 0.881, "res": 20},
        {"id": "B9", "name": "water vapour", "abbrev": "WV", "bw_start": 0.931, "bw_end": 0.958, "res": 60},
        {"id": "B11", "name": "shortwave infrared 1", "abbrev": "SWIR1", "bw_start": 1.539, "bw_end": 1.681, "res": 20},
        {"id": "B12", "name": "shortwave infrared 2", "abbrev": "SWIR2", "bw_start": 2.072, "bw_end": 2.312, "res": 20}
    ],
    "properties": [
        {"PROPERTY": "system:id", "ABBREV": "ID", "DESCRIPTION": "Earth Engine image id"},
        {"PROPERTY": "system:time_start", "ABBREV": "DATE", "DESCRIPTION": "Image capture date/time (UTC)"},
        {"PROPERTY": "VALID_PORTION", "ABBREV": "VALID", "DESCRIPTION": "Portion of cloud and shadow free pixels (%)"},
        {"PROPERTY": "AVG_SCORE", "ABBREV": "SCORE", "DESCRIPTION": "Average distance to cloud/shadow (m)"},
        {"PROPERTY": "RADIOMETRIC_QUALITY", "ABBREV": "RADQ", "DESCRIPTION": "Radiometric quality check"},
        {"PROPERTY": "GEOMETRIC_QUALITY", "ABBREV": "GEOMQ", "DESCRIPTION": "Geometric quality check"},
        {"PROPERTY": "MEAN_SOLAR_AZIMUTH_ANGLE", "ABBREV": "SAA", "DESCRIPTION": "Solar azimuth angle (deg)"},
        {"PROPERTY": "MEAN_SOLAR_ZENITH_ANGLE", "ABBREV": "SZA", "DESCRIPTION": "Solar zenith angle (deg)"},
        {"PROPERTY": "MEAN_INCIDENCE_AZIMUTH_ANGLE_B1", "ABBREV": "VAA", "DESCRIPTION": "View (B1) azimuth angle (deg)"},
        {"PROPERTY": "MEAN_INCIDENCE_ZENITH_ANGLE_B1", "ABBREV": "VZA", "DESCRIPTION": "View (B1) zenith angle (deg)"}
    ]},
"modis_nbar": {
    "start_date": "2000-02-18",
    "end_date": None,
    "ee_coll_name": "MODIS/006/MCD43A4",
    "bands": [
        {"id": "Nadir_Reflectance_Band1", "name": "red", "abbrev": "R", "bw_start": 0.620, "bw_end": 0.670, "res": 500},
        {"id": "Nadir_Reflectance_Band2", "name": "near infrared", "abbrev": "NIR", "bw_start": 0.841, "bw_end": 0.876, "res": 500},
        {"id": "Nadir_Reflectance_Band3", "name": "blue", "abbrev": "B", "bw_start": 0.459, "bw_end": 0.479, "res": 500},
        {"id": "Nadir_Reflectance_Band4", "name": "green", "abbrev": "G", "bw_start": 0.545, "bw_end": 0.565, "res": 500},
        {"id": "Nadir_Reflectance_Band5", "name": "shortwave infrared 1", "abbrev": "SWIR1", "bw_start": 1.230, "bw_end": 1.250, "res": 500},
        {"id": "Nadir_Reflectance_Band6", "name": "shortwave infrared 2", "abbrev": "SWIR2", "bw_start": 1.628, "bw_end": 1.652, "res": 500},
        {"id": "Nadir_Reflectance_Band7", "name": "shortwave infrared 3", "abbrev": "SWIR3", "bw_start": 2.105, "bw_end": 2.155, "res": 500}
    ],
    "properties": [
        {"PROPERTY": "system:id", "ABBREV": "ID", "DESCRIPTION": "Earth Engine image id"},
        {"PROPERTY": "system:time_start", "ABBREV": "DATE", "DESCRIPTION": "Image capture date/time (UTC)"}
    ]}
}

# Dict to convert from geedim to Earth Engine collection names
gd_to_ee = dict([(k, v['ee_coll_name']) for k, v in collection_info.items()])

# Dict to convert from Earth Engine to geedim collection names
ee_to_gd = dict([(v['ee_coll_name'], k) for k, v in collection_info.items()])

# "Two way" dict to convert Earth Engine to/from geedim collection names
coll_names = dict(**gd_to_ee, **ee_to_gd)
