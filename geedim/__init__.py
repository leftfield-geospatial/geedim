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
import json
import os
import pathlib

import ee
from .masked_image import LandsatImage, Sentinel2ClImage, ModisNbarImage
from .image import BaseImage, split_id
from . import info

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())


def _ee_init():
    """
    Initialize earth engine using service account pvt key if it exists (i.e. for GitHub CI)

    adpated from https://gis.stackexchange.com/questions/380664/how-to-de-authenticate-from-earth-engine-api

    """

    if not ee.data._credentials:
        env_key = 'EE_SERVICE_ACC_PRIVATE_KEY'

        if env_key in os.environ:
            # write key val to json file
            key_dict = json.loads(os.environ[env_key])
            filename = '_service.json'
            with open(filename, 'w') as f:
                json.dump(key_dict, f)

            # authenticate with service account and delete json file
            try:
                service_account = key_dict['client_email']
                credentials = ee.ServiceAccountCredentials(service_account, filename)
                ee.Initialize(credentials)
            finally:
                os.remove(filename)
        else:
            ee.Initialize()

def image_from_id(image_id: str, **kwargs):
    ee_coll_name, _ = split_id(image_id)

    masked_image_dict = {
        'LANDSAT/LT04/C02/T1_L2': LandsatImage,
        'LANDSAT/LT05/C02/T1_L2': LandsatImage,
        'LANDSAT/LE07/C02/T1_L2': LandsatImage,
        'LANDSAT/LC08/C02/T1_L2': LandsatImage,
        'COPERNICUS/S2': Sentinel2ClImage,
        'COPERNICUS/S2_SR': Sentinel2ClImage,
        'MODIS/006/MCD43A4': ModisNbarImage
    }
    if ee_coll_name in masked_image_dict:
        return masked_image_dict[ee_coll_name](ee.Image(image_id), **kwargs)
    else:
        if len(kwargs) > 0:
            raise ValueError(f'{list(kwargs.keys())} arguments are not supported for {ee_coll_name} collection')
        return BaseImage(ee.Image(image_id))
