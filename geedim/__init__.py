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
from typing import Union, List

import ee

from . import info
from .image import BaseImage, split_id
from .masked_image import LandsatImage, Sentinel2ClImage, ModisNbarImage, MaskedImage

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


def class_from_id(image_id: str) -> Union[BaseImage, MaskedImage]:
    """Return the *Image class that corresponds to the provided EE image/collection ID."""

    masked_image_dict = {
        'LANDSAT/LT04/C02/T1_L2': LandsatImage,
        'LANDSAT/LT05/C02/T1_L2': LandsatImage,
        'LANDSAT/LE07/C02/T1_L2': LandsatImage,
        'LANDSAT/LC08/C02/T1_L2': LandsatImage,
        'COPERNICUS/S2': Sentinel2ClImage,
        'COPERNICUS/S2_SR': Sentinel2ClImage,
        'MODIS/006/MCD43A4': ModisNbarImage
    }
    ee_coll_name, _ = split_id(image_id)
    if image_id in masked_image_dict:
        return masked_image_dict[image_id]
    elif ee_coll_name in masked_image_dict:
        return masked_image_dict[ee_coll_name]
    else:
        return BaseImage


def image_from_id(image_id: str, **kwargs) -> BaseImage:
    """Return a *Image instance for a given EE image ID."""
    return class_from_id(image_id).from_id(image_id, **kwargs)


def parse_image_list(im_list: List[Union[BaseImage, str],], **kwargs) -> List[BaseImage,]:
    """ Return a list of Base/MaskedImage objects, given download/export parameters """
    _im_list = []

    for im_obj in im_list:
        if isinstance(im_obj, str):
            _im_list.append(image_from_id(im_obj, **kwargs))
        elif isinstance(im_obj, BaseImage):
            _im_list.append(im_obj)
        else:
            raise ValueError(f'Unknown image object type: {type(im_obj)}')
    return _im_list
