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

from .collection import MaskedCollection
from .mask import MaskedImage

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())


def _ee_init():
    """
    Initialise earth engine using service account pvt key if it exists (i.e. for GitHub CI).
    Adpated from https://gis.stackexchange.com/questions/380664/how-to-de-authenticate-from-earth-engine-api.
    """

    if not ee.data._credentials:
        env_key = 'EE_SERVICE_ACC_PRIVATE_KEY'

        if env_key in os.environ:
            # authenticate with service account
            key_dict = json.loads(os.environ[env_key])
            credentials = ee.ServiceAccountCredentials(key_dict['client_email'], key_data=key_dict['private_key'])
            ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')
        else:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
