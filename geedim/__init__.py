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

import ee

from geedim.collection import MaskedCollection
from geedim.mask import MaskedImage
from geedim.enums import CloudMaskMethod, CompositeMethod, ResamplingMethod


def Initialize():
    """
    Initialise Earth Engine though the `high volume endpoint
    <https://developers.google.com/earth-engine/cloud/highvolume>`_.

    Credentials will be read from the `EE_SERVICE_ACC_PRIVATE_KEY` environment variable, if it exists.  This is
    useful for integrating with e.g. GitHub actions.
    """

    if not ee.data._credentials:
        # Adpated from https://gis.stackexchange.com/questions/380664/how-to-de-authenticate-from-earth-engine-api.
        env_key = 'EE_SERVICE_ACC_PRIVATE_KEY'

        if env_key in os.environ:
            # authenticate with service account
            key_dict = json.loads(os.environ[env_key])
            credentials = ee.ServiceAccountCredentials(key_dict['client_email'], key_data=key_dict['private_key'])
            ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')
        else:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
