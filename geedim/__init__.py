"""
    Geedim: Download surface reflectance imagery with Google Earth Engine
    Copyright (C) 2021 Dugal Harris
    Email: dugalh@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import base64
import os
import pathlib
import json
import ee

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())


def _ee_init():
    """
    Initialize earth engine using service account pvt key if it exists (i.e. for GitHub CI)

    adpated from https://gis.stackexchange.com/questions/380664/how-to-de-authenticate-from-earth-engine-api

    """

    # with open('service.json') as f:
    #     key_dict = json.load(f)
    #
    # service_str = json.dumps(key_dict)
    # os.environ.update(EE_SERVICE_ACC_PRIVATE_KEY=service_str)

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
                service_account = key_dict['client_email']    #'geedim-service-account@geedim.iam.gserviceaccount.com'
                credentials = ee.ServiceAccountCredentials(service_account, filename)
                ee.Initialize(credentials)
            finally:
                os.remove(filename)
        else:
            ee.Initialize()
