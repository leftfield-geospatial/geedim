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
import logging
import pathlib
from typing import Dict

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from geedim import root_path
from geedim.utils import requests_retry_session, split_id

import requests
logger = logging.getLogger(__name__)


def traverse_stac(url: str, stac_dict: Dict, lock: threading.Lock, session: requests.Session = None) -> Dict:
    """
    Traverse the EE STAC, returning a dict of image and image collection names, and their corresponding json URLs.
    """
    # TODO: is the session helpful or does it refer to a single URL
    session = session or requests
    response = session.get(url)
    if not response.ok:
        logger.warning(f'Error reading {url}: ' + str(response.content))
        return stac_dict
    response_dict = response.json()
    if 'type' in response_dict:
        if (response_dict['type'].lower() == 'collection'):
            if ('gee:type' in response_dict) and (response_dict['gee:type'].lower() in ['image_collection', 'image']):
                with lock:
                    stac_dict[response_dict['id']] = url
                    logger.debug(f'ID: {response_dict["id"]}, Type: {response_dict["gee:type"]}, URL: {url}')
            return stac_dict

        with ThreadPoolExecutor() as executor:
            futures = []
            for link in response_dict['links']:
                if link['rel'].lower() == 'child':
                    futures.append(executor.submit(traverse_stac, link['href'], stac_dict, lock, session))
            for future in as_completed(futures):
                stac_dict = future.result()
    return stac_dict


def write_stac_dict(
    stac_url: str='https://earthengine-stac.storage.googleapis.com/catalog/catalog.json',
    stac_file: pathlib.Path=None
):
    """ Retrieve the STAC dict and write to file. """
    lock = threading.Lock()
    if not stac_file:
        stac_file = root_path.joinpath('geedim/data/ee_image_stac.json')
    stac_dict = {}
    stac_dict = traverse_stac(stac_url, stac_dict, lock, session=requests_retry_session())

    with open(stac_file, 'w') as f:
        json.dump(stac_dict, f, sort_keys=True, indent=4)

    return stac_dict


class STAC:
    def __init__(self, session=requests_retry_session()):
        stac_file = root_path.joinpath('geedim/data/ee_image_stac.json')

        if not stac_file.exists():
            logger.warning(f'Requesting STAC dictionary, STAC file does not exist: {stac_file}')
            write_stac_dict(stac_file=stac_file)

        with open(stac_file, 'r') as f:
            self._stac_dict = json.load(f)
        self._session = session or requests

    def get_band_info(self, name):
        if not name in self._stac_dict:
            coll_name = split_id(name)[0]
            if not coll_name in self._stac_dict:
                raise ValueError(f'Unknown EE image or collection: {name}')
            name = coll_name

        response = self._session.get(self._stac_dict[name])
        stac_item_dict = response.json()
        summaries = stac_item_dict['summaries']
        eo_bands = summaries['eo:bands']
        global_gsd = summaries['gsd'] if 'gsd' in summaries else None
        band_info = []
        key_list = [
            'name', 'description', 'center_wavelength', 'gee:wavelength', 'gee:units', 'units', 'gee:scale', 'gee:offset'
        ]
        for eo_band in eo_bands:
            band_dict = {key:eo_band[key] for key in key_list if key in eo_band}
            gsd = eo_band['gsd'] if 'gsd' in eo_band else global_gsd
            gsd = gsd[0] if isinstance(gsd, (list, tuple)) else gsd
            band_dict.update(gsd=gsd) if gsd else None
            band_info.append(band_dict)
        return band_info


def testing():
    # Notes:
    # - If all bands are same gsd, the  gsd is in info['summaries]['gsd]
    # - If bands have different gsd's, the gsd is in info['summaries']['eo:bands'][0..N]['gsd']
    # - If band has scale and offset, these are in info['summaries']['eo:bands'][0..N]['gee:scale'] and ['gee:offset'],
    #   but only if they have a scale/offset
    # - Similarly, iff it is a spectral band it has center_wavelength and gee:wavelength keys
    # - info['gee:schema'] lists image properties with name and description keys. But some products don't have this
    # e.g. MODIS.
    stac = STAC()
    name = 'COPERNICUS/S2_SR'
    name = 'LANDSAT/LC08/C02/T1_L2'
    name = 'MODIS/006/MCD43A4'
    name = 'TRMM/3B42'
    name = 'LANDSAT/LC08/C01/T1_8DAY_EVI'
    stac.get_band_info(name)
