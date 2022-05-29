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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import numpy as np

from geedim import root_path
from geedim.utils import requests_retry_session

logger = logging.getLogger(__name__)

root_stac_url = 'https://earthengine-stac.storage.googleapis.com/catalog/catalog.json'


class STACitem:
    def __init__(self, name, item_dict: Dict):
        self._name = name
        self._item_dict = item_dict
        self._prop_descriptions = self._get_prop_descriptions(item_dict)
        self._bands_dict = self._get_bands_dict(item_dict)

    def _get_prop_descriptions(self, item_dict: Dict) -> Dict[str, str]:
        if 'summaries' in item_dict and 'gee:schema' in item_dict['summaries']:
            gee_schema = item_dict['summaries']['gee:schema']
            prop_descriptions = {item['name']: item['description'] for item in gee_schema}
            return prop_descriptions
        else:
            return None

    def _get_bands_dict(self, item_dict: Dict) -> Dict:
        if 'summaries' in item_dict and 'eo:bands' in item_dict['summaries']:
            summaries = item_dict['summaries']
            eo_bands = summaries['eo:bands']
            global_gsd = summaries['gsd'] if 'gsd' in summaries else None
            bands_dict = {}
            prop_keys = [
                'name', 'description', 'center_wavelength', 'gee:wavelength', 'gee:units', 'units', 'gee:scale',
                'gee:offset'
            ]
            for eo_band in eo_bands:
                band_dict = {prop_key: eo_band[prop_key] for prop_key in prop_keys if prop_key in eo_band}
                gsd = eo_band['gsd'] if 'gsd' in eo_band else global_gsd
                gsd = gsd[0] if isinstance(gsd, (list, tuple)) else gsd
                band_dict.update(gsd=gsd) if gsd else None
                bands_dict[eo_band['name']] = band_dict
            return bands_dict
        else:
            logger.warning(f'There is no STAC band information for {self._name}')
            return None

    @property
    def prop_descriptions(self) -> Dict[str, str]:
        return self._prop_descriptions

    @property
    def bands_dict(self) -> Dict:
        return self._bands_dict


class STAC(object):
    _filename = root_path.joinpath('geedim/data/ee_image_stac.json')
    _session = requests_retry_session()
    _url_dict = None
    _cache = {}
    _lock = threading.Lock()

    def __new__(cls):
        """ Singleton constructor. """
        if not hasattr(cls, 'instance'):
            cls.instance = object.__new__(cls)
        return cls.instance

    @property
    def url_dict(self) -> Dict[str, str]:
        """ A dictionary with image/collection IDs/names as keys, and STAC URLs as values. """
        if not self._url_dict:  # delayed read
            with open(self._filename, 'r') as f:
                self._url_dict = json.load(f)
        return self._url_dict

    def _traverse_stac(self, url: str, url_dict: Dict) -> Dict:
        """
        Recursive & threaded EE STAC traversal that returns the `url_dict` i.e. a dict with image/collection
        IDs/names as keys, and the corresponding json STAC URLs as values.
        """
        response = self._session.get(url)
        if not response.ok:
            logger.warning(f'Error reading {url}: ' + str(response.content))
            return url_dict
        response_dict = response.json()
        if 'type' in response_dict:
            if (response_dict['type'].lower() == 'collection'):
                if ('gee:type' in response_dict) and (
                    response_dict['gee:type'].lower() in ['image_collection', 'image']):
                    with self._lock:
                        url_dict[response_dict['id']] = url
                        logger.debug(f'ID: {response_dict["id"]}, Type: {response_dict["gee:type"]}, URL: {url}')
                return url_dict

            with ThreadPoolExecutor() as executor:
                futures = []
                for link in response_dict['links']:
                    if link['rel'].lower() == 'child':
                        futures.append(executor.submit(STAC._traverse_stac, link['href'], url_dict))
                for future in as_completed(futures):
                    url_dict = future.result()
        return url_dict

    def write_url_dict_file(self, filename=None):
        """ Gets the latest url_dict from EE STAC and writes it to file. """
        filename = filename or self._filename
        url_dict = {}
        url_dict = self._traverse_stac(root_stac_url, url_dict)
        with open(filename, 'w') as f:
            json.dump(url_dict, f, sort_keys=True, indent=4)

    def get_item_dict(self, name) -> Dict:
        """ Returns the raw STAC dict for a given an image/collection name/ID. """
        if name not in self._cache:
            if name not in self.url_dict:
                logger.warning(f'There is no STAC entry for: {name}')
                self._cache[name] = None
            else:
                response = self._session.get(self.url_dict[name])
                self._cache[name] = response.json()
        return self._cache[name]

    def get_item(self, name) -> STACitem:
        item_dict = self.get_item_dict(name)
        return STACitem(name, item_dict)


if False:
    def testing():
        # Notes:
        # - If all bands are same gsd, the  gsd is in info['summaries]['gsd]
        # - If bands have different gsd's, the gsd is in info['summaries']['eo:bands'][0..N]['gsd']
        # - If band has scale and offset, these are in info['summaries']['eo:bands'][0..N]['gee:scale'] and [
        # 'gee:offset'],
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

        gsd_dict = {}
        for name in stac._stac_dict.keys():
            band_info = stac.get_band_info(name)
            gsds = np.array([bi['gsd'] for bi in band_info])
            gsd_dict[name] = [min(gsds), max(gsds)]
            print(name)


    def aggr_props():
        stac = STAC()

        def get_properties(name):
            try:
                print(name)
                return stac.get_properties(name)
            except Exception as ex:
                print('Error: ' + name)
                print(stac.get_coll_im_stac(name)['summaries'])
                return None

        props_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = []
            for name in stac._stac_dict.keys():
                futures.append(executor.submit(get_properties, name))
            for future in as_completed(futures):
                res = future.result()
                if res:
                    props_dict[res['name']] = res

        with open('gee_stac.json', 'w') as f:
            json.dump(props_dict, f, sort_keys=True, indent=4)


    def aggr_gsds():
        stac = STAC()

        def get_gsd_info(name):
            try:
                band_info = stac.get_band_info(name)
                gsds = np.array([bi['gsd'] for bi in band_info])
                return [name, min(gsds), min(gsds[gsds > 0]), max(gsds)]
            except Exception as ex:
                print(name)
                print(stac.get_coll_im_stac(name)['summaries'])
                return None

        gsd_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = []
            for name in stac._stac_dict.keys():
                futures.append(executor.submit(get_gsd_info, name))
            for future in as_completed(futures):
                res = future.result()
                if res:
                    gsd_dict[res[0]] = res
        import pandas as pd
        gsd_df = pd.DataFrame(gsd_dict.values(), columns=['ID', 'min', 'min>0', 'max'])
