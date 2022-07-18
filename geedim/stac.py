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
from typing import Dict, Union

from geedim import utils

logger = logging.getLogger(__name__)
root_stac_url = 'https://earthengine-stac.storage.googleapis.com/catalog/catalog.json'


class StacItem:

    def __init__(self, name: str, item_dict: Dict):
        """
        Image/collection STAC container class.  Provides access to band properties and root property descriptions.

        Parameters
        ----------
        name: str
            The image/collection ID/name
        item_dict: dict
            The raw STAC dict for the image/collection.
        """
        self._name = name
        self._item_dict = item_dict
        self._descriptions = self._get_descriptions(item_dict)
        self._band_props = self._get_band_props(item_dict)

    def _get_descriptions(self, item_dict: Dict) -> Union[Dict[str, str], None]:
        """ Return a dictionary with property names as keys, and descriptions as values. """
        if 'summaries' not in item_dict:
            return None
        if 'gee:schema' in item_dict['summaries']:
            gee_schema = item_dict['summaries']['gee:schema']
        elif 'gee:collection_schema' in item_dict['summaries']:
            gee_schema = item_dict['summaries']['gee:collection_schema']
        else:
            return None
        descriptions = {item['name']: item['description'] for item in gee_schema}
        return descriptions

    def _get_band_props(self, item_dict: Dict) -> Union[Dict[str, Dict], None]:
        """
        Return a dictionary of band properties, with band names as keys, and properties as values.
        """
        if not ('summaries' in item_dict and 'eo:bands' in item_dict['summaries']):
            logger.warning(f'There is no STAC band information for {self._name}')
            return None

        summaries = item_dict['summaries']
        ee_band_props = summaries['eo:bands']
        # if the gsd is the same across all bands, there is a `gsd` key in summaries, otherwise there are `gsd` keys
        # for each item in ee_band_props
        global_gsd = summaries['gsd'] if 'gsd' in summaries else None
        band_props = {}
        # a list of the EE band properties we want to copy
        prop_keys = [
            'name', 'description', 'center_wavelength', 'gee:wavelength', 'gee:units', 'gee:scale', 'gee:offset'
        ]

        def strip_gee(key: str):
            """ Remove 'gee:' from the start of `key` if it is there. """
            return key[4:] if key.startswith('gee:') else key

        for ee_band_dict in ee_band_props:
            band_dict = {
                strip_gee(prop_key): ee_band_dict[prop_key] for prop_key in prop_keys if prop_key in ee_band_dict
            }
            gsd = ee_band_dict['gsd'] if 'gsd' in ee_band_dict else global_gsd
            gsd = gsd[0] if isinstance(gsd, (list, tuple)) else gsd
            band_dict.update(gsd=gsd) if gsd else None
            band_props[ee_band_dict['name']] = band_dict
        return band_props

    @property
    def name(self) -> str:
        """ ID/name of the contained image/collection STAC. """
        return self._name

    @property
    def descriptions(self) -> Union[Dict[str, str], None]:
        """ Dictionary of property descriptions with property names as keys, and descriptions as values. """
        return self._descriptions

    @property
    def band_props(self) -> Union[Dict[str, Dict], None]:
        """ Dictionary of band properties, with band names as keys, and properties as values. """
        return self._band_props

    @property
    def license(self) -> Union[str, None]:
        """ Terms of use / license. """
        url = None
        if 'links' in self._item_dict:
            for link in self._item_dict['links']:
                if ('rel' in link) and (link['rel'] == 'license') and ('href' in link):
                    url = link['href']
                    break
        return url


@utils.singleton
class StacCatalog:

    def __init__(self):
        """ Singleton class to interface to the EE STAC, and retrieve image/collection STAC data. """
        self._filename = utils.root_path.joinpath('geedim/data/ee_stac_urls.json')
        self._session = utils.retry_session()
        self._url_dict = None
        self._cache = {}
        self._lock = threading.Lock()

    @property
    def url_dict(self) -> Dict[str, str]:
        """ Dictionary with image/collection IDs/names as keys, and STAC URLs as values. """
        if not self._url_dict:
            # delay reading the json file until it is needed.
            with open(self._filename, 'r') as f:
                self._url_dict = json.load(f)
        return self._url_dict

    def _traverse_stac(self, url: str, url_dict: Dict) -> Dict:
        """
        Recursive & threaded EE STAC tree traversal that returns the `url_dict` i.e. a dict with image/collection
        IDs/names as keys, and the corresponding json STAC URLs as values.
        """
        response = self._session.get(url)
        if not response.ok:
            logger.warning(f'Error reading {url}: ' + str(response.content))
            return url_dict
        response_dict = response.json()
        if 'type' in response_dict:
            if (response_dict['type'].lower() == 'collection'):
                # we have reached a leaf node
                if (('gee:type' in response_dict) and
                    (response_dict['gee:type'].lower() in ['image_collection', 'image'])):
                    # we have reached an image / image collection leaf node
                    with self._lock:
                        url_dict[response_dict['id']] = url
                        logger.debug(f'ID: {response_dict["id"]}, Type: {response_dict["gee:type"]}, URL: {url}')
                return url_dict

            with ThreadPoolExecutor() as executor:
                # traverse the sub-tree links in a thread pool
                futures = []
                for link in response_dict['links']:
                    if link['rel'].lower() == 'child':
                        futures.append(executor.submit(self._traverse_stac, link['href'], url_dict))
                for future in as_completed(futures):
                    url_dict = future.result()
        return url_dict

    def refresh_url_dict(self):
        """ Update `url_dict` with the latest from EE STAC. """
        url_dict = {}
        url_dict = self._traverse_stac(root_stac_url, url_dict)
        self._url_dict = dict(sorted(url_dict.items()))

    def write_url_dict(self, filename=None):
        """ Write the ``url_dict`` to file. """
        if filename is None:
            filename = self._filename
        with open(filename, 'w') as f:
            json.dump(self.url_dict, f)

    def get_item_dict(self, name: str):
        """
        Get the raw STAC dict for a given an image/collection name/ID.

        Parameters
        ----------
        name: str
            ID/name of the image/collection whose STAC data to retrieve.

        Returns
        -------
        dict
            Image/collection STAC data in a dict, if it exists, otherwise None.
        """
        coll_name = utils.split_id(name)[0]
        if coll_name in self.url_dict:
            name = coll_name

        # store item dicts in a cache so we don't have to request them more than once
        if name not in self._cache:
            if name not in self.url_dict:
                logger.warning(f'There is no STAC entry for: {name}')
                self._cache[name] = None
            else:
                response = self._session.get(self.url_dict[name])
                self._cache[name] = response.json()
        return self._cache[name]

    def get_item(self, name: str) -> StacItem:
        """
        Get a STAC container instance for a given an image/collection name/ID.

        Parameters
        ----------
        name: str
            The ID/name of the image/collection whose STAC container to retrieve.

        Returns
        -------
        StacItem
            image/collection STAC container, if it exists, otherwise None.
        """
        coll_name = utils.split_id(name)[0]
        if coll_name in self.url_dict:
            name = coll_name
        item_dict = self.get_item_dict(name)
        return StacItem(name, item_dict) if item_dict else None
