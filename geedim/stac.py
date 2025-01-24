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

from __future__ import annotations

import logging
import warnings
from typing import Any

import aiohttp

from geedim import utils

logger = logging.getLogger(__name__)
_root_stac_url = 'https://earthengine-stac.storage.googleapis.com/catalog/catalog.json'


@utils.singleton
class STACClient:
    def __init__(self):
        """Singleton class to retrieve and cache Earth Engine STAC data."""
        self._urls: dict[str, Any] = {'ROOT': _root_stac_url}
        self._cache = {}

    async def _get(
        self, ee_id: str | None, session: aiohttp.ClientSession
    ) -> dict[str, Any] | None:
        """Return the STAC dictionary for ``ee_id`` or ``None`` if it can't be found."""
        # test ee_id is valid
        parts = ee_id.strip('/').split('/') if isinstance(ee_id, str) else []
        if not len(parts) > 1:
            return None

        # URL dictionary keys to traverse to get to the dictionary containing the URL for ee_id
        titles = ['ROOT', parts[0]]

        # traverse the URL dictionary
        url_dict = self._urls
        for title in titles:
            if title not in url_dict:
                return None
            # populate the URL dictionary at this level if it has not been populated already
            if isinstance(url_dict[title], str):
                logger.debug(f"Requesting STAC entry for '{title}' from '{url_dict[title]}'.")
                async with session.get(url_dict[title]) as response:
                    catalog = await response.json(content_type='text/plain')
                url_dict[title] = {
                    link['title']: link['href']
                    for link in catalog.get('links', [])
                    if link['rel'].lower() == 'child'
                }
            url_dict = url_dict[title]

        # get the URL for ee_id (for ee_id if it has an entry, or for its parent collection
        # otherwise)
        id_title = '_'.join(parts)
        parent_title = '_'.join(parts[:-1])
        if id_title in url_dict:
            url = url_dict[id_title]
        elif parent_title in url_dict:
            url = url_dict[parent_title]
        else:
            return None

        # get and cache the STAC dictionary for ee_id
        async with session.get(url) as response:
            self._cache[ee_id] = await response.json(content_type='text/plain')
        return self._cache[ee_id]

    def get(self, ee_id: str) -> dict[str, Any] | None:
        """
        Get the STAC dictionary for an Earth Engine image or collection.

        :param ee_id:
            Image or collection ID.

        :return:
            STAC dictionary if it exists, otherwise ``None``.
        """
        if ee_id not in self._cache:
            runner = utils.AsyncRunner()
            self._cache[ee_id] = runner.run(self._get(ee_id, runner.session))
            if self._cache[ee_id] is None:
                warnings.warn(f"Couldn't find STAC entry for: '{ee_id}'.", category=RuntimeWarning)
        return self._cache[ee_id]
