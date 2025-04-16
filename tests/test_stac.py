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

import pytest

from geedim.stac import STACClient
from geedim.utils import split_id


def test_singleton(landsat_ndvi_image_id: str):
    """Test STACClient() is a singleton."""
    assert id(STACClient()) == id(STACClient())


def test_get_valid(s2_sr_hm_image_id: str):
    """Test STACClient().get() with a valid EE ID."""
    coll_id = split_id(s2_sr_hm_image_id)[0]
    stac = STACClient().get(coll_id)

    assert stac['id'] == coll_id
    # test for nested dict used by ImageAccessor
    assert 'summaries' in stac and 'eo:bands' in stac['summaries']
    # test passing the image ID to get()
    assert STACClient().get(s2_sr_hm_image_id) == stac
    # test repeat get() for same ID is retrieved from the cache
    assert id(STACClient().get(coll_id)) == id(stac)


def test_get_invalid():
    """Test STACClient().get() with an invalid EE ID."""
    with pytest.warns(RuntimeWarning, match="Couldn't find STAC entry"):
        assert STACClient().get('unknown/unknown') is None
