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

import time

import pytest

from geedim.utils import Spinner, asset_id, split_id


@pytest.mark.parametrize(
    'id, exp_split', [('A/B/C', ('A/B', 'C')), ('ABC', ('', 'ABC')), (None, (None, None))]
)
def test_split_id(id, exp_split):
    """Test split_id()."""
    assert split_id(id) == exp_split


def test_spinner():
    """Test Spinner class."""
    spinner = Spinner(label='test', interval=0.1)
    assert not spinner.is_alive()
    with spinner:
        assert spinner._run
        assert spinner.is_alive()
        time.sleep(0.5)
    assert not spinner._run
    assert not spinner.is_alive()


@pytest.mark.parametrize(
    'filename, folder, exp_id',
    [
        ('file', 'folder', 'projects/folder/assets/file'),
        ('fp1/fp2/fp3', 'folder', 'projects/folder/assets/fp1/fp2/fp3'),
        ('file', 'folder/sub-folder', 'projects/folder/assets/sub-folder/file'),
        ('file', None, 'file'),
        ('projects/folder/assets/file', None, 'projects/folder/assets/file'),
    ],
)
def test_asset_id(filename: str, folder: str, exp_id: str):
    """Test asset_id() works as expected."""
    id = asset_id(filename, folder)
    assert id == exp_id
