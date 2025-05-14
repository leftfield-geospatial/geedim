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

import asyncio

import pytest

from geedim import utils


@pytest.mark.parametrize(
    'id, exp_split', [('A/B/C', ('A/B', 'C')), ('ABC', ('', 'ABC')), (None, (None, None))]
)
def test_split_id(id, exp_split):
    """Test utils.split_id()."""
    assert utils.split_id(id) == exp_split


def test_spinner(capsys: pytest.CaptureFixture):
    """Test utils.Spinner class."""
    desc = 'test'
    with utils.Spinner(desc=desc, leave=True) as spinner:
        spinner.update()
    cap = capsys.readouterr().err
    assert desc in cap and '\n' in cap

    leave = 'done'
    with utils.Spinner(desc=desc, leave=leave) as spinner:
        spinner.update()
    cap = capsys.readouterr().err
    assert desc in cap and leave in cap and '\n' in cap

    with utils.Spinner(desc=desc, leave=False):
        spinner.update()
    assert '\n' not in capsys.readouterr().err


@pytest.mark.parametrize(
    'filename, folder, exp_id',
    [('sub2/file', 'folder/sub1', 'projects/folder/assets/sub1/sub2/file'), ('file', None, 'file')],
)
def test_asset_id(filename: str, folder: str, exp_id: str):
    """Test utils.asset_id()."""
    id = utils.asset_id(filename, folder)
    assert id == exp_id


def test_register_accessor():
    """Test utils.register_accessor()."""

    class Obj:
        pass

    @utils.register_accessor('acc', Obj)
    class Acc:
        def __init__(self, obj):
            self.obj = obj

    obj = Obj()

    # test accessor is created and cached
    assert obj.acc.obj == obj
    assert obj.acc is obj.acc


@pytest.mark.parametrize('kwargs', [dict(desc='desc', unit='unit'), dict(desc='desc')])
def test_get_tqdm_kwargs(kwargs: dict):
    """Test utils.get_tqdm_kwargs()."""
    tqdm_kwargs = utils.get_tqdm_kwargs(**kwargs)
    assert tqdm_kwargs['desc'].lstrip() == kwargs['desc']
    assert tqdm_kwargs['unit'] == kwargs['unit'] if 'unit' in kwargs else 'unit' not in tqdm_kwargs
    assert tqdm_kwargs['dynamic_ncols'] is True
    assert tqdm_kwargs['leave'] is None


def test_async_runner(monkeypatch: pytest.MonkeyPatch):
    """Test the utils.AsyncRunner() singleton class."""
    runner = utils.AsyncRunner()

    # test it is a singleton
    assert runner is utils.AsyncRunner()
    # test creation of the event loop and session
    assert runner.loop and runner.session

    # test running a coroutine in this thread
    async def coro(**kwargs) -> dict:
        return kwargs

    coro_kwargs = dict(a=1, b=2)
    assert runner.run(coro(**coro_kwargs)) == coro_kwargs

    # test running a coroutine when another event loop is already running in this thread (e.g.
    # jupyter notebook)
    async def existing_loop_run(**kwargs):
        return runner.run(coro(**kwargs))

    assert asyncio.run(existing_loop_run(**coro_kwargs)) == coro_kwargs
