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
import unittest
from datetime import datetime, timedelta

import pandas as pd
from click.testing import CliRunner

import geedim.image
from geedim import root_path, cli, collection, masked_image, image_from_id
from tests.util import _test_image_file, _test_search_results, _setup_test


class TestCli(unittest.TestCase):
    """ Test geedim  CLI """

    @classmethod
    def setUpClass(cls):
        """ Initialise Earth Engine once for all the tests here. """
        _setup_test()

    def test_search(self):
        """ Test search command with different options. """

        results_filename = root_path.joinpath('data/outputs/tests/search_results.json')
        region_filename = root_path.joinpath('data/inputs/tests/region.geojson')
        start_date = datetime.strptime('2019-01-01', '%Y-%m-%d')
        end_date = datetime.strptime('2019-03-01', '%Y-%m-%d')
        search_param_list = [
            ['search', '-c', 'landsat8_c2_l2', '-b', 23.9, -33.6, 24, -33.5, '-s', start_date.strftime("%Y-%m-%d"),
             '-e', end_date.strftime("%Y-%m-%d"),
             '-o', f'{results_filename}'],
            ['search', '-c', 'sentinel2_toa', '-r', str(region_filename), '-s', start_date.strftime("%Y-%m-%d"), '-e',
             end_date.strftime("%Y-%m-%d"),
             '-vp', 30, '-o', str(results_filename)]
        ]

        for search_params in search_param_list:
            with self.subTest('Search', cli_params=search_params):
                result = CliRunner().invoke(cli.cli, search_params, terminal_width=100)

                self.assertTrue(result.exit_code == 0, result.exception)  # search returned ok
                self.assertTrue(results_filename.exists(), 'Search results written to file')

                # read json search results into a pandas dataframe
                with open(results_filename) as f:
                    res_dict = json.load(f)
                res_df = pd.DataFrame.from_dict(res_dict, orient='index')
                res_df.DATE = [datetime.utcfromtimestamp(ts / 1000) for ts in res_df.DATE.values]
                _test_search_results(self, res_df, start_date, end_date)  # test search results

    def test_download(self):
        """ Test download command on one image """

        image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190112'
        region_filename = root_path.joinpath('data/inputs/tests/region.geojson')
        download_dir = root_path.joinpath('data/outputs/tests')
        crs = 'EPSG:3857'
        scale = 60
        im_param_list = [
            ['download', '-i', image_id, '-r', str(region_filename), '-dd',
             str(download_dir), '--crs', crs, '--scale', scale, '-o', '-m'],
        ]

        for im_params in im_param_list:
            with self.subTest('Download', cli_params=im_params):
                result = CliRunner().invoke(cli.cli, im_params, terminal_width=80)
                self.assertTrue(result.exit_code == 0, result.exception)

                with open(region_filename) as f:
                    region = json.load(f)

                filename = download_dir.joinpath(image_id.replace('/', '-') + '.tif')
                ee_coll_name = geedim.image.split_id(image_id)[0]
                gd_image = masked_image.get_class(ee_coll_name)._from_id(image_id, mask=True, region=region)
                _test_image_file(self, image_obj=gd_image, filename=filename, region=region, crs=crs, scale=scale,
                                 mask=True)

    def test_export(self):
        """ Test export command on one image, without waiting for completion """
        image_id = 'MODIS/006/MCD43A4/2019_01_02'
        bbox = (23.9, -33.6, 24, -33.5)
        crs = 'EPSG:3857'
        scale = 1000

        result = CliRunner().invoke(cli.cli, ['export', '-i', image_id, '-b', *bbox, '-df', 'geedim_test', '-nw',
                                              '--crs', crs, '--scale', scale, '-m', '-rs', 'bilinear'],
                                    terminal_width=80)
        self.assertTrue(result.exit_code == 0, result.exception)

    def test_composite_download(self):
        """ Test chaining of composite and download commands, to create and download one composite image.  """

        region_filename = root_path.joinpath('data/inputs/tests/region.geojson')
        download_dir = root_path.joinpath('data/outputs/tests')
        comp_ids = ['LANDSAT/LE07/C02/T1_L2/LE07_171083_20190129', 'LANDSAT/LE07/C02/T1_L2/LE07_171083_20190214',
                    'LANDSAT/LE07/C02/T1_L2/LE07_171083_20190302']
        pref_ids = [item for tup in zip(['-i'] * len(comp_ids), comp_ids) for item in tup]
        method = 'q_mosaic'
        pdict = dict(mask=True, crs='EPSG:3857', scale=60)

        cli_params = ['composite', *pref_ids, '-cm', method, '-m' if pdict['mask'] else '-nm',
                      '--resampling', 'bilinear', 'download', '-r', str(region_filename), '-dd', str(download_dir),
                      '--crs', pdict['crs'], '--scale', pdict['scale'], '-o']
        result = CliRunner().invoke(cli.cli, cli_params, terminal_width=100)

        self.assertTrue(result.exit_code == 0, result.exception)

        # recreate composite image and check against downloaded file
        gd_collection = collection.MaskedCollection.from_ids(comp_ids, mask=pdict['mask'])
        comp_im = gd_collection.composite(method)
        comp_fn = download_dir.joinpath(comp_im.name + '.tif')
        with open(region_filename) as f:
            region = json.load(f)
        _test_image_file(self, image_obj=comp_im, filename=comp_fn, region=region, **pdict)

    def test_search_composite_download(self):
        """ Test chaining of search, composite and download commands, to create and download one composite image. """

        region_filename = root_path.joinpath('data/inputs/tests/region.geojson')
        results_filename = root_path.joinpath('data/outputs/tests/search_results.json')
        download_dir = root_path.joinpath('data/outputs/tests')
        start_date = datetime.strptime('2019-03-11', '%Y-%m-%d')
        end_date = start_date + timedelta(days=6)
        method = 'q_mosaic'
        pdict = dict(mask=True, crs='EPSG:3857', scale=50)

        cli_params = ['search', '-c', 'sentinel2_sr', '-r', str(region_filename), '-s',
                      start_date.strftime("%Y-%m-%d"), '-e', end_date.strftime("%Y-%m-%d"), '-o', f'{results_filename}',
                      'composite', '-cm', method, '-m' if pdict['mask'] else '-nm', 'download', '-dd',
                      str(download_dir),
                      '--crs', pdict['crs'], '--scale', pdict['scale'], '-o']

        result = CliRunner().invoke(cli.cli, cli_params, terminal_width=100)

        self.assertTrue(result.exit_code == 0, result.exception)

        # recreate search results and composite image, and check against file
        with open(results_filename) as f:
            res_dict = json.load(f)
        res_df = pd.DataFrame.from_dict(res_dict, orient='index')
        res_df.DATE = [datetime.utcfromtimestamp(ts / 1000) for ts in res_df.DATE.values]
        _test_search_results(self, res_df, start_date, end_date)  # check results

        gd_collection = collection.MaskedCollection.from_ids(res_df.ID.values, mask=pdict['mask'])
        comp_im = gd_collection.composite(method)
        comp_fn = download_dir.joinpath(comp_im.name + '.tif')
        with open(region_filename) as f:
            region = json.load(f)
        _test_image_file(self, image_obj=comp_im, filename=comp_fn, region=region, **pdict)
