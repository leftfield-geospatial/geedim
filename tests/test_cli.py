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
from datetime import datetime

import pandas as pd
from click.testing import CliRunner

from geedim import root_path, cli
from tests import test_api

class TestCli(unittest.TestCase):
    """
    Test geedim download/export CLI
    """

    def test_search(self):
        """ test `geedim search` with --bbox option"""

        results_filename = root_path.joinpath('data/outputs/tests/search_results.json')
        region_filename = root_path.joinpath('data/inputs/tests/region.geojson')
        start_date = datetime.strptime('2019-01-01', '%Y-%m-%d')
        end_date = datetime.strptime('2019-03-01', '%Y-%m-%d')
        search_param_list = [
            ['search', '-c', 'landsat8_c2_l2', '-b', 23.9, -33.6, 24, -33.5, '-s', start_date.strftime("%Y-%m-%d"), '-e', end_date.strftime("%Y-%m-%d"),
             '-o', f'{results_filename}'],
            ['search', '-c', 'sentinel2_toa', '-r', str(region_filename), '-s', start_date.strftime("%Y-%m-%d"), '-e', end_date.strftime("%Y-%m-%d"),
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
                test_api.TestApi._test_search_results(self, res_df, start_date, end_date)  # check results


    def test_download(self):
        """
        Test `geedim download` with --region, --crs and --scale options
        """
        image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190112'
        region_filename = root_path.joinpath('data/inputs/tests/region.geojson')
        bbox = (23.95, -33.6, 24, -33.55)
        download_dir = root_path.joinpath('data/outputs/tests')
        crs = 'EPSG:3857'
        scale = 60
        im_param_list = [
            ['download', '-i', 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190112', '-r', str(region_filename), '-dd', str(download_dir), '--crs', crs, '--scale', scale, '-o', '-m', '-sr'],
            # ['download', '-i', 'COPERNICUS/S2/20190321T075619_20190321T081839_T35HKC', '-b', *bbox, '-dd', str(download_dir), '--crs', crs, '--scale', scale, '-o', '-nm'],
        ]

        for im_params in im_param_list:
            with self.subTest('Download', cli_params=im_params):
                result = CliRunner().invoke(cli.cli, im_params, terminal_width=80)
                self.assertTrue(result.exit_code == 0, result.exception)

                with open(region_filename) as f:
                    region = json.load(f)

                filename = download_dir.joinpath(image_id.replace('/','-') + '.tif')
                test_api.TestApi._test_image_file(self, image_obj=image_id, filename=filename, region=region, crs=crs, scale=scale, mask=True)

    def test_export(self):
        """
        Test `geedim export` with --bbox option
        """
        image_id = 'MODIS/006/MCD43A4/2019_01_02'
        bbox = (23.9, -33.6, 24, -33.5)
        crs = 'EPSG:3857'
        scale = 1000

        # invoke CLI
        result = CliRunner().invoke(cli.cli, ['export', '-i', image_id, '-b', *bbox, '-df', 'geedim_test', '-nw',
                                              '--crs', crs, '--scale', scale, '-m'],
                                    terminal_width=80)
        self.assertTrue(result.exit_code == 0, result.exception)
