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


class TestGeeDimCli(unittest.TestCase):
    """
    Test command line interface
    """

    def _test_search_results(self, collection, start_date, end_date, results_filename):
        """ checking search results from geojson file are valid"""

        # check file exists
        self.assertTrue(results_filename.exists(), 'Search results written to file')

        # read json search results into a pandas dataframe
        with open(results_filename) as f:
            res_dict = json.load(f)

        res_df = pd.DataFrame.from_dict(res_dict)
        res_df.DATE = [datetime.utcfromtimestamp(ts / 1000) for ts in res_df.DATE.values]
        imseach_obj = cli.cls_col_map[collection](collection=collection)

        # check results have correct columns, and sensible values
        self.assertGreater(res_df.shape[0], 0, 'Search returned one or more results')
        self.assertGreater(res_df.shape[1], 1, 'Search results contain two or more columns')
        self.assertTrue(set(res_df.columns) == set(imseach_obj._im_props.ABBREV),
                        'Search results have correct columns')
        self.assertTrue(all(res_df.DATE >= start_date) and all(res_df.DATE <= end_date),
                        'Search results are in correct date range')
        self.assertTrue(all([imseach_obj._collection_info['ee_collection'] in im_id for im_id in res_df.ID.values]),
                        'Search results have correct EE ID')
        self.assertTrue(all(res_df.VALID >= 0) and all(res_df.VALID <= 100),
                        'Search results have correct validity range')
        self.assertTrue(all(res_df.SCORE >= 0) and all(res_df.SCORE <= 100),
                        'Search results have correct q score range')

    def test_search_bbox(self):
        """ test `geedim search` with --bbox option"""

        # setup parmeters
        collection = 'landsat8_c2_l2'
        results_filename = root_path.joinpath('data/outputs/tests/search_results.json')
        start_date = datetime.strptime('2019-01-01', '%Y-%m-%d')
        end_date = datetime.strptime('2019-03-01', '%Y-%m-%d')

        # run command with parameters
        result = CliRunner().invoke(cli.cli, ['search', '-c', collection, '-b', 23.9, 33.5, 24, 33.6, '-s',
                                              start_date.strftime("%Y-%m-%d"), '-e', end_date.strftime("%Y-%m-%d"),
                                              '-o',
                                              f'{results_filename}'], terminal_width=80)

        self.assertTrue(result.exit_code == 0, result.exception)  # search returned ok

        self._test_search_results(collection, start_date, end_date, results_filename)  # check results

    def test_search_region(self):
        """ test `geedim search` with --region option"""

        # setup parmeters
        collection = 'sentinel2_sr'
        results_filename = root_path.joinpath('data/outputs/tests/search_results.json')
        start_date = datetime.strptime('2019-01-01', '%Y-%m-%d')
        end_date = datetime.strptime('2019-01-15', '%Y-%m-%d')
        region_filename = root_path.joinpath('data/inputs/tests/region.geojson')

        result = CliRunner().invoke(cli.cli, ['search', '-c', collection, '-r', str(region_filename), '-s',
                                              start_date.strftime("%Y-%m-%d"), '-e', end_date.strftime("%Y-%m-%d"),
                                              '-o',
                                              str(results_filename)], terminal_width=80)

        self.assertTrue(result.exit_code == 0, result.exception)  # search returned ok

        self._test_search_results(collection, start_date, end_date, results_filename)  # check results
##
