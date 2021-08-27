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
import pathlib
import unittest
from datetime import datetime

import pandas as pd
import rasterio as rio
from click.testing import CliRunner
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

import geedim.collection
from geedim import root_path, cli


class TestSearchCli(unittest.TestCase):
    """
    Test geedim search CLI
    """

    def _test_search_results(self, gd_coll_name, start_date, end_date, results_filename):
        """ checking search results from geojson file are valid"""

        # check file exists
        self.assertTrue(results_filename.exists(), 'Search results written to file')

        # read json search results into a pandas dataframe
        with open(results_filename) as f:
            res_dict = json.load(f)

        res_df = pd.DataFrame.from_dict(res_dict, orient='index')
        res_df.DATE = [datetime.utcfromtimestamp(ts / 1000) for ts in res_df.DATE.values]
        gd_collection = geedim.collection.cls_col_map[gd_coll_name]()

        # check results have correct columns, and sensible values
        self.assertGreater(res_df.shape[0], 0, 'Search returned one or more results')
        self.assertGreater(res_df.shape[1], 1, 'Search results contain two or more columns')
        self.assertTrue(set(res_df.columns) == set(gd_collection._im_props.ABBREV),
                        'Search results have correct columns')
        self.assertTrue(all(res_df.DATE >= start_date) and all(res_df.DATE <= end_date),
                        'Search results are in correct date range')
        self.assertTrue(all([gd_collection.ee_coll_name in im_id for im_id in res_df.ID.values]),
                        'Search results have correct EE ID')
        self.assertTrue(all(res_df.VALID >= 0) and all(res_df.VALID <= 100),
                        'Search results have correct validity range')
        # self.assertTrue(all(res_df.SCORE >= 0) and all(res_df.SCORE <= 100),
        #                 'Search results have correct q score range')

    def test_search_bbox(self):
        """ test `geedim search` with --bbox option"""

        # setup parmeters
        collection = 'landsat8_c2_l2'
        results_filename = root_path.joinpath('data/outputs/tests/search_results.json')
        start_date = datetime.strptime('2019-01-01', '%Y-%m-%d')
        end_date = datetime.strptime('2019-03-01', '%Y-%m-%d')

        # run command with parameters
        result = CliRunner().invoke(cli.cli, ['search', '-c', collection, '-b', 23.9, -33.6, 24, -33.5, '-s',
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


# TODO: consider decreasing testing here as there is somw overlap with test_api
class TestDownloadCli(unittest.TestCase):
    """
    Test geedim download/export CLI
    """

    def _test_download_files(self, ids, download_dir, region_bounds, crs=None, scale=None):
        """
        Test downloaded image file(s) for validity

        Parameters
        ----------
        ids : list
              List of EE ids passed to `geedim download --id ...`
        download_dir : str, pathlib.Path
                       Download directory passed to `geedim download`
        region_bounds : rasterio.coords.BoundingBox
                        Image region as a rasterio bbox
        crs : str, optional
              CRS string passed to `geedim download` if any
        scale : float, optional
                Pixel resolution passed to `geedim download` if any
        """

        for _id in ids:
            image_filename = pathlib.Path(download_dir).joinpath(_id.replace('/', '-') + '.tif')
            self.assertTrue(image_filename.exists(), 'Downloaded image exists')

            with rio.open(image_filename) as im:
                im_bounds_wgs84 = transform_bounds(im.crs, 'WGS84', *im.bounds)  # convert to WGS84
                self.assertFalse(rio.coords.disjoint_bounds(region_bounds, im_bounds_wgs84),
                                 msg='CLI and image region match')
                self.assertTrue(im.count > 0, 'Image has more than one band')

                if crs is not None:
                    self.assertEqual(CRS.from_string(crs).to_proj4(), im.crs.to_proj4(),
                                     msg='CLI and download image CRS match')

                if scale is not None:
                    self.assertAlmostEqual(scale, im.res[0], places=3, msg='CLI and download image scale match')
                # TODO: test masking when that is done, perhaps comparing to VALID_PORTION or similar

                # if 'MODIS' not in _id:
                #     if 'VALID_MASK' in im.descriptions:
                #         valid_mask = im.read(im.descriptions.index('VALID_MASK') + 1)
                #     else:
                #         valid_mask = im.read_masks(1) != 0
                #
                #     self.assertAlmostEqual(100 * valid_mask.mean(), float(im.get_tag_item('VALID_PORTION')), delta=5,
                #                            msg=f'VALID_PORTION matches mask mean for {_id}')

    def test_download_bbox(self):
        """
        Test `geedim download` with --bbox option
        """
        # COPERNICUS/S2_SR/20190108T090339_20190108T090341_T35SKT
        # ids = ['COPERNICUS/S2_SR/20190115T080251_20190115T082230_T35HKC', 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190128',
        #        'MODIS/006/MCD43A4/2019_01_01']
        ids = ['COPERNICUS/S2_SR/20190125T080221_20190125T082727_T35HKC']
        download_dir = root_path.joinpath('data/outputs/tests')
        bbox = (23.95, -33.6, 24, -33.55)
        prefixed_ids = [val for tup in zip(['-i'] * len(ids), ids) for val in tup]

        # invoke CLI
        result = CliRunner().invoke(cli.cli, ['download', *prefixed_ids, '-b', *bbox, '-dd', str(download_dir), '-o',
                                              '-m'], terminal_width=80)
        self.assertTrue(result.exit_code == 0, result.exception)

        # check downloaded images
        region_bounds = rio.coords.BoundingBox(*bbox)
        self._test_download_files(ids, download_dir, region_bounds)

    def test_download_region(self):
        """
        Test `geedim download` with --region, --crs and --scale options
        """
        # ids = ['COPERNICUS/S2_SR/20190120T080239_20190120T082812_T35HKC', 'LANDSAT/LC08/C02/T1_L2/LC08_182037_20190118',
        #        'MODIS/006/MCD43A4/2019_01_02']
        ids = ['LANDSAT/LC08/C02/T1_L2/LC08_172083_20190112']
        region_filename = root_path.joinpath('data/inputs/tests/region.geojson')
        download_dir = root_path.joinpath('data/outputs/tests')
        crs = 'EPSG:3857'
        scale = 30

        prefixed_ids = [val for tup in zip(['-i'] * len(ids), ids) for val in tup]

        # invoke CLI
        result = CliRunner().invoke(cli.cli, ['download', *prefixed_ids, '-r', str(region_filename),
                                              '-dd', str(download_dir), '--crs', crs, '--scale', scale, '-o', '-m'],
                                    terminal_width=80)
        self.assertTrue(result.exit_code == 0, result.exception)

        # convert region json file into rasterio bounds
        with open(region_filename) as f:
            region_bounds_geojson = json.load(f)

        region_arr = pd.DataFrame(region_bounds_geojson['coordinates'][0], columns=['x', 'y'])  # avoid numpy dependency
        region_bounds = rio.coords.BoundingBox(region_arr.x.min(), region_arr.y.min(), region_arr.x.max(),
                                               region_arr.y.max())

        # check downloaded images
        self._test_download_files(ids, download_dir, region_bounds, crs=crs, scale=scale)

    def test_export(self):
        """
        Test `geedim export` with --bbox option
        """
        # test export of one image only to save time, and because we can't check the exported image validity
        ids = ['MODIS/006/MCD43A4/2019_01_02']
        bbox = (23.9, -33.6, 24, -33.5)
        prefixed_ids = [val for tup in zip(['-i'] * len(ids), ids) for val in tup]
        crs = 'EPSG:3857'
        scale = 100

        # invoke CLI
        result = CliRunner().invoke(cli.cli, ['export', *prefixed_ids, '-b', *bbox, '-df', 'geedim_test', '-w',
                                              '--crs', crs, '--scale', scale, '-m'],
                                    terminal_width=80)
        self.assertTrue(result.exit_code == 0, result.exception)
