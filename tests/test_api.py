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

import unittest
from datetime import datetime, timedelta

import ee
import numpy as np
import pandas as pd

from geedim import export, collection, root_path, info, image
from tests.util import _test_image_file, _test_search_results


class TestApi(unittest.TestCase):
    """ Class to test backend (API) search, composite and export functionality. """

    @classmethod
    def setUpClass(cls):
        """ Initialise Earth Engine once for all the tests here. """
        ee.Initialize()

    def _test_image(self, image_id, mask=False, scale_refl=False):
        """ Test the validity of a geedim.image.MaskedImage by checking metadata.  """

        ee_coll_name = image.split_id(image_id)[0]
        gd_coll_name = info.ee_to_gd[ee_coll_name]
        gd_image = image.get_class(gd_coll_name).from_id(image_id, mask=mask, scale_refl=scale_refl)
        self.assertTrue(gd_image.id == image_id, 'IDs match')

        sr_band_df = pd.DataFrame.from_dict(info.collection_info[gd_coll_name]['bands'])
        for key in ['bands', 'properties', 'id', 'crs', 'scale']:
            self.assertTrue(key in gd_image.info.keys(), msg='Image gd_info complete')
            self.assertTrue(gd_image.info[key] is not None, msg='Image gd_info complete')

        self.assertTrue(gd_image.scale > 0 and gd_image.scale < 5000, 'Scale in range')
        self.assertTrue(gd_image.crs != 'EPSG:4326', 'Non wgs84')

        im_band_df = pd.DataFrame.from_dict(gd_image.info['bands'])

        self.assertTrue(im_band_df.shape[0] >= sr_band_df.shape[0], 'Enough bands')
        for id in ['VALID_MASK', 'CLOUD_MASK', 'SHADOW_MASK', 'FILL_MASK', 'SCORE']:
            self.assertTrue(id in im_band_df.id.values, msg='Image has auxiliary bands')
        for id in sr_band_df.id.values:
            self.assertTrue(id in im_band_df.id.values, msg='Image has SR bands')

        # test landsat reflectance statistics for a specific region
        region = {"type": "Polygon",
                  "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}
        if scale_refl and ('landsat' in gd_coll_name):
            sr_band_ids = sr_band_df[sr_band_df.id.str.startswith('SR_B')].id.tolist()
            sr_image = gd_image.ee_image.select(sr_band_ids)
            max_refl = sr_image.reduceRegion(reducer='max', geometry=region, scale=2 * gd_image.scale).getInfo()
            self.assertTrue(all(np.array(list(max_refl.values())) <= 11000), 'Scaled reflectance in range')

    def test_image(self):
        """ Test geedim.image.MaskedImage sub-classes. """
        im_param_list = [
            {'image_id': 'COPERNICUS/S2_SR/20190321T075619_20190321T081839_T35HKC', 'mask': False, 'scale_refl': False},
            {'image_id': 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190301', 'mask': True, 'scale_refl': True},
            {'image_id': 'MODIS/006/MCD43A4/2019_01_01', 'mask': True, 'scale_refl': False},
        ]

        for im_param_dict in im_param_list:
            with self.subTest('Image', **im_param_dict):
                self._test_image(**im_param_dict)

    def test_search(self):
        """ Test search on all supported image collections.  """
        region = {"type": "Polygon",
                  "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}
        start_date = datetime.strptime('2019-02-01', '%Y-%m-%d')
        end_date = start_date + timedelta(days=32)
        valid_portion = 10
        for gd_coll_name in info.gd_to_ee.keys():
            with self.subTest('Search', gd_coll_name=gd_coll_name):
                gd_collection = collection.Collection(gd_coll_name)
                res_df = gd_collection.search(start_date, end_date, region, valid_portion=valid_portion)
                _test_search_results(self, res_df, start_date, end_date, valid_portion=valid_portion)

    def test_download(self):
        """ Test download of images from different collections, and with different crs, scale and scale_refl params. """

        region = {"type": "Polygon",
                  "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}
        im_param_list = [
            {'image_id': 'COPERNICUS/S2_SR/20190321T075619_20190321T081839_T35HKC', 'mask': True, 'scale_refl': False,
             'crs': None, 'scale': 30},
            {'image_id': 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190301', 'mask': True, 'scale_refl': True, 'crs': None,
             'scale': None},
            {'image_id': 'MODIS/006/MCD43A4/2019_01_01', 'mask': True, 'scale_refl': False, 'crs': 'EPSG:3857',
             'scale': 500},
        ]

        for impdict in im_param_list:
            ee_coll_name = image.split_id(impdict['image_id'])[0]
            gd_coll_name = info.ee_to_gd[ee_coll_name]
            with self.subTest('Download', **impdict):
                # create image.MaskedImage
                gd_image = image.get_class(gd_coll_name).from_id(impdict["image_id"], mask=impdict['mask'],
                                                                 scale_refl=impdict['scale_refl'])
                # create a filename for these parameters
                name = impdict["image_id"].replace('/', '-')
                crs_str = impdict["crs"].replace(':', '_') if impdict["crs"] else 'None'
                filename = root_path.joinpath(f'data/outputs/tests/{name}_{crs_str}_{impdict["scale"]}m.tif')
                export.download_image(gd_image, filename, region=region, crs=impdict["crs"], scale=impdict["scale"],
                                      overwrite=True)
                impdict.pop('image_id')
                _test_image_file(self, image_obj=gd_image, filename=filename, region=region, **impdict)

    def test_export(self):
        """ Test export of an image, without waiting for completion. """

        region = {"type": "Polygon",
                  "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}
        image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190128'
        ee_image = ee.Image(image_id)
        export.export_image(ee_image, image_id.replace('/', '-'), folder='geedim_test', region=region, wait=False)

    def _test_composite(self, ee_image, mask=False, scale_refl=False):
        """ Test the metadata of a composite ee.Image for validity. """

        gd_image = image.Image(ee_image)
        ee_coll_name = image.split_id(gd_image.id)[0]
        gd_coll_name = info.ee_to_gd[ee_coll_name]

        sr_band_df = pd.DataFrame.from_dict(info.collection_info[gd_coll_name]['bands'])
        for key in ['bands', 'properties', 'id']:
            self.assertTrue(key in gd_image.info.keys(), msg='Image gd_info complete')
            self.assertTrue(gd_image.info[key] is not None, msg='Image gd_info complete')

        for key in ['crs', 'scale']:
            self.assertTrue(gd_image.info[key] is None, msg='Composite in WGS84')

        im_band_df = pd.DataFrame.from_dict(gd_image.info['bands'])

        self.assertTrue(im_band_df.shape[0] >= sr_band_df.shape[0], 'Enough bands')
        for id in ['VALID_MASK', 'CLOUD_MASK', 'SHADOW_MASK', 'FILL_MASK', 'SCORE']:
            self.assertTrue(id in im_band_df.id.values, msg='Image has auxiliary bands')
        for id in sr_band_df.id.values:
            self.assertTrue(id in im_band_df.id.values, msg='Image has SR bands')

        # test landsat reflectance statistics for a specific region
        region = {"type": "Polygon",
                  "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}
        if scale_refl and ('landsat' in gd_coll_name):
            sr_band_ids = sr_band_df[sr_band_df.id.str.startswith('SR_B')].id.tolist()
            sr_image = gd_image.ee_image.select(sr_band_ids)
            max_refl = sr_image.reduceRegion(reducer='max', geometry=region, scale=60).getInfo()
            self.assertTrue(all(np.array(list(max_refl.values())) <= 10000), 'Scaled reflectance in range')

    def test_composite(self):
        """ Test each composite method on different collections. """

        methods = collection.Collection.composite_methods
        param_list = [
            {'image_ids': ['LANDSAT/LE07/C02/T1_L2/LE07_171083_20190129', 'LANDSAT/LE07/C02/T1_L2/LE07_171083_20190214',
                           'LANDSAT/LE07/C02/T1_L2/LE07_171083_20190302'], 'scale_refl': True, 'mask': True},
            {'image_ids': ['COPERNICUS/S2_SR/20190311T075729_20190311T082820_T35HKC',
                           'COPERNICUS/S2_SR/20190316T075651_20190316T082220_T35HKC',
                           'COPERNICUS/S2_SR/20190321T075619_20190321T081839_T35HKC'], 'scale_refl': True,
             'mask': True},
        ]

        for param_dict in param_list:
            for method in methods:
                with self.subTest('Composite', method=method, **param_dict):
                    gd_collection = collection.Collection.from_ids(**param_dict)
                    comp_im, comp_id = gd_collection.composite(method=method)
                    self._test_composite(comp_im, mask=param_dict['mask'], scale_refl=param_dict['scale_refl'])

##
