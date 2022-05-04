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

import ee
import numpy as np
import pandas as pd

import geedim.image
from geedim import image, collection, root_path, info
from geedim.enums import CompositeMethod, ResamplingMethod
from geedim.masked_image import MaskedImage
from tests.util import _test_image_file, _test_search_results, _setup_test


class TestApi(unittest.TestCase):
    """ Class to test backend (API) search, composite and export functionality. """

    @classmethod
    def setUpClass(cls):
        """ Initialise Earth Engine once for all the tests here. """
        _setup_test()

    def _test_image(self, image_id, mask=MaskedImage._default_mask):
        """ Test the validity of a geedim.image.MaskedImage by checking metadata.  """

        ee_coll_name = geedim.image.split_id(image_id)[0]
        gd_image = MaskedImage.from_id(image_id, mask=mask)
        self.assertTrue(gd_image.id == image_id, 'IDs match')

        sr_band_df = pd.DataFrame.from_dict(info.collection_info[ee_coll_name]['bands'])
        for key in ['bands', 'properties', 'id', 'crs', 'scale']:
            self.assertTrue(key in gd_image.info.keys(), msg='Image gd_info complete')
            self.assertTrue(gd_image.info[key] is not None, msg='Image gd_info complete')

        self.assertTrue(gd_image.scale > 0 and gd_image.scale < 5000, 'Scale in range')
        self.assertTrue(gd_image.crs != 'EPSG:4326', 'Non wgs84')

        im_band_df = pd.DataFrame.from_dict(gd_image.info['bands'])

        self.assertTrue(im_band_df.shape[0] >= sr_band_df.shape[0], 'Enough bands')
        for id in ['CLOUDLESS_MASK', 'CLOUD_MASK', 'SHADOW_MASK', 'FILL_MASK', 'CLOUD_DIST']:
            self.assertTrue(id in im_band_df.id.values, msg='Image has auxiliary bands')
        for id in sr_band_df.id.values:
            self.assertTrue(id in im_band_df.id.values, msg='Image has SR bands')

        # test reflectance statistics for a specific region
        region = {
            "type": "Polygon",
            "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]
        }
        sr_band_ids = sr_band_df.id.tolist()
        sr_image = gd_image.ee_image.select(sr_band_ids)
        std_refl = sr_image.reduceRegion(reducer='stdDev', geometry=region, scale=2 * gd_image.scale).getInfo()
        self.assertTrue(all(np.array(list(std_refl.values())) > 100), 'Std(SR) > 100')

        # test quality score for a specific region
        sr_image = gd_image.ee_image.select('CLOUD_DIST')
        max_score = sr_image.reduceRegion(reducer='max', geometry=region, scale=2 * gd_image.scale).getInfo()
        self.assertTrue(max_score['CLOUD_DIST'] < 5000 * 1.1, 'Max(CLOUD_DIST) < 1.1*CLOUD_DIST')

    def test_image(self):
        """ Test geedim.image.MaskedImage sub-classes. """
        im_param_list = [
            {'image_id': 'COPERNICUS/S2_SR/20190321T075619_20190321T081839_T35HKC', 'mask': False},
            {'image_id': 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190301', 'mask': True},
            # {'image_id': 'MODIS/006/MCD43A4/2019_01_01', 'mask': True, 'cloud_dist': 5000},
        ]

        for im_param_dict in im_param_list:
            with self.subTest('Image', **im_param_dict):
                self._test_image(**im_param_dict)

    def test_search(self):
        """ Test search on supported image collections.  """
        region = {
            "type": "Polygon",
            "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]
        }
        search_date_dict = {
            'LANDSAT/LT05/C02/T1_L2': ['2005-01-01', '2005-06-01'],
            'LANDSAT/LE07/C02/T1_L2': ['2019-01-01', '2019-02-01'],
            'LANDSAT/LC08/C02/T1_L2': ['2019-01-01', '2019-02-01'],
            'COPERNICUS/S2': ['2019-01-01', '2019-02-01'],
            'COPERNICUS/S2_SR': ['2019-01-01', '2019-02-01'],
            'MODIS/006/MCD43A4': ['2019-01-01', '2019-02-01']
        }
        cloudless_portion = 10
        for ee_coll_name, search_dates in search_date_dict.items():
            # find search start / end dates based on collection start / end
            with self.subTest('Search', ee_coll_name=ee_coll_name):
                gd_collection = collection.MaskedCollection.from_name(ee_coll_name)
                gd_collection = gd_collection.search(
                    search_dates[0], search_dates[1], region, cloudless_portion=cloudless_portion
                )
                _test_search_results(
                    self, gd_collection.properties, search_dates[0], search_dates[1],
                    cloudless_portion=cloudless_portion
                )

    def test_download(self):
        """ Test download of images from different collections, and with different crs, and scale params. """

        region = {
            "type": "Polygon",
            "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]
        }
        im_param_list = [
            {
                'image_id': 'COPERNICUS/S2_SR/20190321T075619_20190321T081839_T35HKC', 'mask': True, 'crs': None,
                'scale': 30, 'resampling': ResamplingMethod.near
            },
            {
                'image_id': 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190301', 'mask': True, 'crs': None, 'scale': None,
                'resampling': ResamplingMethod.near
            },
            # {'image_id': 'MODIS/006/MCD43A4/2019_01_01', 'mask': True, 'crs': 'EPSG:3857', 'scale': 500,
            #  'resampling': ResamplingMethod.near},
        ]

        for impdict in im_param_list:
            ee_coll_name = geedim.image.split_id(impdict['image_id'])[0]
            with self.subTest('Download', **impdict):
                # create image.MaskedImage
                gd_image = MaskedImage.from_id(impdict["image_id"], mask=impdict['mask'], region=region)
                # create a filename for these parameters
                name = impdict["image_id"].replace('/', '-')
                crs_str = impdict["crs"].replace(':', '_') if impdict["crs"] else 'None'
                filename = root_path.joinpath(f'data/outputs/tests/{name}_{crs_str}_{impdict["scale"]}m.tif')
                gd_image.download(
                    filename, region=region, crs=impdict["crs"], scale=impdict["scale"],
                    resampling=impdict["resampling"], overwrite=True
                )
                impdict.pop('image_id')
                _test_image_file(self, image_obj=gd_image, filename=filename, region=region, **impdict)

    def test_export(self):
        """ Test export of an image, without waiting for completion. """

        region = {
            "type": "Polygon",
            "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]
        }
        image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190128'
        ee_image = ee.Image(image_id)
        image.BaseImage(ee_image).export(image_id.replace('/', '-'), folder='geedim_test', region=region, wait=False)

    def _test_composite(self, gd_image):
        """ Test the metadata of a composite ee.Image for validity. """

        ee_coll_name = geedim.image.split_id(gd_image.id)[0]
        sr_band_df = pd.DataFrame.from_dict(info.collection_info[ee_coll_name]['bands'])
        for key in ['bands', 'properties', 'id']:
            self.assertTrue(key in gd_image.info.keys(), msg='Image gd_info complete')
            self.assertTrue(gd_image.info[key] is not None, msg='Image gd_info complete')

        for key in ['crs', 'scale']:
            self.assertTrue(gd_image.info[key] is None, msg='Composite in WGS84')

        im_band_df = pd.DataFrame.from_dict(gd_image.info['bands'])

        self.assertTrue(im_band_df.shape[0] >= sr_band_df.shape[0], 'Enough bands')
        for id in ['CLOUDLESS_MASK', 'CLOUD_MASK', 'SHADOW_MASK', 'FILL_MASK']:
            self.assertTrue(id in im_band_df.id.values, msg='Image has auxiliary bands')
        for id in sr_band_df.id.values:
            self.assertTrue(id in im_band_df.id.values, msg='Image has SR bands')

        # test image content for a specific region
        region = {
            "type": "Polygon",
            "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]
        }
        sr_band_ids = sr_band_df[~sr_band_df.abbrev.str.startswith('BT')].id.tolist()
        sr_image = gd_image.ee_image.select(sr_band_ids)
        mean_refl = sr_image.reduceRegion(reducer='mean', geometry=region, scale=100).getInfo()
        self.assertTrue(all(np.array(list(mean_refl.values())) > 100), 'Mean reflectance > 100')
        count_distinct_refl = sr_image.reduceRegion(ee.Reducer.countDistinct(), geometry=region, scale=100).getInfo()
        self.assertTrue(all(np.array(list(count_distinct_refl.values())) > 100), 'Distinct reflectance values > 100')

    def test_composite(self):
        """ Test each composite method on different collections. """

        param_list = [
            {
                'image_ids': ['LANDSAT/LE07/C02/T1_L2/LE07_171083_20190129',
                              'LANDSAT/LE07/C02/T1_L2/LE07_171083_20190214',
                              'LANDSAT/LE07/C02/T1_L2/LE07_171083_20190302'], 'mask': True
            },
            {
                'image_ids': ['COPERNICUS/S2_SR/20190311T075729_20190311T082820_T35HKC',
                              'COPERNICUS/S2_SR/20190316T075651_20190316T082220_T35HKC',
                              'COPERNICUS/S2_SR/20190321T075619_20190321T081839_T35HKC'], 'mask': True
            },
        ]

        for param_dict in param_list:
            for method in CompositeMethod:
                with self.subTest('Composite', method=method, **param_dict):
                    gd_collection = collection.MaskedCollection.from_list(param_dict['image_ids'])
                    comp_im = gd_collection.composite(
                        method=method, resampling=ResamplingMethod.bilinear, mask=param_dict['mask']
                    )
                    self._test_composite(comp_im)

##
