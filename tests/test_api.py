import unittest
from datetime import datetime, timedelta

import ee
import pandas as pd
import importlib
import numpy as np
from geedim import export, collection, root_path, info, image

if importlib.util.find_spec("rasterio"):
    import rasterio as rio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
else:
    raise ModuleNotFoundError('Rasterio is needed to run the unit tests: `conda install -c conda-forge rasterio`')

class TestApi(unittest.TestCase):
    """
    Test backend functionality in search and download modules
    """

    @classmethod
    def setUpClass(cls):
        ee.Initialize()

    def _test_image(self, image_id, mask=False, scale_refl=False):
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

        region = {"type": "Polygon",
                  "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}
        if scale_refl and ('landsat' in gd_coll_name):
            sr_band_ids = sr_band_df[sr_band_df.id.str.startswith('SR_B')].id.tolist()
            sr_image = gd_image.ee_image.select(sr_band_ids)
            max_refl = sr_image.reduceRegion(reducer='max', geometry=region, scale=2 * gd_image.scale).getInfo()
            self.assertTrue(all(np.array(list(max_refl.values())) <= 11000), 'Scaled reflectance in range')

    def test_image(self):
        im_param_list = [
            {'image_id': 'COPERNICUS/S2_SR/20190321T075619_20190321T081839_T35HKC', 'mask': False, 'scale_refl': False},
            {'image_id': 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190301', 'mask': True, 'scale_refl': True},
            {'image_id': 'MODIS/006/MCD43A4/2019_01_01', 'mask': True, 'scale_refl': False},
        ]

        for im_param_dict in im_param_list:
            with self.subTest('Image', **im_param_dict):
                self._test_image(**im_param_dict)

    @staticmethod
    def _test_search_results(test_case, res_df, start_date, end_date, valid_portion=0):
        """ checking search results from geojson file are valid"""
        # check results have correct columns, and sensible values
        test_case.assertGreater(res_df.shape[0], 0, 'Search returned one or more results')
        test_case.assertGreater(res_df.shape[1], 1, 'Search results contain two or more columns')

        image_id = res_df.ID[0]
        ee_coll_name = image.split_id(image_id)[0]
        gd_coll_name = info.ee_to_gd[ee_coll_name]
        summary_key_df = pd.DataFrame(info.collection_info[gd_coll_name]['properties'])

        test_case.assertTrue(set(res_df.columns) == set(summary_key_df.ABBREV),
                        'Search results have correct columns')
        test_case.assertTrue(all(res_df.DATE >= start_date) and all(res_df.DATE <= end_date),
                        'Search results are in correct date range')
        test_case.assertTrue(all([ee_coll_name in im_id for im_id in res_df.ID.values]),
                        'Search results have correct EE ID')
        if gd_coll_name != 'modis_nbar':
            test_case.assertTrue(all(res_df.VALID >= valid_portion) and all(res_df.VALID <= 100),
                            'Search results have correct validity range')
            test_case.assertTrue(all(res_df.SCORE >= 0), 'Search results have correct q score range')


    def test_search(self):
        region = {"type": "Polygon",
                  "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}
        start_date = datetime.strptime('2019-02-01', '%Y-%m-%d')
        end_date = start_date + timedelta(days=32)
        valid_portion = 10
        for gd_coll_name in info.gd_to_ee.keys():
            with self.subTest('Search', gd_coll_name=gd_coll_name):
                gd_collection = collection.Collection(gd_coll_name)
                res_df = gd_collection.search(start_date, end_date, region, valid_portion=valid_portion)
                self._test_search_results(self, res_df, start_date, end_date, valid_portion=valid_portion)

    # TODO: separate API search and download testing (?) and or make one _test_download fn that works from api and cli
    @staticmethod
    def _test_image_file(test_case, image_obj, filename, region, crs=None, scale=None, mask=False, scale_refl=False):
        """ Test downloaded file against image.MaskedImage object """
        if isinstance(image_obj, str):
            ee_coll_name = image.split_id(image_obj)[0]
            gd_coll_name = info.ee_to_gd[ee_coll_name]
            gd_image = image.get_class(gd_coll_name).from_id(image_obj, mask=mask, scale_refl=scale_refl)
        elif isinstance(image_obj, image.Image):
            gd_image = image_obj
            ee_coll_name = image.split_id(gd_image.id)[0]
            gd_coll_name = info.ee_to_gd[ee_coll_name]
        else:
            raise TypeError(f'Unsupported image_obj type: {image_obj.__class__}')

        gd_info = gd_image.info
        sr_band_df = pd.DataFrame.from_dict(info.collection_info[gd_coll_name]['bands'])

        exp_image = export._ExportImage(gd_image, name=gd_image.id, exp_region=region, exp_crs=crs, exp_scale=scale)
        exp_image.parse_attributes()

        region_arr = pd.DataFrame(region['coordinates'][0], columns=['x', 'y'])  # avoid numpy dependency
        region_bounds = rio.coords.BoundingBox(region_arr.x.min(), region_arr.y.min(), region_arr.x.max(),
                                               region_arr.y.max())

        # check the validity of the downloaded file
        test_case.assertTrue(filename.exists(), msg='Download file exists')

        with rio.open(filename) as im:
            test_case.assertEqual(len(gd_info['bands']), im.count, msg='EE and download image band count match')
            exp_epsg = CRS.from_string(exp_image.exp_crs).to_epsg()
            test_case.assertEqual(exp_epsg, im.crs.to_epsg(), msg='EE and download image CRS match')
            test_case.assertAlmostEqual(exp_image.exp_scale, im.res[0], places=3,
                                        msg='EE and download image scale match')

            im_bounds_wgs84 = transform_bounds(im.crs, 'WGS84', *im.bounds)  # convert to WGS84 geojson
            test_case.assertFalse(rio.coords.disjoint_bounds(region_bounds, im_bounds_wgs84),
                                  msg='Search and image bounds match')

            if scale_refl and ('landsat' in gd_coll_name):
                sr_band_ids = sr_band_df[sr_band_df.id.str.startswith('SR_B')].id.tolist()
                sr_band_idx = [im.descriptions.index(sr_id) + 1 for sr_id in sr_band_ids]
                sr_bands = im.read(sr_band_idx)
                test_case.assertTrue(sr_bands.max() <= 11000, 'Scaled reflectance in range')

            if mask:
                im_mask = im.read_masks(im.descriptions.index('VALID_MASK') + 1).astype(bool)
                valid_mask = im.read(im.descriptions.index('VALID_MASK') + 1, masked=False).astype(bool)
                test_case.assertTrue(np.all(im_mask == valid_mask), 'mask == VALID_MASK')

    def test_download(self):
        # construct image filename based on id, crs and scale
        # delay setting crs etc if its None, so we can test download_image(...crs=None,scale=None)

        region = {"type": "Polygon",
                  "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}
        # region = {"type": "Polygon",
        #           "coordinates": [[[24.1, -33.7], [24.1, -33.5], [23.9, -33.5], [23.9, -33.7], [24.1, -33.7]]]}
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
                gd_image = image.get_class(gd_coll_name).from_id(impdict["image_id"], mask=impdict['mask'],
                                                                 scale_refl=impdict['scale_refl'])
                name = impdict["image_id"].replace('/', '-')
                crs_str = impdict["crs"].replace(':', '_') if impdict["crs"] else 'None'
                filename = root_path.joinpath(f'data/outputs/tests/{name}_{crs_str}_{impdict["scale"]}m.tif')
                export.download_image(gd_image, filename, region=region, crs=impdict["crs"], scale=impdict["scale"],
                                      overwrite=True)
                impdict.pop('image_id')
                self._test_image_file(self, image_obj=gd_image, filename=filename, region=region, **impdict)


    def test_export(self):
        region = {"type": "Polygon",
                  "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}
        image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20190128'
        ee_image = ee.Image(image_id)
        export.export_image(ee_image, image_id.replace('/', '-'), folder='geedim_test', region=region, wait=False)

    def _test_composite(self, ee_image, mask=False, scale_refl=False):
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

        region = {"type": "Polygon",
                  "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}
        if scale_refl and ('landsat' in gd_coll_name):
            sr_band_ids = sr_band_df[sr_band_df.id.str.startswith('SR_B')].id.tolist()
            sr_image = gd_image.ee_image.select(sr_band_ids)
            max_refl = sr_image.reduceRegion(reducer='max', geometry=region, scale=60).getInfo()
            self.assertTrue(all(np.array(list(max_refl.values())) <= 10000), 'Scaled reflectance in range')

    def test_composite(self):
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
