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

import glob
import os
import warnings

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from geedim import info, root_path, _ee_init
from geedim.image import split_id, BaseImage
from geedim.masked_image import MaskedImage


def _setup_test():
    """ Test initialisation """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    _ee_init()
    test_out_dir = root_path.joinpath('data/outputs/tests/')
    if not test_out_dir.exists():
        os.makedirs(test_out_dir)
    file_list = glob.glob(str(test_out_dir.joinpath('*')))
    for f in file_list:
        os.remove(f)


def nan_equals(a, b):
    """Compare two numpy objects a & b, returning true where elements of both a & b are nan"""
    return (a == b) | (np.isnan(a) & np.isnan(b))


def _test_search_results(test_case, results, start_date, end_date, cloudless_portion=0):
    """ Test the validity of a search results dataframe against the search parameters. """
    res_df = pd.DataFrame(results)
    test_case.assertGreater(res_df.shape[0], 0, 'Search returned one or more results')
    test_case.assertGreater(res_df.shape[1], 1, 'Search results contain two or more columns')

    image_id = res_df['system:id'][0]
    ee_coll_name = split_id(image_id)[0]
    summary_key_df = pd.DataFrame(info.collection_info[ee_coll_name]['properties'])
    dates = pd.to_datetime(res_df['system:time_start'], unit='ms')
    test_case.assertTrue(set(res_df.columns) == set(summary_key_df.PROPERTY),
                         'Search results have correct columns')
    test_case.assertTrue(all(dates >= start_date) and all(dates <= end_date),
                         'Search results are in correct date range')
    test_case.assertTrue(all([ee_coll_name in im_id for im_id in res_df['system:id'].values]),
                         'Search results have correct EE ID')
    if ee_coll_name != 'MODIS/006/MCD43A4':
        test_case.assertTrue(all(res_df.CLOUDLESS_PORTION >= cloudless_portion) and all(res_df.CLOUDLESS_PORTION <= 100),
                             'Search results have correct validity range')
        test_case.assertTrue(all(res_df.FILL_PORTION >= cloudless_portion) and all(res_df.FILL_PORTION <= 100),
                             'Search results have correct validity range')
        # test_case.assertTrue(all(res_df.CLOUD_DIST >= 0), 'Search results have correct q score range')


def _test_image_file(test_case, image_obj, filename, region, crs=None, scale=None,
                     mask=MaskedImage._default_mask, resampling=BaseImage._default_resampling):
    """ Test downloaded image file against corresponding image object """

    # create objects to test against
    if isinstance(image_obj, str):  # create image.MaskedImage from ID
        ee_coll_name = split_id(image_obj)[0]
        gd_image = MaskedImage.from_id(image_obj, mask=mask)
    elif isinstance(image_obj, BaseImage):
        gd_image = image_obj
        ee_coll_name = split_id(gd_image.id)[0]
    else:
        raise TypeError(f'Unsupported image_obj type: {image_obj.__class__}')

    gd_info = gd_image.info

    exp_image = gd_image._prepare_for_export(region=region, crs=crs, scale=scale)

    region_arr = pd.DataFrame(region['coordinates'][0], columns=['x', 'y'])  # avoid numpy dependency
    region_bounds = rio.coords.BoundingBox(region_arr.x.min(), region_arr.y.min(), region_arr.x.max(),
                                           region_arr.y.max())

    # check the validity of the downloaded file
    test_case.assertTrue(filename.exists(), msg='Download file exists')
    with rio.open(filename) as im:
        # check bands, crs and scale
        test_case.assertEqual(len(gd_info['bands']), im.count, msg='EE and download image band count match')
        exp_epsg = CRS.from_string(crs if crs else gd_image.crs).to_epsg()
        test_case.assertEqual(exp_epsg, im.crs.to_epsg(), msg='EE and download image CRS match')
        test_case.assertAlmostEqual(scale if scale else gd_image.scale, im.res[0], places=3,
                                    msg='EE and download image scale match')

        # check image bounds coincide with region
        im_bounds_wgs84 = transform_bounds(im.crs, 'WGS84', *im.bounds)  # convert to WGS84 geojson
        test_case.assertFalse(rio.coords.disjoint_bounds(region_bounds, im_bounds_wgs84),
                              msg='Search and image bounds match')

        if mask:  # and not ('sentinel2' in gd_coll_name):  # check mask is same as CLOUDLESS_MASK band
            im_mask = ~nan_equals(im.read(1), im.nodata)
            cloudless_mask = im.read(im.descriptions.index('CLOUDLESS_MASK') + 1)
            cloudless_mask[nan_equals(cloudless_mask, im.nodata)] = 0
            test_case.assertTrue(np.all(cloudless_mask.astype('bool') == im_mask), 'mask == CLOUDLESS_MASK')
        else:
            valid_mask = im.read(im.descriptions.index('CLOUDLESS_MASK') + 1).astype(bool)
            cloud_mask = im.read(im.descriptions.index('CLOUD_MASK') + 1).astype(bool)
            shadow_mask = im.read(im.descriptions.index('SHADOW_MASK') + 1).astype(bool)
            fill_mask = im.read(im.descriptions.index('FILL_MASK') + 1).astype(bool)
            _cloudless_mask = ~(cloud_mask | shadow_mask) & fill_mask
            test_case.assertTrue(np.all(_cloudless_mask[valid_mask]), 'mask contains CLOUDLESS_MASK')
            # pyplot.figure();pyplot.subplot(2,2,1);pyplot.imshow(im_mask);pyplot.subplot(2,2,2);pyplot.imshow(valid_mask);pyplot.subplot(2,2,3);pyplot.imshow(cloud_mask);pyplot.subplot(2,2,4);pyplot.imshow(shadow_mask)

        # do basic checks on image content
        sr_band_df = pd.DataFrame.from_dict(info.collection_info[ee_coll_name]['bands'])
        for band_i, band_row in sr_band_df.iterrows():
            if 'BT' not in band_row.abbrev and band_row.bw_start < 5:  # exclude mid-far IR
                sr_band = im.read(im.descriptions.index(band_row.id) + 1, masked=True)
                test_case.assertTrue(sr_band.mean() > 100, f'Mean {band_row.id} reflectance > 100')
                test_case.assertTrue(len(np.unique(sr_band)) > 100, f'Distinct {band_row.id} reflectance values > 100')

        # where search stats exist, check they match image content
        if 'FILL_PORTION' in gd_info['properties'] and 'CLOUDLESS_PORTION' in gd_info['properties']:
            fill_portion = gd_info['properties']['FILL_PORTION']
            cloudless_portion = gd_info['properties']['CLOUDLESS_PORTION']
            fill_mask = im.read(im.descriptions.index('FILL_MASK') + 1)
            fill_mask[nan_equals(fill_mask, im.nodata)] = 0
            cloudless_mask = im.read(im.descriptions.index('CLOUDLESS_MASK') + 1)
            cloudless_mask[nan_equals(cloudless_mask, im.nodata)] = 0
            test_case.assertAlmostEqual(cloudless_portion if mask else fill_portion, 100 * fill_mask.mean(), delta=5,
                                        msg='EE and file fill portion match')
            test_case.assertAlmostEqual(cloudless_portion, 100 * cloudless_mask.mean(), delta=5,
                                        msg='EE and file cloudless portions match')
