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

import importlib
import numpy as np
import pandas as pd

from geedim import export, info, image

if importlib.util.find_spec("rasterio"):
    import rasterio as rio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
else:
    raise ModuleNotFoundError('Rasterio is needed to run the unit tests: `conda install -c conda-forge rasterio`')


def _test_search_results(test_case, res_df, start_date, end_date, valid_portion=0):
    """ Test the validity of a search results dataframe against the search parameters. """

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


def _test_image_file(test_case, image_obj, filename, region, crs=None, scale=None, mask=False, scale_refl=False):
    """ Test downloaded image file against corresponding image object """

    # create objects to test against
    if isinstance(image_obj, str):  # create image.MaskedImage from ID
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
        # check bands, crs and scale
        test_case.assertEqual(len(gd_info['bands']), im.count, msg='EE and download image band count match')
        exp_epsg = CRS.from_string(exp_image.exp_crs).to_epsg()
        test_case.assertEqual(exp_epsg, im.crs.to_epsg(), msg='EE and download image CRS match')
        test_case.assertAlmostEqual(exp_image.exp_scale, im.res[0], places=3,
                                    msg='EE and download image scale match')

        # check image bounds coincide with region
        im_bounds_wgs84 = transform_bounds(im.crs, 'WGS84', *im.bounds)  # convert to WGS84 geojson
        test_case.assertFalse(rio.coords.disjoint_bounds(region_bounds, im_bounds_wgs84),
                              msg='Search and image bounds match')

        if scale_refl and ('landsat' in gd_coll_name):  # check surface reflectance in range
            sr_band_ids = sr_band_df[sr_band_df.id.str.startswith('SR_B')].id.tolist()
            sr_band_idx = [im.descriptions.index(sr_id) + 1 for sr_id in sr_band_ids]
            sr_bands = im.read(sr_band_idx)
            test_case.assertTrue(sr_bands.max() <= 11000, 'Scaled reflectance in range')

        if mask:    # check mask is same as VALID_MASK band
            im_mask = im.read_masks(im.descriptions.index('VALID_MASK') + 1).astype(bool)
            valid_mask = im.read(im.descriptions.index('VALID_MASK') + 1, masked=False).astype(bool)
            test_case.assertTrue(np.all(im_mask == valid_mask), 'mask == VALID_MASK')