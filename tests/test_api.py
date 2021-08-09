import unittest
from datetime import datetime, timedelta

import ee
import pandas as pd
import rasterio as rio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

import math

from geedim import download
from geedim import root_path
from geedim import search


class TestGeeDimApi(unittest.TestCase):
    """
    Test backend functionality in search and download modules
    """

    def _test_download(self, image, image_id, region, crs=None, scale=None, band_df=None):
        """
        Test image download

        Parameters
        ----------
        image: ee.Image
                The image to download
        image_id : str
                   A string describing the image, will be used as filename
        region : dict, geojson, ee.Geometry
                 geojson region of interest to download
        crs : str, optional
              str compatible with rasterio.rs.CRS.from_string() specifying download CRS
        scale : float, optional
               Image target resolution
        band_df : pandas.DataFrame, optional
                  DataFrame specifying band metadata to be copied to downloaded file.  'id' column should contain band
                  id's that match the ee.Image band id's
        """
        # check image info
        im_info_dict, band_info_df = download.get_image_info(image)

        for key in ['bands', 'properties']:
            self.assertTrue(key in im_info_dict.keys(), msg='Image info has bands and properties')
        self.assertGreater(band_info_df.shape[0], 1, msg='Image has more than one band')

        # construct image filename based on id, crs and scale
        # delay setting crs etc if its None, so we can test download_image(...crs=None,scale=None)
        if crs is not None:
            crs_str = f'EPSG{CRS.from_string(crs).to_epsg()}'
        else:
            crs_str = 'None'

        image_filename = root_path.joinpath(f'data/outputs/tests/{image_id}_{crs_str}_{scale}m.tif')

        # run the download
        download.download_image(image, image_filename, region=region, crs=crs, scale=scale, band_df=band_df)

        # now set scale and crs to their image defaults as necessary
        if scale is None:
            scale = band_info_df['scale'].min()
        if crs is None:
            crs = CRS.from_wkt(download.get_min_projection(image).wkt().getInfo())
        elif isinstance(crs, str):
            crs = CRS.from_string(crs)
        else:
            raise TypeError('CRS must be None or string')

        region_arr = pd.DataFrame(region['coordinates'][0], columns=['x', 'y'])  # avoid numpy dependency
        region_bounds = rio.coords.BoundingBox(region_arr.x.min(), region_arr.y.min(), region_arr.x.max(),
                                               region_arr.y.max())

        # check the validity of the downloaded file
        self.assertTrue(image_filename.exists(), msg='Download file exists')

        with rio.open(image_filename) as im:
            self.assertEqual(band_info_df.shape[0], im.count, msg='EE and download image band count match')
            self.assertEqual(crs.to_proj4(), im.crs.to_proj4(), msg='EE and download image CRS match')
            self.assertAlmostEqual(scale, im.res[0], places=3, msg='EE and download image scale match')
            im_bounds_wgs84 = transform_bounds(im.crs, 'WGS84', *im.bounds)  # convert to WGS84 geojson

            self.assertFalse(rio.coords.disjoint_bounds(region_bounds, im_bounds_wgs84),
                             msg='Search and image bounds match')

    def _test_imsearch_obj(self, imsearch_obj):
        """
        Test search and download/export on a specified *ImSearch object

        Parameters
        ----------
        imsearch_obj : geedim.search.ImSearch
                       A *ImSearch object to test
        """

        # GEF Baviaanskloof region
        region = {'type': 'Polygon',
                  'coordinates': [[(24.018987152147467, -33.58425124616373),
                                   (24.01823400920558, -33.52008125251578),
                                   (23.927741869101595, -33.52079213034971),
                                   (23.92842810355374, -33.58496384476743),
                                   (24.018987152147467, -33.58425124616373)]]}
        date = datetime.strptime('2019-02-01', '%Y-%m-%d')
        band_df = pd.DataFrame.from_dict(imsearch_obj._collection_info['bands'])

        image_df = imsearch_obj.search(date, date + timedelta(days=32), region)

        # check search results
        self.assertGreater(image_df.shape[0], 0, msg='Search returned one or more images')
        for im_prop in imsearch_obj._im_props.PROPERTY.values:
            self.assertTrue(im_prop in image_df.columns, msg='Search results contain specified properties')

        # select an image to download/export
        im_idx = math.ceil(image_df.shape[0] / 2)
        image = image_df.IMAGE.iloc[im_idx]
        image_id = image_df['system:id'].iloc[im_idx].replace('/', '_')

        export_tasks = []
        if self.test_export:  # start export tasks
            export_tasks.append(
                download.export_image(image, f'{image_id}_None_None', folder='GeedimTest', region=region,
                                      crs=None, scale=None, wait=False))
            export_tasks.append(download.export_image(image, f'{image_id}_Epsg32635_240m', folder='GeedimTest',
                                                      region=region, crs='EPSG:32635', scale=240, wait=False))

        # download in native crs and scale, and validate
        self._test_download(image, image_id, region, crs=None, scale=None, band_df=band_df)
        # download in specified crs and scale, and validate
        self._test_download(image, image_id, region, crs='EPSG:32635', scale=240, band_df=band_df)  # UTM zone 35N

        return export_tasks

    def test_api(self):
        """
        Test search and download/export for each *ImSearch class
        """
        self.test_export = True

        ee.Initialize()

        # *ImSearch objects to test
        test_objs = [  # search.ModisNbarImSearch(),
            search.LandsatImSearch(collection='landsat8_c2_l2'),
            search.LandsatImSearch(collection='landsat7_c2_l2'),
            search.Sentinel2ImSearch(collection='sentinel2_toa'),
            search.Sentinel2ImSearch(collection='sentinel2_sr'),
            search.Sentinel2CloudlessImSearch(collection='sentinel2_toa'),
            search.Sentinel2CloudlessImSearch(collection='sentinel2_sr')]

        # run tests on each object, accumulating export tasks to check on later
        export_tasks = []
        for test_obj in test_objs:
            export_tasks += self._test_imsearch_obj(test_obj)

        if self.test_export:  # check on export tasks (we can't check on exported files, only the export completed)
            for export_task in export_tasks:
                download.monitor_export_task(export_task)

##
