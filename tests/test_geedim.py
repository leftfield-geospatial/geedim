import unittest
from datetime import datetime

import ee
import numpy as np
import rasterio as rio

from geedim import download
from geedim import root_path
from geedim import search


class TestGeeDim(unittest.TestCase):
    """
    Test backend functionality in search and download modules
    """

    def _test_download(self, image, image_id, region, crs=None, scale=None):
        """
        Test image download

        Parameters
        ----------
        image: ee.Image
        image_id : str
        region : dict, geojson, ee.Geometry
        crs : str, optional
              str compatible with rasterio.rs.CRS.from_string() specifying download CRS
        scale : float, optional
        """
        im_info_dict, band_info_df = download.get_image_info(image)

        for key in ['bands', 'properties']:
            self.assertTrue(key in im_info_dict.keys(), msg='Image info has bands and properties')
        self.assertGreater(band_info_df.shape[0], 1, msg='Image has more than one band')

        if crs is not None:
            crs_str = f'EPSG{rio.crs.CRS.from_string(crs).to_epsg()}'
        else:
            crs_str = 'None'

        image_filename = root_path.joinpath(f'data/outputs/tests/{image_id}_{crs_str}_{scale}m.tif')
        download.download_image(image, image_filename, region=region, crs=crs, scale=scale)
        self.assertTrue(image_filename.exists(), msg='Download file exists')

        # im_region, im_crs = search.get_image_bounds(image_filename, expand=0)

        region_arr = np.array(region['coordinates'][0])
        region_bounds = rio.coords.BoundingBox(region_arr[:,0].min(), region_arr[:,1].min(), region_arr[:,0].max(),
                                               region_arr[:,1].max())

        if scale is None:
            scale = band_info_df['scale'].min()
        if crs is None:
            crs = rio.crs.CRS.from_wkt(download.get_min_projection(image).wkt().getInfo())
        elif isinstance(crs, str):
            crs = rio.crs.CRS.from_string(crs)
        else:
            raise TypeError('CRS must be None or string')

        with rio.open(image_filename) as im:
            self.assertEqual(band_info_df.shape[0], im.count, msg='EE and download image band count match')
            self.assertEqual(crs.to_proj4(), im.crs.to_proj4(), msg='EE and download image CRS match')
            self.assertAlmostEqual(scale, im.res[0], places=3, msg='EE and download image CRS match')
            im_bounds_wgs84 = rio.warp.transform_bounds(im.crs, 'WGS84', *im.bounds)  # convert to WGS84 geojson

            self.assertFalse(rio.coords.disjoint_bounds(region_bounds, im_bounds_wgs84), msg='Search and image bounds match')


    def _test_search_and_download(self, imsearch_obj):
        """
        Test search and download on a specifified *ImSearch object

        Parameters
        ----------
        imsearch_obj : geedim.search.ImSearch
                       A *ImSearch object to test
        """

        # image_filename = root_path.joinpath(r'data/outputs/test_example')
        # region = {'type': 'Polygon',
        #           'coordinates': [[(24.023552906883516, -33.587422354357045),
        #                            (24.02272076989141, -33.516835460612),
        #                            (23.923183092469884, -33.51761733158424),
        #                            (23.923934270192053, -33.58820630746053),
        #                            (24.023552906883516, -33.587422354357045)]]}
        region = {'type': 'Polygon',
                  'coordinates': [[(24.018987152147467, -33.58425124616373),
                                   (24.01823400920558, -33.52008125251578),
                                   (23.927741869101595, -33.52079213034971),
                                   (23.92842810355374, -33.58496384476743),
                                   (24.018987152147467, -33.58425124616373)]]}

        date = datetime.strptime('2019-02-01', '%Y-%m-%d')

        image_df = imsearch_obj.search(date, region, day_range=32)

        self.assertGreater(image_df.shape[0], 0, msg='Search returned one or more images')
        self.assertTrue(('IMAGE' in image_df.columns) and ('ID' in image_df.columns),
                        msg='Search results have image and id fields')
        for im_prop in imsearch_obj._im_props:
            self.assertTrue(im_prop in image_df.columns, msg='Search results contain specified properties')

        im_idx = np.ceil(image_df.shape[0] / 2).astype(int)
        image = image_df.IMAGE.iloc[im_idx]
        image_id =  image_df.ID.iloc[im_idx].replace('/','_')
        self._test_download(image, image_id, region, crs=None, scale=None)
        self._test_download(image, image_id, region, crs='EPSG:32635', scale=240)  #UTM zone 35N


    def test_geedim(self):
        """
        Test search and download for each *ImSearch class
        """

        ee.Initialize()

        test_objs = [#search.ModisNbarImSearch(),
                     search.LandsatImSearch(collection='landsat8'),
                     search.LandsatImSearch(collection='landsat7'),
                     search.Sentinel2ImSearch(collection='sentinel2_toa'),
                     search.Sentinel2ImSearch(collection='sentinel2_sr'),
                     search.Sentinel2CloudlessImSearch(collection='sentinel2_toa'),
                     search.Sentinel2CloudlessImSearch(collection='sentinel2_sr')]

        for test_obj in test_objs:
            self._test_search_and_download(test_obj)


##

