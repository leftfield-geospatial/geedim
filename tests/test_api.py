import math
import unittest
from datetime import datetime, timedelta

import ee
import pandas as pd
import rasterio as rio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from geedim import export, collection, root_path, info, image


class TestGeeDimApi(unittest.TestCase):
    """
    Test backend functionality in search and download modules
    """

    # TODO: separate API search and download testing (?) and or make one _test_download fn that works from api and cli
    def _test_download(self, ee_image, image_id, region, crs=None, scale=None, band_df=None):
        """
        Test image download

        Parameters
        ----------
        ee_image: ee.Image
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
        im_info_dict, band_info_df = image.get_image_info(ee_image)

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
        export.download_image(ee_image, image_filename, region=region, crs=crs, scale=scale, band_df=band_df,
                              overwrite=True)

        # now set scale and crs to their image defaults as necessary
        min_proj = image.get_projection(ee_image)
        if scale is None:
            scale = min_proj.nominalScale().getInfo()    #band_info_df['scale'].min()
        if crs is None:
            crs = CRS.from_wkt(min_proj.wkt().getInfo())
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

            # if 'MODIS' not in image_id:
            #     if 'VALID_MASK' in im.descriptions:
            #         valid_mask = im.read(im.descriptions.index('VALID_MASK') + 1)
            #     else:
            #         valid_mask = im.read_masks(1)
            #
            #     self.assertAlmostEqual(100 * valid_mask.mean(), float(im.get_tag_item('VALID_PORTION')), delta=5,
            #                            msg=f'VALID_PORTION matches mask mean for {image_id}')

    def _test_search(self, gd_coll_name):
        """
        Test search and download/export on a specified *ImColllection object

        Parameters
        ----------
        gd_coll_name : str
        """

        # GEF Baviaanskloof region
        region = {"type": "Polygon",
                  "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}
        date = datetime.strptime('2019-02-01', '%Y-%m-%d')
        gd_collection = collection.Collection(gd_coll_name)
        image_df = gd_collection.search(date, date + timedelta(days=32), region)

        # check search results
        self.assertGreater(image_df.shape[0], 0, msg='Search returned one or more images')
        for im_prop in gd_collection.prop_df.ABBREV.values:
            self.assertTrue(im_prop in image_df.columns, msg='Search results contain specified properties')

        # select an image to download/export
        im_idx = math.ceil(image_df.shape[0] / 2)
        image_id = str(image_df['ID'].iloc[im_idx])
        ee_image = image.get_class(gd_coll_name).from_id(image_id, mask=False, scale_refl=False).ee_image
        image_name = image_id.replace('/', '-')

        # force CRS for MODIS as workaround for GEE CRS bug
        if gd_coll_name == 'modis_nbar':
            _crs = 'EPSG:3857'
            _scale = 500
        else:
            _crs = None
            _scale = None

        export_tasks = []
        if self.test_export:  # start export tasks
            export_tasks.append(
                export.export_image(ee_image, f'{image_name}_None_None', folder='GeedimTest', region=region,
                                    crs=_crs, scale=None, wait=False))
            export_tasks.append(export.export_image(ee_image, f'{image_name}_Epsg32635_240m', folder='GeedimTest',
                                                    region=region, crs='EPSG:32635', scale=240, wait=False))

        # download in native crs and scale, and validate
        band_df = gd_collection.band_df
        self._test_download(ee_image, image_name, region, crs=_crs, scale=None, band_df=band_df)
        # download in specified crs and scale, and validate
        self._test_download(ee_image, image_name, region, crs='EPSG:32635', scale=240, band_df=band_df)  # UTM zone 35N

        return export_tasks

    def test_api(self):
        """
        Test search and download/export for each *ImSearch class
        """
        self.test_export = True

        ee.Initialize()

        # *ImSearch objects to test
        gd_coll_names = info.collection_info.keys()

        # run tests on each object, accumulating export tasks to check on later
        export_tasks = []
        for gd_coll_name in gd_coll_names:
            export_tasks += self._test_search(gd_coll_name)

        if self.test_export:  # check on export tasks (we can't check on exported files, only the export completed)
            for export_task in export_tasks:
                export.monitor_export_task(export_task)

##
