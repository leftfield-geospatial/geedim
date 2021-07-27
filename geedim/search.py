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

## Classes for searching GEE image collections

from datetime import timedelta, datetime

import ee
import pandas
import rasterio as rio
from rasterio.warp import transform_geom

from geedim import download
from geedim import get_logger

# from shapely import geometry

##
logger = get_logger(__name__)

def get_image_bounds(filename, expand=5):
    """
    Get a WGS84 geojson polygon representing the optionally expanded bounds of an image

    Parameters
    ----------
    filename :  str
                name of the image file whose bounds to find
    expand :    int
                percentage (0-100) by which to expand the bounds (default: 5)

    Returns
    -------
    bounds : geojson
             polygon of bounds in WGS84
    crs: str
         WKT CRS string of image file
    """
    with rio.open(filename) as im:
        bbox = im.bounds
        if (im.crs.linear_units == 'metre') and (expand > 0):   # expand the bounding box
            expand_x = (bbox.right - bbox.left) * expand / 100.
            expand_y = (bbox.top - bbox.bottom) * expand / 100.
            bbox_expand = rio.coords.BoundingBox(bbox.left - expand_x, bbox.bottom - expand_y,
                                                 bbox.right + expand_x, bbox.top + expand_y)
        else:
            bbox_expand = bbox

        coordinates = [[bbox_expand.right, bbox_expand.bottom],
                      [bbox_expand.right, bbox_expand.top],
                      [bbox_expand.left, bbox_expand.top],
                      [bbox_expand.left, bbox_expand.bottom],
                      [bbox_expand.right, bbox_expand.bottom]]

        bbox_expand_dict = dict(type='Polygon', coordinates=[coordinates])
        src_bbox_wgs84 = transform_geom(im.crs, 'WGS84', bbox_expand_dict)   # convert to WGS84 geojson
    return src_bbox_wgs84, im.crs.to_wkt()

##
class ImSearch:
    def __init__(self, collection):
        """
        Base class for searching earth engine image collections

        Parameters
        ----------
        collection : str
                     GEE image collection string e.g. 'MODIS/006/MCD43A4'
        """
        self._collection = collection
        self._im_props = []             # list of image properties to display in search results
        self._im_collection = None
        self._im_df = None
        self._search_region = None
        self._search_date = None

    def _add_timedelta(self, image):
        """
        Finds the time difference between image and search date, and adds this as a property to the image

        Parameters
        ----------
        image : ee.Image
                image to add

        Returns
        -------
        Modified image
        """
        return image.set('TIME_DIST', ee.Number(image.get('system:time_start')).
                         subtract(self._search_date.timestamp()*1000).abs())

    def _get_im_collection(self, start_date, end_date, region):
        """
        Create an image collection filtered by date, bounds and *ImSearch-specific mapping

        Parameters
        ----------
        start_date : str, ee.Date
                     Earliest image date e.g. '2015-05-08'
        end_date : str, ee.Date
                   Latest image date e.g. '2015-05-08'
        region : geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect

        Returns
        -------
        : ee.ImageCollection
        The filtered image collection
        """
        return (ee.ImageCollection(self._collection).
                filterDate(start_date, end_date).
                filterBounds(region).
                map(self._add_timedelta))

    @staticmethod
    def _get_collection_df(im_collection, properties, print=True):
        """
        Convert a filtered image collection to a pandas dataframe of image properties

        Parameters
        ----------
        im_collection : ee.ImageCollection
                        Filtered image collection
        properties : list
                     Image property keys to include in output
        print : bool, optional
                Print a table of image properties

        Returns
        -------
        : pandas.DataFrame
        Table of image properties including the ee.Image objects
        """

        init_list = ee.List([])

        # aggregate relevant properties of im_collection images
        def aggregrate_props(image, im_prop_list):
            prop_dict = ee.Dictionary()
            prop_dict = prop_dict.set('EE_ID', image.get('system:index'))
            prop_dict = prop_dict.set('DATE', image.get('system:time_start'))
            for prop_key in properties:
                prop_dict = prop_dict.set(prop_key,
                                          ee.Algorithms.If(image.get(prop_key), image.get(prop_key), ee.String('None')))
            return ee.List(im_prop_list).add(prop_dict)

        # retrieve list of dicts of collection image properties
        im_prop_list = ee.List(im_collection.iterate(aggregrate_props, init_list)).getInfo()

        im_list = im_collection.toList(im_collection.size())    # image objects

        # add EE image objects and convert ee.Date to python datetime
        for i, prop_dict in enumerate(im_prop_list):
            prop_dict['DATE'] = datetime.utcfromtimestamp(prop_dict['DATE'] / 1000)
            prop_dict['IMAGE'] = ee.Image(im_list.get(i))

        # convert to DataFrame
        im_prop_df = pandas.DataFrame(im_prop_list)
        cols = ['EE_ID', 'DATE'] + properties + ['IMAGE']
        im_prop_df = im_prop_df[cols].sort_values(by='DATE').reset_index(drop=True)
        if print == True:
            logger.info('\n' + im_prop_df[['EE_ID', 'DATE'] + properties].to_string())
        return im_prop_df

    def search(self, date, region, day_range=16):
        """
        Search for images based on date and region

        Parameters
        ----------
        date : datetime.datetime
               Python datetime specifying the desired image capture date
        region : geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect
        day_range : int, optional
                    Number of days before and after `date` to search within

        Returns
        -------
        : pandas.DataFrame
        Dataframe specifying image properties that match the search criteria
        """
        # Initialise
        self._im_df = None
        self._im_collection = None
        self._search_region = region
        self._search_date = date

        start_date = date - timedelta(days=day_range)
        end_date = date + timedelta(days=day_range)

        # filter the image collection
        logger.info(f'Searching for {self._collection} images between {start_date.strftime("%Y-%m-%d")} and '
                      f'{end_date.strftime("%Y-%m-%d")}')
        self._im_collection = self._get_im_collection(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                                                      region)

        num_images = self._im_collection.size().getInfo()

        if num_images == 0:
            logger.info(f'Could not find any images matching those criteria')
            return None

        # print search results
        logger.info(f'Found {num_images} images:')
        self._im_df = ImSearch._get_collection_df(self._im_collection, self._im_props, print=True)

        return self._im_df

    def get_composite_image(self):
        """
        Create a median composite image from search results

        Returns
        -------
        : ee.Image
        Composite image
        """
        if self._im_collection is None or self._im_df is None:
            raise Exception('First generate valid search results with search(...) method')

        comp_image = self._im_collection.median()

        # set metadata to indicate component images
        return comp_image.set('COMPOSITE_IMAGES', self._im_df[['EE_ID', 'DATE'] + self._im_props].to_string())

##
class LandsatImSearch(ImSearch):
    def __init__(self, collection='landsat8'):
        """
        Class for searching Landsat 7-8 earth engine image collections

        Parameters
        ----------
        collection : str, optional
                     'landsat7' or 'landsat8' (default)
        """
        ImSearch.__init__(self, collection=collection)

        if collection == 'landsat8':
            self._collection = 'LANDSAT/LC08/C02/T1_L2'     # EE collection name
            self._im_props = ['VALID_PORTION', 'QA_SCORE_AVG', 'GEOMETRIC_RMSE_VERIFY', 'GEOMETRIC_RMSE_MODEL',
                              'SUN_AZIMUTH', 'SUN_ELEVATION']   # image properties to include in search results
        elif collection == 'landsat7':
            self._collection = 'LANDSAT/LE07/C02/T1_L2'
            self._im_props = ['VALID_PORTION', 'QA_SCORE_AVG', 'GEOMETRIC_RMSE_MODEL', 'SUN_AZIMUTH', 'SUN_ELEVATION']
        else:
            # TODO: add support for landsat 4-5 collection 2 when they are available
            raise ValueError(f'Unsupported landsat collection: {collection}')

        self._valid_portion = 90
        self._apply_valid_mask = False

    def _check_validity(self, image):
        """
        Evaluate image validity and quality

        Parameters
        ----------
        image : ee.Image
                Image to evaluate

        Returns
        -------
        : ee.Image
        Image with added properties and band(s)
        """

        # NOTES
        # - QA_PIXEL The *conf bit pairs (8-9,10-11,12-13,14-15) will always be 1 or more, unless it is a fill pixel -
        # i.e. the fill bit 0 is set.  Values are higher where there are cloud, cloud shadow etc.  The water bit 7, is
        # seems to be set incorrectly quite often, but with the rest of the bits ok/sensible.
        # - SR_QA_AEROSOL bits 6-7 can have a value of 0, and this 0 can occur in e.g. an area of QA_PIXEL=cloud shadow,
        # NB this is not band 9 as on GEE site, but band 8.
        # - The behaviour of updateMask in combination with ImageCollection qualityMosaic (and perhaps median and mosaic
        # ) is weird: updateMask always masks bands added with addBands, but only masks the original SR etc bands after
        # a call to qualityMosaic (or perhaps median etc)
        # - Pixels in Fill bit QA_* masks seem to refer to nodata / uncovered pixels only.  They don't occur amongst
        # valid data

        image = self._add_timedelta(image)  # Add TIME_DIST property from base class

        # create a mask of valid (non cloud, shadow and aerosol) pixels
        # bits 1-4 of QA_PIXEL are dilated cloud, cirrus, cloud & cloud shadow, respectively
        qa_pixel_bitmask = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
        qa_pixel = image.select('QA_PIXEL')
        cloud_mask = qa_pixel.bitwiseAnd(qa_pixel_bitmask).eq(0).rename('CLOUD_MASK')

        # mask of filled (imaged) pixels
        fill_mask = qa_pixel.bitwiseAnd(1).eq(0).rename('FILL_MASK')

        # extract cloud etc quality scores (2 bits)
        cloud_conf = qa_pixel.rightShift(8).bitwiseAnd(3).rename('CLOUD_CONF')
        cloud_shadow_conf = qa_pixel.rightShift(10).bitwiseAnd(3).rename('CLOUD_SHADOW_CONF')
        cirrus_conf = qa_pixel.rightShift(14).bitwiseAnd(3).rename('CIRRUS_CONF')

        if self._collection == 'LANDSAT/LC08/C02/T1_L2':    # landsat8
            # TODO: is SR_QA_AEROSOL helpful? (Looks suspect for GEF region images)
            # include SR_QA_AEROSOL in valid_mask and q_score
            # bits 6-7 of SR_QA_AEROSOL, are aerosol level where 3 = high, 2=medium, 1=low
            sr_qa_aerosol = image.select('SR_QA_AEROSOL')
            aerosol_prob = sr_qa_aerosol.rightShift(6).bitwiseAnd(3)
            aerosol_mask = aerosol_prob.lt(3).rename('AEROSOL_MASK')

            # combine cloud etc masks to create validity mask
            valid_mask = cloud_mask.And(aerosol_mask).And(fill_mask).rename('VALID_MASK')
            # sum the cloud etc probabilities, and convert range to 0-12, where 12 is best
            q_score = cloud_conf.add(cloud_shadow_conf).add(cirrus_conf).add(aerosol_prob).multiply(-1).add(12)
        else:
            # combine cloud etc masks to create validity mask
            valid_mask = cloud_mask.And(fill_mask).rename('VALID_MASK')
            # sum the cloud etc probabilities, and convert range to 0-12, where 12 is best
            q_score = cloud_conf.add(cloud_shadow_conf).add(cirrus_conf).multiply(-1).add(9)

        # calculate the potion of valid image pixels
        # TODO: is self._search_region necessary or can it be passed or otherwise referred to?
        valid_portion = (valid_mask.unmask().
                         multiply(100).
                         reduceRegion(reducer='mean', geometry=self._search_region,
                                      scale=image.projection().nominalScale()).
                         rename(['VALID_MASK'], ['VALID_PORTION']))

        q_score = q_score.where(fill_mask.Not(), 0).rename('QA_SCORE')  # Zero q_score where pixels are unfilled

        # calculate the average quality score
        q_score_avg = (q_score.unmask().
                       reduceRegion(reducer='mean', geometry=self._search_region,
                                    scale=image.projection().nominalScale()).
                       rename(['QA_SCORE'], ['QA_SCORE_AVG']))

        if False:
            # TODO: can these be passed in a list?
            image = image.addBands(cloud_conf)
            image = image.addBands(cloud_shadow_conf)
            image = image.addBands(cirrus_conf)
            image = image.addBands(aerosol_prob)
            image = image.addBands(fill_mask)
            image = image.addBands(valid_mask)

        if self._apply_valid_mask:
            image = image.updateMask(valid_mask)    # mask out cloud, shadow, unfilled etc pixels
        else:
            image = image.updateMask(fill_mask)     # mask out unfilled pixels only

        return image.set(valid_portion).set(q_score_avg).addBands(q_score)

    def _get_im_collection(self, start_date, end_date, region):
        """
        Create an image collection filtered by date, bounds and portion of valid pixels

        Parameters
        ----------
        start_date : str, ee.Date
                     Earliest image date e.g. '2015-05-08'
        end_date : str, ee.Date
                   Latest image date e.g. '2015-05-08'
        region : geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect

        Returns
        -------
        : ee.ImageCollection
        The filtered image collection
        """

        return (ee.ImageCollection(self._collection).
                filterDate(start_date, end_date).
                filterBounds(region).
                map(self._check_validity).
                filter(ee.Filter.gt('VALID_PORTION', self._valid_portion)))

    def search(self, date, region, day_range=16, valid_portion=50, apply_valid_mask=False):
        """
        Search for Landsat images based on date, region etc criteria
        
        Parameters
        date : datetime.datetime
               Python datetime specifying the desired image capture date
        region : geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect
        day_range : int, optional
                    Number of days before and after `date` to search within
        valid_portion: int, optional
                     Minimum portion (%) of image pixels that should be valid (filled, and not cloud or shadow)
        apply_valid_mask : bool, optional
                        Mask out clouds, shadows and aerosols in search result images

        Returns
        -------
        : pandas.DataFrame
        Dataframe specifying image properties that match the search criteria
        """
        self._valid_portion = valid_portion
        self._apply_valid_mask = apply_valid_mask
        return ImSearch.search(self, date, region, day_range=day_range)

    def get_composite_image(self):
        """
        Create a composite image from search results, favouring pixels with the highest quality score

        Returns
        -------
        : ee.Image
        Composite image
        """
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')

        comp_im = self._im_collection.qualityMosaic('QA_SCORE')

        # set metadata to indicate component images
        return comp_im.set('COMPOSITE_IMAGES', self._im_df[['EE_ID', 'DATE'] + self._im_props].to_string()).toUint16()

    def convert_dn_to_sr(self, image):
        """
        Scale and offset landsat pixels in SR bands to surface reflectance (0-10000)

        Parameters
        ----------
        image : ee.Image
                image to scale and offset

        Returns
        -------
        : ee.Image
        Float32 image with SR bands in range 0-10000 & nodata = -inf
        """

        # retrieve the names of SR bands
        all_bands = image.bandNames()
        init_bands = ee.List([])

        def add_refl_bands(band, refl_bands):
            refl_bands = ee.Algorithms.If(ee.String(band).rindex('SR_B').eq(0), ee.List(refl_bands).add(band), refl_bands)
            return refl_bands

        sr_bands = ee.List(all_bands.iterate(add_refl_bands, init_bands))

        non_sr_bands = all_bands.removeAll(sr_bands)    # all the other non-SR bands

        # retrieve the scale (mult) and offset (add) parameters for each band
        param_dict = ee.Dictionary(dict(mult=ee.Image().select([]), add=ee.Image().select([])))

        def add_refl_params(band, param_dict):
            param_dict = ee.Dictionary(param_dict)

            band_num = ee.String(band).slice(-1)
            sr_mult_str = ee.String('REFLECTANCE_MULT_BAND_').cat(band_num)
            sr_add_str = ee.String('REFLECTANCE_ADD_BAND_').cat(band_num)

            # create constant scale (mult) and offset (add) images for this band
            sr_mult = ee.Image.constant(image.get(sr_mult_str)).rename(sr_mult_str)
            sr_add = ee.Image.constant(image.get(sr_add_str)).rename(sr_add_str)

            # add the constant scale/offset images for this band to multi-band scale/offset images
            param_dict = param_dict.set('mult', ee.Image(param_dict.get('mult')).addBands(sr_mult))
            param_dict = param_dict.set('add', ee.Image(param_dict.get('add')).addBands(sr_add))

            return param_dict

        param_dict = ee.Dictionary(sr_bands.iterate(add_refl_params, param_dict))

        # apply the scale and offset
        calib_image = image.select(sr_bands).multiply(param_dict.get('mult'))
        calib_image = (calib_image.add(param_dict.get('add'))).multiply(10000.0)
        calib_image = calib_image.addBands(image.select(non_sr_bands))
        calib_image = calib_image.updateMask(image.mask())

        # call toFloat after updateMask
        return ee.Image(calib_image.copyProperties(image)).toFloat()

##
class Sentinel2ImSearch(ImSearch):
    def __init__(self, collection='sentinel2_toa'):
        """
        Class for searching Sentinel-2 TOA and SR earth engine image collections

        Parameters
        ----------
        collection : str, optional
                     'sentinel_toa' (top of atmosphere - default) or 'sentinel_sr' (surface reflectance)
        """
        ImSearch.__init__(self, collection=collection)

        if collection == 'sentinel2_toa':
            self._collection = 'COPERNICUS/S2'
        elif collection == 'sentinel2_sr':
            self._collection = 'COPERNICUS/S2_SR'
        else:
            raise ValueError(f'Unsupported sentinel2 collection: {collection}')

        self._im_props = ['VALID_PORTION', 'GEOMETRIC_QUALITY_FLAG', 'RADIOMETRIC_QUALITY_FLAG',
                          'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B1',
                          'MEAN_INCIDENCE_ZENITH_ANGLE_B1']

        self._valid_portion = 90
        self._apply_valid_mask = False

    def _check_validity(self, image):
        """
        Evaluate image validity and quality

        Parameters
        ----------
        image : ee.Image
                Image to evaluate

        Returns
        -------
        : ee.Image
        Image with added properties and band(s)
        """

        image = self._add_timedelta(image)
        bit_mask = (1 << 11) | (1 << 10)    # bits 10 and 11 are opaque and cirrus clouds respectively
        qa = image.select('QA60')
        valid_mask = qa.bitwiseAnd(bit_mask).eq(0).rename('VALID_MASK')

        min_scale = download.get_min_projection(image).nominalScale()

        # calculate the potion of valid image pixels
        valid_portion = (valid_mask.unmask().
                         multiply(100).
                         reduceRegion(reducer='mean', geometry=self._search_region, scale=min_scale).
                         rename(['VALID_MASK'], ['VALID_PORTION']))

        # TODO: what happens if/when pixels aren't filled
        if self._apply_valid_mask:
            image = image.updateMask(valid_mask)
        return image.set(valid_portion)

    def _get_im_collection(self, start_date, end_date, region):
        """
        Create an image collection filtered by date, bounds and portion of valid pixels

        Parameters
        ----------
        start_date : str, ee.Date
                     Earliest image date e.g. '2015-05-08'
        end_date : str, ee.Date
                   Latest image date e.g. '2015-05-08'
        region : geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect

        Returns
        -------
        : ee.ImageCollection
        The filtered image collection
        """
        return (ee.ImageCollection(self._collection).
                filterDate(start_date, end_date).
                filterBounds(region).
                map(self._check_validity).
                filter(ee.Filter.gt('VALID_PORTION', self._valid_portion)))

    def search(self, date, region, day_range=16, valid_portion=50, apply_valid_mask = False):
        """
        Search for Sentinel-2 images based on date, region etc criteria

        Parameters
        date : datetime.datetime
               Python datetime specifying the desired image capture date
        region : geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect
        day_range : int, optional
                    Number of days before and after `date` to search within
        valid_portion: int, optional
                     Minimum portion (%) of image pixels that should be valid (not cloud)
        apply_valid_mask : bool, optional
                        Mask out clouds in search result images

        Returns
        -------
        : pandas.DataFrame
        Dataframe specifying image properties that match the search criteria
        """
        self._valid_portion = valid_portion
        self._apply_valid_mask = apply_valid_mask
        return ImSearch.search(self, date, region, day_range=day_range)


    def get_composite_image(self):
        """
        Create a composite image from search results

        Returns
        -------
        : ee.Image
        Composite image
        """
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')

        if self._apply_valid_mask is None:
            logger.warning('Calling search(...) with apply_valid_mask=True is recommended composite creation')

        comp_im = self._im_collection.mosaic()

        # set metadata to indicate component images
        return comp_im.set('COMPOSITE_IMAGES', self._im_df[['EE_ID', 'DATE'] + self._im_props].to_string()).toUint16()

##
class Sentinel2CloudlessImSearch(ImSearch):
    def __init__(self, collection='sentinel2_toa'):
        """
        Class for searching Sentinel-2 TOA and SR earth engine image collections. Uses cloud-probability for masking
        and quality scoring.

        Parameters
        ----------
        collection : str, optional
                     'sentinel_toa' (top of atmosphere - default) or 'sentinel_sr' (surface reflectance)
        """
        ImSearch.__init__(self, collection=collection)

        if collection == 'sentinel2_toa':
            self._collection = 'COPERNICUS/S2'
        elif collection == 'sentinel2_sr':
            self._collection = 'COPERNICUS/S2_SR'
        else:
            raise ValueError(f'Unsupported sentinel2 collection: {collection}')

        self._im_props = ['VALID_PORTION', 'QA_SCORE_AVG', 'GEOMETRIC_QUALITY_FLAG', 'RADIOMETRIC_QUALITY_FLAG',
                          'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B1',
                          'MEAN_INCIDENCE_ZENITH_ANGLE_B1']

        self._valid_portion = 90
        self._apply_valid_mask = False

        self._cloud_filter = 60         # Maximum image cloud cover percent allowed in image collection
        self._cloud_prob_thresh = 40    # Cloud probability (%); values greater than are considered cloud
        # self._nir_drk_thresh = 0.15     # Near-infrared reflectance; values less than are considered potential cloud shadow
        self._cloud_proj_dist = 1       # Maximum distance (km) to search for cloud shadows from cloud edges
        self._buffer = 100              # Distance (m) to dilate the edge of cloud-identified objects

    def _check_validity(self, image):
        """
        Evaluate image validity and quality

        Parameters
        ----------
        image : ee.Image
                Image to evaluate

        Returns
        -------
        : ee.Image
        Image with added properties and band(s)
        """
        # adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless

        image = self._add_timedelta(image)
        min_scale = download.get_min_projection(image).nominalScale()

        # convert cloud probability in 0-100 quality score
        cloud_prob = ee.Image(image.get('s2cloudless')).select('probability')
        q_score = cloud_prob.multiply(-1).add(100).rename('QA_SCORE')

        cloud_mask = cloud_prob.gt(self._cloud_prob_thresh).rename('CLOUD_MASK')
        # See https://en.wikipedia.org/wiki/Solar_azimuth_angle

        # TODO: dilate valid_mask by _buffer ?
        # TODO: does below work in N hemisphere?
        shadow_azimuth = ee.Number(-90).add(ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

        # project the the cloud mask in the direction of shadows for self._cloud_proj_dist
        proj_dist_px = ee.Number(self._cloud_proj_dist * 1000).divide(min_scale)
        proj_cloud_mask = (cloud_mask.directionalDistanceTransform(shadow_azimuth, proj_dist_px).
                           select('distance').mask().rename('PROJ_CLOUD_MASK'))
            # .reproject(**{'crs': image.select(0).projection(), 'scale': 100})

        if self._collection == 'COPERNICUS/S2_SR':  # use SCL to reduce shadow_mask
            # Note: SCL does not classify cloud shadows well, they are often labelled "dark".  Instead of using only
            # cloud shadow areas from this band, we combine it with the projected dark and shadow areas from s2cloudless
            scl = image.select('SCL')
            dark_shadow_mask = scl.eq(3).Or(scl.eq(2)).focal_max(self._buffer, 'circle', 'meters')
            shadow_mask = proj_cloud_mask.And(dark_shadow_mask).rename('SHADOW_MASK')
        else:
            shadow_mask = proj_cloud_mask.rename('SHADOW_MASK')   # mask all areas that could be cloud shadow

        # combine cloud and shadow masks
        valid_mask = (cloud_mask.Or(shadow_mask)).Not().rename('VALID_MASK')

        # calculate the potion of valid image pixels TODO: this is repeated in each class
        valid_portion = (valid_mask.unmask().
                         multiply(100).
                         reduceRegion(reducer='mean', geometry=self._search_region, scale=min_scale).
                         rename(['VALID_MASK'], ['VALID_PORTION']))

        # calculate the average quality score
        q_score_avg = (q_score.unmask().
                       reduceRegion(reducer='mean', geometry=self._search_region, scale=min_scale).
                       rename(['QA_SCORE'], ['QA_SCORE_AVG']))

        if False:
            image = image.addBands(cloud_prob)
            image = image.addBands(cloud_mask)
            image = image.addBands(proj_cloud_mask)
            image = image.addBands(shadow_mask)
            image = image.addBands(valid_mask)

        if self._apply_valid_mask:
            # NOTE: for export_image, updateMask sets pixels to 0,
            # for download_image, it does the same and sets nodata=0
            image = image.updateMask(valid_mask)

        return image.set(valid_portion).set(q_score_avg).addBands(q_score)

    def _get_im_collection(self, start_date, end_date, region):
        """
        Create an image collection filtered by date, bounds and portion of valid pixels. Combines sentinel2 SR/TOA
        collection with separate cloud probability collection.

        Parameters
        ----------
        start_date : str, ee.Date
                     Earliest image date e.g. '2015-05-08'
        end_date : str, ee.Date
                   Latest image date e.g. '2015-05-08'
        region : geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect

        Returns
        -------
        : ee.ImageCollection
        The filtered image collection
        """
        # adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless

        s2_sr_toa_col = (ee.ImageCollection(self._collection)
                         .filterBounds(region)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self._cloud_filter)))

        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                            .filterBounds(region)
                            .filterDate(start_date, end_date))

        # join filtered s2cloudless collection to the SR/TOA collection by the 'system:index' property.
        return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': s2_sr_toa_col,
            'secondary': s2_cloudless_col,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
        })).map(self._check_validity).filter(ee.Filter.gt('VALID_PORTION', self._valid_portion))

    def search(self, date, region, day_range=16, valid_portion=50, apply_valid_mask=False):
        """
        Search for Sentinel-2 images based on date, region etc criteria

        Parameters
        date : datetime.datetime
               Python datetime specifying the desired image capture date
        region : geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect
        day_range : int, optional
                    Number of days before and after `date` to search within
        valid_portion: int, optional
                     Minimum portion (%) of image pixels that should be valid (filled, and not cloud or shadow)
        apply_valid_mask : bool, optional
                        Mask out clouds, shadows and aerosols in search result images

        Returns
        -------
        : pandas.DataFrame
        Dataframe specifying image properties that match the search criteria
        """
        self._valid_portion = valid_portion
        self._apply_valid_mask = apply_valid_mask
        return ImSearch.search(self, date, region, day_range=day_range)


    def get_composite_image(self):
        """
        Create a composite image from search results, favouring pixels with the highest quality score

        Returns
        -------
        : ee.Image
        Composite image
        """
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')

        if self._apply_valid_mask is None:
            logger.warning('Calling search(...) with apply_valid_mask=True is recommended for composite creation')

        comp_im = self._im_collection.mosaic()

        # set metadata to indicate component images
        return comp_im.set('COMPOSITE_IMAGES', self._im_df[['EE_ID', 'DATE'] + self._im_props].to_string()).toUint16()

##
class ModisNbarImSearch(ImSearch):
    def __init__(self):
        """
        Class for searching the MODIS daily NBAR earth engine image collection

        Parameters
        ----------
        collection : str, optional
                     'modis' (default)
        """
        ImSearch.__init__(self, 'modis')
        self._collection = 'MODIS/006/MCD43A4'


##

