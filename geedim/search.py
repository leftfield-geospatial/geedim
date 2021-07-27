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
import numpy as np
import pandas
import rasterio as rio
from rasterio.warp import transform_geom
# from shapely import geometry

from geedim import get_logger

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
            bbox_expand = rio.coords.BoundingBox(bbox.left - expand_x, bbox.bottom + expand_y,
                                                 bbox.right + expand_x, bbox.top - expand_y)
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


class ImSearch:
    def __init__(self, collection):
        # self._search_region, self._crs = get_image_bounds(source_image_filename, expand=10)
        self._collection = collection
        self._im_props = []
        self._im_collection = None
        self._im_df = None
        self._search_region = None
        self._search_date = None

    def _add_timedelta(self, image):
        return image.set('TIME_DIST', ee.Number(image.get('system:time_start')).
                         subtract(self._search_date.timestamp()*1000).abs())

    def _get_im_collection(self, start_date, end_date, region):
        return ee.ImageCollection(self._collection).\
            filterDate(start_date, end_date).\
            filterBounds(region).\
            map(self._add_timedelta)
            # filter(ee.Filter.contains('system:footprint'), self._search_region).\

    @staticmethod
    def _print_proplist(prop_list, properties):
        prop_df = pandas.DataFrame(prop_list)
        cols = ['EE_ID', 'DATE'] + properties
        prop_df = prop_df[cols].sort_values(by='DATE').reset_index(drop=True)
        logger.info('Search results:\n' + prop_df.to_string())
        return prop_df

    @staticmethod
    def _get_collection_df(im_collection, properties, print=True):
        init_list = ee.List([])

        def aggregrate_props(image, im_prop_list):
            prop_dict = ee.Dictionary()
            prop_dict = prop_dict.set('EE_ID', image.get('system:index'))
            prop_dict = prop_dict.set('DATE', image.get('system:time_start'))
            for prop_key in properties:
                prop_dict = prop_dict.set(prop_key,
                                          ee.Algorithms.If(image.get(prop_key), image.get(prop_key), ee.String('None')))
            return ee.List(im_prop_list).add(prop_dict)

        # retrieve properties of search result images for display
        im_prop_list = ee.List(im_collection.iterate(aggregrate_props, init_list)).getInfo()
        im_list = im_collection.toList(im_collection.size())

        # add ee image and convert date to python datetime
        for i, prop_dict in enumerate(im_prop_list):
            prop_dict['DATE'] = datetime.utcfromtimestamp(prop_dict['DATE'] / 1000)
            prop_dict['IMAGE'] = ee.Image(im_list.get(i))

        # create dataframe of search results
        im_prop_df = pandas.DataFrame(im_prop_list)
        cols = ['EE_ID', 'DATE'] + properties + ['IMAGE']
        im_prop_df = im_prop_df[cols].sort_values(by='DATE').reset_index(drop=True)
        if print == True:
            logger.info('\n' + im_prop_df[['EE_ID', 'DATE'] + properties].to_string())
        return im_prop_df

    def search(self, date, region, day_range=16):
        self._im_df = None
        self._im_collection = None
        self._search_region = region
        self._search_date = date

        start_date = date - timedelta(days=day_range)
        end_date = date + timedelta(days=day_range)

        logger.info(f'Searching for {self._collection} images between '
                    f'{start_date.strftime("%Y-%m-%d")} and {end_date.strftime("%Y-%m-%d")}')

        self._im_collection = self._get_im_collection(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), region)
        num_images = self._im_collection.size().getInfo()

        if num_images == 0:
            logger.info(f'Could not find any images in that date range')
            return None

        logger.info(f'Found {num_images} images:')
        self._im_df = ImSearch._get_collection_df(self._im_collection, self._im_props, print=True)

        return self._im_df

    # def get_auto_image(self):
    #     if (self._im_df is None) or (self._im_collection is None):
    #         raise Exception('First generate valid search results with search(...) method')
    #     return self._im_collection.sort('TIME_DIST', True).first()
    #
    # def get_single_image(self, image_num):
    #     if (self._im_df is None) or (self._im_collection is None):
    #         raise Exception('First generate valid search results with search(...) method')
    #
    #     if isinstance(image_num, str):
    #         if image_num not in self._im_df['EE_ID'].values:
    #             raise ValueError(f'{image_num} does not exist in search results')
    #     elif isinstance(image_num, int):
    #         if (image_num >= 0) and (image_num < self._im_df.shape[0]):
    #             image_num = self._im_df.loc[image_num]['EE_ID']
    #         else:
    #             raise ValueError(f'image_num={image_num} out of range')
    #     else:
    #         raise TypeError(f'Unknown image_num type')
    #     return self._im_collection.filterMetadata('system:index', 'equals', image_num).first()

    def get_composite_image(self):
        if self._im_collection is None or self._im_df is None:
            raise Exception('First generate valid search results with search(...) method')

        return self._im_collection.median().set('COMPOSITE_IMAGES', self._im_df[['EE_ID', 'DATE'] + self._im_props].to_string())



class LandsatImSearch(ImSearch):
    def __init__(self, collection='landsat8'):
        ImSearch.__init__(self, collection=collection)
        if collection == 'landsat8':
            self._collection = 'LANDSAT/LC08/C02/T1_L2'
            self._im_props = ['VALID_PORTION', 'QA_SCORE_AVG', 'GEOMETRIC_RMSE_VERIFY', 'GEOMETRIC_RMSE_MODEL', 'SUN_AZIMUTH', 'SUN_ELEVATION']
        elif collection == 'landsat7':
            self._collection = 'LANDSAT/LE07/C02/T1_L2'
            self._im_props = ['VALID_PORTION', 'QA_SCORE_AVG', 'GEOMETRIC_RMSE_MODEL', 'SUN_AZIMUTH', 'SUN_ELEVATION']
        else:
            # TODO: add support for landsat 4-5 collection 2 when they are available
            raise ValueError(f'Unsupported landsat collection: {collection}')

        self._valid_portion = 90
        self._apply_valid_mask = False
        # 'LANDSAT_PRODUCT_ID', 'DATE_ACQUIRED', 'SCENE_CENTER_TIME', 'CLOUD_COVER_LAND', 'IMAGE_QUALITY_OLI', 'ROLL_ANGLE', 'NADIR_OFFNADIR', 'GEOMETRIC_RMSE_MODEL',

    def _check_validity(self, image):
        # Notes
        # - QA_PIXEL The *conf bit pairs (8-9,10-11,12-13,14-15) will always be 1 or more, unless it is a fill pixel -
        # i.e. the fill bit 0 is set.  Values are higher where there are cloud, cloud shadow etc.  The water bit 7, is
        # seems to be set incorrectly quite often, but with the rest of the bits ok/sensible.
        # - SR_QA_AEROSOL bits 6-7 can have a value of 0, and this 0 can occur in e.g. an area of QA_PIXEL=cloud shadow,
        # NB this is not band 9 as on GEE site, but band 8.
        # - The behaviour of updateMask in combination with ImageCollection qualityMosaic (and perhaps median and mosaic
        # ) is weird: updateMask always masks bands added with addBands, but only masks the original SR etc bands after
        # a call to qualityMosaic (or perhaps median etc)
        # - Pixels in Fill bit QA_* masks seem to refer to nodata / uncovered pixels only.  They don't occur amongst valid data

        image = self._add_timedelta(image)
        # create a mask of valid (non cloud, shadow and aerosol) pixels
        # bits 1-4 of QA_PIXEL are dilated cloud, cirrus, cloud & cloud shadow, respectively
        qa_pixel_bitmask = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
        qa_pixel = image.select('QA_PIXEL')
        cloud_mask = qa_pixel.bitwiseAnd(qa_pixel_bitmask).eq(0).rename('CLOUD_MASK')
        fill_mask = qa_pixel.bitwiseAnd(1).eq(0).rename('FILL_MASK')

        # quality scores
        cloud_conf = qa_pixel.rightShift(8).bitwiseAnd(3).rename('CLOUD_CONF')
        cloud_shadow_conf = qa_pixel.rightShift(10).bitwiseAnd(3).rename('CLOUD_SHADOW_CONF')
        cirrus_conf = qa_pixel.rightShift(14).bitwiseAnd(3).rename('CIRRUS_CONF')

        if self._collection == 'LANDSAT/LC08/C02/T1_L2':
            # bits 6-7 of SR_QA_AEROSOL, are aerosol level where 3 = high, 2=medium, 1=low
            sr_qa_aerosol_bitmask = (1 << 6) | (1 << 7)
            sr_qa_aerosol = image.select('SR_QA_AEROSOL')
            # aerosol_prob = sr_qa_aerosol.bitwiseAnd(sr_qa_aerosol_bitmask).rightShift(6)
            aerosol_prob = sr_qa_aerosol.rightShift(6).bitwiseAnd(3)
            aerosol_mask = aerosol_prob.lt(3).rename('AEROSOL_MASK')

            # TODO: is aerosol_mask helpful? it looks suspect for GEF NGI ims
            valid_mask = cloud_mask.And(aerosol_mask).And(fill_mask).rename('VALID_MASK')
            q_score = cloud_conf.add(cloud_shadow_conf).add(cirrus_conf).add(aerosol_prob).multiply(-1).add(12)
        else:
            valid_mask = cloud_mask.And(fill_mask).rename('VALID_MASK')
            q_score = cloud_conf.add(cloud_shadow_conf).add(cirrus_conf).multiply(-1).add(9)

        # use unmask below as a workaround to prevent valid_portion only reducing the region masked below
        valid_portion = valid_mask.unmask().multiply(100).reduceRegion(reducer='mean', geometry=self._search_region,
                                                                       scale=image.projection().nominalScale()).rename(['VALID_MASK'], ['VALID_PORTION'])

        # create a pixel quallity score (higher is better)
        # set q_score lowest where fill_mask==0
        q_score = q_score.where(fill_mask.Not(), 0).rename('QA_SCORE')
        q_score_avg = q_score.unmask().reduceRegion(reducer='mean', geometry=self._search_region, scale=image.projection().nominalScale()).rename(['QA_SCORE'], ['QA_SCORE_AVG'])

        if False:
            image = image.addBands(cloud_conf)
            image = image.addBands(cloud_shadow_conf)
            image = image.addBands(cirrus_conf)
            image = image.addBands(aerosol_prob)
            image = image.addBands(fill_mask)
            image = image.addBands(valid_mask)

        if self._apply_valid_mask:
            image = image.updateMask(valid_mask)
        else:
            image = image.updateMask(fill_mask)

        return image.set(valid_portion).set(q_score_avg).addBands(q_score)

    def _get_im_collection(self, start_date, end_date, region):
        return ee.ImageCollection(self._collection).\
            filterDate(start_date, end_date).\
            filterBounds(region).\
            map(self._check_validity).\
            filter(ee.Filter.gt('VALID_PORTION', self._valid_portion))

    def search(self, date, region, day_range=16, valid_portion=70, apply_valid_mask=False):
        self._valid_portion = valid_portion
        self._apply_valid_mask = apply_valid_mask
        return ImSearch.search(self, date, region, day_range=day_range)

    # def get_single_image(self, image_num):
    #     return ImSearch.get_single_image(self, image_num).toUint16()
    #
    # def get_auto_image(self, key='QA_SCORE_AVG', ascending=False):
    #     if (self._im_df is None) or (self._im_collection is None):
    #         raise Exception('First generate valid search results with search(...) method')
    #     return self._im_collection.sort(key, ascending).first().toUint16()

    def get_composite_image(self):
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')
        return self._im_collection.qualityMosaic('QA_SCORE').set('COMPOSITE_IMAGES', self._im_df[['EE_ID', 'DATE'] + self._im_props].to_string()).toUint16()

    def calibrate(self, image):
        # convert DN to float SR
        # copyProperties
        all_bands = image.bandNames()
        refl_bands = ee.List([])
        def add_refl_bands(band, refl_bands):
            refl_bands = ee.Algorithms.If(ee.String(band).rindex('SR_B').eq(0), ee.List(refl_bands).add(band), refl_bands)
            return refl_bands
        refl_bands = ee.List(all_bands.iterate(add_refl_bands, refl_bands))

        non_refl_bands = all_bands.removeAll(refl_bands)

        refl_mult = ee.Number(image.get('REFLECTANCE_MULT_BAND_1'))
        refl_add = ee.Number(image.get('REFLECTANCE_ADD_BAND_1'))
        calib_image = ((image.select(refl_bands).multiply(refl_mult)).add(refl_add)).multiply(10000.0)
        calib_image = calib_image.addBands(image.select(non_refl_bands))
        calib_image = calib_image.updateMask(image.mask())

        return ee.Image(calib_image.copyProperties(image)).toFloat()    # call toFloat after updateMask

    def calibrate2(self, image):
        # convert DN to float SR
        # copyProperties
        all_bands = image.bandNames()
        refl_bands = ee.List([])

        def add_refl_bands(band, refl_bands):
            refl_bands = ee.Algorithms.If(ee.String(band).rindex('SR_B').eq(0), ee.List(refl_bands).add(band), refl_bands)
            return refl_bands

        refl_bands = ee.List(all_bands.iterate(add_refl_bands, refl_bands))
        non_refl_bands = all_bands.removeAll(refl_bands)

        param_dict = ee.Dictionary(dict(mult=ee.Image().select([]), add=ee.Image().select([])))

        def add_refl_params(band, param_dict):
            param_dict = ee.Dictionary(param_dict)
            band_num = ee.String(band).slice(-1)
            refl_mult_str = ee.String('REFLECTANCE_MULT_BAND_').cat(band_num)
            refl_add_str = ee.String('REFLECTANCE_ADD_BAND_').cat(band_num)
            refl_mult = ee.Image.constant(image.get(refl_mult_str)).rename(refl_mult_str)
            refl_add = ee.Image.constant(image.get(refl_add_str)).rename(refl_add_str)
            param_dict = param_dict.set('mult', ee.Image(param_dict.get('mult')).addBands(refl_mult))
            param_dict = param_dict.set('add', ee.Image(param_dict.get('add')).addBands(refl_add))
            return param_dict

        param_dict = ee.Dictionary(refl_bands.iterate(add_refl_params, param_dict))

        calib_image = image.select(refl_bands).multiply(param_dict.get('mult'))
        calib_image = (calib_image.add(param_dict.get('add'))).multiply(10000.0)
        calib_image = calib_image.addBands(image.select(non_refl_bands))
        calib_image = calib_image.updateMask(image.mask())

        return ee.Image(calib_image.copyProperties(image)).toFloat()    # call toFloat after updateMask


class Sentinel2ImSearch(ImSearch):
    def __init__(self, collection='sentinel2_toa'):
        ImSearch.__init__(self, collection=collection)
        if collection == 'sentinel2_toa':
            self._collection = 'COPERNICUS/S2'
        elif collection == 'sentinel2_sr':
            self._collection = 'COPERNICUS/S2_SR'
        else:
            raise ValueError(f'Unsupported sentinel2 collection: {collection}')

        self._im_props = ['VALID_PORTION', 'GEOMETRIC_QUALITY_FLAG', 'RADIOMETRIC_QUALITY_FLAG', 'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B1', 'MEAN_INCIDENCE_ZENITH_ANGLE_B1']
        self._valid_portion = 90
        self._apply_valid_mask = False

    def _check_validity(self, image):
        image = self._add_timedelta(image)
        bit_mask = (1 << 11) | (1 << 10)
        qa = image.select('QA60')
        valid_mask = qa.bitwiseAnd(bit_mask).eq(0).rename('VALID_MASK')
        valid_portion = valid_mask.unmask().multiply(100).reduceRegion(reducer='mean', geometry=self._search_region,
                                                                       scale=image.select(1).projection().nominalScale()).rename(['VALID_MASK'], ['VALID_PORTION'])
        if self._apply_valid_mask:
            image = image.updateMask(valid_mask)
        return image.set(valid_portion)

    def _get_im_collection(self, start_date, end_date, region):
        return ee.ImageCollection(self._collection).\
            filterDate(start_date, end_date).\
            filterBounds(region).\
            map(self._check_validity).\
            filter(ee.Filter.gt('VALID_PORTION', self._valid_portion))

    def search(self, date, region, day_range=16, valid_portion=90, apply_valid_mask = False):
        self._valid_portion = valid_portion
        self._apply_valid_mask = apply_valid_mask
        return ImSearch.search(self, date, region, day_range=day_range)

    # def get_single_image(self, image_num):
    #     return ImSearch.get_single_image(self, image_num).toUint16()
    #
    # def get_auto_image(self, key='VALID_PORTION', ascending=False):
    #     if (self._im_df is None) or (self._im_collection is None):
    #         raise Exception('First generate valid search results with search(...) method')
    #     return self._im_collection.sort(key, ascending).first().toUint16()

    def get_composite_image(self):
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')
        return self._im_collection.mosaic().set('COMPOSITE_IMAGES', self._im_df[['EE_ID', 'DATE'] + self._im_props].to_string()).toUint16()

class Sentinel2CloudlessImSearch(ImSearch):
    def __init__(self, collection='sentinel2_toa'):
        ImSearch.__init__(self, collection=collection)
        if collection == 'sentinel2_toa':
            self._collection = 'COPERNICUS/S2'
        elif collection == 'sentinel2_sr':
            self._collection = 'COPERNICUS/S2_SR'
        else:
            raise ValueError(f'Unsupported sentinel2 collection: {collection}')
        self.scale = 10
        self._im_props = ['VALID_PORTION', 'QA_SCORE_AVG', 'GEOMETRIC_QUALITY_FLAG', 'RADIOMETRIC_QUALITY_FLAG', 'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B1', 'MEAN_INCIDENCE_ZENITH_ANGLE_B1']
        self._valid_portion = 90
        self._apply_valid_mask = False

        self._cloud_filter = 60         # Maximum image cloud cover percent allowed in image collection
        self._cloud_prob_thresh = 40    # Cloud probability (%); values greater than are considered cloud
        self._nir_drk_thresh = 0.15     # Near-infrared reflectance; values less than are considered potential cloud shadow
        self._cloud_proj_dist = 1       # Maximum distance (km) to search for cloud shadows from cloud edges
        self._buffer = 100               # Distance (m) to dilate the edge of cloud-identified objects

    def _check_validity(self, image):
        # adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
        # highest res of s2 image, that all bands of this image will be reprojected to if it is downloaded
        image = self._add_timedelta(image)
        scale = image.select(1).projection().nominalScale()

        cloud_prob = ee.Image(image.get('s2cloudless')).select('probability')
        q_score = cloud_prob.multiply(-1).add(100).rename('QA_SCORE')

        cloud_mask = cloud_prob.gt(self._cloud_prob_thresh).rename('CLOUD_MASK')
        # https://en.wikipedia.org/wiki/Solar_azimuth_angle perhaps it is the angle between south and shadow cast by vertical rod, clockwise +ve
        # shadow_mask = shadow_mask.multiply(dark_pixels).rename('shadow_mask')
        # TODO: dilate valid_mask by _buffer ?
        # TODO: does below work in N hemisphere?
        shadow_azimuth = ee.Number(-90).add(ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

        # project the the cloud mask in the direction of shadows for self._cloud_proj_dist
        proj_cloud_mask = (cloud_mask.directionalDistanceTransform(shadow_azimuth, self._cloud_proj_dist * 1000/10)
                       # .reproject(**{'crs': image.select(0).projection(), 'scale': 100}) # necessary?
                       .select('distance')
                       .mask()  # applies cloud_mask?
                       .rename('PROJ_CLOUD_MASK'))

        if self._collection == 'COPERNICUS/S2_SR':
            # Note: SCL does not classify cloud shadows well, they are often labelled "dark".  Instead of using only
            # cloud shadow areas from this band, we combine it with the projected dark and shadow areas from SCL band
            scl = image.select('SCL')
            dark_shadow_mask = scl.eq(3).Or(scl.eq(2)).focal_max(self._buffer, 'circle', 'meters')     # dilate
            shadow_mask = proj_cloud_mask.And(dark_shadow_mask).rename('SHADOW_MASK')
        else:
            shadow_mask = proj_cloud_mask.rename('SHADOW_MASK')   # mask all areas that could be cloud shadow

        # combine cloud and cloud shadow masks into one
        valid_mask = (cloud_mask.Or(shadow_mask)).Not().rename('VALID_MASK')

        valid_portion = valid_mask.unmask().multiply(100).reduceRegion(reducer='mean', geometry=self._search_region,
                                                                       scale=scale).rename(['VALID_MASK'], ['VALID_PORTION'])

        # create a pixel quallity score (higher is better)
        # set q_score lowest where fill_mask==0
        # q_score = q_score.where(fill_mask.Not(), 0).rename('QA_SCORE')
        q_score_avg = q_score.unmask().reduceRegion(reducer='mean', geometry=self._search_region, scale=scale).rename(['QA_SCORE'], ['QA_SCORE_AVG'])

        if False:
            image = image.addBands(cloud_prob)
            image = image.addBands(cloud_mask)
            image = image.addBands(proj_cloud_mask)
            image = image.addBands(shadow_mask)
            image = image.addBands(valid_mask)

        if self._apply_valid_mask:
            # NOTE: for export_image, updateMask sets pixels to 0, for download_image, it does the same and sets nodata=0
            image = image.updateMask(valid_mask)
        # else:
        #     image = image.unmask()
        # else:
        #     image = image.updateMask(fill_mask)

        return image.set(valid_portion).set(q_score_avg).addBands(q_score)

    def _get_im_collection(self, start_date, end_date, region):
        # adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
        # Import and filter S2 SR.
        s2_sr_toa_col = (ee.ImageCollection(self._collection)
                     .filterBounds(region)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self._cloud_filter)))

        # Import and filter s2cloudless.
        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                            .filterBounds(region)
                            .filterDate(start_date, end_date))

        # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
        return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': s2_sr_toa_col,
            'secondary': s2_cloudless_col,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
        })).map(self._check_validity).filter(ee.Filter.gt('VALID_PORTION', self._valid_portion))

    def search(self, date, region, day_range=16, valid_portion=90, apply_valid_mask=False):
        self._valid_portion = valid_portion
        self._apply_valid_mask = apply_valid_mask
        return ImSearch.search(self, date, region, day_range=day_range)

    # def get_single_image(self, image_num):
    #     return ImSearch.get_single_image(self, image_num).toUint16()
    #
    # def get_auto_image(self, key='VALID_PORTION', ascending=False):
    #     if (self._im_df is None) or (self._im_collection is None):
    #         raise Exception('First generate valid search results with search(...) method')
    #     return self._im_collection.sort(key, ascending).first().toUint16()

    def get_composite_image(self):
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')
        # return self._im_collection.qualityMosaic('QA_SCORE').set('COMPOSITE_IMAGES', self._im_df.to_string())
        return self._im_collection.median().set('COMPOSITE_IMAGES', self._im_df[['EE_ID', 'DATE'] + self._im_props].to_string()).toUint16()

class ModisNbarImSearch(ImSearch):
    def __init__(self, collection='modis'):
        ImSearch.__init__(self, collection=collection)
        if collection == 'modis':
            self._collection = 'MODIS/006/MCD43A4'
        else:
            raise ValueError(f'Unsupported modis collection: {collection}')


##

