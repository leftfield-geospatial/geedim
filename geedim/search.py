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

##
# Classes for searching GEE image collections
import json
import logging
from datetime import datetime

import click
import ee
import pandas
import pandas as pd
import rasterio as rio
from rasterio.warp import transform_geom

from geedim import download, root_path

# from shapely import geometry

##


def load_collection_info():
    """
    Loads the satellite band etc information from json file into a dict
    """
    with open(root_path.joinpath('data/inputs/collection_info.json')) as f:
        satellite_info = json.load(f)

    return satellite_info


def get_image_bounds(filename, expand=5):
    """
    Get a WGS84 geojson polygon representing the optionally expanded bounds of an image

    Parameters
    ----------
    filename :  str, pathlib.Path
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
    try:
        # GEE sets tif colorinterp tags incorrectly, suppress rasterio warning relating to this:
        # 'Sum of Photometric type-related color channels and ExtraSamples doesn't match SamplesPerPixel'
        logging.getLogger("rasterio").setLevel(logging.ERROR)
        with rio.open(filename) as im:
            bbox = im.bounds
            if (im.crs.linear_units == 'metre') and (expand > 0):  # expand the bounding box
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
            src_bbox_wgs84 = transform_geom(im.crs, 'WGS84', bbox_expand_dict)  # convert to WGS84 geojson
    finally:
        logging.getLogger("rasterio").setLevel(logging.WARNING)

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
        collection_info = load_collection_info()
        if collection not in collection_info:
            raise ValueError(f'Unknown collection: {collection}')
        self.collection_info = collection_info[collection]
        self._valid_portion = 0
        self._apply_mask = False

        # list of image properties to display in search results
        self._im_props = pd.DataFrame(self.collection_info['properties'])
        self._search_date = None
        self._im_transform = lambda image: image

    def _process_image(self, image, region=None, apply_mask=True):
        """
        Finds the time difference between image and search date, and adds this as a property to the image

        Parameters
        ----------
        image : ee.Image
                Image to process
        region : ee.Geometry, dict, geojson
                 Not used
        apply_mask : bool
                     Not used

        Returns
        -------
        : ee.Image
          Processed image
        """
        if self._search_date is not None:
            image = image.set('TIME_DIST', ee.Number(image.get('system:time_start')).
                              subtract(self._search_date.timestamp() * 1000).abs())
        return image

    def _get_im_collection(self, start_date, end_date, region):
        """
        Create an image collection filtered by date, bounds and *ImSearch-specific mapping

        Parameters
        ----------
        start_date : str, ee.Date
                     Earliest image date e.g. '2015-05-08'
        end_date : str, ee.Date
                   Latest image date e.g. '2015-05-08'
        region : dict, geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect

        Returns
        -------
        : ee.ImageCollection
        The filtered image collection
        """
        return (ee.ImageCollection(self.collection_info['ee_collection']).
                filterDate(start_date, end_date).
                filterBounds(region).
                map(lambda image: self._process_image(image, region=region, apply_mask=self._apply_mask)))

    @staticmethod
    def _get_collection_df(im_collection, property_df, im_transform=lambda x: x, do_print=True):
        """
        Convert a filtered image collection to a pandas dataframe of images and their properties

        Parameters
        ----------
        im_collection : ee.ImageCollection
                        Filtered image collection
        property_df : pandas.Dataframe
                        Dateframe of image properties to include in results with ['Property', 'Abbrev', 'Description']
                        columns
        im_transform : lambda, optional
                        Transformation function to apply to search result images
        do_print : bool, optional
                   Print the dataframe

        Returns
        -------
        : pandas.DataFrame
        Dataframe of ee.Image objects and their properties
        """

        init_list = ee.List([])

        # aggregate relevant properties of im_collection images
        def aggregrate_props(image, prop_list):
            prop = ee.Dictionary()
            for prop_key in property_df.PROPERTY.values:
                prop = prop.set(prop_key, ee.Algorithms.If(image.get(prop_key), image.get(prop_key), ee.String('None')))
            return ee.List(prop_list).add(prop)

        # retrieve list of dicts of collection image properties
        im_prop_list = ee.List(im_collection.iterate(aggregrate_props, init_list)).getInfo()

        if len(im_prop_list) == 0:
            click.echo('No images found')
            return pandas.DataFrame([], columns=property_df.ABBREV)

        im_list = im_collection.toList(im_collection.size())  # image objects TODO: exclude IMAGE

        # add EE image objects and convert ee.Date to python datetime
        for i, prop_dict in enumerate(im_prop_list):
            if 'system:time_start' in prop_dict:
                prop_dict['system:time_start'] = datetime.utcfromtimestamp(prop_dict['system:time_start'] / 1000)
            prop_dict['IMAGE'] = im_transform(ee.Image(im_list.get(i)))  # TODO: remove IMAGE ?

        # convert to DataFrame
        im_prop_df = pandas.DataFrame(im_prop_list, columns=im_prop_list[0].keys())
        im_prop_df = im_prop_df.sort_values(by='system:time_start').reset_index(drop=True)
        im_prop_df = im_prop_df.rename(
            columns=dict(zip(property_df.PROPERTY, property_df.ABBREV)))  # rename cols to abbrev
        im_prop_df = im_prop_df[property_df.ABBREV]     # reorder columns

        if do_print:
            click.echo(f'{len(im_prop_list)} images found')
            click.echo('\nImage property descriptions:\n\n' +
                        property_df[['ABBREV', 'DESCRIPTION']].to_string(index=False, justify='right'))

            click.echo('\nSearch Results:\n\n' + im_prop_df.to_string(
                float_format='%.2f',
                formatters={'DATE': lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M')},
                columns=property_df.ABBREV,
                # header=property_df.ABBREV,
                index=False,
                justify='center'))

        return im_prop_df

    def get_image(self, image_id, region=None, apply_mask=False):
        """
        Retrieve an ee.Image object, adding validity and quality metadata where possible

        Parameters
        ----------
        image_id : str
             Earth engine image ID e.g. 'LANDSAT/LC08/C02/T1_L2/LC08_182037_20190118 2019-01-18'
        region : ee.Geometry, dict, geojson
                 Process image over this region
        apply_mask : bool
                     Apply any validity mask to the image by setting nodata

        Returns
        -------
        : ee.Image
          The processed image
        """
        if '/'.join(image_id.split('/')[:-1]) != self.collection_info['ee_collection']:
            raise ValueError(f'{image_id} is not a valid earth engine id for {self.__class__}')

        return self._im_transform(self._process_image(ee.Image(image_id), region=region, apply_mask=apply_mask))

    def search(self, start_date, end_date, region, valid_portion=0, apply_mask=False):
        """
        Search for Sentinel-2 images based on date, region etc criteria

        Parameters
        start_date : datetime.datetime
                     Python datetime specifying the start image capture date
        end_date : datetime.datetime
                   Python datetime specifying the end image capture date (if None, then set to start_date)
        region : dict, geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect
        valid_portion: int, optional
                       Minimum portion (%) of image pixels that should be valid (not cloud)
        apply_mask : bool, optional
                           Mask out clouds in search result images

        Returns
        -------
        image_df : pandas.DataFrame
        Dataframe specifying image properties that match the search criteria
        """
        # Initialise
        self._valid_portion = valid_portion
        self._apply_mask = apply_mask
        if end_date is None:
            end_date = start_date
        self._search_date = start_date + (end_date - start_date) / 2

        # start_date = date - timedelta(days=day_range)
        # end_date = date + timedelta(days=day_range)

        # filter the image collection
        im_collection = self._get_im_collection(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                                                region)

        # convert and print search results
        return ImSearch._get_collection_df(im_collection, self._im_props, do_print=True,
                                                  im_transform=self._im_transform)


##
class LandsatImSearch(ImSearch):
    def __init__(self, collection='landsat8_c2_l2'):
        """
        Class for searching Landsat 7-8 earth engine image collections

        Parameters
        ----------
        collection : str, optional
                     'landsat7_c2_l2' or 'landsat8_c2_l2' (default)
        """
        ImSearch.__init__(self, collection=collection)

        # TODO: add support for landsat 4-5 collection 2 when they are available
        if collection not in ['landsat8_c2_l2', 'landsat7_c2_l2']:
            raise ValueError(f'Unsupported landsat collection: {collection}')

        self._im_transform = ee.Image.toUint16

    def _process_image(self, image, region=None, apply_mask=True):
        """
        Find image validity (cloud, shadow and fill mask) and quality score

        Parameters
        ----------
        image : ee.Image
                Image to evaluate
        region : dict, geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect
        apply_mask : bool
                     Set invalid areas to nodata
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
        if region is None:
            region = image.geometry()

        image = ImSearch._process_image(self, image, region=region,
                                        apply_mask=apply_mask)  # Add TIME_DIST property from base class

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

        if self.collection_info['ee_collection'] == 'LANDSAT/LC08/C02/T1_L2':  # landsat8_c2_l2
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
        valid_portion = (valid_mask.unmask().
                         multiply(100).
                         reduceRegion(reducer='mean', geometry=region,
                                      scale=image.projection().nominalScale()).
                         rename(['VALID_MASK'], ['VALID_PORTION']))

        q_score = q_score.where(fill_mask.Not(), 0).rename('QA_SCORE')  # Zero q_score where pixels are unfilled

        # calculate the average quality score
        q_score_avg = (q_score.unmask().
                       reduceRegion(reducer='mean', geometry=region,
                                    scale=image.projection().nominalScale()).
                       rename(['QA_SCORE'], ['QA_SCORE_AVG']))

        if False:
            image = image.addBands(cloud_conf)
            image = image.addBands(cloud_shadow_conf)
            image = image.addBands(cirrus_conf)
            image = image.addBands(aerosol_prob)
            image = image.addBands(fill_mask)
            image = image.addBands(valid_mask)

        if apply_mask:
            image = image.updateMask(valid_mask)  # mask out cloud, shadow, unfilled etc pixels
        else:
            # image = image.updateMask(fill_mask).addBands(valid_mask)  # mask out unfilled pixels only
            image = image.addBands(valid_mask)  # mask out unfilled pixels only

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
        region : dict, geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect

        Returns
        -------
        : ee.ImageCollection
        The filtered image collection
        """
        # TODO: make self._apply_mask and self._valid_portion parameters?
        return (ee.ImageCollection(self.collection_info['ee_collection']).
                filterDate(start_date, end_date).
                filterBounds(region).
                map(lambda image: self._process_image(image, region=region, apply_mask=self._apply_mask)).
                filter(ee.Filter.gt('VALID_PORTION', self._valid_portion)))


    @staticmethod
    def convert_dn_to_sr(image):
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
        # TODO: can we replace with unitScale?
        # retrieve the names of SR bands
        all_bands = image.bandNames()
        init_bands = ee.List([])

        def add_refl_bands(band, refl_bands):
            refl_bands = ee.Algorithms.If(ee.String(band).rindex('SR_B').eq(0), ee.List(refl_bands).add(band),
                                          refl_bands)
            return refl_bands

        sr_bands = ee.List(all_bands.iterate(add_refl_bands, init_bands))
        non_sr_bands = all_bands.removeAll(sr_bands)  # all the other non-SR bands

        # retrieve the scale (mult) and offset (add) parameters for each band
        param_dict = ee.Dictionary(dict(mult=ee.Image().select([]), add=ee.Image().select([])))

        def add_refl_params(band, param):
            param = ee.Dictionary(param)

            band_num = ee.String(band).slice(-1)
            sr_mult_str = ee.String('REFLECTANCE_MULT_BAND_').cat(band_num)
            sr_add_str = ee.String('REFLECTANCE_ADD_BAND_').cat(band_num)

            # create constant scale (mult) and offset (add) images for this band
            sr_mult = ee.Image.constant(image.get(sr_mult_str)).rename(sr_mult_str)
            sr_add = ee.Image.constant(image.get(sr_add_str)).rename(sr_add_str)

            # add the constant scale/offset images for this band to multi-band scale/offset images
            param = param.set('mult', ee.Image(param.get('mult')).addBands(sr_mult))
            param = param.set('add', ee.Image(param.get('add')).addBands(sr_add))

            return param

        param_dict = ee.Dictionary(sr_bands.iterate(add_refl_params, param_dict))

        # apply the scale and offset
        calib_image = image.select(sr_bands).multiply(param_dict.get('mult'))
        calib_image = (calib_image.add(param_dict.get('add'))).multiply(10000.0)
        calib_image = calib_image.addBands(image.select(non_sr_bands))
        calib_image = calib_image.updateMask(image.mask())  # apply any existing mask to refl image

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

        self._im_transform = ee.Image.toUint16

    def _process_image(self, image, region=None, apply_mask=True):
        """
        Find image validity (cloud mask)

        Parameters
        ----------
        image : ee.Image
                Image to evaluate
        region : dict, geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect
        apply_mask : bool
                     Set invalid areas to nodata
        Returns
        -------
        : ee.Image
        Image with added properties and band(s)
        """
        if region is None:
            region = image.geometry()

        image = ImSearch._process_image(self, image, region=region, apply_mask=apply_mask)

        bit_mask = (1 << 11) | (1 << 10)  # bits 10 and 11 are opaque and cirrus clouds respectively
        qa = image.select('QA60')
        valid_mask = qa.bitwiseAnd(bit_mask).eq(0).rename('VALID_MASK')

        min_scale = download.get_min_projection(image).nominalScale()

        # calculate the potion of valid image pixels
        valid_portion = (valid_mask.unmask().
                         multiply(100).
                         reduceRegion(reducer='mean', geometry=region, scale=min_scale).
                         rename(['VALID_MASK'], ['VALID_PORTION']))

        # TODO: what happens if/when pixels aren't filled
        if apply_mask:
            image = image.updateMask(valid_mask)
        else:
            image = image.addBands(valid_mask)

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
        region : dict, geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect

        Returns
        -------
        : ee.ImageCollection
        The filtered image collection
        """
        return (ee.ImageCollection(self.collection_info['ee_collection']).
                filterDate(start_date, end_date).
                filterBounds(region).
                map(lambda image: self._process_image(image, region=region, apply_mask=self._apply_mask)).
                filter(ee.Filter.gt('VALID_PORTION', self._valid_portion)))


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
        self._im_transform = ee.Image.toUint16

        self._cloud_filter = 60  # Maximum image cloud cover percent allowed in image collection
        self._cloud_prob_thresh = 40  # Cloud probability (%); values greater than are considered cloud
        # self._nir_drk_thresh = 0.15# Near-infrared reflectance; values less than are considered potential cloud shadow
        self._cloud_proj_dist = 1  # Maximum distance (km) to search for cloud shadows from cloud edges
        self._buffer = 100  # Distance (m) to dilate the edge of cloud-identified objects

    def _process_image(self, image, region=None, apply_mask=True):
        """
        Find image validity (cloud, shadow and fill mask) and quality score

        Parameters
        ----------
        image : ee.Image
                Image to evaluate
        region : dict, geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect
        apply_mask : bool
                     Set invalid areas to nodata
        Returns
        -------
        : ee.Image
        Image with added properties and band(s)
        """
        # adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
        if region is None:
            region = image.geometry()

        image = ImSearch._process_image(self, image, region=region, apply_mask=apply_mask)
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
                           select('distance').mask().rename('PROJ_CLOUD_MASK'))     # mask converts to boolean?
        # .reproject(**{'crs': image.select(0).projection(), 'scale': 100})

        if self.collection_info['ee_collection'] == 'COPERNICUS/S2_SR':  # use SCL to reduce shadow_mask
            # Note: SCL does not classify cloud shadows well, they are often labelled "dark".  Instead of using only
            # cloud shadow areas from this band, we combine it with the projected dark and shadow areas from s2cloudless
            scl = image.select('SCL')
            dark_shadow_mask = scl.eq(3).Or(scl.eq(2)).focal_max(self._buffer, 'circle', 'meters')
            shadow_mask = proj_cloud_mask.And(dark_shadow_mask).rename('SHADOW_MASK')
        else:
            shadow_mask = proj_cloud_mask.rename('SHADOW_MASK')  # mask all areas that could be cloud shadow

        # combine cloud and shadow masks
        valid_mask = (cloud_mask.Or(shadow_mask)).Not().rename('VALID_MASK')

        # calculate the potion of valid image pixels TODO: this is repeated in each class
        valid_portion = (valid_mask.unmask().
                         multiply(100).
                         reduceRegion(reducer='mean', geometry=region, scale=min_scale).
                         rename(['VALID_MASK'], ['VALID_PORTION']))

        # calculate the average quality score
        q_score_avg = (q_score.unmask().
                       reduceRegion(reducer='mean', geometry=region, scale=min_scale).
                       rename(['QA_SCORE'], ['QA_SCORE_AVG']))

        if False:
            image = image.addBands(cloud_prob)
            image = image.addBands(cloud_mask)
            image = image.addBands(proj_cloud_mask)
            image = image.addBands(shadow_mask)
            image = image.addBands(valid_mask)

        if apply_mask:
            # NOTE: for export_image, updateMask sets pixels to 0,
            # for download_image, it does the same and sets nodata=0
            image = image.updateMask(valid_mask)
        else:
            image = image.addBands(valid_mask)

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
        region : dict, geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect

        Returns
        -------
        : ee.ImageCollection
        The filtered image collection
        """
        # adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless

        s2_sr_toa_col = (ee.ImageCollection(self.collection_info['ee_collection'])
                         .filterBounds(region)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self._cloud_filter)))

        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                            .filterBounds(region)
                            .filterDate(start_date, end_date))

        # join filtered s2cloudless collection to the SR/TOA collection by the 'system:index' property.
        return (ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': s2_sr_toa_col,
            'secondary': s2_cloudless_col,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })})).map(lambda image: self._process_image(image, region=region, apply_mask=self._apply_mask)).
                filter(ee.Filter.gt('VALID_PORTION', self._valid_portion)))

    def get_image(self, image_id, region=None, apply_mask=False):
        """
        Retrieve an ee.Image object, adding validity and quality metadata where possible

        Parameters
        ----------
        image_id : str
             Earth engine image ID e.g. 'LANDSAT/LC08/C02/T1_L2/LC08_182037_20190118 2019-01-18'
        region : ee.Geometry, dict, geojson
                 Process image over this region
        apply_mask : bool
                     Apply any validity mask to the image by setting nodata

        Returns
        -------
        : ee.Image
          The processed image
        """
        index = image_id.split('/')[-1]
        collection = '/'.join(image_id.split('/')[:-1])

        if collection != self.collection_info['ee_collection']:
            raise ValueError(f'{image_id} is not a valid earth engine id for {self.__class__}')

        # combine COPERNICUS/S2* and COPERNICUS/S2_CLOUD_PROBABILITY images
        s2_cloud_prob_image = ee.Image(f'COPERNICUS/S2_CLOUD_PROBABILITY/{index}')
        image = ee.Image(image_id).set('s2cloudless', s2_cloud_prob_image)
        image = self._process_image(image, region=region, apply_mask=apply_mask).set('s2cloudless', None)

        return self._im_transform(image)


##
class ModisNbarImSearch(ImSearch):
    def __init__(self, collection='modis_nbar'):
        """
        Class for searching the MODIS daily NBAR earth engine image collection
        """
        ImSearch.__init__(self, collection)

##
