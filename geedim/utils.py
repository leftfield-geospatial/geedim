"""
    Geedim: Download surface reflectance imagery with Google Earth Engine
    Copyright (C) 2021 Dugal Harris
    Email: dugalh@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import json
import os
import pathlib
import sys
import zipfile
from datetime import timedelta, datetime
import urllib

import ee
import numpy as np
import pandas
import pandas as pd
import rasterio as rio
from geedim import get_logger, root_path
from rasterio.warp import transform_geom
from shapely import geometry
import time

logger = get_logger(__name__)

def load_collection_info():
    """
    Loads the satellite band etc information from json file into a dict
    """
    with open(root_path.joinpath('data/inputs/satellite_info.json')) as f:
        satellite_info = json.load(f)
    return satellite_info

def get_image_bounds(filename, expand=10):
    """
    Get the WGS84 geojson bounds of an image

    Parameters
    ----------
    filename :  str
                name of the image file whose bounds to find
    expand :  int
              percentage (0-100) by which to expand the bounds (default: 10)

    Returns
    -------
    bounds : geojson
             polygon of bounds in WGS84
    crs: str
         WKT CRS of image file
    """
    with rio.open(filename) as im:
        src_bbox = geometry.box(*im.bounds)
        if (im.crs.linear_units == 'metre') and (expand > 0):
            expand_m = np.sqrt(src_bbox.area) * expand / 100.
            src_bbox_expand = src_bbox.buffer(expand_m, join_style=geometry.JOIN_STYLE.mitre)    # expand the bounding box
            # src_bbox = geometry.box(*src_bbox.buffer(expand_m).bounds)    # expand the bounding box
        src_bbox_wgs84 = geometry.shape(transform_geom(im.crs, 'WGS84', src_bbox_expand))
    return geometry.mapping(src_bbox_wgs84), im.crs.to_wkt()

# from https://github.com/gee-community/gee_tools, MIT license
def minscale(image):
    """ Get the minimal scale of an Image, looking at all Image's bands.
    For example if:
        B1 = 30
        B2 = 60
        B3 = 10
    the function will return 10
    :return: the minimal scale
    :rtype: ee.Number
    """
    bands = image.bandNames()

    first = image.select([ee.String(bands.get(0))])
    ini = ee.Number(first.projection().nominalScale())

    def wrap(name, i):
        i = ee.Number(i)
        scale = ee.Number(image.select([name]).projection().nominalScale())
        condition = scale.lte(i)
        newscale = ee.Algorithms.If(condition, scale, i)
        return newscale

    return ee.Number(bands.slice(1).iterate(wrap, ini))

# from https://github.com/gee-community/gee_tools, MIT license
def parametrize(image, range_from, range_to, bands=None, drop=False):
    """ Parametrize from a original **known** range to a fixed new range
    :param range_from: Original range. example: (0, 5000)
    :type range_from: tuple
    :param range_to: Fixed new range. example: (500, 1000)
    :type range_to: tuple
    :param bands: bands to parametrize. If *None* all bands will be
        parametrized.
    :type bands: list
    :param drop: drop the bands that will not be parametrized
    :type drop: bool
    :return: the parsed image with the parsed bands parametrized
    :rtype: ee.Image
    """
    original_range = range_from if isinstance(range_from, ee.List) \
        else ee.List(range_from)

    final_range = range_to if isinstance(range_to, ee.List) \
        else ee.List(range_to)

    # original min and max
    min0 = ee.Image.constant(original_range.get(0))
    max0 = ee.Image.constant(original_range.get(1))

    # range from min to max
    rango0 = max0.subtract(min0)

    # final min max images
    min1 = ee.Image.constant(final_range.get(0))
    max1 = ee.Image.constant(final_range.get(1))

    # final range
    rango1 = max1.subtract(min1)

    # all bands
    all = image.bandNames()

    # bands to parametrize
    if bands:
        bands_ee = ee.List(bands)
    else:
        bands_ee = image.bandNames()

    inter = ee_list.intersection(bands_ee, all)
    diff = ee_list.difference(all, inter)
    image_ = image.select(inter)

    # Percentage corresponding to the actual value
    percent = image_.subtract(min0).divide(rango0)

    # Taking count of the percentage of the original value in the original
    # range compute the final value corresponding to the final range.
    # Percentage * final_range + final_min

    final = percent.multiply(rango1).add(min1)

    if not drop:
        # Add the rest of the bands (no parametrized)
        final = image.select(diff).addBands(final)

    # return passProperty(image, final, 'system:time_start')
    return ee.Image(final.copyProperties(source=image))

# from https://github.com/gee-community/gee_tools, MIT license
def unpack(iterable):
    """ Helper function to unpack an iterable """
    unpacked = []
    for tt in iterable:
        for t in tt:
            unpacked.append(t)
    return unpacked

# from https://github.com/gee-community/gee_tools, MIT license
def getRegion(eeobject, bounds=False, error=1):
    """ Gets the region of a given geometry to use in exporting tasks. The
    argument can be a Geometry, Feature or Image
    :param eeobject: geometry to get region of
    :type eeobject: ee.Feature, ee.Geometry, ee.Image
    :param error: error parameter of ee.Element.geometry
    :return: region coordinates ready to use in a client-side EE function
    :rtype: json
    """
    def dispatch(geometry):
        info = geometry.getInfo()
        geomtype = info['type']
        if geomtype == 'GeometryCollection':
            geometries = info['geometries']
            region = []
            for geom in geometries:
                this_type = geom['type']
                if this_type in ['Polygon', 'Rectangle']:
                    region.append(geom['coordinates'][0])
                elif this_type in ['MultiPolygon']:
                    geometries2 = geom['coordinates']
                    region.append(unpack(geometries2))

        elif geomtype == 'MultiPolygon':
            subregion = info['coordinates']
            region = unpack(subregion)
        else:
            region = info['coordinates']

        return region

    # Geometry
    if isinstance(eeobject, ee.Geometry):
        geometry = eeobject.bounds() if bounds else eeobject
        region = dispatch(geometry)
    # Feature and Image
    elif isinstance(eeobject, (ee.Feature, ee.Image)):
        geometry = eeobject.geometry(error).bounds() if bounds else eeobject.geometry(error)
        region = dispatch(geometry)
    # FeatureCollection and ImageCollection
    elif isinstance(eeobject, (ee.FeatureCollection, ee.ImageCollection)):
        if bounds:
            geometry = eeobject.geometry(error).bounds()
        else:
            geometry = eeobject.geometry(error).dissolve()
        region = dispatch(geometry)
    # List
    elif isinstance(eeobject, list):
        condition = all([type(item) == list for item in eeobject])
        if condition:
            region = eeobject
    else:
        region = eeobject

    return region

class EeRefImage:
    def __init__(self, collection=''):
        self.collection = collection
        # self._search_region, self._crs = get_image_bounds(source_image_filename, expand=10)
        collection_info = load_collection_info()
        if not collection in collection_info.keys():
            raise ValueError(f'Unknown collection: {collection}')
        self._collection_info = collection_info[collection]
        self._band_df = pd.DataFrame(self._collection_info['bands'])
        self._display_properties = []
        self._im_collection = None
        self._im_df = None
        self._search_region = None
        self._search_date = None

    def _add_timedelta(self, image):
        return image.set('TIME_DIST', ee.Number(image.get('system:time_start')).
                         subtract(self._search_date.timestamp()*1000).abs())

    def _get_im_collection(self, start_date, end_date):
        return ee.ImageCollection(self._collection_info['ee_collection']).\
            filterDate(start_date, end_date).\
            filterBounds(self._search_region).\
            map(self._add_timedelta)
            # filter(ee.Filter.contains('system:footprint'), self._search_region).\

    def get_image_info(self, image):
        im_info = image.getInfo()

        band_info_df = pd.DataFrame(im_info['bands'])
        crs_transforms = np.array(band_info_df['crs_transform'].to_list())
        scales = np.abs(crs_transforms[:, 0])
        min_scale_i = np.argmin(scales)
        # crs = image.select(min_scale_i).projection().crs()
        # scale = image.select(min_scale_i).projection().nominalScale()
        min_crs = band_info_df.iloc[min_scale_i]['crs']
        min_scale = scales[min_scale_i]

        return im_info, min_crs, min_scale


    def search(self, date, region, day_range=16):
        self._search_region = region
        self._search_date = date
        self._im_df = None

        start_date = date - timedelta(days=day_range)
        end_date = date + timedelta(days=day_range)
        num_images = 0

        logger.info(f'Searching for {self._collection_info["ee_collection"]} images between '
                    f'{start_date.strftime("%Y-%m-%d")} and {end_date.strftime("%Y-%m-%d")}')
        self._im_collection = self._get_im_collection(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        num_images = self._im_collection.size().getInfo()

        if num_images == 0:
            logger.info(f'Could not find any images in that date range')
            return None

        logger.info(f'Found {num_images} images:')
        self._im_df = self._display_search_results()

        return self._im_df.to_dict(orient='index')

    def _display_search_results(self):
        init_list = ee.List([])
        display_properties = self._display_properties

        def aggregrate_props(image, res_list):
            res_dict = ee.Dictionary()
            res_dict = res_dict.set('EE_ID', image.get('system:index'))
            res_dict = res_dict.set('DATE', image.get('system:time_start'))
            for prop_key in display_properties:
                res_dict = res_dict.set(prop_key, ee.Algorithms.If(image.get(prop_key), image.get(prop_key), ee.String('None')))
            return ee.List(res_list).add(res_dict)

        # retrieve properties of search result images for display
        res_list = ee.List(self._im_collection.iterate(aggregrate_props, init_list)).getInfo()

        # create dataframe of search results
        im_df = pandas.DataFrame(res_list)
        im_df['DATE'] = [datetime.utcfromtimestamp(ts/1000) for ts in im_df['DATE']]    # convert timestamp to datetime
        cols = ['EE_ID', 'DATE'] + self._display_properties     # re-order columns
        im_df = im_df[cols].sort_values(by='DATE').reset_index(drop=True)
        logger.info('Search results:\n' + im_df.to_string())
        return im_df

    def get_auto_image(self):
        if (self._im_df is None) or (self._im_collection is None):
            raise Exception('First generate valid search results with search(...) method')
        return self._im_collection.sort('TIME_DIST', True).first()

    def get_single_image(self, image_num):
        if (self._im_df is None) or (self._im_collection is None):
            raise Exception('First generate valid search results with search(...) method')

        image = None
        if isinstance(image_num, str):
            if image_num not in self._im_df['EE_ID']:
                raise ValueError(f'{image_num} does not exist in search results')
            image = self._im_collection.filterMetadata('system:index', 'equals', image_num).first()
        elif isinstance(image_num, int):
            if (image_num >= 0) and (image_num < self._im_df.shape[0]):
                image_num = self._im_df.loc[image_num]['EE_ID']
                image = self._im_collection.filterMetadata('system:index', 'equals', image_num).first()
            else:
                raise ValueError(f'image_num={image_num} out of range')
        else:
            raise TypeError(f'Unknown image_num type')
        return image

    def get_composite_image(self):
        if self._im_collection is None or self._im_df is None:
            raise Exception('First generate valid search results with search(...) method')

        return self._im_collection.median().set('COMPOSITE_IMAGES', self._im_df.to_string())

    def export_image(self, image, description, folder=None, crs=None, region=None, scale=None, wait=True):

        im_info, min_crs, min_scale = self.get_image_info(image)

        # check if scale is same across bands
        band_info_df = pd.DataFrame(im_info['bands'])
        scales = np.array(band_info_df['crs_transform'].to_list())[:, 0]

        # min_scale_i = np.argmin(scales) #.astype(float)
        if np.all(band_info_df['crs'] == 'EPSG:4326') and np.all(scales == 1):
            # set the crs and and scale if it is a composite image
            if crs is None or scale is None:
                _image = self._im_collection.first()
                _im_info, min_crs, min_scale = self.get_image_info(_image)
                logger.warning(f'This appears to be a composite image in WGS84, reprojecting all bands to {min_crs} at {min_scale}m resolution')

        if crs is None:
            # crs = image.select(min_scale_i).projection().crs()
            crs = min_crs
        if region is None:
            region = ee.Geometry(self._search_region)
        if scale is None:
            # scale = image.select(min_scale_i).projection().nominalScale()
            scale = float(min_scale)

        if (band_info_df['crs'].unique().size > 1) or (np.unique(scales).size > 1):
            logger.warning(f'Image bands have different scales, reprojecting all to {crs} at {scale}m resolution')

        # force all bands into same crs and scale
        band_info_df['crs'] = crs
        band_info_df['scale'] = scale
        bands_dict = band_info_df[['id', 'crs', 'scale', 'data_type']].to_dict('records')

        # TODO: make sure the cast to uint16 below is valid for new collections
        task = ee.batch.Export.image.toDrive(image=image.toUint16(),
                                             region=region,
                                             description=description,
                                             folder=folder,
                                             fileNamePrefix=description,
                                             scale=scale,
                                             crs=crs,
                                             maxPixels=1e9)

        logger.info(f'Starting export task {description}...')
        task.start()
        if wait:
            status = ee.data.getOperation(task.name)
            toggles = '-\|/'
            toggle_count = 0
            while ('done' not in status) or (not status['done']):
                time.sleep(.5)
                status = ee.data.getOperation(task.name)
                # if ('stages' in status['metadata']):
                #     stage_name = status['metadata']['stages'][-1]['displayName']
                #     status_str = status_str + f': {stage_name}'
                # sys.stdout.write(f'\rExport image status: {str(status["metadata"]["state"]).lower()} {toggles[toggle_count%4]}')
                # TODO: interpret totalWorkUnits and completeWorkUnits
                sys.stdout.write(f'\rExport image {str(status["metadata"]["state"]).lower()} [ {toggles[toggle_count%4]} ]')
                sys.stdout.flush()
                toggle_count += 1
            sys.stdout.write(f'\rExport image {str(status["metadata"]["state"]).lower()}\n')
            if status['metadata']['state'] != 'SUCCEEDED':
                logger.error(f'Export failed \n{status}')
                raise Exception(f'Export failed \n{status}')


    def download_image(self, image, filename, crs=None, region=None, scale=None):

        im_info, min_crs, min_scale = self.get_image_info(image)

        # check if scale is same across bands
        band_info_df = pd.DataFrame(im_info['bands'])
        scales = np.array(band_info_df['crs_transform'].to_list())[:, 0]

        # min_scale_i = np.argmin(scales) #.astype(float)
        if np.all(band_info_df['crs'] == 'EPSG:4326') and np.all(scales == 1):
            # set the crs and and scale if it is a composite image
            if crs is None or scale is None:
                _image = self._im_collection.first()
                _im_info, min_crs, min_scale = self.get_image_info(_image)
                logger.warning(f'This appears to be a composite image in WGS84, reprojecting all bands to {min_crs} at {min_scale}m resolution')

        if crs is None:
            # crs = image.select(min_scale_i).projection().crs()
            crs = min_crs
        if region is None:
            region = self._search_region
        if scale is None:
            # scale = image.select(min_scale_i).projection().nominalScale()
            scale = float(min_scale)

        if (band_info_df['crs'].unique().size > 1) or (np.unique(scales).size > 1):
            logger.warning(f'Image bands have different scales, reprojecting all to {crs} at {scale}m resolution')

        # force all bands into same crs and scale
        band_info_df['crs'] = crs
        band_info_df['scale'] = scale
        bands_dict = band_info_df[['id', 'crs', 'scale', 'data_type']].to_dict('records')

        # TODO: make sure the cast to uint16 is ok for any new collections
        link = image.getDownloadURL({
            'scale': scale,
            'crs': crs,
            'fileFormat': 'GeoTIFF',
            'bands':  bands_dict,
            'filePerBand': False,
            'region': region})

        logger.info(f'Opening link: {link}')

        try:
            file_link = urllib.request.urlopen(link)
        except urllib.error.HTTPError as ex:
            logger.error(f'Could not open URL: HHTP error {ex.code} - {ex.reason}')
            response = json.loads(ex.read())
            if ('error' in response) and ('message' in response['error']):
                msg = response['error']['message']
                logger.error(msg)
                if (msg == 'User memory limit exceeded.'):
                    logger.error('There is a 10MB Earth Engine limit on image downloads, either decrease image size, or use export(...)')
                    return
            raise ex

        meta = file_link.info()
        file_size = int(meta['Content-Length'])
        logger.info(f'Download size: {file_size / (1024 ** 2):.2f} MB')

        tif_filename = pathlib.Path(filename)
        tif_filename = tif_filename.joinpath(tif_filename.parent, tif_filename.stem + '.tif')    # force to zip file
        zip_filename = tif_filename.parent.joinpath('gee_image_download.zip')

        if zip_filename.exists():
            logger.warning(f'{zip_filename} exists, overwriting...')

        logger.info(f'Downloading to {zip_filename}')

        with open(zip_filename, 'wb') as f:
            file_size_dl = 0
            block_size = 8192
            while (file_size_dl <= file_size):
                buffer = file_link.read(block_size)
                if not buffer:
                    break
                file_size_dl += len(buffer)
                f.write(buffer)

                progress = (file_size_dl / file_size)
                sys.stdout.write('\r')
                sys.stdout.write('[%-50s] %d%%' % ('=' * int(50 * progress), 100 * progress))
                sys.stdout.flush()
            sys.stdout.write('\n')

        # extract download.zip -> download.tif and rename to tif_filename
        logger.info(f'Extracting {zip_filename}')
        with zipfile.ZipFile(zip_filename, "r") as zip_file:
            zip_file.extractall(zip_filename.parent)

        if tif_filename.exists():
            logger.warning(f'{tif_filename} exists, overwriting...')
            os.remove(tif_filename)

        _tif_filename = zipfile.ZipFile(zip_filename, "r").namelist()[0]
        os.rename(zip_filename.parent.joinpath(_tif_filename), tif_filename)

        if ('properties' in im_info) and ('system:footprint' in im_info['properties']):
            im_info['properties'].pop('system:footprint')

        with rio.open(tif_filename, 'r+') as im:
            if 'properties' in im_info:
                im.update_tags(**im_info['properties'])
            # im.profile['photometric'] = None    # fix warning
            if 'bands' in im_info:
                for band_i, band_info in enumerate(im_info['bands']):
                    im.set_band_description(band_i + 1, band_info['id'])
                    im.update_tags(band_i + 1, ID=band_info['id'])
                    if band_info['id'] in self._band_df['id'].to_list():
                        band_row = self._band_df.loc[self._band_df['id'] == band_info['id']].iloc[0]
                        # band_row = band_row[['abbrev', 'name', 'bw_start', 'bw_end']]
                        im.update_tags(band_i + 1, ABBREV=band_row['abbrev'])
                        im.update_tags(band_i + 1, NAME=band_row['name'])
                        im.update_tags(band_i + 1, BW_START=band_row['bw_start'])
                        im.update_tags(band_i + 1, BW_END=band_row['bw_end'])
        return link


    def _download_im_collection(self, filename, band_list=None):
        """
        Download all images in the collection

        Parameters
        ----------
        filename :  str
                    Base filename to use.

        Returns
        -------
        """
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')

        num_images = self._im_collection.size().getInfo()
        ee_im_list = self._im_collection.toList(num_images)
        filename = pathlib.Path(filename)

        for i in range(num_images):
            im_i = ee.Image(ee_im_list.get(i))
            if band_list is not None:
                im_i = im_i.select(*band_list)
            im_date_i = ee.Date(im_i.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            filename_i = filename.parent.joinpath(f'{filename.stem}_{i}_{im_date_i}.tif')
            logger.info(f'Downloading {filename_i.stem}...')
            self.download_image(im_i, filename_i)



class LandsatEeImage(EeRefImage):
    def __init__(self, apply_valid_mask=True, collection='landsat8', valid_portion=90):
        EeRefImage.__init__(self, collection=collection)
        self.apply_valid_mask = apply_valid_mask
        if self.collection == 'landsat8':
            self._display_properties = ['VALID_PORTION', 'QA_SCORE_AVG', 'GEOMETRIC_RMSE_VERIFY', 'GEOMETRIC_RMSE_MODEL', 'SUN_AZIMUTH', 'SUN_ELEVATION']
        else:
            self._display_properties = ['VALID_PORTION', 'QA_SCORE_AVG', 'GEOMETRIC_RMSE_MODEL', 'SUN_AZIMUTH', 'SUN_ELEVATION']

        self._valid_portion = valid_portion
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

        if self.collection == 'landsat8':
            # bits 6-7 of SR_QA_AEROSOL, are aerosol level where 3 = high, 2=medium, 1=low
            sr_qa_aerosol_bitmask = (1 << 6) | (1 << 7)
            sr_qa_aerosol = image.select('SR_QA_AEROSOL')
            # aerosol_prob = sr_qa_aerosol.bitwiseAnd(sr_qa_aerosol_bitmask).rightShift(6)
            aerosol_prob = sr_qa_aerosol.rightShift(6).bitwiseAnd(3)
            aerosol_mask = aerosol_prob.lt(3).rename('AEROSOL_MASK')

            # TODO: is aerosol_mask helpful in general, it looks suspect for GEF NGI ims
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

        if self.apply_valid_mask:
            image = image.updateMask(valid_mask)
        else:
            image = image.updateMask(fill_mask)

        return image.set(valid_portion).set(q_score_avg).addBands(q_score)

    def _get_im_collection(self, start_date, end_date):
        return ee.ImageCollection(self._collection_info['ee_collection']).\
            filterDate(start_date, end_date).\
            filterBounds(self._search_region).\
            map(self._check_validity).\
            filter(ee.Filter.gt('VALID_PORTION', self._valid_portion))

    def search(self, date, region, day_range=16, valid_portion=70):
        self._valid_portion = valid_portion
        EeRefImage.search(self, date, region, day_range=day_range)

    # TODO: consider making a generic version of this in the base class, perhaps with a self.key=... defaults
    def get_auto_image(self, key='QA_SCORE_AVG', ascending=False):
        if (self._im_df is None) or (self._im_collection is None):
            raise Exception('First generate valid search results with search(...) method')
        return self._im_collection.sort(key, ascending).first()

    def get_composite_image(self):
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')
        return self._im_collection.qualityMosaic('QA_SCORE').set('COMPOSITE_IMAGES', self._im_df.to_string())

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


class Sentinel2EeImage(EeRefImage):
    def __init__(self, collection='sentinel2_toa', valid_portion=90):
        EeRefImage.__init__(self, collection=collection)
        self._valid_portion = valid_portion
        self._display_properties = ['VALID_PORTION', 'GEOMETRIC_QUALITY_FLAG', 'RADIOMETRIC_QUALITY_FLAG', 'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B1', 'MEAN_INCIDENCE_ZENITH_ANGLE_B1']

    def _check_validity(self, image):
        image = self._add_timedelta(image)
        bit_mask = (1 << 11) | (1 << 10)
        qa = image.select('QA60')
        valid_mask = qa.bitwiseAnd(bit_mask).eq(0).rename('VALID_MASK')
        valid_portion = valid_mask.unmask().multiply(100).reduceRegion(reducer='mean', geometry=self._search_region,
                                                                       scale=image.select(1).projection().nominalScale()).rename(['VALID_MASK'], ['VALID_PORTION'])

        return image.set(valid_portion).updateMask(valid_mask)

    def _get_im_collection(self, start_date, end_date):
        return ee.ImageCollection(self._collection_info['ee_collection']).\
            filterDate(start_date, end_date).\
            filterBounds(self._search_region).\
            map(self._check_validity).\
            filter(ee.Filter.gt('VALID_PORTION', self._valid_portion))

    def search(self, date, region, day_range=16, valid_portion=90):
        self._valid_portion = valid_portion
        EeRefImage.search(self, date, region, day_range=day_range)

    # TODO: consider making a generic version of this in the base class, perhaps with a self.key=... defaults
    def get_auto_image(self, key='VALID_PORTION', ascending=False):
        if (self._im_df is None) or (self._im_collection is None):
            raise Exception('First generate valid search results with search(...) method')
        return self._im_collection.sort(key, ascending).first()

    def get_composite_image(self):
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')
        return self._im_collection.mosaic().set('COMPOSITE_IMAGES', self._im_df.to_string())

class Sentinel2CloudlessEeImage(EeRefImage):
    def __init__(self, apply_valid_mask=True, collection='sentinel2_toa', valid_portion=60):
        EeRefImage.__init__(self, collection=collection)
        self.apply_valid_mask = apply_valid_mask
        self.scale = 10
        self._valid_portion = valid_portion
        self._display_properties = ['VALID_PORTION', 'QA_SCORE_AVG', 'GEOMETRIC_QUALITY_FLAG', 'RADIOMETRIC_QUALITY_FLAG', 'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B1', 'MEAN_INCIDENCE_ZENITH_ANGLE_B1']
        self._cloud_filter = 60         # Maximum image cloud cover percent allowed in image collection
        self._cloud_prob_thresh = 40    # Cloud probability (%); values greater than are considered cloud
        self._nir_drk_thresh = 0.15     # Near-infrared reflectance; values less than are considered potential cloud shadow
        self._cloud_proj_dist = 1       # Maximum distance (km) to search for cloud shadows from cloud edges
        self._buffer = 100               # Distance (m) to dilate the edge of cloud-identified objects

    def _check_validity(self, image):
        # adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
        # highest res of s2 image, that all bands of this image will be reprojected to if it is downloaded
        scale = image.select(1).projection().nominalScale()
        image = self._add_timedelta(image)

        cloud_prob = ee.Image(image.get('s2cloudless')).select('probability')
        q_score = cloud_prob.multiply(-1).add(100).rename('QA_SCORE')

        cloud_mask = cloud_prob.gt(self._cloud_prob_thresh).rename('CLOUD_MASK')
        # https://en.wikipedia.org/wiki/Solar_azimuth_angle perhaps it is the angle between south and shadow cast by vertical rod, clockwise +ve
        # shadow_mask = shadow_mask.multiply(dark_pixels).rename('shadow_mask')
        # TODO: dilate valid_mask by _buffer ?
        # TODO: do we need fill_mask with s2 ?
        # TODO: does below work in N hemisphere?
        shadow_azimuth = ee.Number(-90).add(ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

        # project the the cloud mask in the direction of shadows for self._cloud_proj_dist
        proj_cloud_mask = (cloud_mask.directionalDistanceTransform(shadow_azimuth, self._cloud_proj_dist * 1000/10)
                       # .reproject(**{'crs': image.select(0).projection(), 'scale': 100}) # necessary?
                       .select('distance')
                       .mask()  # applies cloud_mask?
                       .rename('PROJ_CLOUD_MASK'))

        if self.collection == 'sentinel2_sr':
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

        if self.apply_valid_mask:
            # NOTE: for export_image, updateMask sets pixels to 0, for download_image, it does the same and sets nodata=0
            image = image.updateMask(valid_mask)
        # else:
        #     image = image.unmask()
        # else:
        #     image = image.updateMask(fill_mask)

        return image.set(valid_portion).set(q_score_avg).addBands(q_score)

    def _get_im_collection(self, start_date, end_date):
        # adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
        # Import and filter S2 SR.
        s2_sr_toa_col = (ee.ImageCollection(self._collection_info['ee_collection'])
                     .filterBounds(self._search_region)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self._cloud_filter)))

        # Import and filter s2cloudless.
        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                            .filterBounds(self._search_region)
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

    def search(self, date, region, day_range=16, valid_portion=90):
        self._valid_portion = valid_portion
        EeRefImage.search(self, date, region, day_range=day_range)

    # TODO: consider making a generic version of this in the base class, perhaps with a self.key=... defaults
    def get_auto_image(self, key='VALID_PORTION', ascending=False):
        if (self._im_df is None) or (self._im_collection is None):
            raise Exception('First generate valid search results with search(...) method')
        return self._im_collection.sort(key, ascending).first()

    def get_composite_image(self):
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')
        # return self._im_collection.qualityMosaic('QA_SCORE').set('COMPOSITE_IMAGES', self._im_df.to_string())
        return self._im_collection.median().set('COMPOSITE_IMAGES', self._im_df.to_string())


##

