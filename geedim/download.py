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
import pandas as pd
import rasterio as rio
from geedim import get_logger, root_path
from rasterio.warp import transform_geom
from shapely import geometry
import time

logger = get_logger(__name__)


def get_image_info(image):
    im_info_dict = image.getInfo()

    band_info_df = pd.DataFrame(im_info_dict['bands'])
    crs_transforms = np.array(band_info_df['crs_transform'].to_list())
    scales = np.abs(crs_transforms[:, 0]).astype(float)
    band_info_df['scale'] = scales

    return im_info_dict, band_info_df

# Adapted from from https://github.com/gee-community/gee_tools, MIT license
def get_min_projection(image):
    bands = image.bandNames()
    # init_dict = ee.Dictionary(dict(scale=ee.Number(1e99), crs=ee.String('')))
    init_proj = image.select(0).projection()

    def compare_scale(name, prev_proj):
        prev_proj = ee.Projection(prev_proj)
        prev_scale = prev_proj.nominalScale()

        curr_proj = image.select([name]).projection()
        curr_scale = ee.Number(curr_proj.nominalScale())

        min_proj = ee.Algorithms.If(curr_scale.lte(prev_scale), curr_proj, prev_proj)
        return ee.Projection(min_proj)

    return ee.Projection(bands.iterate(compare_scale, init_proj))
    # return ee.Number(bands.iterate(compare_scale, ee.Number(1e99)))

# TODO: minimise as far as possible getInfo() calls below
def export_image(image, description, folder=None, region=None, crs=None, scale=None, wait=True):
    # TODO: can we avoid getInfo here?
    im_info_dict, band_info_df = get_image_info(image)

    if np.all(band_info_df['crs'] == 'EPSG:4326') and np.all(band_info_df['scale'] == 1) and \
            (crs is None or scale is None):
            raise Exception(f'This appears to be a composite image in WGS84, specify a target scale and CRS')

    if crs is None:
        crs = band_info_df['crs'].iloc[band_info_df['scale'].argmin()]
    if region is None:
        region = image.geometry().bounds()
        logger.warning('Region not specified, setting to image bounds')
    if scale is None:
        scale = band_info_df['scale'].min()

    if (band_info_df['crs'].unique().size > 1) or (band_info_df['scale'].unique().size > 1):
        logger.warning(f'Image bands have different scales, reprojecting all to {crs} at {scale}m resolution')

    task = ee.batch.Export.image.toDrive(image=image,
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
            sys.stdout.write(
                f'\rExport image {str(status["metadata"]["state"]).lower()} [ {toggles[toggle_count % 4]} ]')
            sys.stdout.flush()
            toggle_count += 1
        sys.stdout.write(f'\rExport image {str(status["metadata"]["state"]).lower()}\n')
        if status['metadata']['state'] != 'SUCCEEDED':
            logger.error(f'Export failed \n{status}')
            raise Exception(f'Export failed \n{status}')

    return task

def download_image(image, filename, region=None, crs=None, scale=None):

    # TODO: this first section of code is the same as for export_image
    im_info_dict, band_info_df = get_image_info(image)

    if np.all(band_info_df['crs'] == 'EPSG:4326') and np.all(band_info_df['scale'] == 1) and \
            (crs is None or scale is None):
            raise Exception(f'This appears to be a composite image in WGS84, specify a target scale and CRS')

    if crs is None:
        crs = band_info_df['crs'].iloc[band_info_df['scale'].argmin()]
    if region is None:
        region = image.geometry().bounds()
        logger.warning('Region not specified, setting to granule bounds')
    if scale is None:
        scale = band_info_df['scale'].min()

    if (band_info_df['crs'].unique().size > 1) or (band_info_df['scale'].unique().size > 1):
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
        'bands': bands_dict,
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
                logger.error(
                    'There is a 10MB Earth Engine limit on image downloads, either decrease image size, or use export(...)')
                return
        raise ex

    meta = file_link.info()
    file_size = int(meta['Content-Length'])
    logger.info(f'Download size: {file_size / (1024 ** 2):.2f} MB')

    tif_filename = pathlib.Path(filename)
    tif_filename = tif_filename.joinpath(tif_filename.parent, tif_filename.stem + '.tif')  # force to zip file
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

    if ('properties' in im_info_dict) and ('system:footprint' in im_info_dict['properties']):
        im_info_dict['properties'].pop('system:footprint')

    with rio.open(tif_filename, 'r+') as im:
        if 'properties' in im_info_dict:
            im.update_tags(**im_info_dict['properties'])
        # im.profile['photometric'] = None    # fix warning
        if 'bands' in im_info_dict:
            for band_i, band_info in enumerate(im_info_dict['bands']):
                im.set_band_description(band_i + 1, band_info['id'])
                im.update_tags(band_i + 1, ID=band_info['id'])
                # TODO: use image id to get _band_df from json file
                # if band_info['id'] in self._band_df['id'].to_list():
                #     band_row = self._band_df.loc[self._band_df['id'] == band_info['id']].iloc[0]
                #     # band_row = band_row[['abbrev', 'name', 'bw_start', 'bw_end']]
                #     im.update_tags(band_i + 1, ABBREV=band_row['abbrev'])
                #     im.update_tags(band_i + 1, NAME=band_row['name'])
                #     im.update_tags(band_i + 1, BW_START=band_row['bw_start'])
                #     im.update_tags(band_i + 1, BW_END=band_row['bw_end'])
    return link


def download_im_collection(im_collection, path, region=None, crs=None, scale=None):

    num_images = im_collection.size().getInfo()
    ee_im_list = im_collection.toList(num_images)
    path = pathlib.Path(path)

    for i in range(num_images):
        image = ee.Image(ee_im_list.get(i))
        # im_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        id = ee.String(image.get('system:id')).getInfo().replace('/','_')
        filename = path.joinpath(f'{id}.tif')
        logger.info(f'Downloading {filename.stem}...')
        download_image(image, filename, region=region, crs=crs, scale=scale)

def export_im_collection(im_collection, path, folder=None, region=None, crs=None, scale=None):

    num_images = im_collection.size().getInfo()
    ee_im_list = im_collection.toList(num_images)
    path = pathlib.Path(path)

    task_list = []
    for i in range(num_images):
        image = ee.Image(ee_im_list.get(i))
        # im_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        id = ee.String(image.get('system:id')).getInfo().replace('/','_')
        filename = path.joinpath(f'{id}.tif')
        logger.info(f'Downloading {filename.stem}...')
        task = export_image(image, filename, folder=folder, region=region, crs=crs, scale=scale, wait=False)
        task_list.append(task)

    return task_list
