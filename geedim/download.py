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

## Functions to download and export GEE images

import json
import os
import pathlib
import sys
import time
import urllib
import zipfile

import ee
import numpy as np
import pandas as pd
import rasterio as rio

from geedim import get_logger, root_path

logger = get_logger(__name__)

def load_collection_info():
    """
    Loads the satellite band etc information from json file into a dict
    """
    with open(root_path.joinpath('data/inputs/satellite_info.json')) as f:
        satellite_info = json.load(f)
    return satellite_info


def get_image_info(image):
    """
    Retrieve image info, and create a pandas DataFrame of band properties

    Parameters
    ----------
    image : ee.Image

    Returns
    -------
    im_info_dict : dict
                   Image properties
    band_info_df : pandas.DataFrame
                   Band properties including scale
    """
    im_info_dict = image.getInfo()

    band_info_df = pd.DataFrame(im_info_dict['bands'])
    crs_transforms = np.array(band_info_df['crs_transform'].to_list())
    scales = np.abs(crs_transforms[:, 0]).astype(float)
    band_info_df['scale'] = scales

    return im_info_dict, band_info_df

def get_min_projection(image):
    """
    Server side operations to find the minimum scale projection from image bands.  No calls to getInfo().

    Parameters
    ----------
    image : ee.Image

    Returns
    -------
    : ee.Projection
      The projection with the smallest scale
    """

    # Adapted from from https://github.com/gee-community/gee_tools, MIT license
    bands = image.bandNames()
    init_proj = image.select(0).projection()

    def compare_scale(name, prev_proj):
        prev_proj = ee.Projection(prev_proj)
        prev_scale = prev_proj.nominalScale()

        curr_proj = image.select([name]).projection()
        curr_scale = ee.Number(curr_proj.nominalScale())

        min_proj = ee.Algorithms.If(curr_scale.lte(prev_scale), curr_proj, prev_proj)
        return ee.Projection(min_proj)

    return ee.Projection(bands.iterate(compare_scale, init_proj))

def export_image(image, filename, folder=None, region=None, crs=None, scale=None, wait=True):
    """
    Export an image to a GeoTiff in Google Drive

    Parameters
    ----------
    image : ee.Image
            The image to export
    filename : str
               The name of the task and destination file
    folder : str, optional
             Google Drive folder to export to (default: root).
    region : geojson, ee.Geometry, optional
             Region of interest (WGS84) to export (default: export the entire image granule if it has one).
    crs : str, optional
          WKT, EPSG etc specification of CRS to export to (default: use the image CRS if it has one).
    scale : float, optional
            Pixel resolution (m) to export to (default: use the highest resolution of the image bands).
    wait : bool
           Wait for the export to complete before returning (default: True)

    Returns
    -------
    task : EE task object
    """
    # TODO: minimise as far as possible getInfo() calls below
    im_info_dict, band_info_df = get_image_info(image)

    # if the image is in WGS84 and has no scale (probable composite), then exit
    if np.all(band_info_df['crs'] == 'EPSG:4326') and np.all(band_info_df['scale'] == 1) and \
            (crs is None or scale is None):
            raise Exception(f'This appears to be a composite image in WGS84, specify a destination scale and CRS')

    if crs is None:
        crs = band_info_df['crs'].iloc[band_info_df['scale'].argmin()]  # CRS corresponding to minimum scale
    if region is None:
        region = image.geometry()   # not recommended
        logger.warning('Region not specified, setting to image bounds')
    if scale is None:
        scale = band_info_df['scale'].min()     # minimum scale

    # warn if some band scales will be changed
    if (band_info_df['crs'].unique().size > 1) or (band_info_df['scale'].unique().size > 1):
        logger.warning(f'Image bands have different scales, reprojecting all to {crs} at {scale}m resolution')

    if isinstance(region, dict):
        region = ee.Geometry(region)

    # create export task and start
    task = ee.batch.Export.image.toDrive(image=image,
                                         region=region,
                                         description=filename,
                                         folder=folder,
                                         fileNamePrefix=filename,
                                         scale=scale,
                                         crs=crs,
                                         maxPixels=1e9)

    logger.info(f'Starting export task {filename}...')
    task.start()

    if wait:    # wait for completion
        status = ee.data.getOperation(task.name)
        toggles = '-\|/'
        toggle_count = 0

        while ('done' not in status) or (not status['done']):
            time.sleep(.5)
            status = ee.data.getOperation(task.name)    # get task status
            # TODO: interpret totalWorkUnits and completeWorkUnits

            # display progress state and toggle
            sys.stdout.write(f'\rExport image {str(status["metadata"]["state"]).lower()} '
                             f'[ {toggles[toggle_count % 4]} ]')
            sys.stdout.flush()
            toggle_count += 1

        sys.stdout.write(f'\rExport image {str(status["metadata"]["state"]).lower()}\n')
        if status['metadata']['state'] != 'SUCCEEDED':
            logger.error(f'Export failed \n{status}')
            raise Exception(f'Export failed \n{status}')

    return task

def download_image(image, filename, region=None, crs=None, scale=None, band_df=None):
    """
    Download an image as a GeoTiff

    Parameters
    ----------
    image : ee.Image
            The image to export
    filename : str, pathlib.Path
               Name of the destination file
    region : geojson, ee.Geometry, optional
             Region of interest (WGS84) to export (default: export the entire image granule if it has one).
    crs : str, optional
          WKT, EPSG etc specification of CRS to export to (default: use the image CRS if it has one).
    scale : float, optional
            Pixel resolution (m) to export to (default: use the highest resolution of the image bands).
    band_df : pandas.DataFrame, optional
              DataFrame specifying band metadata to be copied to downloaded file.  'id' column should contain band id's
              that match the ee.Image band id's
    """

    # TODO: this first section of code is the same as for export_image
    im_info_dict, band_info_df = get_image_info(image)

    # if the image is in WGS84 and has no scale (probable composite), then exit
    if np.all(band_info_df['crs'] == 'EPSG:4326') and np.all(band_info_df['scale'] == 1) and\
            (crs is None or scale is None):
            raise Exception(f'This appears to be a composite image in WGS84, specify a destination scale and CRS')

    if crs is None:
        crs = band_info_df['crs'].iloc[band_info_df['scale'].argmin()]
    if region is None:
        region = image.geometry()
        logger.warning('Region not specified, setting to granule bounds')
    if scale is None:
        scale = band_info_df['scale'].min()

    # warn if some band scales will be changed
    if (band_info_df['crs'].unique().size > 1) or (band_info_df['scale'].unique().size > 1):
        logger.warning(f'Image bands have different scales, reprojecting all to {crs} at {scale}m resolution')

    # force all bands into same crs and scale
    band_info_df['crs'] = crs
    band_info_df['scale'] = scale
    bands_dict = band_info_df[['id', 'crs', 'scale', 'data_type']].to_dict('records')

    # get download link
    link = image.getDownloadURL({
        'scale': scale,
        'crs': crs,
        'fileFormat': 'GeoTIFF',
        'bands': bands_dict,
        'filePerBand': False,
        'region': region})

    logger.info(f'Opening link: {link}')

    # open the link
    try:
        file_link = urllib.request.urlopen(link)
    except urllib.error.HTTPError as ex:
        logger.error(f'Could not open URL: HHTP error {ex.code} - {ex.reason}')

        # check for size limit error
        response = json.loads(ex.read())
        if ('error' in response) and ('message' in response['error']):
            msg = response['error']['message']
            logger.error(msg)
            if (msg == 'User memory limit exceeded.'):
                logger.error('There is a 10MB Earth Engine limit on image downloads, either decrease image size, or use export(...)')
        raise ex

    # setup destination zip and tif filenames
    meta = file_link.info()
    file_size = int(meta['Content-Length'])
    logger.info(f'Download size: {file_size / (1024 ** 2):.2f} MB')

    tif_filename = pathlib.Path(filename)
    tif_filename = tif_filename.joinpath(tif_filename.parent, tif_filename.stem + '.tif')  # force to tif file
    zip_filename = tif_filename.parent.joinpath('gee_image_download.zip')

    if zip_filename.exists():
        logger.warning(f'{zip_filename} exists, overwriting...')

    # download the zip file
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

            # display progress bar
            progress = (file_size_dl / file_size)
            sys.stdout.write('\r')
            sys.stdout.write('[%-50s] %d%%' % ('=' * int(50 * progress), 100 * progress))
            sys.stdout.flush()
        sys.stdout.write('\n')

    # extract tif from zip file
    logger.info(f'Extracting {zip_filename}')
    with zipfile.ZipFile(zip_filename, "r") as zip_file:
        zip_file.extractall(zip_filename.parent)

    if tif_filename.exists():
        logger.warning(f'{tif_filename} exists, overwriting...')
        os.remove(tif_filename)

    _tif_filename = zipfile.ZipFile(zip_filename, "r").namelist()[0]
    os.rename(zip_filename.parent.joinpath(_tif_filename), tif_filename)

    # remove footprint property from im_info_dict before copying to tif file
    if ('properties' in im_info_dict) and ('system:footprint' in im_info_dict['properties']):
        im_info_dict['properties'].pop('system:footprint')

    # copy metadata to downloaded tif file
    with rio.open(tif_filename, 'r+') as im:
        if 'properties' in im_info_dict:
            im.update_tags(**im_info_dict['properties'])
        # im.profile['photometric'] = None    # TODO: fix warning

        if 'bands' in im_info_dict:
            for band_i, band_info in enumerate(im_info_dict['bands']):
                im.set_band_description(band_i + 1, band_info['id'])
                im.update_tags(band_i + 1, ID=band_info['id'])

                if (band_df is not None) and (band_info['id'] in band_df['id'].values):
                    band_row = band_df.loc[band_df['id'] == band_info['id']].iloc[0].drop('id')
                    for key, val in band_row.iteritems():
                        im.update_tags(band_i + 1, **{str(key).upper(): val})
    return link


def download_im_collection(im_collection, path, region=None, crs=None, scale=None, band_df=None):
    """
    Download each image in a collection

    Parameters
    ----------
    im_collection : ee.ImageCollection
                    The image collection to download
    path : str, pathlib.Path
           Directory to download image files to.  Image filenames will be derived from their earth engine IDs.
    region : geojson, ee.Geometry, optional
             Region of interest (WGS84) to export (default: export the entire image granule if it has one).
    crs : str, optional
          WKT, EPSG etc specification of CRS to export to (default: use the image CRS if it has one).
    scale : float, optional
            Pixel resolution (m) to export to (default: use the highest resolution of the image bands).
    band_df : pandas.DataFrame, optional
              DataFrame specifying band metadata to be copied to downloaded file.  'id' column should contain band id's
              that match the ee.Image band id's
    """

    num_images = im_collection.size().getInfo()
    ee_im_list = im_collection.toList(im_collection.size())
    path = pathlib.Path(path)

    for i in range(num_images):
        image = ee.Image(ee_im_list.get(i))
        id = ee.String(image.get('system:id')).getInfo().replace('/','_')   # TODO: can we remove getInfo() here?
        filename = path.joinpath(f'{id}.tif')
        logger.info(f'Downloading {filename.stem}...')
        download_image(image, filename, region=region, crs=crs, scale=scale, band_df=band_df)

def export_im_collection(im_collection, folder=None, region=None, crs=None, scale=None):
    """
    Export each image in a collection to Google Drive

    Parameters
    ----------
    im_collection : ee.ImageCollection
                    The image collection to download
    folder : str, optional
             Google Drive folder to export to (default: root).
    region : geojson, ee.Geometry, optional
             Region of interest (WGS84) to export (default: export the entire image granule if it has one).
    crs : str, optional
          WKT, EPSG etc specification of CRS to export to (default: use the image CRS if it has one).
    scale : float, optional
            Pixel resolution (m) to export to (default: use the highest resolution of the image bands).
    """

    num_images = im_collection.size().getInfo()
    ee_im_list = im_collection.toList(num_images)

    task_list = []
    for i in range(num_images):
        image = ee.Image(ee_im_list.get(i))
        id = ee.String(image.get('system:id')).getInfo().replace('/','_')
        filename = pathlib.Path(f'{id}.tif')
        logger.info(f'Exporting {filename}...')
        task = export_image(image, filename, folder=folder, region=region, crs=crs, scale=scale, wait=False)
        task_list.append(task)

    return task_list
