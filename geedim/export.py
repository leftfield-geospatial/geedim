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
# Functions to download and export GEE images

import json
import os
import pathlib
import sys
import time
from urllib import request
from urllib.error import HTTPError
import zipfile
import logging
import click

import ee
import pandas as pd
import rasterio as rio
from rasterio.enums import ColorInterp


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
    crs_transforms = band_info_df['crs_transform'].values
    scales = [abs(float(crs_transform[0])) for crs_transform in crs_transforms]
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


def export_image(image, filename, folder='', region=None, crs=None, scale=None, wait=True):
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
    region : dict, geojson, ee.Geometry, optional
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
    im_id = im_info_dict['id'] if 'id' in im_info_dict else ''
    click.echo(f'\nExporting {im_id} to Google Drive:{folder}/{filename}.tif')

    # if the image is in WGS84 and has no scale (probable composite), then exit
    if all(band_info_df['crs'] == 'EPSG:4326') and all(band_info_df['scale'] == 1) and \
            (crs is None or scale is None):
        raise Exception(f'Image appears to be a composite in WGS84, specify a destination scale and CRS')

    # if it is a native MODIS CRS then warn about GEE bug
    if any(band_info_df['crs'] == 'SR-ORG:6974') and (crs is None):
        raise Exception(f'There is an earth engine bug exporting in SR-ORG:6974: '
                        f'https://issuetracker.google.com/issues/194561313')

    if crs is None:
        crs = band_info_df['crs'].iloc[band_info_df['scale'].argmin()]  # CRS corresponding to minimum scale
    if region is None:
        region = image.geometry()  # not recommended
        click.secho('Warning: region not specified, setting to granule bounds', fg='red')
    if scale is None:
        scale = band_info_df['scale'].min()  # minimum scale

    # warn if some band scales will be changed
    if (band_info_df['crs'].unique().size > 1) or (band_info_df['scale'].unique().size > 1):
        click.echo(f'Re-projecting all bands to {crs} at {scale}m resolution')

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

    task.start()

    if wait:  # wait for completion
        monitor_export_task(task)

    return task


def monitor_export_task(task):
    """

    Parameters
    ----------
    task : ee task to monitor
    """
    status = ee.data.getOperation(task.name)
    toggles = r'-\|/'
    toggle_count = 0

    while ('done' not in status) or (not status['done']):
        time.sleep(.5)
        status = ee.data.getOperation(task.name)  # get task status
        # TODO: interpret totalWorkUnits and completeWorkUnits

        # display progress state and toggle
        sys.stdout.write(f'\rExport image {str(status["metadata"]["state"]).lower()} '
                         f'[ {toggles[toggle_count % 4]} ]')
        sys.stdout.flush()
        toggle_count += 1

    sys.stdout.write(f'\rExport image {str(status["metadata"]["state"]).lower()}\n')
    if status['metadata']['state'] != 'SUCCEEDED':
        raise Exception(f'{task.name} export failed \n{status}')


def download_image(image, filename, region=None, crs=None, scale=None, band_df=None, overwrite=False):
    """
    Download an image as a GeoTiff

    Parameters
    ----------
    image : ee.Image
            The image to export
    filename : str, pathlib.Path
               Name of the destination file
    region : dict, geojson, ee.Geometry, optional
             Region of interest (WGS84) to export (default: export the entire image granule if it has one).
    crs : str, optional
          WKT, EPSG etc specification of CRS to export to (default: use the image CRS if it has one).
    scale : float, optional
            Pixel resolution (m) to export to (default: use the highest resolution of the image bands).
    band_df : pandas.DataFrame, optional
              DataFrame specifying band metadata to be copied to downloaded file.  'id' column should contain band id's
              that match the ee.Image band id's
    overwrite : bool, optional
                Overwrite the destination file if it exists (default: prompt)
    """
    # get ee image info which is used in setting crs, scale and tif metadata (no further calls to getInfo)
    im_info_dict, band_info_df = get_image_info(image)
    im_id = im_info_dict['id'] if 'id' in im_info_dict else ''
    click.echo(f'\nDownloading {im_id} to {filename.name}')

    # if the image is in WGS84 and has no scale (probable composite), then exit
    if all(band_info_df['crs'] == 'EPSG:4326') and all(band_info_df['scale'] == 1) and \
            (crs is None or scale is None):
        raise Exception(f'Image appears to be a composite in WGS84, specify a destination scale and CRS')

    # if it is a native MODIS CRS then warn about GEE bug
    if any(band_info_df['crs'] == 'SR-ORG:6974') and (crs is None):
        raise Exception(f'There is an earth engine bug exporting in SR-ORG:6974: '
                        f'https://issuetracker.google.com/issues/194561313')

    if crs is None:
        crs = band_info_df['crs'].iloc[band_info_df['scale'].argmin()]  # CRS corresponding to minimum scale
    if region is None:
        region = image.geometry()  # not recommended
        click.secho('Warning: region not specified, setting to granule bounds', fg='red')
    if scale is None:
        scale = band_info_df['scale'].min()  # minimum scale

    # warn if some band scales will be changed
    if (band_info_df['crs'].unique().size > 1) or (band_info_df['scale'].unique().size > 1):
        click.echo(f'Re-projecting all bands to {crs} at {scale}m resolution')

    if isinstance(region, dict):
        region = ee.Geometry(region)

    # force all bands into same crs and scale
    band_info_df['crs'] = str(crs)
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

    file_link = None
    try:
        # setup the download
        file_link = request.urlopen(link)
        file_size = int(file_link.info()['Content-Length'])
        click.echo(f'Download size: {file_size / (1024 ** 2):.2f} MB')

        tif_filename = pathlib.Path(filename)
        tif_filename = tif_filename.parent.joinpath(tif_filename.stem + '.tif')  # force to tif file
        zip_filename = tif_filename.parent.joinpath('geedim_download.zip')

        # download the file
        with open(zip_filename, 'wb') as f:
            file_size_dl = 0
            with click.progressbar(length=file_size,
                                   label='Downloading') as bar:
                while file_size_dl <= file_size:
                    buffer = file_link.read(8192)
                    if not buffer:
                        break
                    file_size_dl += len(buffer)
                    f.write(buffer)
                    bar.update(file_size_dl)

    except HTTPError as ex:
        # check for size limit error
        response = json.loads(ex.read())
        if ('error' in response) and ('message' in response['error']):
            if response['error']['message'] == 'User memory limit exceeded.':
                click.secho('There is a 10MB Earth Engine limit on image downloads, '
                             'either decrease image size, or use export(...)', fg='red')
        raise ex
    finally:
        if file_link is not None:
            file_link.close()

    # extract tif from zip file
    _tif_filename = zip_filename.parent.joinpath(zipfile.ZipFile(zip_filename, "r").namelist()[0])
    with zipfile.ZipFile(zip_filename, "r") as zip_file:
        zip_file.extractall(zip_filename.parent)
    os.remove(zip_filename)     # clean up zip file

    # rename to extracted tif file to filename
    if (_tif_filename != tif_filename):
        if tif_filename.exists():
            if overwrite or click.confirm(f'{tif_filename.name} exists, do you want to overwrite?'):
                os.remove(tif_filename)
            else:
                return link
        os.rename(_tif_filename, tif_filename)


    # remove footprint property from im_info_dict before copying to tif file
    if ('properties' in im_info_dict) and ('system:footprint' in im_info_dict['properties']):
        im_info_dict['properties'].pop('system:footprint')

    # copy ee image metadata to downloaded tif file
    try:
        # suppress rasterio warnings relating to GEE tag (?) issue
        logging.getLogger("rasterio").setLevel(logging.ERROR)

        with rio.open(tif_filename, 'r+') as im:
            if 'properties' in im_info_dict:
                im.update_tags(**im_info_dict['properties'])

            im.colorinterp = [ColorInterp.undefined] * im.count

            if 'bands' in im_info_dict:
                for band_i, band_info in enumerate(im_info_dict['bands']):
                    im.set_band_description(band_i + 1, band_info['id'])
                    im.update_tags(band_i + 1, ID=band_info['id'])

                    if (band_df is not None) and (band_info['id'] in band_df['id'].values):
                        band_row = band_df.loc[band_df['id'] == band_info['id']].iloc[0].drop('id')
                        for key, val in band_row.iteritems():
                            im.update_tags(band_i + 1, **{str(key).upper(): val})
    finally:
        logging.getLogger("rasterio").setLevel(logging.WARNING)

    return link
