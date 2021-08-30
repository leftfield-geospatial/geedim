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
import time
from urllib import request
from urllib.error import HTTPError
import zipfile
import logging
import click

import ee
import rasterio as rio
from rasterio.enums import ColorInterp
from geedim import image


def _parse_export_args(ee_image, filename=None, region=None, crs=None, scale=None):
    """
    Download an image as a GeoTiff

    Parameters
    ----------
    ee_image : ee.Image
            The image to export
    region : dict, geojson, ee.Geometry, optional
             Region of interest (WGS84) to export (default: export the entire image granule if it has one).
    crs : str, optional
          WKT, EPSG etc specification of CRS to export to (default: use the image CRS if it has one).
    scale : float, optional
            Pixel resolution (m) to export to (default: use the highest resolution of the image bands).
    """
    # get ee image info which is used in setting crs, scale and tif metadata (no further calls to getInfo)
    im_info_dict, band_info_df = image.get_image_info(ee_image)

    # filename = pathlib.Path(filename)
    if filename is None:
        im_id = im_info_dict['id'] if 'id' in im_info_dict else 'Image'
    else:
        im_id = pathlib.Path(filename).stem

    # if the image is in WGS84 and has no scale (probable composite), then exit
    if crs is None or scale is None:
        _band_info_df = band_info_df[(band_info_df.crs != 'EPSG:4326') & (band_info_df.scale != 1)]
        if _band_info_df.shape[0]==0:
            raise Exception(f'{im_id} appears to be a composite in WGS84, specify a destination scale and CRS')

        # get minimum scale and corresponding crs, excluding WGS84 bands
        min_scale_idx = _band_info_df.scale.idxmin()
        min_crs, min_scale = band_info_df.loc[min_scale_idx, ['crs', 'scale']]

    # if it is a native MODIS CRS then warn about GEE bug
    if any(band_info_df['crs'] == 'SR-ORG:6974') and (crs is None):
        raise Exception(f'There is an earth engine bug exporting in SR-ORG:6974, specify another CRS: '
                        f'https://issuetracker.google.com/issues/194561313')

    if crs is None:
        crs = min_crs        # CRS corresponding to minimum scale
    if region is None:
        if 'system:footprint' in im_info_dict:
            region = im_info_dict['system:footprint']
            click.secho(f'{im_id}: region not specified, setting to image footprint')
        else:
            raise AttributeError(f'{im_id} does not have a footprint, specify a region to download')

    if scale is None:
        scale = min_scale     # minimum scale

    # warn if some band scales will be changed
    if (band_info_df['crs'].unique().size > 1) or (band_info_df['scale'].unique().size > 1):
        click.echo(f'{im_id}: re-projecting all bands to {crs} at {scale:.1f}m')

    if isinstance(region, dict):
        region = ee.Geometry(region)

    return region, crs, scale, im_info_dict


def export_image(ee_image, filename, folder='', region=None, crs=None, scale=None, wait=True):
    """
    Export an image to a GeoTiff in Google Drive

    Parameters
    ----------
    ee_image : ee.Image
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
    # im_id = im_info_dict['id'] + ' ' if 'id' in im_info_dict else ''
    # click.echo(f'Exporting to Google Drive:{folder}/{filename}.tif')
    region, crs, scale, im_info_dict = _parse_export_args(ee_image, filename=filename, region=region, crs=crs, scale=scale)


    # create export task and start
    task = ee.batch.Export.image.toDrive(image=ee_image,
                                         region=region,
                                         description=filename[:100],
                                         folder=folder,
                                         fileNamePrefix=filename,
                                         scale=scale,
                                         crs=crs,
                                         maxPixels=1e9)

    task.start()

    if wait:  # wait for completion
        # click.echo(f'Waiting for Google Drive:{folder}/{filename}.tif ...')
        monitor_export_task(task)

    return task


def monitor_export_task(task, label=None):
    """

    Parameters
    ----------
    task : ee task to monitor
    """
    toggles = r'-\|/'
    toggle_count = 0
    pause = 0.5
    bar_len = 100
    status = ee.data.getOperation(task.name)

    if label is None:
        label = f'{status["metadata"]["description"][:80]}:'

    while (not 'progress' in status['metadata']):
        time.sleep(pause)
        status = ee.data.getOperation(task.name)  # get task status
        click.echo(f'\rPreparing {label}: {toggles[toggle_count % 4]}', nl='')
        toggle_count += 1
    click.echo(f'\rPreparing {label}:  done')

    with click.progressbar(length=bar_len, label=f'Exporting {label}:') as bar:
        while ('done' not in status) or (not status['done']):
            time.sleep(pause)
            status = ee.data.getOperation(task.name)  # get task status
            progress = status['metadata']['progress']*bar_len
            bar.update(progress - bar.pos)
        bar.update(bar_len - bar.pos)

    if status['metadata']['state'] != 'SUCCEEDED':
        raise Exception(f'Export failed \n{status}')

def download_image(ee_image, filename, region=None, crs=None, scale=None, band_df=None, overwrite=False):
    """
    Download an image as a GeoTiff

    Parameters
    ----------
    ee_image : ee.Image
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
    filename = pathlib.Path(filename)
    # im_id = im_info_dict['id'] + ' ' if 'id' in im_info_dict else ''
    # click.echo(f'Downloading to {filename.name}')
    region, crs, scale, im_info_dict = _parse_export_args(ee_image, filename=filename, region=region, crs=crs, scale=scale)


    # get download link
    link = ee_image.getDownloadURL({
        'scale': scale,
        'crs': crs,
        'fileFormat': 'GeoTIFF',
        # 'bands': bands_dict,
        'filePerBand': False,
        'region': region})  # TODO: file size error

    file_link = None
    try:
        # setup the download
        file_link = request.urlopen(link)
        file_size = int(file_link.info()['Content-Length'])
        # click.echo(f'Download size: {file_size / (1024 ** 2):.2f} MB')

        tif_filename = filename
        tif_filename = tif_filename.parent.joinpath(tif_filename.stem + '.tif')  # force to tif file
        zip_filename = tif_filename.parent.joinpath('geedim_download.zip')

        # download the file
        with open(zip_filename, 'wb') as f:
            file_size_dl = 0
            with click.progressbar(length=file_size, label=f'{filename.stem[:80]}:', show_pos=True) as bar:
                bar.format_pos = lambda : f'{bar.pos/(1024**2):.1f}/{bar.length/(1024**2):.1f} MB'
                while file_size_dl <= file_size:
                    buffer = file_link.read(8192)
                    if not buffer:
                        break
                    file_size_dl += len(buffer)
                    f.write(buffer)
                    bar.update(len(buffer))

    except HTTPError as ex:
        # check for size limit error
        response = json.loads(ex.read())
        # TODO: catch this in CLI and print a message
        if ('error' in response) and ('message' in response['error']):
            if response['error']['message'] == 'User memory limit exceeded.':
                click.echo('There is a 10MB Earth Engine limit on image downloads, '
                             'either decrease image size, or use export(...)', err=True)
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
                click.secho(f'Warning: {tif_filename} exists, exiting', fg='red')   # TODO: get another filename
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
            if 'id' in im_info_dict:
                im.update_tags(id=im_info_dict['id'])

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
