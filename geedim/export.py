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
import re
# from urllib import request
# from urllib.error import HTTPError
import requests
import zipfile
import logging
import click
from xml.etree import ElementTree as etree
from xml.dom import minidom

import ee
import rasterio as rio
from rasterio.enums import ColorInterp

from geedim import image, info

def write_pam_xml(obj, filename):
    if isinstance(obj, dict):
        gd_info = obj
    elif isinstance(obj, ee.Image):
        gd_info = image.get_info(obj)
    elif isinstance(obj, image.Image):
        gd_info = obj.info
    else:
        raise TypeError(f'Unsupported type: {obj.__class__}')

    if 'system:footprint' in gd_info['properties']:
        gd_info['properties'].pop('system:footprint')

    root = etree.Element('PAMDataset')
    prop_meta = etree.SubElement(root, 'Metadata')
    for key, val in gd_info['properties'].items():
        item = etree.SubElement(prop_meta, 'MDI', attrib=dict(key=key))
        item.text = str(val)

    for band_i, band_dict in enumerate(gd_info['bands']):
        band_elem = etree.SubElement(root, 'PAMRasterBand', attrib=dict(band=str(band_i+1)))
        if 'id' in band_dict:
            desc = etree.SubElement(band_elem, 'Description')
            desc.text = band_dict['id']
        band_meta = etree.SubElement(band_elem, 'Metadata')
        for key, val in band_dict.items():
            item = etree.SubElement(band_meta, 'MDI', attrib=dict(key=key.upper()))
            item.text = str(val)

    xml_str = minidom.parseString(etree.tostring(root)).childNodes[0].toprettyxml(indent="   ")
    with open(filename, 'w') as f:
        f.write(xml_str)



def band_df_to_pamxml(band_df, filename):
    root = etree.Element('PAMDataset')

    for i, row in band_df.iterrows():
        band = etree.SubElement(root, 'PAMRasterBand', attrib=dict(band=str(i+1)))
        if 'id' in row:
            desc = etree.SubElement(band, 'Description')
            desc.text = row.id
        metadata = etree.SubElement(band, 'Metadata')
        for col_name, col_val in row.iteritems():
            item = etree.SubElement(metadata, 'MDI', attrib=dict(key=col_name.upper()))
            item.text = str(col_val)

    xml_str = minidom.parseString(etree.tostring(root)).toprettyxml(indent="   ")
    with open(filename, 'w') as f:
        f.write(xml_str)


def _parse_export_args(ee_image, filename=None, region=None, crs=None, scale=None):
    """
    Download an image as a GeoTiff

    Parameters
    ----------
    ee_image : ee.Image, geedim.image.Image
               The image to export
    region : dict, geojson, ee.Geometry, optional
             Region of interest (WGS84) to export (default: export the entire image granule if it has one).
    crs : str, optional
          WKT, EPSG etc specification of CRS to export to (default: use the image CRS if it has one).
    scale : float, optional
            Pixel resolution (m) to export to (default: use the highest resolution of the image bands).
    """
    # get ee image info which is used in setting crs, scale and tif metadata (no further calls to getInfo)
    if isinstance(ee_image, image.Image):
        gd_info = ee_image.info
    else:
        gd_info = image.get_info(ee_image)

    # filename = pathlib.Path(filename)
    if gd_info['id'] is None:
        gd_info['id'] = pathlib.Path(filename).stem

    # if the image is in WGS84 and has no scale (probable composite), then exit
    if gd_info['crs'] is None or gd_info['scale'] is None:
        raise Exception(f'{gd_info["id"]} appears to be a composite in WGS84, specify a scale and CRS')

    # if it is a native MODIS CRS then warn about GEE bug
    if (gd_info['crs'] == 'SR-ORG:6974') and (crs is None):
        raise Exception(f'There is an earth engine bug exporting in SR-ORG:6974, specify another CRS: '
                        f'https://issuetracker.google.com/issues/194561313')

    if crs is None:
        crs = gd_info['crs']        # CRS corresponding to minimum scale
    if region is None:
        if 'system:footprint' in gd_info['properties']:
            region = gd_info['properties']['system:footprint']
            click.secho(f'{gd_info["id"]}: region not specified, setting to image footprint')
        else:
            raise AttributeError(f'{gd_info["id"]} does not have a footprint, specify a region to download')

    if scale is None:
        scale = gd_info['scale']     # minimum scale

    # warn if some band scales will be changed
    # if (band_info_df['crs'].unique().size > 1) or (band_info_df['scale'].unique().size > 1):
    #     click.echo(f'{im_id}: re-projecting all bands to {crs} at {scale:.1f}m')

    if isinstance(region, dict):
        region = ee.Geometry(region)

    return region, crs, scale, gd_info


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
          WKT, EPSG etc specification of CRS to export to.  Where image bands have different CRSs, all are re-projected
          to this CRS.
          (default: use the CRS of the minimum scale band if available).
    scale : float, optional
            Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this scale.
            (default: use the minimum scale of image bands if available).
    wait : bool
           Wait for the export to complete before returning (default: True)

    Returns
    -------
    task : EE task object
    """
    region, crs, scale, _ = _parse_export_args(ee_image, filename=filename, region=region, crs=crs, scale=scale)


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
        click.echo(f'\rPreparing {label} {toggles[toggle_count % 4]}', nl='')
        toggle_count += 1
    click.echo(f'\rPreparing {label}  done')

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
          WKT, EPSG etc specification of CRS to export to.  Where image bands have different CRSs, all are re-projected
          to this CRS.
          (default: use the CRS of the minimum scale band if available).
    scale : float, optional
            Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this scale.
            (default: use the minimum scale of image bands if available).
    band_df : pandas.DataFrame, optional
              DataFrame specifying band metadata to be copied to downloaded file.  'id' column should contain band id's
              that match the ee.Image band id's
    overwrite : bool, optional
                Overwrite the destination file if it exists, otherwise prompt the user (default: True)
    """
    filename = pathlib.Path(filename)
    region, crs, scale, gd_info = _parse_export_args(ee_image, filename=filename, region=region, crs=crs, scale=scale)


    # get download link
    try:
        link = ee_image.getDownloadURL({
            'scale': scale,
            'crs': crs,
            'fileFormat': 'GeoTIFF',
            # 'bands': bands_dict,
            'filePerBand': False,
            'region': region})
    except ee.ee_exception.EEException as ex:
        if re.match(r'Total request size \(.*\) must be less than or equal to .*', str(ex)):
            raise Exception(f'The requested image is too large, reduce its size, or use `export`\n({str(ex)})')
        else:
            raise ex

    # download zip file
    tif_filename = filename.parent.joinpath(filename.stem + '.tif')  # force to tif file
    zip_filename = tif_filename.parent.joinpath('geedim_download.zip')
    with requests.get(link, stream=True) as r:
        r.raise_for_status()
        with open(zip_filename, 'wb') as f:
            csize = 8192
            with click.progressbar(r.iter_content(chunk_size=csize),
                                   label=f'{filename.stem[:80]}:',
                                   length=int(r.headers['Content-length'])/csize,
                                   show_pos=True) as bar:
                bar.format_pos = lambda: f'{bar.pos * csize / (1024**2):.1f}/{bar.length * csize / (1024**2):.1f} MB'
                for chunk in bar:
                    f.write(chunk)

    # extract tif from zip file
    zip_tif_filename = zip_filename.parent.joinpath(zipfile.ZipFile(zip_filename, 'r').namelist()[0])
    with zipfile.ZipFile(zip_filename, 'r') as zip_file:
        zip_file.extractall(zip_filename.parent)
    os.remove(zip_filename)     # clean up zip file

    # rename to extracted tif file to tif_filename
    if (zip_tif_filename != tif_filename):
        while tif_filename.exists():
            if overwrite or click.confirm(f'{tif_filename.name} exists, do you want to overwrite?', default='n'):
                os.remove(tif_filename)
            else:
                tif_filename = click.prompt('Please enter another filename', type=str, default=None)
                tif_filename = pathlib.Path(tif_filename)
        os.rename(zip_tif_filename, tif_filename)

    # write image metadata to pam xml file
    write_pam_xml(gd_info, str(tif_filename) + '.aux.xml')

    return link
