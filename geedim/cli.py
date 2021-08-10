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
import json
import os
import pathlib

import click
import ee
import pandas as pd

from geedim import download as download_api
from geedim import get_logger
from geedim import search as search_api

logger = get_logger(__name__)


# map collection keys to classes
cls_col_map = {'landsat7_c2_l2': search_api.LandsatImSearch,
               'landsat8_c2_l2': search_api.LandsatImSearch,
               'sentinel2_toa': search_api.Sentinel2CloudlessImSearch,
               'sentinel2_sr': search_api.Sentinel2CloudlessImSearch,
               'modis_nbar': search_api.ModisNbarImSearch}

def _parse_region_bbox(region=None, bbox=None, region_buf=5):
    """ create geojson dict from region or bbox """

    if (bbox is None) and (region is None):
        raise click.BadParameter('Either --region or --bbox must be passed', region)

    if region is not None:  # read region file/string
        region = pathlib.Path(region)
        if 'json' in region.suffix:  # read region from gejson file
            with open(region) as f:
                region_geojson = json.load(f)
        else:  # read region from raster file
            try:
                region_geojson, _ = search_api.get_image_bounds(region, region_buf)
            except Exception as ex:
                raise click.BadParameter(f'{region} is not a valid geojson or raster file. \n{ex}')
    else:  # convert bbox to geojson
        xmin, ymin, xmax, ymax = bbox
        coordinates = [[xmax, ymax], [xmax, ymin], [xmin, ymin], [xmin, ymax], [xmax, ymax]]
        region_geojson = dict(type='Polygon', coordinates=[coordinates])

    return region_geojson

def _export_download(id, bbox=None, region=None, path=None, crs=None, scale=None, wait=True, download=True):
    """ download or export image(s), with cloud and shadow masking """

    ee.Initialize()

    region_geojson = _parse_region_bbox(region=region, bbox=bbox)

    collection_info = search_api.load_collection_info()
    collection_df = pd.DataFrame.from_dict(collection_info, orient='index')

    for _id in id:
        ee_collection = '/'.join(_id.split('/')[:-1])
        if not (ee_collection in collection_df.ee_collection.values):
            logger.warning(f'Skipping {_id}: Unknown collection')
            continue

        collection = collection_df.index[collection_df.ee_collection == ee_collection][0]

        if collection == 'modis_nbar' and crs is None:  # workaround MODIS native CRS export issue
            crs = 'EPSG:3857'
            logger.warning(f'Re-projecting {_id} to {crs} to avoid GEE MODIS CRS bug: https://issuetracker.google.com/issues/194561313.')

        imsearch_obj = cls_col_map[collection](collection=collection)
        image = imsearch_obj.get_image(_id, region=region_geojson)

        if download:
            filename = pathlib.Path(path).joinpath(_id.replace('/', '_') + '.tif')
            download_api.download_image(image, filename, region=region_geojson, crs=crs, scale=scale)
        else:
            filename = _id.replace('/', '_')
            download_api.export_image(image, filename, folder=path, region=region_geojson, crs=crs, scale=scale,
                                  wait=wait)

# define options common to >1 command
bbox_option = click.option(
    "-b",
    "--bbox",
    type=click.FLOAT,
    nargs=4,
    help="Region defined by bounding box co-ordinates in WGS84 (xmin, ymin, xmax, ymax).  "
         "[One of --bbox or --region is required.]",
    required=False
)
region_option = click.option(
    "-r",
    "--region",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False, readable=True, resolve_path=True,
                    allow_dash=False),
    help="Region defined by geojson or raster file.  [One of --bbox or --region is required.]",
    required=False
)
region_buf_option = click.option(
    "-rb",
    "--region_buf",
    type=click.FLOAT,
    default=5,
    help="If --region is a raster file, extend the region bounds by region_buf %",
    required=False,
    show_default=True
)
image_id_option = click.option(
    "-i",
    "--id",
    type=click.STRING,
    help="Earth engine image ID(s).",
    required=True,
    multiple=True
)
crs_option = click.option(
    "-c",
    "--crs",
    type=click.STRING,
    default=None,
    help="Reproject image(s) to this CRS, specified as WKT or EPSG string. \n[default: source CRS]",
    required=False
)
scale_option = click.option(
    "-s",
    "--scale",
    type=click.FLOAT,
    default=None,
    help="Resample image bands to this pixel resolution (m). \n[default: minimum of the source band resolutions]",
    required=False
)

# define the click cli
@click.group()
def cli():
    pass


@click.command()
@click.option(
    "-c",
    "--collection",
    type=click.Choice(list(cls_col_map.keys()), case_sensitive=False),
    help="Earth Engine image collection to search.",
    default="landsat8_c2_l2",
    required=True
)
@click.option(
    "-s",
    "--start-date",
    type=click.DateTime(),
    help="Start date (UTC).",
    required=True
)
@click.option(
    "-e",
    "--end-date",
    type=click.DateTime(),
    help="End date (UTC).  \n[default: start_date]",
    required=False,
)
@bbox_option
@region_option
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, readable=False, resolve_path=True,
                    allow_dash=False),
    default=None,
    help="Write search results to this filename. File type inferred from extension: [.csv|.json]",
    required=False
)
@region_buf_option
def search(collection, start_date, end_date=None, bbox=None, region=None, output=None, region_buf=5):
    """ Search for images """

    ee.Initialize()

    if end_date is None:
        end_date = start_date


    imsearch = cls_col_map[collection](collection=collection)
    region_geojson = _parse_region_bbox(region=region, bbox=bbox, region_buf=region_buf)

    im_df = imsearch.search(start_date, end_date, region_geojson)

    if (output is not None) and (im_df is not None):
        if 'IMAGE' in im_df.columns:
            im_df = im_df.drop(columns='IMAGE')
        output = pathlib.Path(output)
        if output.suffix == '.csv':
            im_df.to_csv(output)
        elif output.suffix == '.json':
            im_df.to_json(output)
        else:
            raise ValueError(f'Unknown output file extension: {output.suffix}')
    return 0


cli.add_command(search)

@click.command()
@image_id_option
@bbox_option
@region_option
@click.option(
    "-dd",
    "--download-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True, readable=False, resolve_path=True,
                    allow_dash=False),
    default=os.getcwd(),
    help="Download image file(s) to this directory",
    required=False,
    show_default=True
)
@crs_option
@scale_option
def download(id, bbox=None, region=None, download_dir=os.getcwd(), crs=None, scale=None):
    """ Download image(s), with cloud and shadow masking """
    _export_download(id, bbox=bbox, region=region, path=download_dir, crs=crs, scale=scale, download=True)

cli.add_command(download)

@click.command()
@image_id_option
@bbox_option
@region_option
@click.option(
    "-df",
    "--drive-folder",
    type=click.STRING,
    default=None,
    help="Export image(s) to this Google Drive folder. [default: root]",
    required=False,
    show_default=True
)
@crs_option
@scale_option
@click.option(
    "-w/-nw",
    "--wait/--no-wait",
    default=True,
    help="Wait / don't wait for export to complete.  [default: wait]",
    required=False,
)
def export(id, bbox=None, region=None, drive_folder='', crs=None, scale=None, wait=True):
    """ Export image(s) to Google Drive, with cloud and shadow masking """
    _export_download(id, bbox=bbox, region=region, path=drive_folder, crs=crs, scale=scale, wait=wait, download=False)

cli.add_command(export)