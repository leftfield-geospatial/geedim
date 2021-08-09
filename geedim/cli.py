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

from geedim import search as search_api
from geedim import download as download_api
from geedim import get_logger

logger = get_logger(__name__)


# map collection keys to classes
cls_col_map = {'landsat7_c2_l2': search_api.LandsatImSearch,
               'landsat8_c2_l2': search_api.LandsatImSearch,
               'sentinel2_toa': search_api.Sentinel2CloudlessImSearch,
               'sentinel2_sr': search_api.Sentinel2CloudlessImSearch,
               'modis_nbar': search_api.ModisNbarImSearch}

def _parse_region_bbox(region=None, bbox=None):
    """ create geojson dict from region or bbox """

    if (bbox is None) and (region is None):
        raise click.BadParameter('Either --region or --bbox must be passed', region)

    if region is not None:  # read region file/string
        if os.path.isfile(region):
            region = pathlib.Path(region)
            if 'json' in region.suffix:  # read region from gejson file
                with open(region) as f:
                    region_geojson = json.load(f)
            else:  # read region from raster file
                try:
                    region_geojson, _ = search_api.get_image_bounds(region, region_buf)
                except Exception as ex:
                    raise click.BadParameter(f'{region} is not a valid geojson or raster file. \n{ex}')
        else:
            region_geojson = json.loads(region)
    else:  # convert bbox to geojson
        xmin, ymin, xmax, ymax = bbox
        coordinates = [[xmax, ymax], [xmax, ymin], [xmin, ymin], [xmin, ymax], [xmax, ymax]]
        region_geojson = dict(type='Polygon', coordinates=[coordinates])

    return region_geojson

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
@click.option(
    "-b",
    "--bbox",
    type=click.FLOAT,
    nargs=4,
    help="Search region bounding box co-ordinates in WGS84 (xmin, ymin, xmax, ymax).",
    required=False
)
@click.option(
    "-r",
    "--region",
    type=click.STRING,
    help="Geojson or raster filename providing search region.  One of bbox or region must be passed.",
    required=False
)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, readable=False, resolve_path=True,
                    allow_dash=False),
    default=None,
    help="Write search results to this filename. File type inferred from extension: [.csv|.json]",
    required=False
)
@click.option(
    "-rb",
    "--region_buf",
    type=click.FLOAT,
    default=5,
    help="If --region is a raster file, extend the image bounds by region_buf %",
    required=False,
    show_default=True
)
def search(collection, start_date, end_date=None, bbox=None, region=None, output=None, region_buf=5):
    """ Search for Earth Engine images """

    if end_date is None:
        end_date = start_date

    ee.Initialize()

    imsearch = cls_col_map[collection](collection=collection)
    region_geojson = _parse_region_bbox(region=region, bbox=bbox)

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
@click.option(
    "-i",
    "--id",
    type=click.STRING,
    help="Earth engine ID(s) of image(s) to download.",
    required=True,
    multiple=True
)
@click.option(
    "-b",
    "--bbox",
    type=click.FLOAT,
    nargs=4,
    help="Search region bounding box co-ordinates in WGS84 (xmin, ymin, xmax, ymax).",
    required=False
)
@click.option(
    "-r",
    "--region",
    type=click.STRING,
    help="Geojson or raster filename providing search region.  One of bbox or region must be passed.",
    required=False
)
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
@click.option(
    "-c",
    "--crs",
    type=click.STRING,
    default=None,
    help="Reproject image(s) to this CRS, specified as WKT or EPSG string. \n[default: download in source CRS]",
    required=False
)
@click.option(
    "-s",
    "--scale",
    type=click.FLOAT,
    default=None,
    help="Resample image bands to this pixel resolution (m). \n[default: download in minimum source band resolution]",
    required=False
)
def download(id, bbox=None, region=None, download_dir=os.getcwd(), crs=None, scale=None):
    """ Download image with optional cloud and shadow mask """

    ee.Initialize()

    region_geojson = _parse_region_bbox(region=region, bbox=bbox)

    collection_info = search_api.load_collection_info()
    collection_df = pd.DataFrame.from_dict(collection_info, orient='index')

    for _id in id:
        ee_collection = '/'.join(_id.split('/')[:-1])
        if not (ee_collection in collection_df.ee_collection.values):
            logger.warning(f'Skipping {_id}, unknown collection')
            continue
        collection = collection_df.index[collection_df.ee_collection == ee_collection][0]
        filename = pathlib.Path(download_dir).joinpath(_id.replace('/','_') + '.tif')

        imsearch_obj = cls_col_map[collection](collection=collection)
        image = imsearch_obj.get_image(_id, region=region_geojson)

        download_api.download_image(image, filename, region=region_geojson, crs=None, scale=None)

cli.add_command(download)