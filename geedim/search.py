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
from datetime import datetime, timedelta

import click
import ee
import pandas
import pandas as pd
import rasterio as rio
from rasterio.warp import transform_geom

from geedim import export, root_path, collection


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
def search(collection, start_date, end_date, region, valid_portion=0, apply_mask=False):
    """
    Search for images based on date, region etc criteria

    Parameters
    collection : geedim.collection.ImCollection
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
    if (end_date is None):
        end_date = start_date + timedelta(days=1)
    if (end_date <= start_date):
        raise Exception('`end_date` must be at least a day later than `start_date`')

    click.echo(f'\nSearching for {collection.collection_info["ee_collection"]} images between '
               f'{start_date.strftime("%Y-%m-%d")} and {end_date.strftime("%Y-%m-%d")}...')

    # filter the image collection
    im_collection = (collection.get_ee_collection().
                     filterDate(start_date, end_date).
                     filterBounds(region).
                     map(lambda image : collection.set_image_valid_portion(image, region=region)).
                     filter(ee.Filter.gt('VALID_PORTION', valid_portion)))

    # convert and print search results
    return collection._get_collection_df(im_collection, do_print=True)
