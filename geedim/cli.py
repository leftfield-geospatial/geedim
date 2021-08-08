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

import click
import pathlib
import pandas as pd

from geedim import search
import ee

# map collection keys to classes
# cls_col_map = {'landsat7_c2_l2': lambda: search.LandsatImSearch(collection='landsat7_c2_l2'),
#                  'landsat8_c2_l2': lambda: search.LandsatImSearch(collection='landsat8_c2_l2'),
#                  'sentinel2_toa': lambda: search.Sentinel2CloudlessImSearch(collection='sentinel2_toa'),
#                  'sentinel2_sr': lambda: search.Sentinel2CloudlessImSearch(collection='sentinel2_sr'),
#                  'modis_nbar': lambda: search.LandsatImSearch(collection='modis_nbar')}

cls_col_map = {'landsat7_c2_l2': search.LandsatImSearch,
               'landsat8_c2_l2': search.LandsatImSearch,
               'sentinel2_toa': search.Sentinel2CloudlessImSearch,
               'sentinel2_sr': search.Sentinel2CloudlessImSearch,
               'modis_nbar': search.ModisNbarImSearch}

# pd.set_option("display.precision", 2)
# pd.set_option("display.max_colwidth", 50)

@click.group()
def cli():
    pass

@click.command()
@click.option(
    "-c",
    "--collection",
    type=click.Choice(list(cls_col_map.keys()), case_sensitive=False),
    help="Image collection.",
    default="landsat8_c2_l2",
    required=True
)
@click.option(
    "-b",
    "--bbox",
    type=click.FLOAT,
    nargs=4,
    help="Bounding box in WGS84 (xmin, ymin, xmax, ymax).",
    required=False
)
@click.option(
    "-r",
    "--region",
    type=click.STRING,
    help="File specifying region.",
    required=False
)
@click.option("-s", "--start_date", type=click.DateTime(), help="Start date.", required=True)
@click.option("-e", "--end_date", type=click.DateTime(), help="End date.", required=True)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, readable=False, resolve_path=True,
                    allow_dash=False),
    default=None,
    help="Output results to this filename. Type inferred from extension [.csv|.json]",
    required=False
)
def search(collection, bbox, region, start_date, end_date, output):
    """ Search for Earth Engine images """
    ee.Initialize()

    imsearch = cls_col_map[collection](collection=collection)

    xmin, ymin, xmax, ymax = bbox
    coordinates = [[xmax, ymax], [xmax, ymin], [xmin, ymin], [xmin, ymax], [xmax, ymax]]
    bbox_dict = dict(type='Polygon', coordinates=[coordinates])

    im_df = imsearch.search(start_date, end_date, bbox_dict)

    if output is not None:
        output = pathlib.Path(output)
        if output.suffix == '.csv':
            im_df.to_csv(output)
        elif output.suffix == '.json':
            im_df.to_json(output)
        else:
            raise ValueError(f'Unknown output file extension: {output.suffix}')

cli.add_command(search)

