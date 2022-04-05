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
import importlib
import json
import os
import pathlib
from collections import namedtuple

import click
import ee

from geedim import collection as coll_api
from geedim import export as export_api
from geedim import info, image, _ee_init
from geedim.download import Image
from rasterio.dtypes import dtype_ranges


class _CmdChainResults(object):
    """ Class to hold results for command chaining """

    def __init__(self):
        self.search_ids = None
        self.search_region = None
        self.comp_image = None
        self.comp_id = None


def _extract_region(region=None, bbox=None, region_buf=10):
    """ Return geojson dict from region or bbox parameters """

    if (bbox is None or len(bbox) == 0) and (region is None):
        return None

    if isinstance(region, dict):
        region_dict = region
    elif region is not None:  # read region file/string
        region = pathlib.Path(region)
        if not region.exists():
            raise click.BadParameter(f'{region} does not exist.')
        if 'json' in region.suffix:
            with open(region) as f:
                region_dict = json.load(f)
        elif importlib.util.find_spec("rasterio"):  # rasterio is installed, extract region from raster file
            region_dict, _ = image.get_bounds(region, expand=region_buf)
        else:
            raise click.BadParameter(f'{region} is not a valid geojson or raster file.')
    else:  # convert bbox to geojson
        xmin, ymin, xmax, ymax = bbox
        coordinates = [[xmax, ymax], [xmax, ymin], [xmin, ymin], [xmin, ymax], [xmax, ymax]]
        region_dict = dict(type='Polygon', coordinates=[coordinates])

    return region_dict


def _export_im_list(im_list, path='', wait=True, overwrite=False, do_download=True, **kwargs):
    """ Export/download image(s) """
    click.echo('\nDownloading:\n') if do_download else click.echo('\nExporting:\n')
    export_tasks = []

    for im_dict in im_list:
        if do_download:
            filename = pathlib.Path(path).joinpath(im_dict['image'].name + '.tif')
            im_dict['image'].download(filename, overwrite=overwrite, **kwargs)
        else:
            task = im_dict['image'].export(im_dict['image'].name, folder=path, wait=False, **kwargs)
            export_tasks.append(task)

    if wait:
        for task in export_tasks:
            Image.monitor_export_task(task)


def _create_im_list(ids, **kwargs):
    """ Return a list of Image objects and names, given download/export CLI parameters """
    im_list = []

    for im_id in ids:
        ee_coll_name, im_idx = image.split_id(im_id)
        if ee_coll_name not in info.ee_to_gd:
            im_list.append(dict(image=Image(ee.Image(im_id))))
        else:
            gd_image = image.get_class(ee_coll_name).from_id(im_id, **kwargs)
            im_list.append(dict(image=gd_image))

    return im_list


def _interpret_crs(crs):
    """ return a CRS string from WKT file / EPSG string. """
    if crs is not None:
        wkt_fn = pathlib.Path(crs)
        if wkt_fn.exists():  # read WKT from file, if it exists
            with open(wkt_fn, 'r') as f:
                crs = f.read()

        if importlib.util.find_spec("rasterio"):  # clean WKT with rasterio if it is installed
            from rasterio import crs as rio_crs
            crs = rio_crs.CRS.from_string(crs).to_wkt()
    return crs


def _export_download(res=_CmdChainResults(), do_download=True, **kwargs):
    """ Helper function to execute export/download commands """

    arg_tuple = namedtuple('ArgTuple', kwargs)
    params = arg_tuple(**kwargs)

    # get the download/export region
    region = _extract_region(region=params.region, bbox=params.bbox) or res.search_region
    if region is None:
        if res.comp_image is not None:
            raise click.BadOptionUsage('region', 'One of --region or --box is required for a composite image.')
        else:
            pass  # TODO log message that footprint will be used

    # interpret the CRS
    crs = _interpret_crs(params.crs)

    # create a list of Image objects and names
    im_list = []
    if res.comp_image is not None:  # download/export chained with composite command
        im_list.append(dict(image=Image(res.comp_image)))
    elif res.search_ids is not None:  # download/export chained with search command
        im_list = _create_im_list(res.search_ids, mask=params.mask, cloud_dist=params.cloud_dist)
    elif len(params.image_id) > 0:  # download/export image ids specified on command line
        im_list = _create_im_list(params.image_id, mask=params.mask, cloud_dist=params.cloud_dist)
    else:
        raise click.BadOptionUsage('image_id',
                                   'Either pass --id, or chain this command with a successful `search` or `composite`')

    # download/export the image list
    if do_download:
        _export_im_list(im_list, region=region, path=params.download_dir, crs=crs, scale=params.scale,
                        resampling=params.resampling, overwrite=params.overwrite, dtype=params.dtype, do_download=True)
    else:
        _export_im_list(im_list, region=region, path=params.drive_folder, crs=crs, scale=params.scale,
                        resampling=params.resampling, dtype=params.dtype, do_download=False, wait=params.wait)


""" Define click options that are common to more than one command """
bbox_option = click.option(
    "-b",
    "--bbox",
    type=click.FLOAT,
    nargs=4,
    help="Region defined by bounding box co-ordinates in WGS84 (xmin, ymin, xmax, ymax).  "
         "[One of --bbox or --region is required.]",
    required=False,
    default=None,
)
region_option = click.option(
    "-r",
    "--region",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False, readable=True, resolve_path=True,
                    allow_dash=False),
    help="Region defined by geojson or raster file.  [One of --bbox or --region is required.]",
    required=False,
)
image_id_option = click.option(
    "-i",
    "--id",
    "image_id",
    type=click.STRING,
    help="Earth engine image ID(s).",
    required=False,
    multiple=True,
)
crs_option = click.option(
    "-c",
    "--crs",
    type=click.STRING,
    default=None,
    help="Reproject image(s) to this CRS (EPSG string or path to text file containing WKT). \n[default: source CRS]",
    required=False,
)
scale_option = click.option(
    "-s",
    "--scale",
    type=click.FLOAT,
    default=None,
    help="Resample image bands to this pixel resolution (m). \n[default: minimum of the source band resolutions]",
    required=False,
)
dtype_option = click.option(
    "-dt",
    "--dtype",
    type=click.Choice(list(dtype_ranges.keys()), case_sensitive=False),
    default=None,
    help="Convert image(s) to this data type.",
    required=False,
)
mask_option = click.option(
    "-m/-nm",
    "--mask/--no-mask",
    default=image.MaskedImage._default_params['mask'],
    help="Do/don't apply (cloud and shadow) nodata mask(s).  [default: --no-mask]",
    required=False,
)
resampling_option = click.option(
    "-rs",
    "--resampling",
    type=click.Choice(["near", "bilinear", "bicubic"], case_sensitive=True),
    help="Resampling method.",
    default="near",
    show_default=True,
)
cloud_dist_option = click.option(
    "-cd",
    "--cloud-dist",
    type=click.FLOAT,
    default=image.MaskedImage._default_params['cloud_dist'],
    help="Search for cloud/shadow inside this radius (m) to determine compositing quality score.",
    show_default=True,
    required=False,
)

# Define the geedim CLI and chained command group
@click.group(chain=True)
@click.pass_context
def cli(ctx):
    _ee_init()
    ctx.obj = _CmdChainResults()  # object to hold chained results


# Define search command options
@click.command()
@click.option(
    "-c",
    "--collection",
    type=click.Choice(list(info.gd_to_ee.keys()), case_sensitive=False),
    help="Earth Engine image collection to search.",
    default="landsat8_c2_l2",
    show_default=True,
)
@click.option(
    "-s",
    "--start-date",
    type=click.DateTime(),
    help="Start date (UTC).",
    required=True,
)
@click.option(
    "-e",
    "--end-date",
    type=click.DateTime(),
    help="End date (UTC).  \n[default: start_date + 1 day]",
    required=False,
)
@bbox_option
@region_option
@click.option(
    "-vp",
    "--valid-portion",
    type=click.FloatRange(min=0, max=100),
    default=0,
    help="Lower limit of the portion of valid (cloud and shadow free) pixels (%).",
    required=False,
    show_default=True,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, readable=False, resolve_path=True,
                    allow_dash=False),
    default=None,
    help="Write results to this filename, file type inferred from extension: [.csv|.json]",
    required=False,
)
@click.pass_obj
def search(res, collection, start_date, end_date, bbox, region, valid_portion, output):
    """ Search for images """

    res.search_region = _extract_region(region=region, bbox=bbox)  # store region for chaining
    if res.search_region is None:
        raise click.BadOptionUsage('region', 'Either pass --region or --bbox')
    res.search_ids = None

    click.echo(f'\nSearching for {info.gd_to_ee[collection]} images between '
               f'{start_date.strftime("%Y-%m-%d")} and {end_date.strftime("%Y-%m-%d")}...')

    # create collection wrapper and search
    gd_collection = coll_api.Collection(collection)
    im_df = gd_collection.search(start_date, end_date, res.search_region, valid_portion=valid_portion)

    if im_df.shape[0] == 0:
        click.echo('No images found\n')
    else:
        res.search_ids = im_df.ID.values.tolist()  # store ids for chaining
        click.echo(f'{len(res.search_ids)} images found\n')
        click.echo(f'Image property descriptions:\n\n{gd_collection.summary_key}\n')
        click.echo(f'Search Results:\n\n{gd_collection.summary}')

    # write results to file
    if output is not None:
        output = pathlib.Path(output)
        if output.suffix == '.csv':
            im_df.to_csv(output, index=False)
        elif output.suffix == '.json':
            im_df.to_json(output, orient='index')
        else:
            raise ValueError(f'Unknown output file extension: {output.suffix}')


cli.add_command(search)


# Define download command options
@click.command()
@image_id_option
@bbox_option
@region_option
@click.option(
    "-dd",
    "--download-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True, readable=False, resolve_path=True),
    default=os.getcwd(),
    help="Download image file(s) to this directory.  [default: cwd]",
    required=False,
)
@crs_option
@scale_option
@dtype_option
@mask_option
@resampling_option
@cloud_dist_option
@click.option(
    "-o",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite the destination file if it exists.  [default: prompt the user for confirmation]",
    required=False,
    show_default=False,
)
@click.pass_context
def download(ctx, image_id, bbox, region, download_dir, crs, scale, dtype, mask, resampling, cloud_dist,
             overwrite=False):
    """ Download image(s), with cloud and shadow masking """
    _export_download(res=ctx.obj, do_download=True, **ctx.params)


cli.add_command(download)


# Define export command options
@click.command()
@image_id_option
@bbox_option
@region_option
@click.option(
    "-df",
    "--drive-folder",
    type=click.STRING,
    default='',
    help="Export image(s) to this Google Drive folder. [default: root]",
    required=False,
)
@crs_option
@scale_option
@dtype_option
@mask_option
@resampling_option
@cloud_dist_option
@click.option(
    "-w/-nw",
    "--wait/--no-wait",
    default=True,
    help="Wait / don't wait for export to complete.  [default: --wait]",
    required=False,
)
@click.pass_context
def export(ctx, image_id, bbox, region, drive_folder, crs, scale, dtype, mask, resampling, cloud_dist, wait):
    """ Export image(s) to Google Drive, with cloud and shadow masking """
    _export_download(res=ctx.obj, do_download=False, **ctx.params)


cli.add_command(export)


# Define composite command options
@click.command()
@image_id_option
@click.option(
    "-cm",
    "--method",
    type=click.Choice(coll_api.Collection.composite_methods, case_sensitive=False),
    help="Compositing method to use.",
    default="q_mosaic",
    show_default=True,
    required=False,
)
@click.option(
    "-m/-nm",
    "--mask/--no-mask",
    default=True,
    help="Do/don't apply (cloud and shadow) nodata mask(s) before compositing.  [default: --mask]",
    required=False,
)
@resampling_option
@cloud_dist_option
@click.pass_obj
def composite(res, image_id, mask, method, resampling, cloud_dist):
    """ Create a cloud-free composite image """

    # get image ids from command line or chained search command
    image_id = list(image_id)
    if image_id is None or len(image_id) == 0:
        if res.search_ids is None:
            raise click.BadOptionUsage('image_id', 'Either pass --id, or chain this command with a successful `search`')
        else:
            image_id = res.search_ids
            res.search_ids = None

    gd_collection = coll_api.Collection.from_ids(image_id, mask=mask, cloud_dist=cloud_dist)
    res.comp_image, res.comp_id = gd_collection.composite(method=method, resampling=resampling)


cli.add_command(composite)

##
