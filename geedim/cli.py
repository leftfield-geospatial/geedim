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
import logging
import os
import pathlib
import sys
from types import SimpleNamespace
from typing import List

import click
import rasterio.crs as rio_crs
from click.core import ParameterSource
from rasterio.dtypes import dtype_ranges
from rasterio.errors import CRSError

from geedim import collection as coll_api, info, _ee_init, version
from geedim.collection import MaskedCollection
from geedim.download import BaseImage, get_bounds
from geedim.enums import CloudMaskMethod, CompositeMethod, ResamplingMethod
from geedim.mask import MaskedImage

logger = logging.getLogger(__name__)


class PlainInfoFormatter(logging.Formatter):
    """logging formatter to format INFO logs without the module name etc prefix"""

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "%(levelname)s:%(name)s: %(message)s"
        return super().format(record)


class ChainedCommand(click.Command):
    """
    click Command class for managing parameters shared between chained commands,
    and formatting single newlines in help strings as single newlines.
    """

    def get_help(self, ctx):
        """Format help strings with single newlines as single newlines."""
        # adapted from https://stackoverflow.com/questions/55585564/python-click-formatting-help-text
        orig_wrap_text = click.formatting.wrap_text

        def wrap_text(text, width, **kwargs):
            text = text.replace('\n', '\n\n')
            return orig_wrap_text(text, width, **kwargs).replace('\n\n', '\n')

        click.formatting.wrap_text = wrap_text
        return click.Command.get_help(self, ctx)

    def invoke(self, ctx):
        """Manage shared `image_list` and `region` parameters."""

        # initialise earth engine (do it here, rather than in cli() so that it does not delay --help)
        _ee_init()

        # combine `region` and `bbox` into a single region in the context object
        region = ctx.params['region'] if 'region' in ctx.params else None
        bbox = ctx.params['bbox'] if 'bbox' in ctx.params else None
        region = region or bbox
        if region is not None:
            ctx.obj.region = region

        if 'image_id' in ctx.params:
            # append any image id's to the image_list
            ctx.obj.image_list += list(ctx.params['image_id'])

        return click.Command.invoke(self, ctx)


def _configure_logging(verbosity):
    """configure python logging level"""
    # adapted from rasterio https://github.com/rasterio/rasterio
    log_level = max(10, 20 - 10 * verbosity)

    # limit logging config to homonim by applying to package logger, rather than root logger
    # pkg_logger level etc are then 'inherited' by logger = getLogger(__name__) in the modules
    pkg_logger = logging.getLogger(__package__)
    formatter = PlainInfoFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(log_level)
    logging.captureWarnings(True)


def _collection_cb(ctx, param, value):
    """click callback to validate collection name"""
    if value in info.gd_to_ee:
        value = info.gd_to_ee[value]
    return value


def _crs_cb(ctx, param, crs):
    """click callback to validate and parse the CRS."""
    if crs is not None:
        try:
            wkt_fn = pathlib.Path(crs)
            if wkt_fn.exists():  # read WKT from file, if it exists
                with open(wkt_fn, 'r') as f:
                    crs = f.read()

            crs = rio_crs.CRS.from_string(crs).to_wkt()
        except CRSError as ex:
            raise click.BadParameter(f'Invalid CRS value: {crs}.\n {str(ex)}', param=param)
    return crs


def _bbox_cb(ctx, param, value):
    """click callback to validate and parse --bbox"""
    if isinstance(value, tuple) and len(value) == 4:  # --bbox
        xmin, ymin, xmax, ymax = value
        coordinates = [[xmax, ymax], [xmax, ymin], [xmin, ymin], [xmin, ymax], [xmax, ymax]]
        value = dict(type='Polygon', coordinates=[coordinates])
    elif value is not None and len(value) != 0:
        raise click.BadParameter(f'Invalid bbox: {value}.', param=param)
    return value


def _region_cb(ctx, param, value):
    """click callback to validate and parse --region"""
    filename = value
    if isinstance(value, str):  # read region file/string
        if value == '-' or 'json' in value:
            with click.open_file(value, encoding='utf-8') as f:
                value = json.load(f)
        else:
            value = get_bounds(value, expand=10)
    elif value is not None and len(value) != 0:
        raise click.BadParameter(f'Invalid region: {filename}.', param=param)
    return value


def _mask_method_cb(ctx, param, value):
    """click callback to convert cloud mask method string to enum."""
    return CloudMaskMethod(value)


def _resampling_method_cb(ctx, param, value):
    """click callback to convert resampling string to enum."""
    return ResamplingMethod(value)


def _comp_method_cb(ctx, param, value):
    """click callback to convert composite method string to enum."""
    return CompositeMethod(value) if value else None


def _prepare_image_list(obj: SimpleNamespace, mask=False) -> List[MaskedImage,]:
    """Validate and prepare the obj.image_list for export/download.  Returns a list of MaskedImage objects."""
    if len(obj.image_list) == 0:
        raise click.BadOptionUsage(
            'image_id', 'Either pass --id, or chain this command with a successful `search` or `composite`'
        )
    image_list = []
    for im_obj in obj.image_list:
        if isinstance(im_obj, str):
            im_obj = MaskedImage.from_id(im_obj, mask=mask, **obj.cloud_kwargs)
        elif isinstance(im_obj, MaskedImage):
            if mask:
                im_obj.mask_clouds()
        else:
            raise ValueError(f'Unsupported image object type: {type(im_obj)}')
        image_list.append(im_obj)

    if obj.region is None and any([not im.has_fixed_projection for im in image_list]):
        raise click.BadOptionUsage('region', 'One of --region or --bbox is required for a composite image.')
    return image_list


# Define click options that are common to more than one command
bbox_option = click.option(
    '-b', '--bbox', type=click.FLOAT, nargs=4, default=None, callback=_bbox_cb,
    help='Region defined by bounding box co-ordinates in WGS84 (xmin, ymin, xmax, ymax).'
)
region_option = click.option(
    '-r', '--region', type=click.Path(exists=True, dir_okay=False, allow_dash=True), default=None, callback=_region_cb,
    help='Region defined by geojson or raster file.  Use "-" to read geojson from stdin.'
)
image_id_option = click.option(
    '-i', '--id', 'image_id', type=click.STRING, multiple=True, help='Earth engine image ID(s).'
)
crs_option = click.option(
    '-c', '--crs', type=click.STRING, default=None, callback=_crs_cb,
    help='Reproject image(s) to this CRS (EPSG string or path to text file containing WKT). \n[default: source CRS]'
)
scale_option = click.option(
    '-s', '--scale', type=click.FLOAT, default=None,
    help='Resample image bands to this pixel resolution (m). \n[default: minimum of the source band resolutions]'
)
dtype_option = click.option(
    '-dt', '--dtype', type=click.Choice(list(dtype_ranges.keys()), case_sensitive=False), default=None,
    help='Convert image(s) to this data type.'
)
mask_option = click.option(
    '-m/-nm', '--mask/--no-mask', default=MaskedImage._default_mask,
    help='Do/don\'t apply (cloud and shadow) nodata mask(s).  [default: --no-mask]'
)
resampling_option = click.option(
    '-rs', '--resampling', type=click.Choice([rm.value for rm in ResamplingMethod], case_sensitive=True),
    default=BaseImage._default_resampling.value, show_default=True, callback=_resampling_method_cb,
    help='Resampling method.',
)


# geedim CLI and chained command group
@click.group(chain=True)
@click.option('--verbose', '-v', count=True, help="Increase verbosity.")
@click.option('--quiet', '-q', count=True, help="Decrease verbosity.")
@click.version_option(version=version.__version__, message='%(version)s')
@click.pass_context
def cli(ctx, verbose, quiet):
    ctx.obj = SimpleNamespace(image_list=[], region=None, cloud_kwargs={})
    verbosity = verbose - quiet
    _configure_logging(verbosity)


# config command
@click.command(cls=ChainedCommand, context_settings=dict(auto_envvar_prefix='GEEDIM'))
@click.option(
    '-mc/-nmc', '--mask-cirrus/--no-mask-cirrus', default=True,
    help='Whether to mask cirrus clouds. For sentinel2 collections this is valid just for method = `qa`.  '
         '[default: --mask-cirrus]'
)
@click.option(
    '-ms/-nms', '--mask-shadows/--no-mask-shadows', default=True,
    help='Whether to mask cloud shadows. [default: --mask-shadows]'
)
@click.option(
    '-mm', '--mask-method', type=click.Choice([cmm.value for cmm in CloudMaskMethod], case_sensitive=True),
    default=CloudMaskMethod.cloud_prob.value, show_default=True, callback=_mask_method_cb,
    help='Method used to cloud mask Sentinel-2 images.'
)
@click.option(
    '-p', '--prob', type=click.FloatRange(min=0, max=100), default=60, show_default=True,
    help='Cloud probability threshold. Valid just for --mask-method `cloud-prob`. (%).'
)
@click.option(
    '-d', '--dark', type=click.FloatRange(min=0, max=1), default=.15, show_default=True,
    help='NIR reflectance threshold [0-1] for shadow masking Sentinel-2 images. NIR values below this threshold are '
         'potential cloud shadows.'
)
@click.option(
    '-sd', '--shadow-dist', type=click.INT, default=1000, show_default=True,
    help='Maximum distance in meters (m) to look for cloud shadows from cloud edges.  Valid for Sentinel-2 images.'
)
@click.option(
    '-b', '--buffer', type=click.INT, default=250, show_default=True,
    help='Distance in meters (m) to dilate cloud and cloud shadows objects.  Valid for Sentinel-2 images.'
)
@click.option(
    '-cdi', '--cdi-thresh', type=click.FloatRange(min=-1, max=1), default=None,
    help='Cloud Displacement Index threshold. Values below this threshold are considered potential clouds.  A '
         'cdi-thresh = None means that the index is not used.  Valid for Sentinel-2 images.  [default: None]'
)
@click.option(
    '-mcd', '--max-cloud-dist', type=click.INT, default=5000, show_default=True,
    help='Maximum distance in meters (m) to look for clouds.  Used to form the cloud distance band for `q-mosaic` '
         'compositing. Valid for Sentinel-2 images.'
)
@click.pass_context
def config(ctx, mask_cirrus, mask_shadows, mask_method, prob, dark, shadow_dist, buffer, cdi_thresh, max_cloud_dist):
    """Configure cloud/shadow masking."""
    # store commandline configuration (only) in the context object for use by other commands
    for key, val in ctx.params.items():
        if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE:
            ctx.obj.cloud_kwargs[key] = val


cli.add_command(config)


# search command
@click.command(cls=ChainedCommand)
@click.option(
    '-c', '--collection', type=click.STRING, default='landsat8_c2_l2', show_default=True, callback=_collection_cb,
    help=f'Earth Engine image collection to search.  [{"|".join(list(info.gd_to_ee.keys())[:-1])}], or '
         f'any valid Earth Engine image collection ID.'
)
@click.option(
    '-s', '--start-date', type=click.DateTime(), required=True, help='Start date (UTC).'
)
@click.option(
    '-e', '--end-date', type=click.DateTime(), required=False, help='End date (UTC).  \n[default: start_date + 1 day]'
)
@bbox_option
@click.option(
    '-r', '--region', type=click.Path(exists=True, dir_okay=False, allow_dash=True), callback=_region_cb,
    help='Region defined by geojson or raster file. Use "-" to read geojson from stdin.  [One of --bbox or --region is '
         'required]'
)
@click.option(
    '-cp', '--cloudless-portion', type=click.FloatRange(min=0, max=100), default=0, show_default=True,
    help='Lower limit on the cloud/shadow free portion of the region (%).'
)
@click.option(
    '-o', '--output', type=click.Path(exists=False, dir_okay=False, writable=True), default=None,
    help='Write search results to this json file'
)
@click.pass_obj
def search(obj, collection, start_date, end_date, bbox, region, cloudless_portion, output):
    """Search for images."""
    if not obj.region:
        raise click.BadOptionUsage('region', 'Either pass --region or --bbox')

    # TODO: add spinner
    logger.info(
        f'\nSearching for {collection} images between {start_date.strftime("%Y-%m-%d")} and '
        f'{end_date.strftime("%Y-%m-%d")}...'
    )

    # create collection wrapper and search
    gd_collection = coll_api.MaskedCollection.from_name(collection)
    gd_collection = gd_collection.search(
        start_date, end_date, obj.region, cloudless_portion=cloudless_portion, **obj.cloud_kwargs
    )

    if len(gd_collection.properties) == 0:
        logger.info('No images found\n')
    else:
        obj.image_list += list(gd_collection.properties.keys())  # store image ids for chained commands
        logger.info(f'{len(gd_collection.properties)} images found\n')
        logger.info(f'Image property descriptions:\n\n{gd_collection.key_table}\n')
        logger.info(f'Search Results:\n\n{gd_collection.properties_table}')

    # write results to file
    if output is not None:
        output = pathlib.Path(output)
        with open(output, 'w', encoding='utf8', newline='') as f:
            json.dump(gd_collection.properties, f)


cli.add_command(search)


# download command
@click.command(cls=ChainedCommand)
@image_id_option
@bbox_option
@region_option
@click.option(
    '-dd', '--download-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=os.getcwd(), help='Download image file(s) to this directory.  [default: cwd]'
)
@crs_option
@scale_option
@dtype_option
@mask_option
@resampling_option
@click.option(
    '-o', '--overwrite', is_flag=True, default=False,
    help='Overwrite the destination file if it exists.  [default: don\'t overwrite]'
)
@click.pass_obj
def download(obj, image_id, bbox, region, download_dir, mask, overwrite, **kwargs):
    """Download image(s)."""
    logger.info('\nDownloading:\n')
    image_list = _prepare_image_list(obj, mask=mask)
    for im in image_list:
        filename = pathlib.Path(download_dir).joinpath(im.name + '.tif')
        im.download(filename, overwrite=overwrite, region=obj.region, **kwargs)


cli.add_command(download)


# export command
@click.command(cls=ChainedCommand)
@image_id_option
@bbox_option
@region_option
@click.option(
    '-df', '--drive-folder', type=click.STRING, default='',
    help='Export image(s) to this Google Drive folder. [default: root]'
)
@crs_option
@scale_option
@dtype_option
@mask_option
@resampling_option
@click.option(
    '-w/-nw', '--wait/--no-wait', default=True, help='Wait / don\'t wait for export to complete.  [default: --wait]'
)
@click.pass_obj
def export(obj, image_id, bbox, region, drive_folder, mask, wait, **kwargs):
    """Export image(s) to Google Drive."""
    logger.info('\nExporting:\n')
    image_list = _prepare_image_list(obj, mask=mask)
    export_tasks = []
    for im in image_list:
        task = im.export(im.name, folder=drive_folder, wait=False, region=obj.region, **kwargs)
        export_tasks.append(task)
        logger.info(f'Started {im.name}') if not wait else None

    if wait:
        for task in export_tasks:
            BaseImage.monitor_export_task(task)


cli.add_command(export)


# composite command
@click.command(cls=ChainedCommand)
@image_id_option
@click.option(
    '-cm', '--method', type=click.Choice([cm.value for cm in CompositeMethod], case_sensitive=False),
    default=None, callback=_comp_method_cb,
    help='Compositing method to use.  [default: `q_mosaic` for supported collections, otherwise `mosaic` ]'
)
@click.option(
    '-m/-nm', '--mask/--no-mask', default=True,
    help='Do/don\'t apply (cloud and shadow) nodata mask(s) before compositing.  [default: --mask]'
)
@resampling_option
@bbox_option  # TODO: the implications of these options here needs to be explained.  Likewise for method.
@region_option
@click.option(
    '-d', '--date', type=click.DateTime(),
    help='Give preference to images closest to this date (UTC).  [Supported by `mosaic` and `q-mosaic` methods.]'
)
@click.pass_obj
def composite(obj, image_id, mask, method, resampling, bbox, region, date):
    """Create a cloud-free composite image."""

    # get image ids from command line or chained search command
    if len(obj.image_list) == 0:
        raise click.BadOptionUsage('image_id', 'Either pass --id, or chain this command with a successful `search`')

    gd_collection = MaskedCollection.from_list(obj.image_list)
    obj.image_list = [gd_collection.composite(
        method=method, mask=mask, resampling=resampling, region=obj.region, date=date, **obj.cloud_kwargs
    )]


cli.add_command(composite)

##
