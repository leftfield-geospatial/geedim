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

from __future__ import annotations

import json
import logging
import posixpath
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import click
import ee
import fsspec
import rasterio as rio
from click.core import ParameterSource
from fsspec.core import OpenFile
from rasterio.warp import transform_bounds
from tqdm.auto import tqdm

from geedim import enums, schema, utils, version
from geedim.image import ImageAccessor, _nodata_vals
from geedim.tile import Tiler

logger = logging.getLogger(__name__)


class _EECommand(click.Command):
    """click Command sub-class for initialising Earth Engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ee_init = False

    def invoke(self, ctx: click.Context) -> Any:
        if not self._ee_init:
            utils.Initialize()
            self._ee_init = True
        return super().invoke(ctx)


def _collection_cb(ctx: click.Context, param: click.Parameter, name: str) -> str:
    """click callback to convert abbreviated to full collection name."""
    return schema.gd_to_ee.get(name, name)


def _crs_cb(ctx: click.Context, param: click.Parameter, crs: str) -> str | None:
    """click callback to read --crs."""
    if crs:
        crs_path = Path(crs)
        try:
            if crs_path.suffix.lower() in ['.tif', '.tiff', '.gtiff']:
                # read CRS from geotiff path / URI
                with rio.open(fsspec.open(crs, 'rb'), 'r') as im:
                    crs = im.crs.to_wkt()
            else:
                if crs_path.exists() or urlparse(crs).scheme in fsspec.available_protocols():
                    # read WKT string from text file path / URI
                    with fsspec.open(crs, 'rt', encoding='utf-8') as f:
                        crs = f.read()
        except Exception as ex:
            raise click.BadParameter(str(ex)) from ex
    return crs


def _bbox_cb(ctx: click.Context, param: click.Parameter, bounds: Sequence[float]) -> None:
    """click callback to convert --bbox to a GeoJSON polygon."""
    # --bbox and --region are not exposed but combined into 'geometry' which is passed to the
    # command
    ctx.params.setdefault('geometry', None)
    if bounds == '-':
        if 'geometry' not in ctx.obj:
            raise click.BadParameter("No piped bounds or region available.")
        ctx.params['geometry'] = ctx.obj['geometry']
    elif bounds:
        bounds = _bounds_to_geojson(*bounds)
        ctx.params['geometry'] = ctx.obj['geometry'] = bounds


def _region_cb(ctx: click.Context, param: click.Parameter, region: str) -> None:
    """click callback to read --region and convert to a GeoJSON polygon."""
    # --bbox and --region are not exposed but combined into 'geometry' which is passed to the
    # command
    ctx.params.setdefault('geometry', None)
    if region == '-':
        if 'geometry' not in ctx.obj:
            raise click.BadParameter("No piped bounds or region available.")
        ctx.params['geometry'] = ctx.obj['geometry']
    elif region:
        try:
            if region.lower().endswith('json'):
                with fsspec.open(region, 'rt', encoding='utf-8') as f:
                    region = json.load(f)
            else:
                with rio.open(fsspec.open(region, 'rb'), 'r') as ds:
                    bounds = transform_bounds(ds.crs, 'EPSG:4326', *ds.bounds)
                    region = _bounds_to_geojson(*bounds)
        except Exception as ex:
            raise click.BadParameter(str(ex)) from ex
        ctx.params['geometry'] = ctx.obj['geometry'] = region


def _dir_cb(ctx: click.Context, param: click.Parameter, uri_path: str) -> OpenFile:
    """Click callback to convert a directory path / URI to an fsspec OpenFile, and validate."""
    try:
        ofile = fsspec.open(uri_path)
    except Exception as ex:
        raise click.BadParameter(str(ex)) from ex

    # isdir() requires a trailing slash on some file systems (e.g. gcs)
    if not ofile.fs.isdir(posixpath.join(ofile.path, '')):
        raise click.BadParameter(f"'{uri_path}' is not a directory or cannot be accessed.")
    return ofile


def _like_cb(ctx: click.Context, param: click.Parameter, uri_path: str) -> dict[str, Any] | None:
    """Click callback to read --like."""
    if uri_path:
        try:
            with rio.open(fsspec.open(uri_path, 'rb'), 'r') as ds:
                if not ds.crs:
                    raise click.BadParameter('Image is not georeferenced.')
                return dict(crs=ds.crs.to_wkt(), crs_transform=ds.transform[:6], shape=ds.shape)
        except Exception as ex:
            raise click.BadParameter(str(ex)) from ex
    return None


def _bounds_to_geojson(left: float, bottom: float, right: float, top: float) -> dict[str, Any]:
    """Convert bounds to a GeoJSON polygon."""
    coords = [[left, top], [left, bottom], [right, bottom], [right, top]]
    return dict(type="Polygon", coordinates=[coords], evenOdd=True)


def _configure_logging(verbosity: int):
    """Configure logging level, redirecting warnings to the logger and logs to ``tqdm.write()``."""
    # configure the package logger (adapted from rasterio:
    # https://github.com/rasterio/rasterio/blob/main/rasterio/rio/main.py)
    pkg_logger = logging.getLogger(__package__)
    log_level = max(10, 20 - 10 * verbosity)
    # route logs through tqdm handler so they don't interfere with progress bars
    handler = utils.TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(log_level)

    # redirect warnings to package logger
    def showwarning(message, category, filename, lineno, file=None, line=None):
        """Redirect warnings to the package logger."""
        pkg_logger.warning(message)

    warnings.showwarning = showwarning


def _get_images(obj: dict[str, Any], image_ids: tuple[str]) -> list[ee.Image]:
    """Combine piped images in the click context object with images created from --id image IDs."""
    images = [ee.Image(ee_id) for ee_id in image_ids] + obj.get('images', [])
    if len(images) == 0:
        raise click.UsageError(
            "No images - either supply '--id' / '-i' or pipe images into this command."
        )
    return images


def _prepare_export_collection(
    images: list[ee.Image],
    cloud_kwargs: dict | None = None,
    geometry: dict[str, Any] | None = None,
    buffer: float | None = None,
    like: dict[str, Any] | None = None,
    mask: bool = False,
    **kwargs,
) -> ee.ImageCollection:
    """Create an image collection and prepare it for export."""
    # resolve and buffer region
    if buffer is not None:
        if not geometry:
            raise click.UsageError(
                "'-b' / '--bbox' or '-r' / '--region' is required with '-buf' / '--buffer'"
            )
        geometry = ee.Geometry(geometry).buffer(buffer)

    if like:
        # overwrite crs, crs_transform and shape from --like
        kwargs.update(**like)

    coll = ee.ImageCollection.gd.fromImages(images)
    coll = coll.gd.addMaskBands(**(cloud_kwargs or {}))
    if mask:
        coll = coll.gd.maskClouds()
    coll = coll.gd.prepareForExport(region=geometry, **kwargs)
    return coll


# Define click options that are common to more than one command.
bbox_option = click.option(
    '-b',
    '--bbox',
    type=click.FLOAT,
    nargs=4,
    default=None,
    show_default='source image bounds',
    callback=_bbox_cb,
    expose_value=False,  # callback exposes 'geometry' to the command
    metavar='LEFT BOTTOM RIGHT TOP',
    help="WGS84 geographic coordinates of the export bounds.  Use'-' to read from previous --bbox / "
    "--region options in the pipeline.",
)
region_option = click.option(
    '-r',
    '--region',
    type=click.Path(dir_okay=False),
    default=None,
    show_default='source image bounds',
    callback=_region_cb,
    expose_value=False,  # callback exposes 'geometry' to the command
    help="Path / URI of a GeoJSON or georeferenced image file defining the export bounds.  Use "
    "'-' to read from previous --bbox / --region options in the pipeline.",
)
buffer_option = click.option(
    '-buf',
    '--buffer',
    type=click.FLOAT,
    default=None,
    show_default=True,
    help='Distance (m) to buffer --bbox / --region with.',
)
crs_option = click.option(
    '-c',
    '--crs',
    type=click.STRING,
    default=None,
    show_default='source image CRS',
    callback=_crs_cb,
    help='CRS of the export image(s) as a well known authority (e.g. EPSG) or WKT string, path / '
    'URI of text file containing a string, or path / URI of an image with metadata CRS.',
)
scale_option = click.option(
    '-s',
    '--scale',
    type=click.FLOAT,
    default=None,
    show_default='minimum scale of source image bands',
    help='Pixel scale of the export image(s) (m).',
)
dtype_option = click.option(
    '-dt',
    '--dtype',
    type=click.Choice(_nodata_vals.keys(), case_sensitive=True),
    default=None,
    show_default='auto',
    help='Data type of the export image(s).',
)
mask_option = click.option(
    '-m/-nm',
    '--mask/--no-mask',
    default=False,
    show_default=True,
    help='Whether to mask the export image(s).  Cloud/shadow mask(s) are used when supported, '
    'otherwise fill mask(s).',
)
resampling_option = click.option(
    '-rs',
    '--resampling',
    type=click.Choice(enums.ResamplingMethod, case_sensitive=True),
    default=ImageAccessor._default_resampling,
    show_default=True,
    help='Resampling method to use when reprojecting.',
)
scale_offset_option = click.option(
    '-so/-nso',
    '--scale-offset/--no-scale-offset',
    default=False,
    show_default=True,
    help='Whether to apply any STAC scales and offsets to the export image(s) (e.g. to convert '
    'digital numbers to physical units).',
)
crs_transform_option = click.option(
    '-ct',
    '--crs-transform',
    type=click.FLOAT,
    nargs=6,
    default=None,
    metavar='XSCALE XSHEAR XTRANSLATION YSHEAR YSCALE YTRANSLATION',
    help='Georeferencing transform of the export image(s).',
)
shape_option = click.option(
    '-sh',
    '--shape',
    type=click.INT,
    nargs=2,
    default=None,
    metavar='HEIGHT WIDTH',
    help='Dimensions of the export image(s) (pixels).',
)
like_option = click.option(
    '-l',
    '--like',
    type=click.Path(dir_okay=False),
    default=None,
    callback=_like_cb,
    help='Path / URI of a georeferenced image file defining --crs, --crs-transform & --shape.',
)
image_ids_option = click.option(
    '-i',
    '--id',
    'image_ids',
    type=click.STRING,
    multiple=True,
    help='Earth Engine ID(s) of image(s) to export.',
)
band_name_option = click.option(
    '-bn',
    '--band-name',
    'bands',
    type=click.STRING,
    multiple=True,
    default=None,
    show_default='all bands',
    help='Band name(s) to export.',
)
split_option = click.option(
    '-sp',
    '--split',
    type=click.Choice(enums.SplitType, case_sensitive=True),
    default=enums.SplitType.images,
    show_default=True,
    help="Whether to export an image file for each input image, or for each band of the input "
    "image(s).",
)


# geedim CLI and chained command group
@click.group(chain=True)
@click.option('--verbose', '-v', count=True, help="Increase verbosity.")
@click.option('--quiet', '-q', count=True, help="Decrease verbosity.")
@click.version_option(version=version.__version__, message='%(version)s')
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: int):
    """Search, composite and export Google Earth Engine imagery."""
    ctx.obj = {}
    _configure_logging(verbose - quiet)
    # prevent GDAL from searching for sidecar files (speeds up reads of remote files)
    ctx.with_resource(rio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'))


# config command
@cli.command(epilog='See https://geedim.readthedocs.io/ for more details.')
@click.option(
    '-mc/-nmc',
    '--mask-cirrus/--no-mask-cirrus',
    default=True,
    show_default=True,
    help="Whether to mask cirrus clouds.  Valid for Landsat 8-9 images and for Sentinel-2 "
    "images with the 'qa' --mask-method.",
)
@click.option(
    '-ms/-nms',
    '--mask-shadows/--no-mask-shadows',
    default=True,
    show_default=True,
    help="Whether to mask cloud shadows.  Valid for Landsat images and for Sentinel-2 images "
    "with the 'qa' or 'cloud-prob' --mask-method.",
)
@click.option(
    '-mm',
    '--mask-method',
    type=click.Choice(enums.CloudMaskMethod, case_sensitive=True),
    default=enums.CloudMaskMethod.cloud_score,
    show_default=True,
    help='Method used to mask clouds.  Valid for Sentinel-2 images.',
)
@click.option(
    '-p',
    '--prob',
    type=click.FloatRange(min=0, max=100),
    default=60,
    show_default=True,
    help="Cloud Probability threshold (%). Valid for Sentinel-2 images with the 'cloud-prob' "
    "--mask-method.",
)
@click.option(
    '-d',
    '--dark',
    type=click.FloatRange(min=0, max=1),
    default=0.15,
    show_default=True,
    help="NIR reflectance threshold for shadow masking. NIR values below this threshold are "
    "potential cloud shadows.  Valid for Sentinel-2 images with the 'qa' or 'cloud-prob' "
    "--mask-method.",
)
@click.option(
    '-sd',
    '--shadow-dist',
    type=click.INT,
    default=1000,
    show_default=True,
    help="Maximum distance (m) to look for cloud shadows from cloud edges.  Valid for Sentinel-2 "
    "images with the 'qa' or 'cloud-prob' --mask-method.",
)
@click.option(
    '-b',
    '--buffer',
    type=click.INT,
    default=50,
    show_default=True,
    help="Distance (m) to dilate cloud/shadow.  Valid for Sentinel-2 images with the 'qa' or "
    "'cloud-prob' --mask-method.",
)
@click.option(
    '-cdi',
    '--cdi-thresh',
    type=click.FloatRange(min=-1, max=1),
    default=None,
    help="Cloud Displacement Index (CDI) threshold.  Values below this threshold are considered "
    "potential clouds.  Valid for Sentinel-2 images with the 'qa' or 'cloud-prob' "
    '--mask-method.  By default, the CDI is not used.',
)
@click.option(
    '-mcd',
    '--max-cloud-dist',
    type=click.INT,
    default=5000,
    show_default=True,
    help="Maximum distance (m) to look for clouds.  Used to form the cloud distance band for the "
    "'q-mosaic' compositing --method.",
)
@click.option(
    '-s',
    '--score',
    type=click.FloatRange(min=0, max=1),
    default=0.6,
    show_default=True,
    help="Cloud Score+ threshold.  Valid for Sentinel-2 images with the 'cloud-score' "
    "--mask-method.",
)
@click.option(
    '-cb',
    '--cs-band',
    type=click.Choice(enums.CloudScoreBand, case_sensitive=True),
    default=enums.CloudScoreBand.cs,
    show_default=True,
    help="Cloud Score+ band to threshold. Valid for Sentinel-2 images with the 'cloud-score' "
    '--mask-method.',
)
@click.pass_context
def config(ctx: click.Context, **kwargs):
    """
    Configure cloud/shadow masking.

    Cloud/shadow configuration is piped out of this command and used by subsequent pipeline
    commands that require it.
    """
    # store configuration in the context object for use by other commands (only include command
    # line options so that the mask._CloudlessImage._get_mask_bands() does not get unexpected
    # default value kwargs)
    ctx.obj['cloud_kwargs'] = {
        k: v
        for k, v in ctx.params.items()
        if ctx.get_parameter_source(k) == ParameterSource.COMMANDLINE
    }


# search command
@cli.command(cls=_EECommand, epilog='See https://geedim.readthedocs.io/ for more details.')
@click.option(
    '-c',
    '--collection',
    'coll_id',
    type=click.STRING,
    required=True,
    callback=_collection_cb,
    help='Earth Engine / Geedim ID of the image collection to search.',
)
@click.option(
    '-s',
    '--start-date',
    type=click.DateTime(),
    required=False,
    default=None,
    help='Start date (UTC).',
)
@click.option(
    '-e',
    '--end-date',
    type=click.DateTime(),
    required=False,
    default=None,
    show_default='one millisecond after --start-date',
    help='End date (UTC).',
)
@bbox_option
@region_option
@click.option(
    '-fp',
    '--fill-portion',
    '--fill',
    metavar='VALUE',
    type=click.FloatRange(min=0, max=100),
    default=None,
    is_flag=False,
    flag_value=0,
    show_default="don't calculate or filter on fill portion",
    help='Lower limit on the portion of --bbox / --region that contains filled pixels (%). '
    'Uses zero if VALUE is not supplied.',
)
@click.option(
    '-cp',
    '--cloudless-portion',
    '--cloudless',
    metavar='VALUE',
    type=click.FloatRange(min=0, max=100),
    default=None,
    is_flag=False,
    flag_value=0,
    show_default="don't calculate or filter on cloudless portion",
    help='Lower limit on the portion of filled pixels in --bbox / --region that are cloud/shadow '
    'free (%).  Uses zero if VALUE is not supplied. Has no effect if cloud/shadow masking '
    ' is not supported for --collection.',
)
@click.option(
    '-cf',
    '--custom-filter',
    type=click.STRING,
    default=None,
    help='Custom image property filter e.g. "property > value".',
)
@click.option(
    '-ap',
    '--add-property',
    'add_props',
    type=click.STRING,
    default=None,
    multiple=True,
    help='Additional image property name(s) to include in search results.',
)
# TODO: deprecate this?
@click.option(
    '-op',
    '--output',
    type=click.Path(dir_okay=False),
    default=None,
    help='Path / URI of a file to write JSON search results to.',
)
@click.pass_context
def search(
    ctx: click.Context,
    coll_id: str,
    geometry: dict[str, Any] | None,
    output: str | None,
    add_props: tuple[str] | None,
    **kwargs,
):
    """
    Search a collection for images.

    Search result images are added to images piped in from previous commands, and piped out for
    use by subsequent commands.
    """
    # raise an error if --fill-portion / --cloudless-portion were supplied without --bbox / --region
    if not geometry:
        for option, option_str in zip(
            ['fill_portion', 'cloudless_portion'],
            ["'-fp' / '--fill-portion'", "'-cp' / '--cloudless-portion'"],
        ):
            if ctx.get_parameter_source(option) is ParameterSource.COMMANDLINE:
                raise click.UsageError(
                    f"'-b' / '--bbox' or '-r' / '--region' is required with {option_str}"
                )

    # create collection and search
    coll = ee.ImageCollection(coll_id)
    label = f'Searching for {coll_id} images: '
    with utils.Spinner(desc=label, leave=' '):
        coll = coll.gd.filter(region=geometry, **kwargs, **ctx.obj.get('cloud_kwargs', {}))

        # retrieve search result properties from EE
        coll.gd.schemaPropertyNames += add_props
        num_images = len(coll.gd.properties)

        # update context object images for chained commands
        images = [ee.Image(f'{coll_id}/{im_idx}') for im_idx in coll.gd.properties.keys()]
        ctx.obj['images'] = ctx.obj.get('images', []) + images

    # print results
    if num_images == 0:
        tqdm.write('No images found\n')
    else:
        tqdm.write(f'{num_images} images found\n')
        tqdm.write(f'Image property descriptions:\n\n{coll.gd.schemaTable}\n')
        tqdm.write(f'Search Results:\n\n{coll.gd.propertiesTable}')

    # write results to file
    if output:
        with fsspec.open(output, 'wt', encoding='utf8', newline='') as f:
            json.dump(coll.gd.properties, f)


# download command
@cli.command(cls=_EECommand, epilog='See https://geedim.readthedocs.io/ for more details.')
@image_ids_option
@crs_option
@bbox_option
@region_option
@buffer_option
@scale_option
@crs_transform_option
@shape_option
@like_option
@dtype_option
@band_name_option
@mask_option
@resampling_option
@scale_offset_option
@split_option
@click.option(
    '-n/-nn',
    '--nodata/--no-nodata',
    type=click.BOOL,
    default=True,
    show_default=True,
    help='Set the nodata tag of downloaded file(s) to the --dtype dependent value provided by '
    'Earth Engine (--nodata), or leave it unset (--no-nodata).',
)
@click.option(
    '-dv',
    '--driver',
    type=click.Choice(enums.Driver, case_sensitive=True),
    default=enums.Driver.gtiff,
    show_default=True,
    help='Format driver for the downloaded file(s).',
)
@click.option(
    '-mts',
    '--max-tile-size',
    type=click.FLOAT,
    default=Tiler._default_max_tile_size,
    show_default=True,
    help='Maximum download tile size (MB).',
)
@click.option(
    '-mtd',
    '--max-tile-dim',
    type=click.INT,
    default=Tiler._ee_max_tile_dim,
    show_default=True,
    help='Maximum download tile dimension (pixels).',
)
@click.option(
    '-mr',
    '--max-requests',
    type=click.INT,
    default=Tiler._max_requests,
    show_default=True,
    help='Maximum concurrent tile downloads.',
)
@click.option(
    '-mc',
    '--max-cpus',
    type=click.INT,
    default=None,
    show_default='auto',
    help='Maximum number of tiles to decompress concurrently.',
)
@click.option(
    '-dd',
    '--download-dir',
    type=click.Path(file_okay=False),
    default=str(Path.cwd()),
    show_default='current working',
    callback=_dir_cb,
    help='Path / URI of the download directory.',
)
@click.option(
    '-o',
    '--overwrite',
    is_flag=True,
    default=False,
    help='Overwrite destination file(s) if they exist.',
)
@click.pass_obj
def download(
    obj: dict[str, Any],
    image_ids: tuple[str],
    split: enums.SplitType,
    nodata: bool,
    driver: enums.Driver,
    max_tile_size: float,
    max_tile_dim: int,
    max_requests: int,
    max_cpus: int,
    download_dir: OpenFile,
    overwrite: bool,
    **kwargs,
):
    """
    Export image(s) to GeoTIFF file(s).

    Images piped from previous commands will be exported, in addition to any images specified
    with --id.  All input images are piped out of this command for use by subsequent commands.

    Exported images include a fill (validity) mask band, and cloud/shadow related bands when
    supported.

    Images are retrieved as separate tiles which are downloaded and decompressed
    concurrently.  Tile size can be controlled with --max-tile-size and --max-tile-dim, and
    download / decompress concurrency with --max-requests and --max-cpus.

    Files are named with the Earth Engine index of their source image, and file band descriptions
    set to image band names with --split images.  Otherwise, files are named with their band
    name, and file band descriptions set to the Earth Engine index of the band's source image
    with --split bands.
    """
    obj['images'] = _get_images(obj, image_ids=image_ids)
    coll = _prepare_export_collection(obj['images'], cloud_kwargs=obj.get('cloud_kwargs'), **kwargs)

    tqdm.write('\nDownloading:\n')
    try:
        coll.gd.toGeoTIFF(
            download_dir,
            overwrite=overwrite,
            split=split,
            nodata=nodata,
            driver=driver,
            max_tile_size=max_tile_size,
            max_tile_dim=max_tile_dim,
            max_requests=max_requests,
            max_cpus=max_cpus,
        )
    except FileExistsError as ex:
        raise click.UsageError(str(ex)) from ex


# export command
@cli.command(cls=_EECommand, epilog='See https://geedim.readthedocs.io/ for more details.')
@image_ids_option
@click.option(
    '-t',
    '--type',
    type=click.Choice(enums.ExportType, case_sensitive=True),
    default=ImageAccessor._default_export_type.value,
    show_default=True,
    help='Export type.',
)
@click.option(
    '-f',
    '-df',
    '--folder',
    '--drive-folder',
    type=click.STRING,
    default=None,
    help='Google Drive folder, Earth Engine asset project, or Google Cloud Storage bucket to '
    'export image(s) to.  Can include sub-folders.  Interpretation based on --type.',
)
@crs_option
@bbox_option
@region_option
@scale_option
@crs_transform_option
@shape_option
@like_option
@dtype_option
@band_name_option
@mask_option
@resampling_option
@scale_offset_option
@split_option
@click.option(
    '-w/-nw',
    '--wait/--no-wait',
    default=True,
    show_default=True,
    help='Whether to wait for the export to complete.',
)
@click.pass_obj
def export(
    obj: dict[str, Any],
    image_ids: tuple[str],
    type: enums.ExportType,
    folder: str,
    split: enums.SplitType,
    wait: bool,
    **kwargs,
):
    """
    Export image(s) to Google cloud platforms.

    Images piped from previous commands will be exported, in addition to any images specified
    with --id.  All input images are piped out of this command for use by subsequent commands.

    Exported images include a fill (validity) mask band, and cloud/shadow related bands when
    supported.

    Images are retrieved as separate tiles which are downloaded and decompressed
    concurrently.  Tile size can be controlled with --max-tile-size and --max-tile-dim, and
    download / decompress concurrency with --max-requests and --max-cpus.

    Files are named with the Earth Engine index of their source image, and file band descriptions
    set to image band names with --split images.  Otherwise, files are named with their band
    name, and file band descriptions set to the Earth Engine index of the band's source image
    with --split bands.
    """
    if (type in [enums.ExportType.asset, enums.ExportType.cloud]) and not folder:
        raise click.MissingParameter(param_hint="'-f' / '--folder'", param_type='option')

    obj['images'] = _get_images(obj, image_ids=image_ids)
    coll = _prepare_export_collection(obj['images'], cloud_kwargs=obj.get('cloud_kwargs'), **kwargs)

    tqdm.write('\nExporting:\n')
    try:
        coll.gd.toGoogleCloud(type=type, folder=folder, wait=wait, split=split)
    except (ee.EEException, OSError) as ex:
        raise click.ClickException(str(ex)) from ex


# composite command
@cli.command(cls=_EECommand, epilog='See https://geedim.readthedocs.io/ for more details.')
@click.option(
    '-i',
    '--id',
    'image_ids',
    type=click.STRING,
    multiple=True,
    help='Earth Engine ID(s) of the component image(s).',
)
@click.option(
    '-cm',
    '--method',
    'method',
    type=click.Choice(enums.CompositeMethod, case_sensitive=False),
    default=None,
    show_default="'q-mosaic' for cloud/shadow supported collections, 'mosaic' otherwise.",
    help="Compositing method.",
)
@click.option(
    '-m/-nm',
    '--mask/--no-mask',
    default=True,
    show_default=True,
    help='Whether to mask component images.  Cloud/shadow masks are used when supported, otherwise '
    'fill masks.',
)
@click.option(
    '-rs',
    '--resampling',
    type=click.Choice(enums.ResamplingMethod, case_sensitive=True),
    default=ImageAccessor._default_resampling,
    show_default=True,
    help='Resampling method for component images.',
)
@click.option(
    '-b',
    '--bbox',
    type=click.FLOAT,
    nargs=4,
    default=None,
    callback=_bbox_cb,
    expose_value=False,  # callback exposes 'geometry' parameter for the command
    metavar='LEFT BOTTOM RIGHT TOP',
    help="Prioritise component images by their cloudless / filled portion inside these bounds.  "
    "Valid for the 'mosaic' and 'q-mosaic' --method.  Use '-' to read from previous --bbox / "
    "--region options in the pipeline.",
)
@click.option(
    '-r',
    '--region',
    type=click.Path(dir_okay=False),
    default=None,
    callback=_region_cb,
    expose_value=False,  # callback exposes 'geometry' parameter for the command
    help="Prioritise component images by their cloudless / filled portion inside the bounds "
    "defined by this GeoJSON / georeferenced image file.  Valid for the 'mosaic' and "
    "'q-mosaic' --method.  Use '-' to read from previous --bbox / --region options in the "
    "pipeline.",
)
@click.option(
    '-d',
    '--date',
    type=click.DateTime(),
    help="Prioritise component images closest to this date (UTC).  Valid for the 'mosaic' and "
    "'q-mosaic' --method.",
)
@click.pass_obj
def composite(
    obj: dict[str, Any],
    image_ids: tuple[str],
    geometry: ee.Geometry | None,
    **kwargs,
):
    """
    Create a composite image.

    Images piped from previous commands are used to create the composite, in addition to any
    images specified with --id.  The composite image is piped out of this command for use by
    subsequent commands.

    Input images should belong to the same collection or to spectrally compatible Landsat
    collections i.e. Landsat-4 with Landsat-5, or Landsat-8 with Landsat-9.

    When supported, the default is to cloud/shadow mask input images.  This can be turned off
    with --no-mask.  The 'q-mosaic' --method uses distance to the nearest cloud as the quality
    measure and requires cloud/shadow support.
    """
    images = _get_images(obj, image_ids)
    coll = ee.ImageCollection.gd.fromImages(images)
    comp_im = coll.gd.composite(
        region=geometry,
        **kwargs,
        **obj.get('cloud_kwargs', {}),
    )

    obj['images'] = [comp_im]


if __name__ == '__main__':
    cli()
