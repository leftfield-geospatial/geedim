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

from geedim import schema, utils, version
from geedim.enums import (
    CloudMaskMethod,
    CloudScoreBand,
    CompositeMethod,
    Driver,
    ExportType,
    ResamplingMethod,
)
from geedim.image import ImageAccessor, _nodata_vals
from geedim.tile import Tiler

logger = logging.getLogger(__name__)


def _convert_bounds_to_geojson(bounds: Sequence[float]) -> dict[str, Any]:
    """Convert WGS84 geographic coordinate bounds to a GeoJSON polygon."""
    return dict(
        type='Polygon',
        coordinates=[
            [
                [bounds[0], bounds[1]],
                [bounds[2], bounds[1]],
                [bounds[2], bounds[3]],
                [bounds[0], bounds[3]],
                [bounds[0], bounds[1]],
            ]
        ],
    )


def _collection_cb(ctx: click.Context, param: click.Parameter, name: str) -> str:
    """click callback to convert abbreviated to full collection name."""
    return schema.gd_to_ee.get(name, name)


def _crs_cb(ctx: click.Context, param: click.Parameter, crs: str) -> str | None:
    """click callback to read --crs."""
    if crs:
        crs_path = Path(crs)
        try:
            if crs_path.suffix.lower() in ['.tif', '.tiff']:
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


def _bbox_cb(
    ctx: click.Context, param: click.Parameter, bounds: Sequence[float]
) -> dict[str, Any] | None:
    """click callback to convert --bbox to a GeoJSON polygon."""
    if bounds:
        bounds = _convert_bounds_to_geojson(bounds)
    return bounds


def _region_cb(ctx: click.Context, param: click.Parameter, region: str) -> dict[str, Any] | None:
    """click callback to read --region and convert to an GeoJSON polygon."""
    if region:
        try:
            if region.lower().endswith('json'):
                with fsspec.open(region, 'rt', encoding='utf-8') as f:
                    region = json.load(f)
            else:
                # TODO: add Env that prevents searching for sidecar files?
                with rio.open(fsspec.open(region, 'rb'), 'r') as ds:
                    bounds = transform_bounds(ds.crs, 'EPSG:4326', *ds.bounds)
                    region = _convert_bounds_to_geojson(bounds)
        except Exception as ex:
            raise click.BadParameter(str(ex)) from ex
    return region


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
    # TODO: test that only the metadata is read from e.g. a large remote file
    if uri_path:
        try:
            with rio.open(fsspec.open(uri_path, 'rb'), 'r') as ds:
                if not ds.crs:
                    raise click.BadParameter('Image is not georeferenced.')
                return dict(crs=ds.crs.to_wkt(), crs_transform=ds.transform[:6], shape=ds.shape)
        except Exception as ex:
            raise click.BadParameter(str(ex)) from ex
    return None


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


def _get_im_name(im: ee.Image) -> str:
    """Return a name for the given image."""
    return im.gd.id.replace('/', '-')


def _get_images(obj: dict[str, Any], image_ids: tuple[str]) -> list[ee.Image]:
    """Return a list of images in the click context object combined with images created from the EE
    IDs.
    """
    images = [ee.Image(ee_id) for ee_id in image_ids] + obj.get('images', [])
    if len(images) == 0:
        raise click.UsageError(
            "No images - either supply '--id' / '-i', or chain this command after 'search'."
        )
    return images


def _prepare_obj_for_export(
    obj: dict[str, Any],
    image_ids: tuple[str] = (),
    bbox: dict[str, Any] | None = None,
    region: dict[str, Any] | None = None,
    like: dict[str, Any] | None = None,
    mask: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Updates the click context object with export-ready images and export region."""
    utils.Initialize()
    # resolve and store region
    obj['region'] = bbox or region or obj.get('region')

    if like:
        # overwrite crs, crs_transform and shape from --like
        kwargs.update(**like)

    # prepare and store images
    obj['images'] = _get_images(obj, image_ids)

    for i, im in enumerate(obj['images']):
        if mask:
            # TODO: previously mask bands were always added
            im = im.gd.addMaskBands(**obj.get('cloud_kwargs', {})).gd.maskClouds()
        im = im.gd.prepareForExport(region=obj['region'], **kwargs)
        obj['images'][i] = im

    return obj


# Define click options that are common to more than one command
# TODO: define separate help for search --bbox and --region... and composite :/.  maybe not.
bbox_option = click.option(
    '-b',
    '--bbox',
    type=click.FLOAT,
    nargs=4,
    default=None,
    show_default='source image bounds',
    callback=_bbox_cb,
    metavar='LEFT BOTTOM RIGHT TOP',
    help='Bounds of the export image(s) in WGS84 geographic coordinates.',
)
region_option = click.option(
    '-r',
    '--region',
    type=click.Path(dir_okay=False),
    default=None,
    callback=_region_cb,
    help='Path / URI of a GeoJSON or georeferenced image file defining the export bounds.',
)
# TODO: implement buffering
buffer_option = click.option(
    '-b',
    '--buffer',
    type=click.FLOAT,
    default=0,
    show_default=True,
    help='Distance (m) to buffer --region / --bbox by.',
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
    show_default='auto',
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
    type=click.Choice(ResamplingMethod, case_sensitive=True),
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
# TODO: move "use with --shape.." help to command docstring
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


# geedim CLI and chained command group
@click.group(chain=True)
@click.option('--verbose', '-v', count=True, help="Increase verbosity.")
@click.option('--quiet', '-q', count=True, help="Decrease verbosity.")
@click.version_option(version=version.__version__, message='%(version)s')
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: int):
    """Search, composite and download Google Earth Engine imagery."""
    ctx.obj = {}
    _configure_logging(verbose - quiet)


# TODO: add clear docs on what is piped out of or into each command.
# TODO: use cloup and linear help layout & move lists of option values from command docsting to corresponding option
#  help


# config command
@cli.command()
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
    type=click.Choice(CloudMaskMethod, case_sensitive=True),
    default=CloudMaskMethod.cloud_score,
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
    type=click.Choice(CloudScoreBand, case_sensitive=True),
    default=CloudScoreBand.cs,
    show_default=True,
    help="Cloud Score+ band to threshold. Valid for Sentinel-2 images with the 'cloud-score' "
    '--mask-method.',
)
@click.pass_context
def config(ctx: click.Context, **kwargs):
    """
    Configure cloud/shadow masking.

    Chain this command with one or more other command(s) to configure cloud/shadow masking for
    those operation(s). A sensible default configuration is used if this command is not specified.

    Cloud/shadow masking is supported for the collections:
    \b

        ===========  ===========================
        geedim name  EE name
        ===========  ===========================
        l4-c2-l2     LANDSAT/LT04/C02/T1_L2
        l5-c2-l2     LANDSAT/LT05/C02/T1_L2
        l7-c2-l2     LANDSAT/LE07/C02/T1_L2
        l8-c2-l2     LANDSAT/LC08/C02/T1_L2
        l9-c2-l2     LANDSAT/LC09/C02/T1_L2
        s2-toa       COPERNICUS/S2
        s2-sr        COPERNICUS/S2_SR
        s2-toa-hm    COPERNICUS/S2_HARMONIZED
        s2-sr-hm     COPERNICUS/S2_SR_HARMONIZED
        ===========  ===========================
    \b

    For Sentinel-2 collections, ``--mask-method`` can be one of:
    \b

        * | `cloud-prob`: Threshold the Sentinel-2 Cloud Probability.
        * | `qa`: Bit mask the `QA60` quality assessment band.
        * | `cloud-score`: Threshold the Sentinel-2 Cloud Score+.
    \b

    Examples
    --------

    Search the Sentinel-2 SR collection for images with a cloudless portion of at least 60%,
    where cloud/shadow is identified with the `qa` ``mask-method``::

        geedim config --mask-method qa search -c s2-sr --cloudless-portion 60 -s 2022-01-01 -e 2022-01-14 --bbox 24 -34 24.5 -33.5

    Download and cloud/shadow mask a Landsat-8 image, where shadows are excluded from the mask::

        geedim config --no-mask-shadows download -i LANDSAT/LC08/C02/T1_L2/LC08_172083_20220104 --mask --bbox 24.25 -34 24.5 -33.75
    """  # noqa: E501
    # store configuration in the context object for use by other commands (only include command
    # line options so that the mask._CloudlessImage._get_mask_bands() does not get unexpected
    # default value kwargs)
    ctx.obj['cloud_kwargs'] = {
        k: v
        for k, v in ctx.params.items()
        if ctx.get_parameter_source(k) == ParameterSource.COMMANDLINE
    }


# search command
@cli.command()
@click.option(
    '-c',
    '--collection',
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
    show_default='one day after --start-date',
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
# TODO: use 'supply/supplied' not 'specified'
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
@click.pass_obj
def search(
    obj: dict[str, Any],
    collection: str,
    bbox: dict[str, Any] | None,
    region: dict[str, Any] | None,
    output: str | None,
    add_props: tuple[str] | None,
    **kwargs,
):
    """
    Search for images.

    Search a Google Earth Engine image collection for images, based on date, region, portion of
    filled pixels in region, and custom filters.  Filtering on cloud/shadow-free (cloudless)
    portion of filled pixels is supported on the following collections:
    \b

        ===========  ===========================
        geedim name  EE name
        ===========  ===========================
        l4-c2-l2     LANDSAT/LT04/C02/T1_L2
        l5-c2-l2     LANDSAT/LT05/C02/T1_L2
        l7-c2-l2     LANDSAT/LE07/C02/T1_L2
        l8-c2-l2     LANDSAT/LC08/C02/T1_L2
        l9-c2-l2     LANDSAT/LC09/C02/T1_L2
        s2-toa       COPERNICUS/S2
        s2-sr        COPERNICUS/S2_SR
        s2-toa-hm    COPERNICUS/S2_HARMONIZED
        s2-sr-hm     COPERNICUS/S2_SR_HARMONIZED
        ===========  ===========================

    The search must be filtered with at least one of the ``--start-date``, ``--bbox`` or
    ``--region`` options.

    Note that filled/cloudless portions are not taken from the granule metadata, but are
    calculated as portions inside the specified search region for improved accuracy.  These
    portions are only found and reported when one or both of ``--fill-portion`` /
    ``--cloudless-portion`` are specified.

    Search speed can be improved by specifying ``--custom-filter``, and or by omitting
    ``--fill-portion`` / ``--cloudless-portion``.
    \b

    Examples
    --------

    Search the GEDI canopy height collection for images with a filled portion of at least 0.5%::

        geedim search -c LARSE/GEDI/GEDI02_A_002_MONTHLY -s 2021-12-01 -e 2022-02-01 --bbox 23 -34 23.2 -33.8 --fill-portion 0.5

    Search the Landsat-9 collection for images, reporting the cloud/shadow free portion::

        geedim search -c l9-c2-l2 -s 2022-01-01 -e 2022-03-01 --bbox 23 -34 23.2 -33.8 --cloudless-portion

    Search the Landsat-8 collection for images whose `CLOUD_COVER_LAND` property is less than
    50%, and include the `CLOUD_COVER_LAND`, and `CLOUD_COVER` image properties in the search
    results::

        geedim search -c l8-c2-l2 -s 2022-01-01 -e 2022-05-01 --bbox 23 -34 23.2 -33.8 -cf "CLOUD_COVER_LAND<50" -ap CLOUD_COVER_LAND -ap CLOUD_COVER
    """  # noqa: E501
    # resolve region and store for chained commands
    obj['region'] = bbox or region or obj.get('region')

    for option, option_str in zip(
        ['fill_portion', 'cloudless_portion'],
        ["'-fp' / '--fill-portion'", "'-cp' / '--cloudless-portion'"],
    ):
        # TODO: refactor error msgs to follow oty
        if option in kwargs and not obj['region']:
            raise click.UsageError(
                f"'-b' / '--bbox' or '-r' / '--region' is required with {option_str}"
            )

    # create collection and search
    utils.Initialize()
    coll = ee.ImageCollection(collection)
    label = f'Searching for {collection} images: '
    with utils.Spinner(label=label, leave=' '):
        coll = coll.gd.filter(region=obj['region'], **kwargs, **obj.get('cloud_kwargs', {}))

        # retrieve search result properties from EE
        coll.gd.schemaPropertyNames += add_props
        num_images = len(coll.gd.properties)

        # store images for chained commands
        images = coll.toList(num_images)
        obj['images'] = obj.get('images', []) + [
            ee.Image(images.get(i)) for i in (range(num_images))
        ]

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
@cli.command()
@image_ids_option
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
@click.option(
    '-dd',
    '--download-dir',
    type=click.Path(file_okay=False),
    default=str(Path.cwd()),
    show_default='current working',
    callback=_dir_cb,
    help='Path / URI of the download directory.',
)
# TODO: add nodata option
@click.option(
    '-dv',
    '--driver',
    type=click.Choice(Driver, case_sensitive=True),
    default=Driver.gtiff,
    show_default=True,
    help='Format driver for the download file.',
)
@click.option(
    '-mts',
    '--max-tile-size',
    type=click.FLOAT,
    default=Tiler._ee_max_tile_size,
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
    '-o',
    '--overwrite',
    is_flag=True,
    default=False,
    help='Overwrite the destination file if it exists.',
)
@click.pass_obj
def download(
    obj: dict[str, Any],
    download_dir: OpenFile,
    driver: Driver,
    max_tile_size: float,
    max_tile_dim: float,
    overwrite: bool,
    **kwargs,
):
    """
    Download image(s).

    Download Earth Engine image(s) to GeoTIFF file(s), allowing optional region of interest,
    and other image formatting options to be specified.  Images larger than the `Earth Engine
    size limit <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`_ are
    split and downloaded as separate tiles, then re-assembled into a single GeoTIFF.  Downloaded
    image files are populated with metadata from the Earth Engine image and STAC.

    This command can be chained after the ``composite`` command, to download the composite image,
    or it can be chained after an asset ``export`` to download the asset image.  It can also be
    chained after the ``search`` command, in which case the search result images will be
    downloaded, without the need to specify image IDs with ``--id``, or region with ``--bbox`` /
    ``--region``.

    The following auxiliary bands are added to images from collections with support for
    cloud/shadow masking:
    \b

        ==============  =========================================
        Band name       Description
        ==============  =========================================
        FILL_MASK       Mask of filled (valid) pixels.
        SHADOW_MASK     Mask of cloud shadows.
        CLOUD_MASK      Mask of clouds.
        CLOUDLESS_MASK  Mask of filled & cloud/shadow-free pixels.
        CLOUD_DIST      Distance to nearest cloud (10m).
        ==============  =========================================

    Images from other collections, will contain the FILL_MASK band only.

    Bounds and resolution of the downloaded image can be specified with ``--region`` / ``--bbox``
    and ``--scale`` / ``--shape``, or ``--crs-transform`` and ``--shape``.  The ``--like`` option
    will automatically derive ``--crs``, ``--crs-transform`` and ``--shape`` from a provided
    template raster.  If no bounds are specified (with either ``--region``,
    or ``--crs-transform`` & ``--shape``), the entire image granule is downloaded.

    When ``--crs``, ``--scale``, ``--crs-transform`` and ``--shape`` are not specified, the pixel
    grids of the downloaded and Earth Engine images will coincide.

    Image filenames are derived from their Earth Engine ID.
    \b

    Examples
    --------

    Download a region of a Landsat-9 image, applying the cloud/shadow mask and converting to
    uint16::

        geedim download -i LANDSAT/LC09/C02/T1_L2/LC09_173083_20220308 --mask --bbox 21.6 -33.5 21.7 -33.4 --dtype uint16

    Download the results of a MODIS NBAR search, specifying a CRS and scale to reproject to::

        geedim search -c MODIS/006/MCD43A4 -s 2022-01-01 -e 2022-01-03 --bbox 23 -34 24 -33 download --crs EPSG:3857 --scale 500
    """  # noqa: E501
    obj = _prepare_obj_for_export(obj, **kwargs)

    tqdm.write('\nDownloading:\n')
    for im in obj['images']:
        joined_path = posixpath.join(download_dir.path, _get_im_name(im) + '.tif')
        ofile = OpenFile(download_dir.fs, joined_path, mode='wb')
        try:
            im.gd.toGeoTIFF(
                ofile,
                overwrite=overwrite,
                driver=driver,
                max_tile_size=max_tile_size,
                max_tile_dim=max_tile_dim,
            )
        except FileExistsError as ex:
            raise click.UsageError(str(ex)) from ex


# export command
@cli.command()
@image_ids_option
@click.option(
    '-t',
    '--type',
    type=click.Choice(ExportType, case_sensitive=True),
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
    type: ExportType,
    folder: str,
    wait: bool,
    **kwargs,
):
    """
    Export image(s).

    Export Earth Engine image(s) to Google Drive, Earth Engine asset, or Google Cloud Storage,
    allowing optional region of interest, and other image formatting options to be specified.

    This command can be chained after the ``composite`` command, to export the composite image.
    It can also be chained after the ``search`` command, in which case the search result images
    will be exported, without the need to specify image IDs with ``--id``, or region with
    ``--bbox`` / ``--region``.

    The following auxiliary bands are added to images from collections with support for
    cloud/shadow masking:
    \b

        ==============  =========================================
        Band name       Description
        ==============  =========================================
        FILL_MASK       Mask of filled (valid) pixels.
        SHADOW_MASK     Mask of cloud shadows.
        CLOUD_MASK      Mask of clouds.
        CLOUDLESS_MASK  Mask of filled & cloud/shadow-free pixels.
        CLOUD_DIST      Distance to nearest cloud (10m).
        ==============  =========================================

    Images from other collections, will contain the FILL_MASK band only.

    Bounds and resolution of the exported image can be specified with ``--region`` / ``--bbox``
    and ``--scale`` / ``--shape``, or ``--crs-transform`` and ``--shape``.  The ``--like`` option
    will automatically derive ``--crs``, ``--crs-transform`` and ``--shape`` from a provided
    template raster.  If no bounds are specified (with either ``--region``,
    or ``--crs-transform`` & ``--shape``), the entire image granule is exported.

    When ``--crs``, ``--scale``, ``--crs-transform`` and ``--shape`` are not specified, the pixel
    grids of the exported and Earth Engine images will coincide.

    Image file or asset names are derived from their Earth Engine ID.
    \b

    Examples
    --------

    Export a region of a Landsat-9 image to an Earth Engine asset, applying the cloud/shadow mask
    and converting to uint16::

        geedim export -i LANDSAT/LC09/C02/T1_L2/LC09_173083_20220308 --type asset --folder <your cloud project> --mask --bbox 21.6 -33.5 21.7 -33.4 --dtype uint16

    Export the results of a MODIS NBAR search to Google Drive in the 'geedim' folder, specifying
    a CRS and scale to reproject to::

        geedim search -c MODIS/006/MCD43A4 -s 2022-01-01 -e 2022-01-03 --bbox 23 -34 24 -33 export --crs EPSG:3857 --scale 500 -df geedim
    """  # noqa: E501
    if (type in [ExportType.asset, ExportType.cloud]) and not folder:
        raise click.MissingParameter(param_hint="'-f' / '--folder'", param_type='option')

    obj = _prepare_obj_for_export(obj, **kwargs)

    tqdm.write('\nExporting:\n')
    export_tasks = []
    for im in obj['images']:
        name = _get_im_name(im)
        try:
            task = im.gd.toGoogleCloud(name, type=type, folder=folder, wait=False)
        except ee.EEException as ex:
            raise click.ClickException(str(ex)) from ex
        export_tasks.append(task)
        tqdm.write(f'Started {name}') if not wait else None

    # TODO: should images be piped out in all cases?
    if wait:
        if type is ExportType.asset:
            # replace context object images with asset images so they can be downloaded in a
            # chained command
            obj['images'] = [
                ee.Image(utils.asset_id(_get_im_name(im), folder)) for im in obj['images']
            ]

        # wait for export tasks
        for task in export_tasks:
            try:
                ImageAccessor.monitorExport(task)
            except (ee.EEException, OSError) as ex:
                raise click.ClickException(str(ex)) from ex


# composite command
@cli.command()
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
    type=click.Choice(CompositeMethod, case_sensitive=False),
    default=None,
    show_default="'q-mosaic' for cloud/shadow supported collections, 'mosaic' otherwise.",
    help='Compositing method.',
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
    type=click.Choice(ResamplingMethod, case_sensitive=True),
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
    metavar='LEFT BOTTOM RIGHT TOP',
    help="Prioritise component images by their cloudless / filled portion inside these bounds.  "
    "Valid for the 'mosaic' and 'q-mosaic' compositing --method.",
)
@click.option(
    '-r',
    '--region',
    type=click.Path(dir_okay=False),
    default=None,
    callback=_region_cb,
    help="Prioritise component images by their cloudless / filled portion inside the region / "
    "bounds defined by this GeoJSON / georeferenced image file.  Valid for the 'mosaic' and "
    "'q-mosaic' --method.",
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
    image_ids: Sequence[str],
    bbox: dict[str, Any] | None,
    region: dict[str, Any] | None,
    **kwargs,
):
    """
    Create a composite image.

    Create cloud/shadow-free and other composite image(s) from specified input images.

    ``download`` or ``export`` commands can be chained after the ``composite`` command to
    download/export the composite image. ``composite`` can also be chained after ``search``,
    ``download`` or ``composite``, in which case it will composite the output image(s) from the
    previous command.  Images specified with the ``--id`` option will be added to any existing
    chained images i.e. images output from previous chained commands.

    In general, input images should belong to the same collection.  In the specific case of
    Landsat, images from spectrally compatible collections can be combined i.e. Landsat-4 with
    Landsat-5, and Landsat-8 with Landsat-9.

    ``--method`` specifies the method for finding a composite pixel from the stack of
    corresponding input image pixels.  The following options are available:
    \b

        ==========  ========================================================
        Method      Description
        ==========  ========================================================
        `q-mosaic`  | Use the unmasked pixel with the highest cloud distance
                    | (i.e. distance to nearest cloud).  Where more than one
                    | pixel has the same cloud distance, the first one in the
                    | stack is selected.
        `mosaic`    Use the first unmasked pixel in the stack.
        `medoid`    | Use the medoid of the unmasked pixels i.e. the pixel
                    | of the image with the minimum sum of spectral distances
                    | to the rest of the input images.
                    | Maintains relationship between bands.
        `median`    Use the median of the unmasked pixels.
        `mode`      Use the mode of the unmasked pixels.
        `mean`      Use the mean of the unmasked pixels.
        ==========  ========================================================

    For the `mosaic`, `q-mosaic` and `medoid` methods there are three ways of ordering (i.e.
    prioritising) images in the stack:
    \b

        * | If ``--date`` is specified, images are sorted by the absolute
          | difference of their capture time from this date.
        * | If either ``--region`` or ``--bbox`` are specified, images are sorted
          | by their cloudless/filled portion inside this region.
        * | If none of the above options are specified, images are sorted by their
          | capture time.

    By default, input images are masked before compositing.  This means that only
    cloud/shadow-free (or filled) pixels are used to make the composite.  You can turn off this
    behaviour with the ``--no-mask`` option.
    \b

    Examples
    --------
    Composite two Landsat-7 images using the default options and download the result::

        geedim composite -i LANDSAT/LE07/C02/T1_L2/LE07_173083_20100203 -i LANDSAT/LE07/C02/T1_L2/LE07_173083_20100219 download --bbox 22 -33.1 22.1 -33 --crs EPSG:3857 --scale 30

    Create and download a composite of a year of GEDI canopy height data, by chaining with
    ``search``::

        geedim search -c LARSE/GEDI/GEDI02_A_002_MONTHLY -s 2021-01-01 -e 2022-01-01 --bbox 23 -34 23.1 -33.9 --fill-portion 0.1 composite -cm mosaic download --crs EPSG:3857 --scale 25

    Create and download a cloud/shadow-free composite of Sentinel-2 SR images, by chaining with
    ``search``::

        geedim search -c s2-sr -s 2021-01-12 -e 2021-01-23 --bbox 23 -33.5 23.1 -33.4 composite -cm q-mosaic download --crs EPSG:3857 --scale 10
    """  # noqa: E501
    utils.Initialize()
    # TODO: should region be chained with e.g. search - it has a different meaning here

    images = _get_images(obj, image_ids)
    coll = ee.ImageCollection.gd.fromImages(images)
    comp_im = coll.gd.composite(
        region=bbox or region,
        **kwargs,
        **obj.get('cloud_kwargs', {}),
    )

    obj['images'] = [comp_im]


if __name__ == '__main__':
    cli()
