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

import itertools
import json
import logging
import os
import pathlib
import sys
import time
from threading import Thread
from typing import Tuple, Optional

import ee
import rasterio as rio
import requests
from geedim.enums import ResamplingMethod
from rasterio.warp import transform_geom
from requests.adapters import HTTPAdapter, Retry
import numpy as np
from rasterio.env import GDALVersion
from tqdm.auto import tqdm

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())

_GDAL_AT_LEAST_35 = GDALVersion.runtime().at_least("3.5")

def Initialize(opt_url: Optional[str]=None, **kwargs):
    """
    Initialise Earth Engine.

    Credentials will be read from the `EE_SERVICE_ACC_PRIVATE_KEY` environment variable if it exists
    (useful for integrating with e.g. GitHub actions).

    .. note::

        Earth Engine recommends using the `high volume endpoint` for applications like ``geedim``.
        See `the docs <https://developers.google.com/earth-engine/cloud/highvolume>`_ for more information.

    Parameters
    ----------
    opt_url: str
        The Earth Engine endpoint to use.  If ``None``, the default is used.
    kwargs
        Optional arguments to pass to `ee.Initialize`.
    """

    if not ee.data._credentials:
        # Adpated from https://gis.stackexchange.com/questions/380664/how-to-de-authenticate-from-earth-engine-api.
        env_key = 'EE_SERVICE_ACC_PRIVATE_KEY'

        if env_key in os.environ:
            # authenticate with service account
            key_dict = json.loads(os.environ[env_key])
            credentials = ee.ServiceAccountCredentials(key_dict['client_email'], key_data=key_dict['private_key'])
            ee.Initialize(credentials, opt_url=opt_url, **kwargs)
        else:
            ee.Initialize(opt_url=opt_url, **kwargs)


def singleton(cls):
    """ Class decorator to make it a singleton. """
    instances = {}

    def getinstance() -> cls:
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


def split_id(image_id: str) -> Tuple[str, str]:
    """
    Split Earth Engine image ID into collection and index components.

    Parameters
    ----------
    image_id: str
        Earth engine image ID.

    Returns
    -------
    tuple(str, str)
        A tuple of strings: (collection name, image index).
    """
    if not image_id:
        return None, None
    index = image_id.split("/")[-1]
    ee_coll_name = "/".join(image_id.split("/")[:-1])
    return ee_coll_name, index


def get_bounds(filename: pathlib.Path, expand: float = 5):
    """
    Get a geojson polygon representing the bounds of an image.

    Parameters
    ----------
    filename: str, pathlib.Path
        Path of the image file whose bounds to find.
    expand : int, optional
        Percentage (0-100) by which to expand the bounds (default: 5).

    Returns
    -------
    dict
        Geojson polygon.
    """
    try:
        # GEE sets tif colorinterp tags incorrectly, suppress rasterio warning relating to this:
        # 'Sum of Photometric type-related color channels and ExtraSamples doesn't match SamplesPerPixel'
        logging.getLogger("rasterio").setLevel(logging.ERROR)
        with rio.Env(GTIFF_FORCE_RGBA=False), rio.open(filename) as im:
            bbox = im.bounds
            if (im.crs.linear_units == "metre") and (expand > 0):  # expand the bounding box
                expand_x = (bbox.right - bbox.left) * expand / 100.0
                expand_y = (bbox.top - bbox.bottom) * expand / 100.0
                bbox_expand = rio.coords.BoundingBox(
                    bbox.left - expand_x, bbox.bottom - expand_y, bbox.right + expand_x, bbox.top + expand_y
                )
            else:
                bbox_expand = bbox

            coordinates = [
                [bbox_expand.right, bbox_expand.bottom], [bbox_expand.right, bbox_expand.top],
                [bbox_expand.left, bbox_expand.top], [bbox_expand.left, bbox_expand.bottom],
                [bbox_expand.right, bbox_expand.bottom],
            ]

            bbox_expand_dict = dict(type="Polygon", coordinates=[coordinates])
            src_bbox_wgs84 = transform_geom(im.crs, "WGS84", bbox_expand_dict)  # convert to WGS84 geojson
    finally:
        logging.getLogger("rasterio").setLevel(logging.WARNING)
    return src_bbox_wgs84


def get_projection(image, min_scale=True):
    """
    Get the min/max scale projection of image bands.  Server side - no calls to getInfo().
    Adapted from from https://github.com/gee-community/gee_tools, MIT license.

    Parameters
    ----------
    image : ee.Image
            Image whose min/max projection to retrieve.
    min_scale: bool, optional
         Retrieve the projection corresponding to the band with the minimum (True) or maximum (False) scale.
         (default: True)

    Returns
    -------
    ee.Projection
        Requested projection.
    """
    if not isinstance(image, ee.Image):
        raise TypeError('image is not an instance of ee.Image')

    bands = image.bandNames()

    compare = ee.Number.lte if min_scale else ee.Number.gte
    init_proj = image.select(0).projection()

    def compare_scale(name, prev_proj):
        """Server side comparison of band scales"""
        prev_proj = ee.Projection(prev_proj)
        prev_scale = prev_proj.nominalScale()

        curr_proj = image.select([name]).projection()
        curr_scale = ee.Number(curr_proj.nominalScale())

        condition = compare(curr_scale, prev_scale)
        comp_proj = ee.Algorithms.If(condition, curr_proj, prev_proj)
        return ee.Projection(comp_proj)

    return ee.Projection(bands.iterate(compare_scale, init_proj))


class Spinner(Thread):

    def __init__(self, label='', interval=0.2, leave=True, **kwargs):
        """
        Thread sub-class to run a non-blocking spinner.

        Parameters
        ----------
        label: str, optional
            Prepend spinner with this label.
        interval: float, optional
            Spinner update interval (s).
        leave: optional, bool, str
            What to do with the spinner display on stop():
                False: clear the label + spinner.
                True:  leave the label + spinner as is.
                <string message>: print this message in place of the spinner
        kwargs: optional
            Additional kwargs to pass to Thread.__init__()
        """
        Thread.__init__(self, **kwargs)
        self._label = label
        self._interval = interval
        self._run = True
        self._leave = leave
        self._file = sys.stderr

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join()

    def run(self):
        """ Run the spinner thread. """
        cursors_it = itertools.cycle('/-\|')

        while self._run:
            cursor = next(cursors_it)
            tqdm.write('\r' + self._label + cursor, file=self._file, end='')
            self._file.flush()
            time.sleep(self._interval)

        if self._leave == True:
            tqdm.write('', file=self._file, end='\n')
        elif self._leave == False:
            tqdm.write('\r', file=self._file, end='')
        elif isinstance(self._leave, str):
            tqdm.write('\r' + self._label + self._leave + ' ', file=self._file, end='\n')
        self._file.flush()

    def start(self):
        """ Start the spinner thread. """
        self._run = True
        Thread.start(self)

    def stop(self):
        """ Stop the spinner thread. """
        self._run = False


def resample(ee_image: ee.Image, method: ResamplingMethod) -> ee.Image:
    """
    Resample an ee.Image. Extends ee.Image.resample by only resampling when the image has a fixed projection, and by
    providing an additional 'average' method for downsampling.  This logic is performed server-side.

    See https://developers.google.com/earth-engine/guides/resample for more info.

    Parameters
    ----------
    ee_image: ee.Image
        Image to resample.
    method: ResamplingMethod
        Resampling method to use.

    Returns
    -------
    ee_image: ee.Image
        Resampled image.
    """
    # TODO : use STAC to only resample continuous qty type bands
    method = ResamplingMethod(method)
    if method == ResamplingMethod.near:
        return ee_image

    # resample the image, if it has a fixed projection
    proj = ee_image.select(0).projection()
    has_fixed_proj = proj.crs().compareTo('EPSG:4326').neq(0).Or(proj.nominalScale().toInt64().neq(111319))

    def _resample(ee_image: ee.Image) -> ee.Image:
        """ Resample the given image, allowing for additional 'average' method. """
        if method == ResamplingMethod.average:
            return ee_image.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        else:
            return ee_image.resample(method.value)

    return ee.Image(ee.Algorithms.If(has_fixed_proj, _resample(ee_image), ee_image))


def retry_session(
    retries: int = 3, backoff_factor: float = 0.3, status_forcelist: Tuple = (429, 500, 502, 503, 504),
    session: requests.Session = None
) -> requests.Session:
    """ requests session configured for retries. """
    session = session or requests.Session()
    retry = Retry(
        total=retries, read=retries, connect=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


