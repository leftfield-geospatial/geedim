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

import logging
from datetime import timedelta, datetime

import ee
import pandas
import rasterio as rio
from rasterio.warp import transform_geom

from geedim import download
from geedim import get_logger

# from shapely import geometry

##
logger = get_logger(__name__)


def get_composite_image(self):
    """
    Create a median composite image from search results

    Returns
    -------
    : ee.Image
    Composite image
    """
    if self._im_collection is None or self._im_df is None:
        raise Exception('First generate valid search results with search(...) method')

    comp_image = self._im_collection.median()

    # set metadata to indicate component images
    return comp_image.set('COMPOSITE_IMAGES', self._im_df[['ID', 'DATE'] + self._im_props].to_string())


def get_composite_image(self):
    """
    Create a composite image from search results, favouring pixels with the highest quality score

    Returns
    -------
    : ee.Image
    Composite image
    """
    if self._im_collection is None:
        raise Exception('First generate a valid image collection with search(...) method')

    comp_im = self._im_collection.qualityMosaic('QA_SCORE')

    # set metadata to indicate component images
    return comp_im.set('COMPOSITE_IMAGES', self._im_df[['ID', 'DATE'] + self._im_props].to_string()).toUint16()


def get_composite_image(self):
    """
    Create a composite image from search results

    Returns
    -------
    : ee.Image
    Composite image
    """
    if self._im_collection is None:
        raise Exception('First generate a valid image collection with search(...) method')

    if self._apply_valid_mask is None:
        logger.warning('Calling search(...) with apply_mask=True is recommended composite creation')

    comp_im = self._im_collection.mosaic()

    # set metadata to indicate component images
    return comp_im.set('COMPOSITE_IMAGES', self._im_df[['ID', 'DATE'] + self._im_props].to_string()).toUint16()


def get_composite_image(self):
    """
    Create a composite image from search results, favouring pixels with the highest quality score

    Returns
    -------
    : ee.Image
    Composite image
    """
    if self._im_collection is None:
        raise Exception('First generate a valid image collection with search(...) method')

    if self._apply_valid_mask is None:
        logger.warning('Calling search(...) with apply_mask=True is recommended for composite creation')

    comp_im = self._im_collection.mosaic()

    # set metadata to indicate component images
    return comp_im.set('COMPOSITE_IMAGES', self._im_df[['ID', 'DATE'] + self._im_props].to_string()).toUint16()
