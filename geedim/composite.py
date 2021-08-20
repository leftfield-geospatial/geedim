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

from geedim import export

# from shapely import geometry

##


'''
# from https://github.com/saveriofrancini/bap/blob/dbdf44df5cdf54cfb0d8eafaa7eeae68f0312467/js/library.js#L145 
var calculateCloudWeightAndDist = function(imageWithCloudMask, cloudDistMax){

  var cloudM = imageWithCloudMask.select('cloudM').unmask(0).eq(0);
  var nPixels = ee.Number(cloudDistMax).divide(30).toInt();
  var cloudDist = cloudM.fastDistanceTransform(nPixels, "pixels",  'squared_euclidean');
  // fastDistanceTransform max distance (i.e. 50*30 = 1500) is approzimate. Correcting it...
  cloudDist = cloudDist.where(cloudDist.gt(ee.Image(cloudDistMax)), cloudDistMax);
  
  var deltaCloud = ee.Image(1).toDouble() .divide((ee.Image(ee.Number(-0.008))
  .multiply(cloudDist.subtract(ee.Number(cloudDistMax/2)))).exp().add(1))
  .unmask(1)
  .select([0], ['cloudScore']);
  
  cloudDist = ee.Image(cloudDist).int16().rename('cloudDist');

  var keys = ['cloudScore', 'cloudDist'];
  var values = [deltaCloud, cloudDist]; 
  
  return ee.Dictionary.fromLists(keys, values);
};
exports.calculateCloudWeightAndDist = calculateCloudWeightAndDist;



        pjeDist = ee.Image().expression('1-exp((-dist+dmin)/(dmax*factor))',
                                        {
                                            'dist': distance,
                                            'dmin': dmini,
                                            'dmax': dmaxi,
                                            'factor': factori
                                        }).rename(bandname)

'''

'''
#from https://github.com/gee-community/gee_tools/blob/master/geetools/composite.py

def medoidScore(collection, bands=None, discard_zeros=False,
                bandname='sumdist', normalize=True):
    """ Compute a score to reflect 'how far' is from the medoid. Same params
     as medoid() """
    first_image = ee.Image(collection.first())
    if not bands:
        bands = first_image.bandNames()

    # Create a unique id property called 'enumeration'
    enumerated = tools.imagecollection.enumerateProperty(collection)
    collist = enumerated.toList(enumerated.size())

    def over_list(im):
        im = ee.Image(im)
        n = ee.Number(im.get('enumeration'))

        # Remove the current image from the collection
        filtered = tools.ee_list.removeIndex(collist, n)

        # Select bands for medoid
        to_process = im.select(bands)

        def over_collist(img):
            return ee.Image(img).select(bands)
        filtered = filtered.map(over_collist)

        # Compute the sum of the euclidean distance between the current image
        # and every image in the rest of the collection
        dist = algorithms.sumDistance(
            to_process, filtered,
            name=bandname,
            discard_zeros=discard_zeros)

        # Mask zero values
        if not normalize:
            # multiply by -1 to get the lowest value in the qualityMosaic
            dist = dist.multiply(-1)

        return im.addBands(dist)

    imlist = ee.List(collist.map(over_list))

    medcol = ee.ImageCollection.fromImages(imlist)

    # Normalize result to be between 0 and 1
    if normalize:
        min_sumdist = ee.Image(medcol.select(bandname).min())\
                        .rename('min_sumdist')
        max_sumdist = ee.Image(medcol.select(bandname).max()) \
                        .rename('max_sumdist')

        def to_normalize(img):
            sumdist = img.select(bandname)
            newband = ee.Image().expression(
                '1-((val-min)/(max-min))',
                {'val': sumdist,
                 'min': min_sumdist,
                 'max': max_sumdist}
            ).rename(bandname)
            return tools.image.replace(img, bandname, newband)

        medcol = medcol.map(to_normalize)

    return medcol


def medoid(collection, bands=None, discard_zeros=False):
    """ Medoid Composite. Adapted from https://www.mdpi.com/2072-4292/5/12/6481
    :param collection: the collection to composite
    :type collection: ee.ImageCollection
    :param bands: the bands to use for computation. The composite will include
        all bands
    :type bands: list
    :param discard_zeros: Masked and pixels with value zero will not be use
        for computation. Improves dark zones.
    :type discard_zeros: bool
    :return: the Medoid Composite
    :rtype: ee.Image
    """
    medcol = medoidScore(collection, bands, discard_zeros)
    comp = medcol.qualityMosaic('sumdist')
    final = tools.image.removeBands(comp, ['sumdist', 'mask'])
    return final
'''

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
