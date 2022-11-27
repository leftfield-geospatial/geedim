"""
   Copyright 2022 Dugal Harris - dugalh@gmail.com

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
import ee
from enums import SpectralDistanceMetric
from typing import Optional, List
"""
    This module contains Medoid related functionality adapted from 'Google Earth Engine tools' under MIT 
    license.  See https://github.com/gee-community/gee_tools.
"""

def sum_distance(
    image: ee.Image, collection: ee.ImageCollection, bands: Optional[List] = None,
    metric: SpectralDistanceMetric = SpectralDistanceMetric.sed, omit_mask: bool = False,
) -> ee.Image:
    """ Find the sum of the euclidean spectral distances between the provided ``image`` and ``collection``. """
    if not bands:
        bands = collection.first().bandNames()

    image = ee.Image(image).select(bands)

    def accum_dist_to_image(to_image: ee.Image, sum_image: ee.Image) -> ee.Image:
        """
        Earth engine iterator function to find the sum of the euclidean spectral distances between ``image`` and
        ``to_image``.
        """
        # Notes on masking:
        # - Where ``image`` is masked, the summed distance should be masked
        # - Where any other image in ``collection`` is masked, the summed distance should omit its contribution.

        # unmask the other image so that it does not mask summed distance when added
        to_image = ee.Image(to_image).unmask(0).select(bands)

        # find the distance between image and unmask_to_image
        dist = image.spectralDistance(to_image, metric.value)
        if omit_mask:
            # zero distances where to_image is masked
            zero_mask = to_image.mask().reduce(ee.Reducer.allNonZero()).Not()
            dist = dist.where(zero_mask, 0)
        # return accumulated distance
        return ee.Image(sum_image).add(dist.unmask())

    sumdist = collection.iterate(accum_dist_to_image, ee.Image(0))
    # TODO: mask sumdist with image mask?
    return ee.Image(sumdist)


def medoid_score(
    collection: ee.ImageCollection, bands: Optional[List] = None, name: str = 'sumdist', omit_mask: bool = False,
) -> ee.ImageCollection:
    """
    Add medoid score band (i.e. summed distance to all other images) to all images in ``collection``.

    Parameters
    ----------
    collection: ee.ImageCollection
        Collection to add medoid score band to.
    bands: list of str, optional
        Bands to calculate the medoid score from (default: use all bands).
    name: str, optional
        Name of score band to add (default: 'sumdist').
    omit_mask: bool, optional
        Whether to omit the contribution of masked image pixels to the summed distance (default: False).

    Returns
    -------
    ee.ImageCollection
        Collection with added medoid score band.
    """

    def add_score_band(image: ee.Image):
        """ Add medoid score band to provided ``image``. """
        image = ee.Image(image)

        # Compute the sum of the euclidean distance between the current image
        # and every image in the rest of the collection
        # TODO: many (~50%) of these distance calcs are duplicates, can we make this more efficient?
        dist = sum_distance(image, collection, bands=bands, omit_mask=omit_mask)

        # multiply by -1 so that highest score is closest distance
        dist = dist.multiply(-1)
        return image.addBands(dist.rename(name))

    # TODO: can we avoid having two copies (selected and unselected) of the image and collection.  would it help
    #  speed things up and reduce mem?
    return collection.map(add_score_band)


def medoid(collection: ee.ImageCollection, bands: Optional[List] = None) -> ee.Image:
    """
    Find the medoid composite of an image collection. Adapted from https://www.mdpi.com/2072-4292/5/12/6481, and
    https://github.com/gee-community/gee_tools.

    Parameters
    ----------
    collection: ee.ImageCollection
        Image collection to composite.
    bands: list of str, optional
        Bands to calculate the medoid score from (default: use all bands).

    Returns
    -------
    ee.Image
        Medoid composite image.
    """
    name = 'sumdist'
    medoid_coll = medoid_score(collection, bands, name=name)
    comp_im = medoid_coll.qualityMosaic(name)
    # remove score band and return
    keep_bands = comp_im.bandNames().remove(name)
    return comp_im.select(keep_bands)
