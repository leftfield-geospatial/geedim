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
from geedim.enums import SpectralDistanceMetric
from typing import Optional, List
"""
    This module contains Medoid related functionality adapted from 'Google Earth Engine tools' under MIT 
    license.  See https://github.com/gee-community/gee_tools.
"""


def sum_distance(
    image: ee.Image, collection: ee.ImageCollection, bands: Optional[List] = None,
    metric: SpectralDistanceMetric = SpectralDistanceMetric.sed,
) -> ee.Image:
    """ Find the sum of the spectral distances between the provided ``image`` and image ``collection``. """
    metric = SpectralDistanceMetric(metric)
    if not bands:
        bands = collection.first().bandNames()
    image = ee.Image(image).select(bands)

    # Notes on masking:
    # - Where ``image`` is masked, the summed distance should be masked.
    # - Where any other image in ``collection`` is masked, the summed distance should omit its contribution.
    #
    # The above requirements are satisfied by leaving images masked, creating an ee.ImageCollection of distances
    # between ``image`` and other images in the collection (these distances are masked where either image is masked),
    # and using ImageCollection.sum() to sum distances, omitting masked areas from the sum.
    # The sum is only masked where all component distances are masked i.e. where ``image`` is masked.

    def accum_dist_to_image(to_image: ee.Image, dist_list: ee.List) -> ee.Image:
        """
        Earth engine iterator function to create a list of spectral distances between ``image`` and ``to_image``.
        """
        to_image = ee.Image(to_image).select(bands)

        # Find the distance between image and to_image.  Both images are not unmasked so that distance will be
        # masked where one or both are masked.
        dist = image.spectralDistance(to_image, metric.value)
        if metric == SpectralDistanceMetric.sed:
            # sqrt scaling is necessary for summing with other distances and equivalence to original method
            dist = dist.sqrt()

        # Append distance to list.
        return ee.List(dist_list).add(dist)

    dist_list = ee.List(collection.iterate(accum_dist_to_image, ee.List([])))
    # TODO: are we better off using mean here to avoid overflow?
    return ee.ImageCollection(dist_list).sum()


def medoid_score(
    collection: ee.ImageCollection, bands: Optional[List] = None, name: str = 'sumdist',
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
        dist = sum_distance(image, collection, bands=bands)

        # multiply by -1 so that highest score is lowest summed distance
        return image.addBands(dist.multiply(-1).rename(name))

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
