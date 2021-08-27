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
import click

import ee
import pandas
import rasterio as rio
from rasterio.warp import transform_geom

import geedim.collection
from geedim import search, cli, collection

# from shapely import geometry

##


# All below medoid related code taken from adapted from https://github.com/gee-community/gee_tools under MIT license
# TODO: rewrite / attribute properly
def enumerate(collection):
    """ Create a list of lists in which each element of the list is:
    [index, element]. For example, if you parse a FeatureCollection with 3
    Features you'll get: [[0, feat0], [1, feat1], [2, feat2]]
    :param collection: can be an ImageCollection or a FeatureCollection
    :return: ee.Collection
    """
    collist = collection.toList(collection.size())

    # first element
    ini = ee.Number(0)
    first_image = ee.Image(collist.get(0))
    first = ee.List([ini, first_image])

    start = ee.List([first])
    rest = collist.slice(1)

    def over_list(im, s):
        im = ee.Image(im)
        s = ee.List(s)
        last = ee.List(s.get(-1))
        last_index = ee.Number(last.get(0))
        index = last_index.add(1)
        return s.add(ee.List([index, im]))

    list = ee.List(rest.iterate(over_list, start))

    return list
def enumerateProperty(collection, name='enumeration'):
    """
    :param collection:
    :param name:
    :return:
    """
    enumerated = enumerate(collection)

    def over_list(l):
        l = ee.List(l)
        index = ee.Number(l.get(0))
        element = l.get(1)
        return ee.Image(element).set(name, index)

    imlist = enumerated.map(over_list)
    return ee.ImageCollection(imlist)

def empty(value=0, names=None, from_dict=None):
    """ Create a constant image with the given band names and value, and/or
    from a dictionary of {name: value}
    :param names: list of names
    :type names: ee.List or list
    :param value: value for every band of the resulting image
    :type value: int or float
    :param from_dict: {name: value}
    :type from_dict: dict
    :rtype: ee.Image
    """
    image = ee.Image.constant(0)
    bandnames = ee.List([])
    if names:
        bandnames = names if isinstance(names, ee.List) else ee.List(names)
        def bn(name, img):
            img = ee.Image(img)
            newi = ee.Image(value).select([0], [name])
            return img.addBands(newi)
        image = ee.Image(bandnames.iterate(bn, image)) \
            .select(bandnames)

    if from_dict:
        from_dict = ee.Dictionary(from_dict)
        image = ee.Image(from_dict.toImage())

    if not from_dict and not names:
        image = ee.Image.constant(value)

    return image

def euclideanDistance(image1, image2, bands=None, discard_zeros=False,
                      name='distance'):
    """ Compute the Euclidean distance between two images. The image's bands
    is the dimension of the arrays.
    :param image1:
    :type image1: ee.Image
    :param image2:
    :type image2: ee.Image
    :param bands: the bands that want to be computed
    :type bands: list
    :param discard_zeros: pixel values equal to zero will not count in the
        distance computation
    :type discard_zeros: bool
    :param name: the name of the resulting band
    :type name: str
    :return: a distance image
    :rtype: ee.Image
    """
    if not bands:
        bands = image1.bandNames()

    image1 = image1.select(bands)
    image2 = image2.select(bands)

    proxy = empty(0, bands)
    image1 = proxy.where(image1.gt(0), image1)
    image2 = proxy.where(image2.gt(0), image2)

    if discard_zeros:
        # zeros
        zeros1 = image1.eq(0)
        zeros2 = image2.eq(0)

        # fill zeros with values from the other image
        image1 = image1.where(zeros1, image2)
        image2 = image2.where(zeros2, image1)

    a = image1.subtract(image2)
    b = a.pow(2)
    c = b.reduce('sum')
    d = c.sqrt()

    return d.rename(name)


def sumDistance(image, collection, bands=None, discard_zeros=False,
                name='sumdist'):
    """ Compute de sum of all distances between the given image and the
    collection passed

    :param image:
    :param collection:
    :return:
    """
    condition = isinstance(collection, ee.ImageCollection)

    if condition:
        collection = collection.toList(collection.size())

    accum = ee.Image(0).rename(name)

    def over_rest(im, ini):
        ini = ee.Image(ini)
        im = ee.Image(im)
        dist = ee.Image(euclideanDistance(image, im, bands, discard_zeros)) \
            .rename(name)
        return ini.add(dist)

    return ee.Image(collection.iterate(over_rest, accum))

def removeIndex(list, index):
    """ Remove an element by its index """
    list = ee.List(list)
    index = ee.Number(index)
    size = list.size()

    def allowed():
        def zerof(list):
            return list.slice(1, list.size())

        def rest(list, index):
            list = ee.List(list)
            index = ee.Number(index)
            last = index.eq(list.size())

            def lastf(list):
                return list.slice(0, list.size().subtract(1))

            def restf(list, index):
                list = ee.List(list)
                index = ee.Number(index)
                first = list.slice(0, index)
                return first.cat(list.slice(index.add(1), list.size()))

            return ee.List(ee.Algorithms.If(last, lastf(list), restf(list, index)))

        return ee.List(ee.Algorithms.If(index, rest(list, index), zerof(list)))

    condition = index.gte(size).Or(index.lt(0))

    return ee.List(ee.Algorithms.If(condition, -1, allowed()))


def intersection(eelist, intersect):
    """ Find matching values. If ee_list1 has duplicated values that are
    present on ee_list2, all values from ee_list1 will apear in the result
    :param intersect: the other Earth Engine List
    :return: list with the intersection (matching values)
    :rtype: ee.List
    """
    eelist = ee.List(eelist)
    intersect = ee.List(intersect)
    newlist = ee.List([])
    def wrap(element, first):
        first = ee.List(first)

        return ee.Algorithms.If(intersect.contains(element),
                                first.add(element), first)

    return ee.List(eelist.iterate(wrap, newlist))


def removeBands(image, bands):
    """ Remove the specified bands from an image """
    bnames = image.bandNames()
    bands = ee.List(bands)
    inter = intersection(bnames, bands)
    diff = bnames.removeAll(inter)
    return image.select(diff)


def replace(image, to_replace, to_add):
    """ Replace one band of the image with a provided band
    :param to_replace: name of the band to replace. If the image hasn't got
        that band, it will be added to the image.
    :type to_replace: str
    :param to_add: Image (one band) containing the band to add. If an Image
        with more than one band is provided, it uses the first band.
    :type to_add: ee.Image
    :return: Same Image provided with the band replaced
    :rtype: ee.Image
    """

    band = to_add.select([0])
    bands = image.bandNames()
    resto = bands.remove(to_replace)
    img_resto = image.select(resto)
    img_final = img_resto.addBands(band)
    return img_final


def medoidScore(collection, bands=None, discard_zeros=False,
                bandname='sumdist', normalize=False):
    """ Compute a score to reflect 'how far' is from the medoid. Same params
     as medoid() """
    first_image = ee.Image(collection.first())
    if not bands:
        bands = first_image.bandNames()

    # Create a unique id property called 'enumeration'
    enumerated = enumerateProperty(collection)
    collist = enumerated.toList(enumerated.size())

    def over_list(im):
        im = ee.Image(im)
        n = ee.Number(im.get('enumeration'))

        # Remove the current image from the collection
        filtered = removeIndex(collist, n)

        # Select bands for medoid
        to_process = im.select(bands)

        def over_collist(img):
            return ee.Image(img).select(bands)
        filtered = filtered.map(over_collist)

        # Compute the sum of the euclidean distance between the current image
        # and every image in the rest of the collection
        dist = sumDistance(
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
            return replace(img, bandname, newband)

        medcol = medcol.map(to_normalize)

    return medcol


def medoid(collection, bands=None, discard_zeros=True):
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
    # final = removeBands(comp, ['sumdist', 'mask'])
    final = removeBands(comp, ['sumdist'])
    return final

'''
    # set metadata to indicate component images
    return comp_im.set('COMPOSITE_IMAGES', self._im_df[['ID', 'DATE'] + self._im_props].to_string()).toUint16()
'''

def collection_from_ids(ids, apply_mask=False, add_aux_bands=False, scale_refl=False):
    """
    Create ee.ImageCollection of masked and scored images, from a list of EE image IDs

    Parameters
    ----------
    ids : list[str]
          list of EE image IDs
    apply_mask : bool, optional
                 Apply any validity mask to the image by setting nodata (default: False)
    add_aux_bands: bool, optional
                   Add auxiliary bands (cloud, shadow, fill & validity masks, and quality score) (default: False)
    scale_refl : bool, optional
                 Scale reflectance values from 0-10000 if they are not in that range already (default: True)

    Returns
    -------
    : ee.ImageCollection
    """

    ee_coll_name = collection.ee_split(ids[0])[0]
    if not ee_coll_name in collection.ee_to_gd_map():
        raise ValueError(f'Unsupported collection: {ee_coll_name}')

    gd_coll_name = collection.ee_to_gd_map()[ee_coll_name]

    id_check = [collection.ee_split(im_id)[0] == ee_coll_name for im_id in ids[1:]]
    if not all(id_check):
        raise ValueError(f'All images must belong to the same collection')

    gd_collection = collection.cls_col_map[gd_coll_name]()

    im_list = ee.List([])
    for im_id in ids:
        im = gd_collection.get_image(im_id, apply_mask=apply_mask, add_aux_bands=add_aux_bands, scale_refl=scale_refl)
        im_list = im_list.add(im)

    return ee.ImageCollection(im_list), gd_collection


def composite(images, method='q_mosaic', apply_mask=True):
    # qualityMosaic will prefer clear pixels based on SCORE and irrespective of mask, for other methods, the mask
    # is needed to avoid including cloudy pixels
    method = str(method).lower()

    ee_collection = None
    if (isinstance(images, list) or isinstance(images, tuple)) and len(images) > 0:
        if isinstance(images[0], str):
            ee_collection, gd_collection = collection_from_ids(images, apply_mask=apply_mask, add_aux_bands=True)
        elif isinstance(images[0], ee.Image):
            im_list = ee.List([])
            for image in images:
                im_list = im_list.add(image)
            ee_collection = ee.ImageCollection(im_list)
    elif isinstance(images, ee.ImageCollection):
        ee_collection = images

    if ee_collection is None:
        raise ValueError(f'Unsupported images type: {type(images)}')

    if method == 'q_mosaic':
        comp_image = ee_collection.qualityMosaic('SCORE')
    elif method == 'mosaic':
        comp_image = ee_collection.mosaic()
    elif method == 'median':
        comp_image = ee_collection.median()
    elif method == 'medoid':
        bands = [band_dict['id'] for band_dict in gd_collection.collection_info['bands']]
        comp_image = medoid(ee_collection, bands=bands)
    else:
        raise ValueError(f'Unsupported composite method: {method}')

    # populate image metadata with info on component images
    im_prop_df = gd_collection._get_collection_df(ee_collection, do_print=False)
    comp_str = im_prop_df.to_string(
                float_format='%.2f',
                formatters={'DATE': lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M')},
                columns=gd_collection._im_props.ABBREV,
                # header=property_df.ABBREV,
                index=False,
                justify='center')

    comp_image = comp_image.set('COMPOSITE_IMAGES', comp_str)

    # name the composite
    start_date = im_prop_df.DATE.iloc[0].strftime('%Y_%m_%d')
    end_date = im_prop_df.DATE.iloc[-1].strftime('%Y_%m_%d')
    ee_coll_name = collection.ee_split(im_prop_df.ID.values[0])[0]
    comp_name = f'{ee_coll_name}/{start_date}-{end_date}-{method.upper()}_COMP'

    return comp_image, comp_name
