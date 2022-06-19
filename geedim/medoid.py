"""
    MIT License

    Copyright (c) 2017 Rodrigo E. Principe

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
import ee
"""
    This module contains Medoid related functionality copied from 'Google Earth Engine tools' under MIT license.
    See https://github.com/gee-community/gee_tools.
"""


# yapf: disable
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

    def over_list(ll):
        ll = ee.List(ll)
        index = ee.Number(ll.get(0))
        element = ll.get(1)
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


def euclideanDistance(
    image1, image2, bands=None, discard_zeros=False,
    name='distance'
):
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


def sumDistance(
    image, collection, bands=None, discard_zeros=False,
    name='sumdist'
):
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

        return ee.Algorithms.If(
            intersect.contains(element),
            first.add(element), first
        )

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


def medoidScore(
    collection, bands=None, discard_zeros=False,
    bandname='sumdist', normalize=False
):
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
            discard_zeros=discard_zeros
        )

        # Mask zero values
        if not normalize:
            # multiply by -1 to get the lowest value in the qualityMosaic
            dist = dist.multiply(-1)

        return im.addBands(dist)

    imlist = ee.List(collist.map(over_list))

    medcol = ee.ImageCollection.fromImages(imlist)

    # Normalize result to be between 0 and 1
    if normalize:
        min_sumdist = ee.Image(medcol.select(bandname).min()) \
            .rename('min_sumdist')
        max_sumdist = ee.Image(medcol.select(bandname).max()) \
            .rename('max_sumdist')

        def to_normalize(img):
            sumdist = img.select(bandname)
            newband = ee.Image().expression(
                '1-((val-min)/(max-min))',
                {
                    'val': sumdist,
                    'min': min_sumdist,
                    'max': max_sumdist
                }
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
    final = removeBands(comp, ['sumdist'])
    return final
# yapf: enable
