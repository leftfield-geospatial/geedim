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

import logging
import re
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List

import ee
import tabulate
from tabulate import TableFormat, Line, DataRow

from geedim import info, medoid
from geedim.download import BaseImage, split_id
from geedim.enums import ResamplingMethod, CompositeMethod
from geedim.errors import UnfilteredError, ComponentImageError
from geedim.mask import MaskedImage, class_from_id

logger = logging.getLogger(__name__)
tabulate.MIN_PADDING = 0

##
# tabulate format for collection properties
_table_fmt = TableFormat(
    lineabove=Line("", "-", " ", ""),
    linebelowheader=Line("", "-", " ", ""),
    linebetweenrows=None,
    linebelow=Line("", "-", " ", ""),
    headerrow=DataRow("", " ", ""),
    datarow=DataRow("", " ", ""),
    padding=0,
    with_header_hide=["lineabove", "linebelow"],
)


def compatible_collections(names: List[str]) -> bool:
    """
    Test if all the collections in a list are spectrally compatible i.e. images from these collections can be
    composited together.
    """
    names = list(set(names))  # reduce to unique values
    start_name = names[0]
    name_matches = [True]
    landsat_regex = re.compile('(LANDSAT/\w{2})(\d{2})(/.*)')
    start_landsat_match = landsat_regex.search(start_name)
    for name in names[1:]:
        name_match = False
        if start_name == name:
            name_match = True
        elif start_landsat_match:
            landsat_regex = re.compile(f'{start_landsat_match.groups()[0]}\d\d{start_landsat_match.groups()[-1]}')
            if landsat_regex.search(name):
                name_match = True
        name_matches.append(name_match)
    return all(name_matches)


class MaskedCollection:
    """
    Class for encapsulating, searching and compositing an Earth Engine image collection, with support for
    cloud/shadow masking.
    """

    def __init__(self, ee_collection):
        """
        Create a MaskedCollection instance.

        Parameters
        ----------
        ee_collection : ee.ImageCollection
            The Earth Engine image collection to encapsulate.
        """
        if not isinstance(ee_collection, ee.ImageCollection):
            raise TypeError(f'`ee_collection` must be an instance of ee.ImageCollection')
        self._name = None
        self._info = None
        self._properties = None
        self._filtered = False
        self._ee_collection = ee_collection
        self._image_type = None

    @classmethod
    def from_name(cls, name):
        """
        Create a MaskedCollection instance from an Earth Engine image collection name.

        Parameters
        ----------
        name: str
            The name of the Earth Engine image collection to create.

        Returns
        -------
        gd_collection: MaskedCollection
            The MaskedCollection instance.
        """
        # this is separate from __init__ for consistency with MaskedImage.from_id()
        if (name == 'COPERNICUS/S2') or (name == 'COPERNICUS/S2_SR'):
            # Recent images in S2_SR do not always have matching images in S2_CLOUD_PROBABILITY (which is needed for
            # 'cloud_prob' cloud masking), so this is a special case to return a filtered S2/S2_SR collection that has
            # matching images in S2_CLOUD_PROBABILITY.
            cloud_prob_coll = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            s2_coll = ee.ImageCollection(name)
            filt = ee.Filter.equals(leftField='system:index', rightField='system:index')
            ee_collection = ee.ImageCollection(ee.Join.simple().apply(s2_coll, cloud_prob_coll, filt))
        else:
            ee_collection = ee.ImageCollection(name)
        gd_collection = cls(ee_collection)
        gd_collection._name = name
        return gd_collection

    @classmethod
    def from_list(cls, image_list):
        """
        Create a MaskedCollection instance from a list of EE image ID strings, ee.Image objects and/or MaskedImage
        objects.

        Any ee.Image objects (including those encapsulated by MaskedImage) must have `system:id` and
        `system:time_start` properties.  This is always the case for images from the EE data catalog, and images
        returned from MaskedCollection.composite(), but user-created images passed to from_list() should have these
        properties set.

        Parameters
        ----------
        image_list : list
            A list of images to include in the collection (must all be from the same EE collection).

        Returns
        -------
        gd_collection: MaskedCollection
            The MaskedCollection instance.
        """
        # build lists of EE images and IDs from image_list
        if len(image_list) == 0:
            raise ValueError('`image_list` is empty.')

        im_dict_list = []
        for image_obj in image_list:
            if isinstance(image_obj, str):
                im_dict_list.append(dict(ee_image=ee.Image(image_obj), id=image_obj, has_date=True))
            elif isinstance(image_obj, ee.Image):
                ee_info = image_obj.getInfo()
                ee_id = ee_info['id'] if 'id' in ee_info else None
                has_date = ('properties' in ee_info) and ('system:time_start' in ee_info['properties'])
                im_dict_list.append(dict(ee_image=ee.Image(image_obj), id=ee_id, has_date=has_date))
            elif isinstance(image_obj, BaseImage):
                im_dict_list.append(
                    dict(ee_image=image_obj.ee_image, id=image_obj.id, has_date=image_obj.date is not None)
                )
            else:
                raise TypeError(f'Unsupported image object type: {type(image_obj)}')

        # check all images have IDs and capture dates
        if any([(im_dict['id'] is None) or (not im_dict['has_date']) for im_dict in im_dict_list]):
            raise ComponentImageError('Image(s) must have "id" and "system:time_start" properties.')

        # check the images all come from the same or compatible collections
        ee_coll_names = [split_id(im_dict['id'])[0] for im_dict in im_dict_list]
        if not compatible_collections(ee_coll_names):
            raise ComponentImageError(
                'All images must belong to the same, or spectrally compatible, collections.'
            )

        # create the collection object, using the name of the first collection in ee_coll_names.
        gd_collection = cls.from_name(ee_coll_names[0])
        gd_collection._ee_collection = ee.ImageCollection(
            ee.List([im_dict['ee_image'] for im_dict in im_dict_list])
        )
        gd_collection._filtered = True
        return gd_collection

    @property
    def ee_collection(self) -> ee.ImageCollection:
        """ The encapsulated Earth Engine image collection. """
        return self._ee_collection

    @property
    def name(self) -> str:
        """ Name of the encapsulated Earth Engine image collection. """
        if not self._name:
            ee_info = self._ee_collection.first().getInfo()
            self._name = split_id(ee_info['id'])[0] if ee_info and 'id' in ee_info else 'None'
        return self._name

    @property
    def info(self) -> Dict:
        """ Search properties and band metadata. """
        if not self._info:
            if self.name in info.collection_info:
                self._info = info.collection_info[self.name]
            else:
                self._info = info.collection_info['*']
        return self._info

    @property
    def image_type(self) -> type:
        """ geedim class to encapsulate images from `ee_collection`. """
        if not self._image_type:
            self._image_type = class_from_id(self.name)
        return self._image_type

    @property
    def properties(self) -> Dict:
        """ Properties for each image in the collection. """
        if not self._filtered:
            raise UnfilteredError(
                '`properties` can only be retrieved for collections returned by `search()` and `from_list()`'
            )
        if not self._properties:
            self._properties = self._get_properties(self._ee_collection)
        return self._properties

    @property
    def properties_table(self) -> str:
        """ `properties` formatted as a table. """
        return self._get_properties_table(self.properties)

    @property
    def properties_key(self) -> Dict:
        """ Abbreviations and descriptions for `properties`. """
        return OrderedDict({key_dict['PROPERTY']: key_dict for key_dict in self.info['properties']})

    @property
    def key_table(self) -> str:
        """ `properties_key` formatted as a table. """
        key_dict = [dict(ABBREV=v['ABBREV'], DESCRIPTION=v['DESCRIPTION']) for v in self.properties_key.values()]
        return tabulate.tabulate(key_dict, headers='keys', floatfmt='.2f', tablefmt=_table_fmt)

    def _get_properties(self, ee_collection: ee.ImageCollection) -> Dict:
        """ Retrieve properties of images in a given Earth Engine image collection. """

        # the properties to retrieve
        prop_key_list = ee.List([item['PROPERTY'] for item in self.info['properties']])

        def aggregrate_props(ee_image, coll_list):
            im_dict = ee_image.toDictionary(prop_key_list)
            return ee.List(coll_list).add(im_dict)

        # retrieve list of dicts of properties of images in ee_collection
        props_list = ee.List(ee_collection.iterate(aggregrate_props, ee.List([]))).getInfo()
        # add image properties to the return dict in the same order as the underlying collection
        props_dict = OrderedDict()
        for prop_dict in props_list:
            props_dict[prop_dict['system:id']] = prop_dict
        return props_dict

    def _get_properties_table(self, properties: Dict, properties_key: Dict = None) -> str:
        """
        Format the given properties into a table.  Orders properties (columns) according `properties_key` and replaces
        long form property names with abbreviations.
        """
        if not properties_key:
            properties_key = self.properties_key

        abbrev_props = []
        for im_id, im_prop_dict in properties.items():
            abbrev_dict = OrderedDict()
            for prop_name, key_dict in properties_key.items():
                if prop_name in im_prop_dict:
                    if prop_name == 'system:time_start':  # convert timestamp to date string
                        dt = datetime.utcfromtimestamp(im_prop_dict[prop_name] / 1000)
                        abbrev_dict[key_dict['ABBREV']] = datetime.strftime(dt, '%Y-%m-%d %H:%M')
                    else:
                        abbrev_dict[key_dict['ABBREV']] = im_prop_dict[prop_name]
            abbrev_props.append(abbrev_dict)
        return tabulate.tabulate(abbrev_props, headers='keys', floatfmt='.2f', tablefmt=_table_fmt)

    def _prepare_for_composite(
        self, method=None, mask=True, resampling=BaseImage._default_resampling, date=None,
        region=None, **kwargs
    ):
        """
        Prepare the Earth Engine collection for compositing. See MaskedCollection.composite() for
        parameter descriptions.
        """

        if not self._filtered:
            raise UnfilteredError(
                'Composites can only be created from collections returned by `search()` and `from_list()`'
            )

        method = CompositeMethod(method)
        resampling = ResamplingMethod(resampling) if resampling else BaseImage._default_resampling
        if (method == CompositeMethod.q_mosaic) and (self.image_type == MaskedImage):
            # TODO get a list of supported collections, report this in CLI help too
            raise ValueError(f'The `q-mosaic` method is not supported for this ("{self.name}") collection.')

        def prepare_image(ee_image: ee.Image):
            """ Prepare an EE image for use in compositing. """
            if date and (method in [CompositeMethod.mosaic, CompositeMethod.q_mosaic]):
                date_dist = ee.Number(ee_image.get('system:time_start')).subtract(ee.Date(date).millis()).abs()
                ee_image = ee_image.set('DATE_DIST', date_dist)

            gd_image = self.image_type(ee_image, **kwargs)
            if region and (method in [CompositeMethod.mosaic, CompositeMethod.q_mosaic]):
                gd_image.set_region_stats(region)
            if mask:
                gd_image.mask_clouds()
            if resampling != BaseImage._default_resampling:
                return gd_image.ee_image.resample(resampling.value)
            else:
                return gd_image.ee_image

        ee_collection = self._ee_collection.map(prepare_image)

        if method in [CompositeMethod.mosaic, CompositeMethod.q_mosaic]:
            if date:
                ee_collection = ee_collection.sort('DATE_DIST', opt_ascending=False)
            elif region:
                ee_collection = ee_collection.sort('CLOUDLESS_PORTION')
            else:
                ee_collection = ee_collection.sort('system:time_start')
        else:
            if date:
                logger.warning('`date` is valid for `mosaic` and `q_mosaic` methods only.')
            elif region:
                logger.warning('`region` is valid for `mosaic` and `q_mosaic` methods only.')

        return ee_collection

    def search(self, start_date, end_date, region, fill_portion=None, cloudless_portion=None, **kwargs):
        """
        Search for images based on date, region and cloudless portion criteria.

        Parameters
        ----------
        start_date : datetime.datetime
            Start date (UTC).
        end_date : datetime.datetime, optional
            End date (UTC). [default: start_date + 1 day]
        region : dict, ee.Geometry
            Polygon in WGS84 specifying a region that images should intersect.
        fill_portion: float, optional
            Minimum portion (%) of valid/filled image pixels.
        cloudless_portion: float, optional
            Minimum portion (%) of cloud/shadow free image pixels.
        kwargs: optional
            Cloud/shadow masking parameters - see MaskedImage.__init__() for details.

        Returns
        -------
        gd_collection: MaskedCollection
            A new MaskedCollection instance containing the search filtered images.
        """
        if end_date is None:
            end_date = start_date + timedelta(days=1)
        if end_date <= start_date:
            raise ValueError('`end_date` must be at least a day later than `start_date`')

        def set_region_stats(ee_image: ee.Image):
            """ Find filled and cloud/shadow free portions inside the search region for a given image.  """
            gd_image = self.image_type(ee_image, **kwargs)
            gd_image.set_region_stats(region)
            return gd_image.ee_image

        # filter the image collection, finding cloud/shadow masks and region stats
        ee_collection = self._ee_collection.filterDate(start_date, end_date).filterBounds(region).map(set_region_stats)
        if fill_portion:
            ee_collection = ee_collection.filter(ee.Filter.gte('FILL_PORTION', fill_portion))
        if cloudless_portion and self.image_type != MaskedImage:
            ee_collection = ee_collection.filter(ee.Filter.gte('CLOUDLESS_PORTION', cloudless_portion))
        ee_collection = ee_collection.sort('system:time_start')

        # return a new MaskedCollection containing the filtered EE collection (the EE collection
        # wrapped by MaskedCollection remains fixed)
        gd_collection = MaskedCollection(ee_collection)
        gd_collection._name = self._name
        gd_collection._filtered = True
        return gd_collection

    def composite(
        self, method=None, mask=True, resampling=BaseImage._default_resampling, date=None,
        region=None, **kwargs
    ):
        """
        Create a composite image from the encapsulated image collection.

        Parameters
        ----------
        method: CompositeMethod, optional
            Method for finding each composite pixel from the collection of corresponding input image pixels.  One of:
                `q_mosaic`: Use the unmasked pixel with the highest quality (i.e. distance to nearest cloud). When more
                    than one pixel shares the highest quality value, the first of the competing pixels is used. Valid
                    for cloud/shadow maskable image collections only (Sentinel-2 TOA and SR, and Landsat4-9 level 2
                    collection 2).
                `mosaic`: Use the first unmasked pixel.
                `medoid`: Use the medoid of the unmasked pixels.  The medoid selects the image pixel (across
                    bands) from the image having the minimum summed diff (across bands) from the median of the
                    collection images. Maintains the original relationship between bands.
                    See https://www.mdpi.com/2072-4292/5/12/6481 for detail.
                `median`: Use the median of the unmasked pixels.
                `mode`: Use the mode of the unmasked pixels.
                `mean`: use the mean of the unmasked pixels.
        mask: bool, optional
            Whether to apply the cloud/shadow mask, or fill (valid pixel) mask, in the case of images without
            support for cloud/shadow masking.  [default: True].
        resampling: ResamplingMethod, optional
            The resampling method to use on collection images prior to compositing.  If 'near', no resampling is done
            [default: 'near'].
        date: datetime.datetime, optional
            Sort collection images by their absolute difference in capture time from this date.  Useful for
            prioritising pixels from images closest to this date.  Valid for the `q-mosaic`
            and `mosaic` methods only.  If None, no time difference sorting is done. [default: None].
        region: dict, optional
            Sort collection images by their cloudless portion inside this geojson polygon (only if `date` is not
            specified).  This is useful to prioritise pixels from the least cloudy image(s).  Valid for the `q-mosaic`
            and `mosaic` methods.  If `date` and `region` are not specified, collection images are sorted by their
            capture date.
        kwargs: optional
            Cloud/shadow masking parameters - see MaskedImage.__init__() for details.

        Returns
        -------
        comp_image: MaskedImage
            The composite image.
        """

        if isinstance(date, str):
            try:
                date = datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                raise ValueError('`date` should be a datetime instance or a string with format: "%Y-%m-%d"')

        if method is None:
            method = CompositeMethod.mosaic if self.image_type == MaskedImage else CompositeMethod.q_mosaic

        # mask, sort & resample the EE collection
        method = CompositeMethod(method)
        ee_collection = self._prepare_for_composite(
            method=method, mask=mask, resampling=resampling, date=date, region=region, **kwargs
        )

        if method == CompositeMethod.q_mosaic:
            comp_image = ee_collection.qualityMosaic('CLOUD_DIST')
        elif method == CompositeMethod.mosaic:
            comp_image = ee_collection.mosaic()
        elif method == CompositeMethod.median:
            # TODO: S2 median gives 'Output of image computation is too large' on download dtype=uint16
            comp_image = ee_collection.median()
        elif method == CompositeMethod.medoid:
            # limit medoid to surface reflectance bands
            # TODO: we need another way to get sr_bands if we are removing collection_info
            sr_bands = [band_dict['id'] for band_dict in self.info['bands']]
            comp_image = medoid.medoid(ee_collection, bands=sr_bands)
        elif method == CompositeMethod.mode:
            comp_image = ee_collection.mode()
        elif method == CompositeMethod.mean:
            comp_image = ee_collection.mean()
        else:
            raise ValueError(f'Unsupported composite method: {method}')

        # populate composite image metadata with info on component images
        props = self._get_properties(ee_collection)
        if len(props) == 0:
            raise ValueError('The collection is empty.')
        props_str = self._get_properties_table(props)
        comp_image = comp_image.set('COMPONENT_IMAGES', 'TABLE:\n' + props_str)

        # construct an ID for the composite
        dates = [datetime.utcfromtimestamp(item['system:time_start'] / 1000) for item in props.values()]
        start_date = min(dates).strftime('%Y_%m_%d')
        end_date = max(dates).strftime('%Y_%m_%d')

        method_str = method.value.upper()
        if method in [CompositeMethod.mosaic, CompositeMethod.q_mosaic] and date:
            method_str += '-' + date.strftime('%Y_%m_%d')

        comp_id = f'{self.name}/{start_date}-{end_date}-{method_str}-COMP'
        comp_image = comp_image.set('system:id', comp_id)  # sets root 'id' property
        comp_image = comp_image.set('system:index', comp_id)  # sets 'properties'->'system:index'

        # set the composite capture time to the capture time of the first component image (sets
        # 'properties'->'system:time_start')
        comp_image = comp_image.set('system:time_start', min(dates).timestamp() * 1000)
        gd_comp_image = self.image_type(comp_image)
        gd_comp_image._id = comp_id  # avoid getInfo() for id property
        return gd_comp_image
