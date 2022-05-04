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
from collections import OrderedDict
##
from datetime import datetime, timedelta
from typing import Dict, List

import ee
import tabulate
from tabulate import TableFormat, Line, DataRow

from geedim import info, medoid
from geedim.enums import ResamplingMethod, CompositeMethod
from geedim.errors import UnfilteredError, UnsupportedValueError, UnsupportedTypeError, OutOfRangeError
from geedim.image import BaseImage, split_id
from geedim.masked_image import MaskedImage, class_from_id

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


class MaskedCollection:
    """
    Class for encapsulating, searching and compositing an Earth Engine image collection, with support for
    cloud/shadow masking.
    """
    _default_comp_method = CompositeMethod.q_mosaic

    def __init__(self, ee_collection):
        """
        Create a MaskedCollection instance.

        Parameters
        ----------
        ee_collection : ee.ImageCollection
            The Earth Engine image collection to encapsulate.
        """
        if not isinstance(ee_collection, ee.ImageCollection):
            raise UnsupportedTypeError(f'`ee_collection` must be an instance of ee.ImageCollection')
        self._name = None
        self._info = None
        self._properties = None
        self._filtered = False
        self._ee_collection = ee_collection

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
        # this is separate from __inti__ for consistency with MaskedImage.from_id()
        gd_collection = cls(ee.ImageCollection(name))
        gd_collection._name = name
        return gd_collection

    @classmethod
    def from_list(cls, image_list):
        """
        Create a MaskedCollection instance from a list of EE image IDs, ee.Image's and/or MaskedImage's.

        Parameters
        ----------
        image_list : List[Union[str, ee.Image, MaskedImage], ...]
            A list of images to include in the collection (must all be from the same EE collection).

        Returns
        -------
        gd_collection: MaskedCollection
            The MaskedCollection instance.
        """
        # build lists of EE images and IDs from image_list
        ee_image_list = []
        ee_id_list = []
        for image_obj in image_list:
            if isinstance(image_obj, str):
                ee_image_list.append(ee.Image(image_obj))
                ee_id_list.append(image_obj)
            elif isinstance(image_obj, ee.Image):
                ee_image_list.append(image_obj)
                ee_id_list.append(image_obj.getInfo()['id'])
            elif isinstance(image_obj, BaseImage):
                ee_image_list.append(image_obj.ee_image)
                ee_id_list.append(image_obj.id)
            else:
                raise UnsupportedTypeError(f'Unsupported image object type: {type(image_obj)}')

        # check the images all come from the same collection
        ee_coll_name = split_id(ee_id_list[0])[0]
        id_check = [split_id(im_id)[0] == ee_coll_name for im_id in ee_id_list[1:]]
        if not all(id_check):
            # TODO: allow images from compatible landsat collections
            raise UnsupportedValueError('All images must belong to the same collection')

        # create the collection object
        gd_collection = cls.from_name(ee_coll_name)
        gd_collection._ee_collection = ee.ImageCollection(ee.List(ee_image_list))
        gd_collection._name = ee_coll_name
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
            image_id = self._ee_collection.first().getInfo()['id']
            self._name = split_id(image_id)[0]
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
        return class_from_id(self.name)

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
        return OrderedDict({prop_dict['system:id']:prop_dict for prop_dict in props_list})

    def _get_properties_table(self, properties: List, properties_key: List = None) -> str:
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
            self, method=_default_comp_method, mask=True, resampling=BaseImage._default_resampling, date=None,
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
            resampling = ResamplingMethod(resampling)
            if (method == CompositeMethod.q_mosaic) and (self.image_type == MaskedImage):
                # TODO get a list of supported collections, report this in CLI help too
                raise UnsupportedValueError(f'The `q-mosaic` method is not supported for the {self.name} collection.')

            ee_collection = self._ee_collection
            if method in [CompositeMethod.mosaic, CompositeMethod.q_mosaic]:
                if date:
                    # sort the collection by time difference to `date`, so that *mosaic uses the closest in time pixels
                    def set_date_dist(ee_image):
                        date_dist = ee.Number(ee_image.get('system:time_start')).subtract(ee.Date(date).millis()).abs()
                        return ee_image.set('DATE_DIST', date_dist)

                    ee_collection = ee_collection.map(set_date_dist).sort('DATE_DIST', opt_ascending=False)
                elif region:
                    # sort the collection by cloud/shadow free portion, so that *mosaic favours pixels from the least
                    # cloudy image
                    def set_cloudless_portion(ee_image):
                        gd_image = self.image_type(ee_image, **kwargs)
                        gd_image.set_region_stats(region)
                        return gd_image.ee_image

                    ee_collection = ee_collection.map(set_cloudless_portion).sort('CLOUDLESS_PORTION')
                else:
                    # sort the collection by capture date.  *mosaic will favour the most recent pixels.
                    ee_collection = ee_collection.sort('system:time_start')

            if mask:
                def mask_clouds(ee_image):
                    gd_image = self.image_type(ee_image, **kwargs)
                    gd_image.mask_clouds()
                    return gd_image.ee_image

                ee_collection = ee_collection.map(mask_clouds)

            if resampling != BaseImage._default_resampling:
                def resample(ee_image):
                    return ee_image.resample(resampling.value)

                ee_collection = ee_collection.map(resample)
            return ee_collection

    def search(self, start_date, end_date, region, cloudless_portion=0, **kwargs):
        """
        Search for images based on date, region and cloudless portion criteria.

        Parameters
        ----------
        start_date : datetime.datetime
            Start image capture date.
        end_date : datetime.datetime
            End image capture date (if None, then set to start_date + 1 day).
        region : dict, ee.Geometry
            Polygon in WGS84 specifying a region that images should intersect.
        cloudless_portion: int, optional
            Minimum portion (%) of image pixels that should be cloud/shadow free.
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
            raise OutOfRangeError('`end_date` must be at least a day later than `start_date`')

        def set_region_stats(ee_image: ee.Image):
            """ Find filled and cloud/shadow free portions inside the search region for a given image.  """
            gd_image = self.image_type(ee_image, **kwargs)
            gd_image.set_region_stats(region)
            return gd_image.ee_image

        # filter the image collection, finding cloud/shadow masks and region stats
        ee_collection = (
            self._ee_collection.filterDate(start_date, end_date).
                filterBounds(region).
                map(set_region_stats).
                filter(ee.Filter.gte('CLOUDLESS_PORTION', cloudless_portion))
        )
        # return a new MaskedCollection containing the filtered EE collection (the EE collection
        # wrapped by MaskedCollection remains fixed)
        gd_collection = MaskedCollection(ee_collection)
        gd_collection._filtered = True
        return gd_collection

    def composite(
        self, method=_default_comp_method, mask=True, resampling=BaseImage._default_resampling, date=None,
        region=None, **kwargs
    ):
        """
        Create a composite image from the encapsulated image collection.

        Parameters
        ----------
        method: CompositeMethod, optional
            The compositing method to use.  One of:
                `q_mosiac`: Select each composite pixel from the collection image with the highest quality (i.e.
                    distance to nearest cloud). When more than one image shares the highest quality value,
                    the first of the competing images is used. Valid for cloud/shadow maskable image collections only
                    (Sentinel-2 TOA and SR, and Landsat4-9 level 2 collection 2).
                `mosaic`: Select each composite pixel from the first unmasked collection image.
                `medoid`: Select each composite pixel as the the image pixel having the minimum summed diff (across
                    bands) from the median of all collection images.  Maintains the original relationship between
                    bands.  See https://www.mdpi.com/2072-4292/5/12/6481 for detail.
                `median`: Median of the collection images.
                `mode`: Mode of the collection images.
                `mean`: Mean of the collection images.
        mask: bool, optional
            Mask cloud/shadow before compositing  [default: True].
        resampling: ResamplingMethod, optional
            The resampling method to use on collection images prior to compositing.  If 'near', no resampling is done
            [default: 'near'].
        date: datetime.datetime, optional
            Sort collection images by their absolute difference in time from this date.  Useful for
            prioritising pixels from images closest to this date.  Valid for the `q-mosaic`
            and `mosaic` methods only.  If None, no time difference sorting is done. [default: None].
        region: dict, geojson, optional
            Sort collection images by their cloudless portion inside this region (only if `date` is not
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
                raise UnsupportedValueError(
                    '`date` should be a datetime instance or a string with format: "%Y-%m-%d"'
                )

        # mask, sort & resample the EE collection
        method = CompositeMethod(method)
        ee_collection = self._prepare_for_composite(
            method=method, mask=mask, resampling=resampling, date=date, region=region, **kwargs
        )

        if method == CompositeMethod.q_mosaic:
            comp_image = ee_collection.qualityMosaic(self.image_type._cloud_dist_band)
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
            raise UnsupportedValueError(f'Unsupported composite method: {method}')

        # populate composite image metadata with info on component images
        props = self._get_properties(ee_collection)
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
        comp_image = comp_image.set('system:id', comp_id)
        comp_image = comp_image.set('system:index', comp_id)

        # set the composite capture time to the capture time of the first component image
        comp_image = comp_image.set('system:time_start', min(dates).timestamp() * 1000)

        return self.image_type(comp_image)
