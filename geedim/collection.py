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
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Union

import ee
import tabulate
import textwrap as wrap
from geedim import schema, medoid
from geedim.download import BaseImage
from geedim.enums import ResamplingMethod, CompositeMethod
from geedim.errors import UnfilteredError, InputImageError
from geedim.mask import MaskedImage, class_from_id
from geedim.stac import StacCatalog, StacItem
from geedim.utils import split_id, resample
from tabulate import TableFormat, Line, DataRow

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
    with_header_hide=["lineabove", "linebelow"]
)  # yapf: disable


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


def parse_date(date: Union[datetime, str], var_name=None) -> datetime:
    """ Convert a string to a datetime, raising an exception if it is in the wrong format. """
    var_name = var_name or 'date'
    if isinstance(date, str):
        try:
            date = datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f'{var_name} should be a datetime instance or a string with format: "%Y-%m-%d"')
    return date


def abbreviate(name: str) -> str:
    """ Return an acronym for a string in camel or snake case. """
    name = name.strip()
    if len(name) <= 5:
        return name
    abbrev = ''
    prev = '_'
    for curr in name:
        if curr.isdigit():
            abbrev += curr
        elif (prev == '_' and curr.isalnum()) or (prev.islower() and curr.isupper()):
            abbrev += curr.upper()
        prev = curr
    return abbrev if len(abbrev) >= 2 else name


class MaskedCollection:

    def __init__(self, ee_collection: ee.ImageCollection, add_props: List[str] = None):
        """
        A class for describing, searching and compositing an Earth Engine image collection, with support for
        cloud/shadow masking.

        Parameters
        ----------
        ee_collection : ee.ImageCollection
            Earth Engine image collection to encapsulate.
        add_props: list of str, optional
            Additional Earth Engine image properties to include in :attr:`properties`.
        """
        if not isinstance(ee_collection, ee.ImageCollection):
            raise TypeError(f'`ee_collection` must be an instance of ee.ImageCollection')
        self._name = None
        self._properties = None
        self._schema = None
        self._filtered = False
        self._ee_collection = ee_collection
        self._add_props = list(add_props) if add_props else None
        self._image_type = None
        self._stac = None
        self._stats_scale = None

    @classmethod
    def from_name(cls, name: str, add_props: List[str] = None) -> 'MaskedCollection':
        """
        Create a MaskedCollection instance from an Earth Engine image collection name.

        Parameters
        ----------
        name: str
            Name of the Earth Engine image collection to create.
        add_props: list of str, optional
            Additional Earth Engine image properties to include in :attr:`properties`.

        Returns
        -------
        MaskedCollection
            A MaskedCollection instance.
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
        gd_collection = cls(ee_collection, add_props=add_props)
        gd_collection._name = name
        return gd_collection

    @classmethod
    def from_list(
        cls, image_list: List[Union[str, MaskedImage, ee.Image]], add_props: List[str] = None
    ) -> 'MaskedCollection':
        """
        Create a MaskedCollection instance from a list of Earth Engine image ID strings, ``ee.Image`` instances and/or
        :class:`~geedim.mask.MaskedImage` instances.  The list may include composite images, as created with
        :meth:`composite`.

        Images from spectrally compatible Landsat collections can be combined i.e. Landsat-4 with Landsat-5,
        and Landsat-8 with Landsat-9.  Otherwise, images should all belong to the same collection.

        Any ``ee.Image`` instances in the list (including those encapsulated by :class:`~geedim.mask.MaskedImage`)
        must have ``system:id`` and ``system:time_start`` properties.  This is always the case for images
        obtained from the Earth Engine catalog, and images returned from :meth:`composite`.  Any other user-created
        images passed to :meth:`from_list` should have these properties set.

        Parameters
        ----------
        image_list : list
            List of images to include in the collection (must all be from the same, or compatible Earth Engine
            collections). List items can be ID strings, instances of :class:`~geedim.mask.MaskedImage`, or instances of
            ``ee.Image``.
        add_props: list of str, optional
            Additional Earth Engine image properties to include in :attr:`properties`.

        Returns
        -------
        MaskedCollection
            A MaskedCollection instance.
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
            raise InputImageError('Image(s) must have "id" and "system:time_start" properties.')

        # check the images all come from the same or compatible collections
        ee_coll_names = [split_id(im_dict['id'])[0] for im_dict in im_dict_list]
        if not compatible_collections(ee_coll_names):
            raise InputImageError('All images must belong to the same, or spectrally compatible, collections.')

        # create the collection object, using the name of the first collection in ee_coll_names.
        gd_collection = cls.from_name(ee_coll_names[0], add_props=add_props)
        gd_collection._ee_collection = ee.ImageCollection(ee.List([im_dict['ee_image'] for im_dict in im_dict_list]))
        gd_collection._filtered = True
        return gd_collection

    @property
    def stac(self) -> Union[StacItem, None]:
        """ STAC info, if any.  """
        if not self._stac and (self.name in StacCatalog().url_dict):
            self._stac = StacCatalog().get_item(self.name)
        return self._stac

    @property
    def stats_scale(self) -> Union[float, None]:
        """ Scale to use for re-projections when finding region statistics. """
        if not self.stac:
            return None
        if not self._stats_scale:
            gsds = [float(band_dict['gsd']) for band_dict in self.stac.band_props.values()]
            max_gsd = max(gsds)
            min_gsd = min(gsds)
            self._stats_scale = min_gsd if (max_gsd > 10 * min_gsd) and (min_gsd > 0) else max_gsd
        return self._stats_scale

    @property
    def ee_collection(self) -> ee.ImageCollection:
        """ Earth Engine image collection. """
        return self._ee_collection

    @property
    def name(self) -> str:
        """ Name of the encapsulated Earth Engine image collection. """
        if not self._name:
            ee_info = self._ee_collection.first().getInfo()
            self._name = split_id(ee_info['id'])[0] if ee_info and 'id' in ee_info else 'None'
        return self._name

    @property
    def image_type(self) -> type:
        """ :class:`~geedim.mask.MaskedImage` class or sub-class corresponding to images in :attr:`ee_collection`. """
        if not self._image_type:
            self._image_type = class_from_id(self.name)
        return self._image_type

    @property
    def properties(self) -> Dict[str, Dict]:
        """
        A dictionary of properties for each image in the collection. Dictionary keys are the image IDs, and values
        are a dictionaries of image properties.
        """
        if not self._filtered:
            raise UnfilteredError(
                '`properties` can only be retrieved for collections returned by `search()` and `from_list()`'
            )
        if not self._properties:
            self._properties = self._get_properties(self._ee_collection)
        return self._properties

    @property
    def properties_table(self) -> str:
        """ :attr:`properties` formatted as a printable table string. """
        return self._get_properties_table(self.properties)

    @property
    def schema(self) -> Dict[str, Dict]:
        """
        A dictionary of abbreviations and descriptions for :attr:`properties`. Keys are the EE property names,
        and values are dictionaries of abbreviations and descriptions.
        """
        if not self._schema:
            if self.name in schema.collection_schema:
                self._schema = schema.collection_schema[self.name]['prop_schema'].copy()
            else:
                self._schema = schema.default_prop_schema.copy()

            # append any additional properties to the schema
            if self._add_props:
                for add_prop in self._add_props:
                    # get a description from STAC where possible
                    description = (
                        self.stac.descriptions[add_prop] if self.stac and add_prop in self.stac.descriptions else ''
                    )
                    # remove newlines from description and crop to the first sentence
                    description = ' '.join(description.strip().splitlines()).split('. ')[0]
                    self._schema[add_prop] = dict(abbrev=abbreviate(add_prop), description=description)
        return self._schema

    @property
    def schema_table(self) -> str:
        """ :attr:`schema` formatted as a printable table string. """
        table_list = []
        for prop_name, prop_dict in self.schema.items():
            description = '\n'.join(wrap.wrap(prop_dict['description'], 50))
            table_list.append(dict(abbrev=prop_dict['abbrev'], name=prop_name, description=description))
        headers = {key: key.upper() for key in table_list[0].keys()}
        return tabulate.tabulate(table_list, headers=headers, floatfmt='.2f', tablefmt='simple')

    @property
    def refl_bands(self) -> Union[List[str], None]:
        """ List of spectral / reflectance band names, if any. """
        if not self.stac:
            return None
        return [bname for bname, bdict in self.stac.band_props.items() if 'center_wavelength' in bdict]

    def _get_properties(self, ee_collection: ee.ImageCollection) -> Dict:
        """ Retrieve properties of images in a given Earth Engine image collection. """

        # the properties to retrieve
        prop_key_list = ee.List(list(self.schema.keys()))

        def aggregrate_props(ee_image, coll_list):
            im_dict = ee_image.toDictionary(prop_key_list)
            return ee.List(coll_list).add(im_dict)

        # retrieve list of dicts of properties of images in ee_collection
        props_list = []
        try:
            props_list = ee.List(ee_collection.iterate(aggregrate_props, ee.List([]))).getInfo()
        except ee.EEException as ex:
            if 'geometry' in str(ex) and 'unbounded' in str(ex):
                raise ValueError('This collection is unbounded and needs a `region` to be specified.')
            else:
                raise
        # add image properties to the return dict in the same order as the underlying collection
        props_dict = OrderedDict()
        for prop_dict in props_list:
            props_dict[prop_dict['system:id']] = prop_dict
        return props_dict

    def _get_properties_table(self, properties: Dict, schema: Dict = None) -> str:
        """
        Format the given properties into a table.  Orders properties (columns) according :attr:`schema` and
        replaces long form property names with abbreviations.
        """
        if not schema:
            schema = self.schema

        abbrev_props = []
        for im_id, im_prop_dict in properties.items():
            abbrev_dict = OrderedDict()
            for prop_name, key_dict in schema.items():
                if prop_name in im_prop_dict:
                    if prop_name == 'system:time_start':  # convert timestamp to date string
                        dt = datetime.utcfromtimestamp(im_prop_dict[prop_name] / 1000)
                        abbrev_dict[key_dict['abbrev']] = datetime.strftime(dt, '%Y-%m-%d %H:%M')
                    else:
                        abbrev_dict[key_dict['abbrev']] = im_prop_dict[prop_name]
            abbrev_props.append(abbrev_dict)
        return tabulate.tabulate(abbrev_props, headers='keys', floatfmt='.2f', tablefmt=_table_fmt)

    def _prepare_for_composite(
        self, method: CompositeMethod, mask: bool = True, resampling: Union[ResamplingMethod, str] = None,
        date: str = None, region: Dict = None, **kwargs
    ) -> ee.ImageCollection:
        """
        Prepare the Earth Engine collection for compositing. See :meth:`~MaskedCollection.composite` for
        parameter descriptions.
        """

        date = parse_date(date, 'date')

        if not self._filtered:
            raise UnfilteredError(
                'Composites can only be created from collections returned by `search()` and `from_list()`'
            )

        method = CompositeMethod(method)
        resampling = ResamplingMethod(resampling) if resampling else BaseImage._default_resampling
        if (method == CompositeMethod.q_mosaic) and (self.image_type == MaskedImage):
            cloud_coll_str = ", ".join(schema.cloud_coll_names)
            raise ValueError(
                f'The `q-mosaic` method is not supported for this collection: {self.name}.  It is supported for the '
                f'{cloud_coll_str} collections only.'
            )

        def prepare_image(ee_image: ee.Image):
            """ Prepare an Earth Engine image for use in compositing. """
            if date and (method in [CompositeMethod.mosaic, CompositeMethod.q_mosaic]):
                date_dist = ee.Number(ee_image.get('system:time_start')).subtract(ee.Date(date).millis()).abs()
                ee_image = ee_image.set('DATE_DIST', date_dist)

            gd_image = self.image_type(ee_image, **kwargs)
            if region and (method in [CompositeMethod.mosaic, CompositeMethod.q_mosaic]):
                gd_image._set_region_stats(region=region, scale=self.stats_scale)
            if mask:
                gd_image.mask_clouds()
            return resample(gd_image.ee_image, resampling)

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

    def search(
        self, start_date: Union[datetime, str] = None, end_date: Union[datetime, str] = None, region: Dict = None,
        fill_portion: float = None, cloudless_portion: float = None, custom_filter: str = None, **kwargs
    ) -> 'MaskedCollection':
        """
        Search for images based on date, region, filled/cloudless portion, and custom criteria.

        Parameters
        ----------
        start_date : datetime, str
            Start date (UTC).  In '%Y-%m-%d' format if a string.
        end_date : datetime, str, optional
            End date (UTC).  In '%Y-%m-%d' format if a string.  If None, ``end_date`` is set to a day after
            ``start_date``.
        region : dict, ee.Geometry
            Polygon in WGS84 specifying a region that images should intersect.
        fill_portion: float, optional
            Minimum portion (%) of filled (valid) image pixels.
        cloudless_portion: float, optional
            Minimum portion (%) of cloud/shadow free image pixels.
        custom_filter: str, optional
            Custom image property filter expression e.g. "property > value".  See the `EE docs
            <https://developers.google.com/earth-engine/apidocs/ee-filter-expression>`_.
        **kwargs
            Optional cloud/shadow masking parameters - see :meth:`geedim.mask.MaskedImage.__init__` for details.

        Returns
        -------
        MaskedCollection
            Filtered MaskedCollection instance containing the search result image(s).
        """
        if not start_date and not region:
            raise ValueError('At least one of `start_date` or `region` should be specified')

        if not start_date or not region:
            logger.warning('Specifying `start_date` and `region` will improve the search speed.')

        def set_region_stats(ee_image: ee.Image):
            """ Find filled and cloud/shadow free portions inside the search region for a given image.  """
            gd_image = self.image_type(ee_image, **kwargs)
            gd_image._set_region_stats(region, scale=self.stats_scale)
            return gd_image.ee_image

        # filter the image collection, finding cloud/shadow masks and region stats
        ee_collection = self._ee_collection
        if start_date:
            start_date = parse_date(start_date, 'start_date')
            if end_date is None:
                # set end_date a day later than start_date
                end_date = start_date + timedelta(days=1)
            else:
                end_date = parse_date(end_date, 'end_date')
            if end_date <= start_date:
                raise ValueError('`end_date` must be at least a day later than `start_date`')
            ee_collection = ee_collection.filterDate(start_date, end_date)

        if region:
            ee_collection = ee_collection.filterBounds(region)

        # set regions stats before filtering on those properties
        ee_collection = ee_collection.map(set_region_stats)
        if fill_portion:
            ee_collection = ee_collection.filter(ee.Filter.gte('FILL_PORTION', fill_portion))

        if cloudless_portion and self.image_type != MaskedImage:
            ee_collection = ee_collection.filter(ee.Filter.gte('CLOUDLESS_PORTION', cloudless_portion))

        if custom_filter:
            # this expression can include properties from set_region_stats
            ee_collection = ee_collection.filter(ee.Filter.expression(custom_filter))

        ee_collection = ee_collection.sort('system:time_start')

        # return a new MaskedCollection containing the filtered EE collection (the EE collection
        # wrapped by MaskedCollection remains fixed)
        gd_collection = MaskedCollection(ee_collection, add_props=self._add_props)
        gd_collection._name = self._name
        gd_collection._filtered = True
        return gd_collection

    def composite(
        self, method: Union[CompositeMethod, str] = None, mask: bool = True,
        resampling: Union[ResamplingMethod, str] = None, date: Union[datetime, str] = None, region: dict = None,
        **kwargs
    ) -> MaskedImage:
        """
        Create a composite image from the encapsulated image collection.

        Parameters
        ----------
        method: CompositeMethod, str, optional
            Method for finding each composite pixel from the stack of corresponding input image pixels. See
            :class:`~geedim.enums.CompositeMethod` for available options.  By default, `q-mosaic` is used for
            cloud/shadow mask supported collections, `mosaic` otherwise.
        mask: bool, optional
            Whether to apply the cloud/shadow mask; or fill (valid pixel) mask, in the case of images without
            support for cloud/shadow masking.
        resampling: ResamplingMethod, str, optional
            Resampling method to use on collection images prior to compositing.  If None, `near` resampling is used
            (the default).  See :class:`~geedim.enums.ResamplingMethod` for available options.
        date: datetime, str, optional
            Sort collection images by their absolute difference in capture time from this date.  Useful for
            prioritising pixels from images closest to this date.  Valid for the `q-mosaic`
            and `mosaic` ``method`` only.  If None, no time difference sorting is done (the default).
        region: dict, optional
            Sort collection images by their cloudless portion inside this geojson polygon (only if ``date`` is not
            specified).  This is useful to prioritise pixels from the least cloudy image(s).  Valid for the `q-mosaic`
            and `mosaic` ``method`` only.  If None, no cloudless portion sorting is done (the default). If ``date`` and
            ``region`` are not specified, collection images are sorted by their capture date.
        **kwargs
            Optional cloud/shadow masking parameters - see :meth:`geedim.mask.MaskedImage.__init__` for details.

        Returns
        -------
        MaskedImage
            Composite image.
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
            comp_image = ee_collection.median()
        elif method == CompositeMethod.medoid:
            # limit medoid to surface reflectance bands
            comp_image = medoid.medoid(ee_collection, bands=self.refl_bands)
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
        comp_image = comp_image.set('INPUT_IMAGES', 'TABLE:\n' + props_str)

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

        # set the composite capture time to the capture time of the first input image.
        # note that we must specify the timezone as utc, otherwise .timestamp() assumes local time.
        timestamp = min(dates).replace(tzinfo=timezone.utc).timestamp() * 1000
        comp_image = comp_image.set('system:time_start', timestamp)
        gd_comp_image = self.image_type(comp_image)
        gd_comp_image._id = comp_id  # avoid getInfo() for id property
        return gd_comp_image
