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

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from functools import cached_property
from typing import List, Union

import ee
import tabulate
from tabulate import DataRow, Line, TableFormat

from geedim import schema
from geedim.download import BaseImage
from geedim.enums import CompositeMethod, ResamplingMethod
from geedim.errors import InputImageError
from geedim.mask import class_from_id, _MaskedImage, MaskedImage
from geedim.medoid import medoid
from geedim.stac import StacCatalog, StacItem
from geedim.utils import split_id, register_accessor

logger = logging.getLogger(__name__)
tabulate.MIN_PADDING = 0

##
# tabulate format for collection properties
_tablefmt = TableFormat(
    lineabove=Line('', '-', ' ', ''),
    linebelowheader=Line('', '-', ' ', ''),
    linebetweenrows=None,
    linebelow=Line('', '-', ' ', ''),
    headerrow=DataRow('', ' ', ''),
    datarow=DataRow('', ' ', ''),
    padding=0,
    with_header_hide=['lineabove', 'linebelow'],
)


def _compatible_collections(names: List[str]) -> bool:
    """Test if the given collection names are spectrally compatible (either identical or
    compatible Landsat collections).
    """
    names = list(set(names))  # reduce to unique values
    landsat_regex = re.compile(r'(LANDSAT/\w{2})(\d{2})(/.*)')
    landsat_match = names[0] and landsat_regex.search(names[0])
    for name in names[1:]:
        if name and landsat_match:
            landsat_regex = re.compile(
                rf'{landsat_match.groups()[0]}\d\d{landsat_match.groups()[-1]}'
            )
            if not landsat_regex.search(name):
                return False
        elif not name == names[0]:
            return False
    return True


def abbreviate(name: str) -> str:
    """Return an acronym for a string in camel or snake case."""
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


@register_accessor('gd', ee.ImageCollection)
class ImageCollectionAccessor:
    # TODO: necessary?
    _sort_methods = [CompositeMethod.mosaic, CompositeMethod.q_mosaic, CompositeMethod.medoid]

    def __init__(self, ee_coll: ee.ImageCollection):
        self._ee_coll = ee_coll
        self._name = None
        self._schema = None
        self._properties = None
        self._stac = None
        self._stats_scale = None
        self._table_properties = None
        self._schema = None

    @staticmethod
    def load(name: str, add_props: List[str] = None) -> ee.ImageCollection:
        ee_coll = ee.ImageCollection(name)
        ee_coll.gd._name = name
        return ee_coll

    @staticmethod
    def fromList(images: list[str | ee.Image]) -> ee.ImageCollection:
        # TODO: previously there was error checking here to see if images were in compatible
        #  collections and all had IDs and dates.  Rather than check here, check/get search and
        #  composite to behave sensibly when this is not the case.
        # TODO: if this is going to be included, make it fromList?
        # Note: I have added this, partially as a testing convenience function, to allow creating a
        # cloud/shadow supported collection from individual images / IDs.
        ee_coll = ee.ImageCollection(images)
        names = [
            split_id(im_props.get('id', None))[0]
            for im_props in ee_coll.gd.properties.get('features', [])
        ]
        if not _compatible_collections(names):
            raise InputImageError(
                'All images must belong to the same, or spectrally compatible, collections.'
            )

        # TODO: should raise an error if any/all names are None.  And what happens if it is a
        #  list of composites with new naming?  Use the term homogenous.  See errors from
        #  reduceRegion.
        # persist the source collection name to enable cloud/shadow masking
        ee_coll = ee_coll.set('system:id', names[0])
        ee_coll.gd._name = names[0]
        return ee_coll

    @property
    def name(self) -> str | None:
        # TODO: rather than getInfo for name and later getInfos for properties, do a getInfo for
        #  ee.ImageCollection.limit, then we have properties and name?  The typical use case
        #  would be to run addAuxBands and maskClouds on the collection first though,
        #  both of which only need name, then get properties on final collection.  Having _name
        #  separate also allows it to be set when returning collections from addAuxBands and
        #  maskClouds as these methods won't affect the collection name.
        if not self._name:
            if self._properties:
                self._name = self._properties.get('id', None)
            else:
                self._name = self._ee_coll.get('system:id').getInfo()
        return self._name

    @cached_property
    def _mi(self) -> type[_MaskedImage]:
        return class_from_id(self.name)

    @property
    def stac(self) -> StacItem | None:
        # TODO: used cached_property decorator here and elsewhere if possible
        if not self._stac and (self.name in StacCatalog().url_dict):
            self._stac = StacCatalog().get_item(self.name)
        return self._stac

    @property
    def stats_scale(self) -> float | None:
        """Scale to use for re-projections when finding region statistics."""
        if not self.stac:
            return None
        if not self._stats_scale:
            gsds = [float(band_dict['gsd']) for band_dict in self.stac.band_props.values()]
            max_gsd = max(gsds)
            min_gsd = min(gsds)
            self._stats_scale = min_gsd if (max_gsd > 10 * min_gsd) and (min_gsd > 0) else max_gsd
        return self._stats_scale

    @property
    def properties(self) -> dict:
        if not self._properties:
            self._properties = self._ee_coll.limit(1000).select(None).getInfo()
        return self._properties

    @cached_property
    def table_properties(self) -> list[str]:
        if self.name in schema.collection_schema:
            default_schema = schema.collection_schema[self.name]['prop_schema']
        else:
            default_schema = schema.default_prop_schema
        return list(default_schema)

    # @property
    # def table_properties(self) -> list[str]:
    #     if not self._table_properties:
    #         if self.name in schema.collection_schema:
    #             default_schema = schema.collection_schema[self.name]['prop_schema']
    #         else:
    #             default_schema = schema.default_prop_schema
    #         self._table_properties = list(default_schema)
    #     return self._table_properties
    #
    # @table_properties.setter
    # def table_properties(self, value: list[str]):
    #     self._table_properties = value

    @property
    def schema(self) -> dict[str, dict]:
        """A dictionary of abbreviations and descriptions for :attr:`properties`. Keys are the EE
        property names, and values are dictionaries of abbreviations and descriptions.
        """
        if not self._schema or list(self._schema.keys()) != self.table_properties:
            if self.name in schema.collection_schema:
                coll_schema = schema.collection_schema[self.name]['prop_schema']
            else:
                coll_schema = schema.default_prop_schema
            self._schema = {}
            for prop_name in self.table_properties:
                if prop_name in coll_schema:
                    prop_schema = coll_schema[prop_name]
                elif prop_name in self.stac.descriptions:
                    prop_schema = dict(
                        abbrev=abbreviate(prop_name), description=self.stac.descriptions[prop_name]
                    )
                else:
                    prop_schema = dict(abbrev=abbreviate(prop_name), description=None)
                self._schema[prop_name] = prop_schema
        return self._schema

    @property
    def images_table(self) -> str:
        table_list = []
        for im_info in self.properties.get('features', []):
            im_props = im_info.get('properties', {})
            row_dict = {}
            for prop_name, prop_dict in self.schema.items():
                prop_val = im_props.get(prop_name, None)
                if prop_name in ['system:time_start', 'system:time_end'] and prop_val:
                    # convert timestamp to date string
                    dt = datetime.fromtimestamp(prop_val / 1000, tz=timezone.utc)
                    row_dict[prop_dict['abbrev']] = datetime.strftime(dt, '%Y-%m-%d %H:%M')
                else:
                    row_dict[prop_dict['abbrev']] = prop_val
            table_list.append(row_dict)

        return tabulate.tabulate(
            table_list, headers='keys', floatfmt='.2f', tablefmt=_tablefmt, missingval='-'
        )

    @property
    def schema_table(self) -> str:
        """:attr:`schema` formatted as a printable table string."""
        # TODO: could tables be a sub-properties of parent item like schema.as_table?  or perhaps
        #  done in another class or function.
        table_list = [
            dict(ABBREV=prop_dict['abbrev'], NAME=prop_name, DESCRIPTION=prop_dict['description'])
            for prop_name, prop_dict in self.schema.items()
        ]
        return tabulate.tabulate(
            table_list,
            headers='keys',
            floatfmt='.2f',
            tablefmt='simple',
            maxcolwidths=50,
        )

    @property
    def refl_bands(self) -> Union[List[str], None]:
        """List of spectral / reflectance band names, if any."""
        if not self.stac:
            return None
        return [
            bname for bname, bdict in self.stac.band_props.items() if 'center_wavelength' in bdict
        ]

    def _get_images_table(self, properties: dict, schema: dict = None) -> str:
        """
        Format the given properties into a table.  Orders properties (columns) according to :attr:`schema` and
        replaces long form property names with abbreviations.
        """
        schema = schema or self.schema
        table_props = []
        for im_info in properties.get('features', []):
            # TODO: give a default enumerated ID/index?
            im_props = im_info.get('properties', {})
            row_props = {}
            for prop_name, prop_schema in schema.items():
                prop_val = im_props.get(prop_name, None)
                if prop_name in ['system:time_start', 'system:time_end'] and prop_val:
                    # convert timestamp to date string
                    dt = datetime.fromtimestamp(prop_val / 1000, tz=timezone.utc)
                    row_props[prop_schema['abbrev']] = datetime.strftime(dt, '%Y-%m-%d %H:%M')
                else:
                    row_props[prop_schema['abbrev']] = prop_val
            table_props.append(row_props)

        return tabulate.tabulate(
            table_props, headers='keys', floatfmt='.2f', tablefmt=_tablefmt, missingval='-'
        )

    def _set_portions(self, ee_image: ee.Image, region: dict | ee.Geometry) -> ee.Image:
        # TODO: make common or protected member of ImageAccessor
        mask_names = ee_image.bandNames().filter(
            ee.Filter.inList('item', ['CLOUDLESS_MASK', 'FILL_MASK'])
        )
        portions = ee_image.select(mask_names).gd.maskCoverRegion(
            region=region, scale=self.stats_scale, maxPixels=1e6
        )
        fill_portion = ee.Number(portions.get('FILL_MASK'))
        cl_portion = ee.Number(portions.get('CLOUDLESS_MASK', fill_portion))
        cl_portion = cl_portion.divide(fill_portion).multiply(100)
        return ee_image.set('FILL_PORTION', fill_portion, 'CLOUDLESS_PORTION', cl_portion)

    def _prepare_for_composite(
        self,
        method: str | CompositeMethod,
        mask: bool = True,
        resampling: str | ResamplingMethod = None,
        date: str | datetime | ee.Date = None,
        region: dict | ee.Geometry = None,
        **kwargs,
    ) -> ee.ImageCollection:
        """Prepare the Earth Engine collection for compositing. See
        :meth:`~MaskedCollection.composite` for parameter descriptions.
        """

        method = CompositeMethod(method)
        resampling = ResamplingMethod(resampling) if resampling else BaseImage._default_resampling
        if (method == CompositeMethod.q_mosaic) and (self.name not in schema.cloud_coll_names):
            # TODO: errors/warnings if cloud/shadow masking not supported and mask is True
            raise ValueError(
                f"The 'q-mosaic' method is not supported for this collection: '{self.name}'.  It is"
                f"supported for the {schema.cloud_coll_names} collections only."
            )

        def prepare_image(ee_image: ee.Image) -> ee.Image:
            """Prepare an Earth Engine image for use in compositing."""
            # TODO: we want aux bands overwritten if it is fixed proj, and not otherwise
            # TODO: always add aux bands, or only when needed?
            if date and (method in self._sort_methods):
                # TODO: nevermind about _sort_methods, just do these things anyway?  i'm assuming
                #  computation time is minimal
                # TODO: timezone awareness & conversion to UTC
                date_dist = (
                    ee.Number(ee_image.date().millis()).subtract(ee.Date(date).millis()).abs()
                )
                ee_image = ee_image.set('DATE_DIST', date_dist)

            ee_image = self._mi.add_mask_bands(ee_image, **kwargs)
            if region and (method in self._sort_methods):
                ee_image = self._mi.set_mask_portions(
                    ee_image, region=region, scale=self.stats_scale
                )

            if mask:
                # TODO: masking must be done after setting portions
                ee_image = self._mi.mask_clouds(ee_image)
            return ee_image.gd.resample(resampling)

        ee_coll = self._ee_coll.map(prepare_image)

        if method in self._sort_methods:
            if date:
                ee_coll = ee_coll.sort('DATE_DIST', ascending=False)
            elif region:
                sort_key = (
                    'CLOUDLESS_PORTION'
                    if 'CLOUDLESS_PORTION' in self.schema.keys()
                    else 'FILL_PORTION'
                )
                ee_coll = ee_coll.sort(sort_key)
            else:
                ee_coll = ee_coll.sort('system:time_start')
        else:
            if date:
                logger.warning(f"'date' is valid for {self._sort_methods} methods only.")
            elif region:
                logger.warning(f"'region' is valid for {self._sort_methods} methods only.")

        return ee_coll

    def addMaskBands(self, **kwargs) -> ee.ImageCollection:
        # TODO: this will keep adding extra aux bands by default.  Is this compatible with old
        #  MaskedImage expected behaviour?  For a composite with existing aux bands, this will
        #  add extra (unusable) aux bands with overwrite=False, but perhaps that doesn't matter
        #  as the original aux bands will get used in maskClouds, and the added aux bands would
        #  be excluded from a further composite.  If the added aux bands for a composite generate
        #  errors, they should not be added of course, but I don't think this is the case.
        ee_coll = self._ee_coll.map(lambda ee_image: self._mi.add_mask_bands(ee_image, **kwargs))
        ee_coll.gd._name = self._name
        return ee_coll

    def maskClouds(self, **kwargs) -> ee.ImageCollection:
        # TODO: is it more efficient to only get mask, excluding cloud distance etc
        # TODO: do we want to mask with FILL_MASK when there is no CLOUDLESS_MASK,
        #  and incorporate it into the CLOUDLESS_MASK when there is?  FILL_MASK is used for
        #  search filtering on fill portion property, but is to useful to mask with it over EE
        #  mask? If FILL_MASK masking is abandoned, we should test the e.g. Landsat-7 mask
        #  defines the invalid areas.
        # TODO: neither this nor addAuxBands can be mapped over an ImageCollection as they
        #  require getInfo
        ee_coll = self._ee_coll.map(lambda ee_image: self._mi.mask_clouds(ee_image))
        ee_coll.gd._name = self._name
        return ee_coll

    def medoid(self, bands: list[str | ee.String] = None) -> ee.Image:
        return medoid(self._ee_coll, bands=bands)

    def search(
        self,
        start_date: str | datetime | ee.Date = None,
        end_date: str | datetime | ee.Date = None,
        region: dict | ee.Geometry = None,
        fill_portion: float | ee.Number = None,
        cloudless_portion: float | ee.Number = None,
        custom_filter: str = None,
        **kwargs,
    ) -> ee.ImageCollection:
        # TODO: test for these errors
        if not start_date and not region:
            raise ValueError("At least one of 'start_date' or 'region' should be specified")
        if (fill_portion is not None or cloudless_portion is not None) and not region:
            raise ValueError(
                "'region' is required when 'fill_portion' or 'cloudless_portion' are specified."
            )

        if not start_date or not region:
            logger.warning("Specifying 'start_date' and 'region' will improve the search speed.")

        # filter the image collection, finding cloud/shadow masks and region stats
        ee_coll = self._ee_coll
        if start_date:
            # default end_date a day later than start_date
            end_date = end_date or ee.Date(start_date).advance(1, unit='day')
            ee_coll = ee_coll.filterDate(start_date, end_date)

        if region:
            ee_coll = ee_coll.filterBounds(region)

        # when possible filter on custom_filter before calling set_region_stats to reduce computation
        if custom_filter and all(
            [prop_key not in custom_filter for prop_key in ['FILL_PORTION', 'CLOUDLESS_PORTION']]
        ):
            ee_coll = ee_coll.filter(ee.Filter.expression(custom_filter))
            custom_filter = None

        if (fill_portion is not None) or (cloudless_portion is not None) or custom_filter:
            # set regions stats before filtering on those properties
            # TODO: we want aux bands overwritten if it is fixed proj image, and not otherwise
            # TODO: add and set portions in one mapping fn.  perhaps don't add bands rather just
            #  set portions from separate aux image.  but if a composite aux bands should come
            #  from itself.
            # TODO: is there a neater & automatic way to copy name here & id in image?
            # copy name to avoid getInfo in addAuxBands
            ee_coll.gd._name = self._name

            def set_portions(ee_image: ee.Image) -> ee.Image:
                ee_image = self._mi.add_mask_bands(ee_image, **kwargs)
                return self._mi.set_mask_portions(ee_image, region=region, scale=self.stats_scale)

            ee_coll = ee_coll.map(set_portions)
            if fill_portion:
                ee_coll = ee_coll.filter(ee.Filter.gte('FILL_PORTION', fill_portion))
            if cloudless_portion:
                ee_coll = ee_coll.filter(ee.Filter.gte('CLOUDLESS_PORTION', cloudless_portion))

            # filter on custom_filter that refers to FILL_ or CLOUDLESS_PORTION
            if custom_filter:
                # this expression can include properties from set_region_stats
                ee_coll = ee_coll.filter(ee.Filter.expression(custom_filter))

        ee_coll = ee_coll.sort('system:time_start')

        ee_coll.gd._name = self._name
        return ee_coll

    def composite(
        self,
        method: CompositeMethod | str = None,
        mask: bool = True,
        resampling: ResamplingMethod | str = None,
        date: str | datetime | ee.Date = None,
        region: dict | ee.Geometry = None,
        **kwargs,
    ) -> ee.Image:
        if method is None:
            method = (
                CompositeMethod.q_mosaic
                if self.name in schema.cloud_coll_names
                else CompositeMethod.mosaic
            )

        # mask, sort & resample the EE collection
        method = CompositeMethod(method)
        ee_coll = self._prepare_for_composite(
            method=method, mask=mask, resampling=resampling, date=date, region=region, **kwargs
        )

        if method == CompositeMethod.q_mosaic:
            comp_image = ee_coll.qualityMosaic('CLOUD_DIST')
        elif method == CompositeMethod.medoid:
            # limit medoid to surface reflectance bands
            comp_image: ee.Image = ee_coll.gd.medoid(bands=self.refl_bands)
        else:
            comp_image = getattr(ee_coll, method.name)()

        # populate composite image metadata
        # TODO: can this be done server side to save time?
        # TODO: don't do this in accessor but in MaskedCollection subclass for backwards
        #  compatibility.  or can properties be set server side in some way (like a dict) that
        #  GDAL / rasterio understands and is friendly to display
        comp_index = ee.String(f'{method.value.upper()}-COMP')
        # set 'properties'->'system:index'
        comp_image = comp_image.set('system:index', comp_index)
        # set root 'id' property
        comp_id = ee.String(self.name + '/') if self.name else ee.String('')
        comp_id = comp_id.cat(comp_index)
        comp_image = comp_image.set('system:id', comp_id)
        # # set composite start-end times
        date_range = ee_coll.reduceColumns(ee.Reducer.minMax(), ['system:time_start'])
        comp_image = comp_image.set('system:time_start', ee.Number(date_range.get('min')))
        comp_image = comp_image.set('system:time_end', ee.Number(date_range.get('max')))
        return comp_image


class MaskedCollection(ImageCollectionAccessor):

    def __init__(self, ee_collection: ee.ImageCollection, add_props: list[str] = None):
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
        super().__init__(ee_collection)
        if add_props:
            self.table_properties += add_props

    @classmethod
    def from_name(cls, name: str, add_props: list[str] = None) -> MaskedCollection:
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
        ee_coll = ee.ImageCollection(name)
        gd_coll = cls(ee_coll, add_props=add_props)
        gd_coll._name = name
        return gd_coll

    @classmethod
    def from_list(
        cls, image_list: list[str | MaskedImage | ee.Image], add_props: list[str] = None
    ) -> MaskedCollection:
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
        if len(image_list) == 0:
            raise ValueError("'image_list' is empty.")

        images = [
            image.ee_image if isinstance(image, BaseImage) else ee.Image(image)
            for image in image_list
        ]
        # TODO: previously there were checks for IDs (should be implicit in
        #  compatible_collections - to check) and capture times. Does seach / composite work OK w/o
        #  these?
        ee_coll = super().fromList(images)
        return cls(ee_coll, add_props=add_props)

    @property
    def ee_collection(self) -> ee.ImageCollection:
        """Earth Engine image collection."""
        return self._ee_coll

    @property
    def properties(self) -> dict:
        """A dictionary of properties for each image in the collection. Dictionary keys are the
        image IDs, and values are a dictionaries of image properties.
        """

        # convert to legacy format
        coll_props = {}
        for i, im_info in enumerate(super().properties.get('features', [])):
            im_props = im_info.get('properties', {})
            im_props_schema = {key: im_props[key] for key in self.schema.keys() if key in im_props}
            im_id = im_info.get('id', i)
            # TODO: the schema system:id needs a workaround now that getInfo is used for
            #  properties in the parent class, and should be changed to avoid this
            if 'system:id' in self.schema.keys():
                im_props_schema['system:id'] = im_id
            coll_props[im_id] = im_props_schema
        return coll_props

    @property
    def properties_table(self) -> str:
        """:attr:`properties` formatted as a printable table string."""
        return self._get_images_table(super().properties, self.schema)

    def search(self, *args, **kwargs) -> MaskedCollection:
        """
        Search for images based on date, region, filled/cloudless portion, and custom criteria.

        Filled and cloudless portions are only calculated and included in collection properties
        when one or both of ``fill_portion`` / ``cloudless_portion`` are specified.

        Search speed can be increased by specifying ``custom_filter``, and or by omitting
        ``fill_portion`` / ``cloudless_portion``.

        Parameters
        ----------
        start_date : datetime, str
            Start date (UTC).  In '%Y-%m-%d' format if a string.

        end_date : datetime, str, optional End date (UTC).
            In '%Y-%m-%d' format if a string.  If None, ``end_date`` is set to a day after
            ``start_date``.
        region: dict, ee.Geometry
            Region that images should intersect as a GeoJSON dictionary or ``ee.Geometry``.
        fill_portion: float, optional
            Lower limit on the portion of region that contains filled/valid image pixels (%).
        cloudless_portion: float, optional
            Lower limit on the portion of filled pixels that are cloud/shadow free (%).
        custom_filter: str, optional
            Custom image property filter expression e.g. "property > value".  See the `EE docs
            <https://developers.google.com/earth-engine/apidocs/ee-filter-expression>`_.
        **kwargs
            Optional cloud/shadow masking parameters - see :meth:`geedim.mask.MaskedImage.__init__`
            for details.

        Returns
        -------
        MaskedCollection
            Filtered MaskedCollection instance containing the search result image(s).
        """
        ee_coll = self._ee_coll.gd.search(*args, **kwargs)
        gd_coll = MaskedCollection(ee_coll)
        gd_coll._name = self._name
        gd_coll.table_properties = self.table_properties
        return gd_coll

    def composite(self, *args, **kwargs) -> MaskedImage:
        """
        Create a composite image from the encapsulated image collection.

        Parameters
        ----------
        method: CompositeMethod, str, optional
            Method for finding each composite pixel from the stack of corresponding input image
            pixels. See :class:`~geedim.enums.CompositeMethod` for available options.  By
            default, `q-mosaic` is used for cloud/shadow mask supported collections, `mosaic`
            otherwise.
        mask: bool, optional
            Whether to apply the cloud/shadow mask; or fill (valid pixel) mask, in the case of
            images without support for cloud/shadow masking.
        resampling: ResamplingMethod, str, optional
            Resampling method to use on collection images prior to compositing.  If None,
            `near` resampling is used (the default).  See :class:`~geedim.enums.ResamplingMethod`
            for available options.
        date: datetime, str, optional
            Sort collection images by their absolute difference in capture time from this date.
            Pioritises pixels from images closest to this date.  Valid for the `q-mosaic`,
            `mosaic` and `medoid` ``method`` only.  If None, no time difference sorting is done (
            the default).
        region: dict, ee.Geometry, optional
            Sort collection images by the portion of their pixels that are cloudless, and inside
            this region.  Can be a GeoJSON dictionary or ee.Geometry.  Prioritises pixels from
            the least cloudy image(s). Valid for the `q-mosaic`, `mosaic`, and `medoid`
            ``method`` only.  If collection has no cloud/shadow mask support, images are sorted
            by the portion of their pixels that are valid, and inside ``region``. If None,
            no cloudless/valid portion sorting is done (the default).  If ``date`` and ``region``
            are not specified, collection images are sorted by their capture date.
        **kwargs
            Optional cloud/shadow masking parameters - see :meth:`geedim.mask.MaskedImage.__init__`
            for details.

        Returns
        -------
        MaskedImage
            Composite image.
        """
        ee_image = self._ee_coll.gd.composite(*args, **kwargs)
        gd_image = MaskedImage(ee_image)
        return gd_image
