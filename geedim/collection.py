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

import json
import logging
import os
import posixpath
import re
import warnings
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from functools import cached_property
from typing import Any

import ee
import fsspec
import numpy as np
import tabulate
from fsspec.core import OpenFile
from tabulate import DataRow, Line, TableFormat
from tqdm.auto import tqdm

from geedim import schema, utils
from geedim.download import BaseImage
from geedim.enums import CompositeMethod, Driver, ExportType, ResamplingMethod, SplitType
from geedim.image import ImageAccessor
from geedim.mask import MaskedImage, _CloudlessImage, _get_class_for_id, _MaskedImage
from geedim.medoid import medoid
from geedim.stac import STACClient
from geedim.tile import Tiler
from geedim.utils import register_accessor, split_id

try:
    import xarray
    from pandas import to_datetime
except ImportError:
    xarray = None

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


def _compatible_collections(ids: list[str]) -> bool:
    """Test if the given collection IDs are spectrally compatible (either identical or compatible
    Landsat collections).
    """
    ids = list(set(ids))  # reduce to unique values
    landsat_regex = re.compile(r'(LANDSAT/\w{2})(\d{2})(/.*)')
    landsat_match = ids[0] and landsat_regex.search(ids[0])
    for name in ids[1:]:
        if name and landsat_match:
            landsat_regex = re.compile(
                rf'{landsat_match.groups()[0]}\d\d{landsat_match.groups()[-1]}'
            )
            if not landsat_regex.search(name):
                return False
        elif not name == ids[0]:
            return False
    return True


def _abbreviate(name: str) -> str:
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
    _max_export_images = 5000

    def __init__(self, ee_coll: ee.ImageCollection):
        """
        Accessor for describing, cloud / shadow masking and exporting an image collection.

        :param ee_coll:
            Image collection to access.
        """
        # TODO: strictly this is an _init_ docstring, not class docstring.  does moving it (and all
        #  other class docstrings) to _init_ work with sphinx?
        self._ee_coll = ee_coll
        self._info = None
        self._schema = None
        self._schema_prop_names = None
        self._properties = None

    @staticmethod
    def fromImages(images: Any) -> ee.ImageCollection:
        """
        Create an image collection with support for cloud/shadow masking, that contains the given
        images.

        Images from spectrally compatible Landsat collections can be combined i.e. Landsat-4 with
        Landsat-5, and Landsat-8 with Landsat-9.  Otherwise, images should all belong to the same
        collection.  Images may include composites as created with :meth:`composite`.  Composites
        are treated as belonging to the collection of their component images.

        Use this method (instead of :meth:`ee.ImageCollection` or
        :meth:`ee.ImageCollection.fromImages`) to support cloud/shadow masking on a collection
        built from a sequence of images.

        :param images:
            Sequence of images, or anything that can be used to construct an image.

        :return:
            Image collection.
        """
        # TODO: previously there was error checking here to see if images all had IDs and dates.
        #  Rather than check here, check/get search and composite to behave sensibly when this is
        #  not the case.
        ee_coll = ee.ImageCollection(images)

        # check the images are from compatible collections
        # TODO: get all the info and set the collection property so it doesn't have to call
        #  getInfo() again.
        info = ee_coll.select(None).getInfo()
        ids = [split_id(im_props.get('id', None))[0] for im_props in info.get('features', [])]
        if not _compatible_collections(ids):
            # TODO: test raises an error if any/all names are None
            raise ValueError(
                'All images must belong to the same, or spectrally compatible, collections.'
            )

        # set the collection ID to enable cloud/shadow masking if supported
        ee_coll = ee_coll.set('system:id', ids[0])
        return ee_coll

    @cached_property
    def _mi(self) -> type[_MaskedImage]:
        """Masking method container."""
        return _get_class_for_id(self.id)

    @cached_property
    def _portion_scale(self) -> float | int | None:
        """Scale to use for finding mask portions.  ``None`` if there is no STAC band information
        for this collection.
        """
        if not self.stac:
            return None

        # TODO: test different global and per-band gsd cases (e.g. MODIS/MCD43A1,
        #  COPERNICUS/S5P/OFFL/L3_O3_TCL, NASA/GSFC/MERRA/aer_nv/2, COPERNICUS/S2_SR_HARMONIZED,
        #  LANDSAT/LC08/C02/T1_L2)
        # derive scale from the global GSD if it exists
        summaries = self.stac.get('summaries', {})
        global_gsd = summaries.get('gsd', None)
        if global_gsd:
            return float(np.sqrt(np.prod(global_gsd))) if len(global_gsd) > 1 else global_gsd[0]

        # derive scale from band GSDs if they exist
        band_props = summaries.get('eo:bands', [])
        band_gsds = [float(bp['gsd']) for bp in band_props if 'gsd' in bp]
        if not band_gsds:
            return None
        max_scale = max(band_gsds)
        min_scale = min(band_gsds)
        return min_scale if (max_scale > 10 * min_scale) and (min_scale > 0) else max_scale

    @cached_property
    def _first(self) -> ImageAccessor:
        """Accessor to the first image of the collection."""
        return ImageAccessor(self._ee_coll.first())

    @cached_property
    def id(self) -> str | None:
        """Earth Engine ID."""
        # Get the ID from self._info if it has been cached.  Otherwise get the ID directly rather
        # than retrieving self._info, which can be time-consuming.
        if self._info is not None:
            return self._info.get('id', None)
        else:
            return self._ee_coll.get('system:id').getInfo()

    @property
    def stac(self) -> dict[str, Any] | None:
        """STAC dictionary.  ``None`` if there is no STAC entry for this collection."""
        return STACClient().get(self.id)

    @property
    def info(self) -> dict[str, Any]:
        """Earth Engine information as returned by :meth:`ee.ImageCollection.getInfo`,
        but limited to the first 5000 images.
        """
        if self._info is None:
            self._info = self._ee_coll.limit(self._max_export_images).getInfo()
        return self._info

    @property
    def schemaPropertyNames(self) -> tuple[str]:
        """:attr:`schema` property names."""
        if self._schema_prop_names is None:
            if self.id in schema.collection_schema:
                self._schema_prop_names = schema.collection_schema[self.id]['prop_schema'].keys()
            else:
                self._schema_prop_names = schema.default_prop_schema.keys()
            # use tuple to prevent in-place mutations that don't go through the setter
            self._schema_prop_names = tuple(self._schema_prop_names)
        return self._schema_prop_names

    @schemaPropertyNames.setter
    def schemaPropertyNames(self, value: tuple[str]):
        if not isinstance(value, Iterable) or not all(isinstance(n, str) for n in value):
            raise ValueError("'schemaPropertyNames' should be an iterable of strings.")
        # remove duplicates, keeping order (https://stackoverflow.com/a/17016257)
        self._schema_prop_names = tuple(dict.fromkeys(value))
        # reset the schema
        self._schema = None

    @property
    def schema(self) -> dict[str, dict]:
        """Dictionary of property abbreviations and descriptions used to form
        :attr:`propertiesTable`.
        """
        if self._schema is None:
            if self.id in schema.collection_schema:
                coll_schema = schema.collection_schema[self.id]['prop_schema']
            else:
                coll_schema = schema.default_prop_schema

            self._schema = {}

            # get STAC property descriptions (if any)
            # TODO: this gets STAC even if its not needed.  can we get it only when all
            #  schemaPropertyNames are not in schema module
            summaries = self.stac.get('summaries', {}) if self.stac else {}
            gee_schema = summaries.get('gee:schema', {})
            gee_descriptions = {item['name']: item['description'] for item in gee_schema}

            for prop_name in self.schemaPropertyNames:
                if prop_name in coll_schema:
                    prop_schema = coll_schema[prop_name]
                elif prop_name in gee_descriptions:
                    descr = gee_descriptions[prop_name]
                    # remove newlines from description and crop to the first sentence
                    descr = ' '.join(descr.strip().splitlines()).split('. ')[0]
                    prop_schema = dict(abbrev=_abbreviate(prop_name), description=descr)
                else:
                    prop_schema = dict(abbrev=_abbreviate(prop_name), description=None)
                self._schema[prop_name] = prop_schema

        return self._schema

    @property
    def schemaTable(self) -> str:
        """:attr:`schema` formatted as a printable table string."""
        if not self.schema:
            return ''
        # cast description to str to work around
        # https://github.com/astanin/python-tabulate/issues/312
        table_list = [
            dict(ABBREV=pd['abbrev'], NAME=pn, DESCRIPTION=str(pd['description']))
            for pn, pd in self.schema.items()
        ]
        return tabulate.tabulate(
            table_list,
            headers='keys',
            floatfmt='.2f',
            tablefmt='simple',
            maxcolwidths=50,
            missingval='-',
        )

    @property
    def properties(self) -> dict[str, dict[str, Any]]:
        """Dictionary of image properties.  Keys are the image indexes and values the image
        property dictionaries.
        """
        if self._properties is None:
            self._properties = {}
            for i, im_info in enumerate(self.info.get('features', [])):
                im_props = im_info.get('properties', {})
                # collection images should always have unique indexes
                im_index = im_props.get('system:index', str(i))
                self._properties[im_index] = im_props
        return self._properties

    @property
    def propertiesTable(self) -> str:
        """:attr:`properties` formatted with :attr:`schema` as a printable table string."""
        coll_schema_props = []
        for im_props in self.properties.values():
            im_schema_props = {}
            for prop_name, prop_schema in self.schema.items():
                prop_val = im_props.get(prop_name, None)
                if prop_val is not None:
                    if prop_name in ['system:time_start', 'system:time_end']:
                        # convert timestamp to date string
                        dt = datetime.fromtimestamp(prop_val / 1000, tz=timezone.utc)
                        im_schema_props[prop_schema['abbrev']] = datetime.strftime(
                            dt, '%Y-%m-%d %H:%M'
                        )
                    else:
                        im_schema_props[prop_schema['abbrev']] = prop_val
            coll_schema_props.append(im_schema_props)
        return tabulate.tabulate(
            coll_schema_props,
            headers='keys',
            floatfmt='.2f',
            tablefmt=_tablefmt,
            missingval='-',
        )

    @property
    def specBands(self) -> list[str] | None:
        """List of spectral band names.  ``None`` if there is no :attr:`stac` entry,
        or no spectral bands.
        """
        if not self.stac:
            return None
        return self._first.specBands

    @property
    def cloudShadowSupport(self) -> bool:
        """Whether this collection has cloud/shadow support."""
        return issubclass(self._mi, _CloudlessImage)

    def _prepare_for_composite(
        self,
        method: CompositeMethod | str,
        mask: bool = True,
        resampling: ResamplingMethod | str = ImageAccessor._default_resampling,
        date: str | datetime | ee.Date = None,
        region: dict | ee.Geometry = None,
        **kwargs,
    ) -> ee.ImageCollection:
        """Return a collection that has been prepared for compositing."""
        method = CompositeMethod(method)
        resampling = ResamplingMethod(resampling)
        sort_methods = [CompositeMethod.mosaic, CompositeMethod.q_mosaic, CompositeMethod.medoid]

        if (method is CompositeMethod.q_mosaic) and (not self.cloudShadowSupport):
            raise ValueError(
                "The 'q-mosaic' method requires cloud / shadow masking support, which this "
                "collection does not have."
            )

        if date and region:
            # TODO: test for this error
            raise ValueError("One of 'date' or 'region' can be supplied, but not both.")

        def prepare_image(ee_image: ee.Image) -> ee.Image:
            """Prepare an Earth Engine image for use in compositing."""
            if date and (method in sort_methods):
                date_dist = (
                    ee.Number(ee_image.date().millis()).subtract(ee.Date(date).millis()).abs()
                )
                ee_image = ee_image.set('DATE_DIST', date_dist)

            ee_image = self._mi.add_mask_bands(ee_image, **kwargs)
            if region and (method in sort_methods):
                ee_image = self._mi.set_mask_portions(
                    ee_image, region=region, scale=self._portion_scale
                )
            if mask:
                ee_image = self._mi.mask_clouds(ee_image)
            return ImageAccessor(ee_image).resample(resampling)

        ee_coll = self._ee_coll.map(prepare_image)

        if method in sort_methods:
            if date:
                ee_coll = ee_coll.sort('DATE_DIST', ascending=False)
            elif region:
                sort_key = 'CLOUDLESS_PORTION' if self.cloudShadowSupport else 'FILL_PORTION'
                ee_coll = ee_coll.sort(sort_key)
            else:
                ee_coll = ee_coll.sort('system:time_start')
        else:
            if date:
                warnings.warn(
                    f"'date' is valid for {sort_methods} methods only.",
                    category=UserWarning,
                    stacklevel=2,
                )
            elif region:
                warnings.warn(
                    f"'region' is valid for {sort_methods} methods only.",
                    category=UserWarning,
                    stacklevel=2,
                )

        return ee_coll

    def _raise_image_consistency(self) -> None:
        """Raise an error if the collection image bands are not consistent (i.e. don't have same
        band names, projections or bounds; or don't have fixed projections).
        """
        first_band_names = first_band = None
        band_compare_keys = ['crs', 'crs_transform', 'dimensions', 'data_type']

        # TODO: this test is stricter than it needs to be.  for image splitting, only the min
        #  scale band of each image should have a fixed projection and match all other min scale
        #  band's projection & bounds.  for band splitting, only the first image should have
        #  all bands with fixed projections, and matching projections and bounds.
        try:
            for im_info in self.info.get('features', []):
                cmp_bands = {bp['id']: bp for bp in im_info.get('bands', [])}

                # test number of bands & names against the first image's bands
                if not first_band_names:
                    first_band_names = cmp_bands.keys()
                elif not cmp_bands.keys() == first_band_names:
                    raise ValueError('Inconsistent number of bands or band names.')

                for band in cmp_bands.values():
                    cmp_band = {k: band.get(k, None) for k in band_compare_keys}
                    # test band has a fixed projections
                    if not cmp_band['dimensions']:
                        raise ValueError('One or more image bands do not have a fixed projection.')
                    # test band projection & bounds against the first image's first band
                    if not first_band:
                        first_band = cmp_band
                    elif cmp_band != first_band:
                        raise ValueError('Inconsistent band projections, bounds or data types.')

        except ValueError as ex:
            raise ValueError(
                f"Cannot export collection: '{ex!s}'.  'prepareForExport()' can be called to "
                f"create an export-ready collection."
            ) from ex

    def _split_images(self, split: SplitType) -> dict[str, ImageAccessor]:
        """Split the collection into images according to ``split``."""
        indexes = [*self.properties.keys()]

        if split is SplitType.bands:
            # split collection into an image per band (i.e. the same band from every collection
            # image form the bands of a new 'band' image)
            first_info = self.info['features'][0] if self.info.get('features') else {}
            first_band_names = [bi['id'] for bi in first_info.get('bands', [])]

            def to_bands(band_name: ee.String) -> ee.Image:
                ee_image = self._ee_coll.select(ee.String(band_name)).toBands()
                # rename system:index to band name & band names to system indexes
                ee_image = ee_image.set('system:index', band_name)
                return ee_image.rename(indexes)

            im_list = ee.List(first_band_names).map(to_bands)
            im_names = first_band_names
        else:
            # split collection into its images
            im_list = self._ee_coll.toList(self._max_export_images)
            im_names = indexes

        # return a dictionary of image name keys, and ImageAccessor values
        return {k: ImageAccessor(ee.Image(im_list.get(i))) for i, k in enumerate(im_names)}

    def addMaskBands(self, **kwargs) -> ee.ImageCollection:
        """
        Return this collection with cloud/shadow masks and related bands added when supported,
        otherwise with fill (validity) masks added.

        Existing mask bands are overwritten, except on images without fixed projections,
        where no mask bands are added or overwritten.

        :param kwargs:
            Cloud/shadow masking arguments - see :meth:`geedim.mask.ImageAccessor.addMaskBands`
            for details.

        :return:
            Image collection with added mask bands.
        """
        return self._ee_coll.map(lambda ee_image: self._mi.add_mask_bands(ee_image, **kwargs))

    def maskClouds(self) -> ee.ImageCollection:
        """
        Return this collection with cloud/shadow masks applied when supported, otherwise with
        fill (validity) masks applied.

        Mask bands should be added with :meth:`addMaskBands` before calling this method.

        :return:
            Masked image collection.
        """
        return self._ee_coll.map(lambda ee_image: self._mi.mask_clouds(ee_image))

    def medoid(self, bands: list | ee.List = None) -> ee.Image:
        """
        Find the medoid composite of the collection images.

        See https://www.mdpi.com/2072-4292/5/12/6481 for a description of the method.

        :param bands:
            List of bands to include in the medoid score.  Defaults to :attr:`reflBands` if
            available, otherwise all bands.

        :return:
            Medoid composite image.
        """
        return medoid(self._ee_coll, bands=bands or self.specBands)

    def filter(
        self,
        start_date: str | datetime | ee.Date = None,
        end_date: str | datetime | ee.Date = None,
        region: dict | ee.Geometry = None,
        fill_portion: float | ee.Number = None,
        cloudless_portion: float | ee.Number = None,
        custom_filter: str | None = None,
        **kwargs,
    ) -> ee.ImageCollection:
        """
        Search the collection for images that satisfy date, region, filled/cloudless portion,
        and custom criteria.

        Filled and cloudless portions are only calculated and included in collection
        :attr:`properties` when one or both of ``fill_portion`` / ``cloudless_portion`` are
        supplied.  If ``fill_portion`` or ``cloudless_portion`` are supplied, ``region`` is
        required.

        Search speeds can be improved by supplying multiple of the ``start_date``, ``end_date``,
        ``region`` and ``custom_filter`` arguments.

        :param start_date:
            Start date, in ISO format if a string.
        :param end_date:
            End date, in ISO format if a string.  Defaults to a millisecond after ``start_date`` if
            ``start_date`` is supplied.  Ignored if ``start_date`` is not supplied.
        :param region:
            Region that images should intersect as a GeoJSON dictionary or ``ee.Geometry``.
        :param fill_portion:
            Lower limit on the portion of ``region`` that contains filled pixels (%).
        :param cloudless_portion:
            Lower limit on the portion of filled pixels that are cloud/shadow free (%).
        :param custom_filter:
            Custom image property filter expression e.g. ``property > value``.  See the `EE docs
            <https://developers.google.com/earth-engine/apidocs/ee-filter-expression>`_ for details.
        :param kwargs:
            Cloud/shadow masking arguments - see :meth:`geedim.mask.ImageAccessor.addMaskBands`
            for details.

        :return:
            Filtered image collection containing search result image(s).
        """
        # TODO: refactor error classes and what gets raised where & test for these errors
        if (fill_portion is not None or cloudless_portion is not None) and not region:
            raise ValueError(
                "'region' is required when 'fill_portion' or 'cloudless_portion' are supplied."
            )

        # filter the image collection, finding cloud/shadow masks and region stats
        ee_coll = self._ee_coll
        if start_date:
            ee_coll = ee_coll.filterDate(start_date, end_date)

        if region:
            ee_coll = ee_coll.filterBounds(region)

        # when possible filter on custom_filter before calling set_region_stats to reduce
        # computation
        if custom_filter and all(
            prop_key not in custom_filter for prop_key in ['FILL_PORTION', 'CLOUDLESS_PORTION']
        ):
            ee_coll = ee_coll.filter(ee.Filter.expression(custom_filter))
            custom_filter = None

        if (fill_portion is not None) or (cloudless_portion is not None) or custom_filter:
            # set regions stats before filtering on those properties

            def set_portions(ee_image: ee.Image) -> ee.Image:
                ee_image = self._mi.add_mask_bands(ee_image, **kwargs)
                return self._mi.set_mask_portions(
                    ee_image, region=region, scale=self._portion_scale
                )

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
        return ee_coll

    def composite(
        self,
        method: CompositeMethod | str = None,
        mask: bool = True,
        resampling: ResamplingMethod | str = ImageAccessor._default_resampling,
        date: str | datetime | ee.Date = None,
        region: dict | ee.Geometry = None,
        **kwargs,
    ) -> ee.Image:
        """
        Create a composite from the images in the collection.

        If both ``date`` and ``region`` are ``None``, images are sorted by their capture date
        (the default).

        :param method:
            Compositing method. By default, :attr:`~geedim.enums.CompositeMethod.q_mosaic` is
            used for cloud/shadow mask supported collections,
            and :attr:`~geedim.enums.CompositeMethod.mosaic` otherwise.
        :param mask:
            Whether to apply the cloud/shadow mask; or fill (valid pixel) mask, in the case of
            images without support for cloud/shadow masking.
        :param resampling:
            Resampling method to use on collection images prior to compositing.
        :param date:
            Sort collection images by their absolute difference in capture time from this date.
            This prioritises pixels from images closest to ``date``.  Valid for the
            :attr:`~geedim.enums.CompositeMethod.q_mosaic`,
            :attr:`~geedim.enums.CompositeMethod.mosaic` and
            :attr:`~geedim.enums.CompositeMethod.medoid` ``method`` only.  If a string, it should
            be in ISO format.  If ``None``, no time difference sorting is done (the default).
        :param region:
            Sort collection images by the portion of valid pixels inside this region that are
            cloudless. Can be a GeoJSON dictionary or ``ee.Geometry``.  This prioritises pixels
            from the least cloudy images. Valid for the
            :attr:`~geedim.enums.CompositeMethod.q_mosaic`,
            :attr:`~geedim.enums.CompositeMethod.mosaic` and
            :attr:`~geedim.enums.CompositeMethod.medoid` ``method`` only.  If the collection has
            no cloud/shadow mask support, images are sorted by the portion of their valid pixels
            that are inside ``region``.  This prioritises pixels from images with the best
            ``region`` coverage. If ``region`` is ``None``, no cloudless/valid portion sorting is
            done (the default).
        :param kwargs:
            Cloud/shadow masking arguments - see :meth:`geedim.mask.ImageAccessor.addMaskBands`
            for details.

        :return:
            Composite image.
        """
        # TODO: allow S2 cloud score to be used as quality band
        if method is None:
            method = (
                CompositeMethod.q_mosaic
                if self.id in schema.cloud_coll_names
                else CompositeMethod.mosaic
            )

        # mask, sort & resample the EE collection
        ee_coll = self._prepare_for_composite(
            method=method, mask=mask, resampling=resampling, date=date, region=region, **kwargs
        )

        method = CompositeMethod(method)
        if method == CompositeMethod.q_mosaic:
            comp_image = ee_coll.qualityMosaic('CLOUD_DIST')
        elif method == CompositeMethod.medoid:
            # limit medoid to surface reflectance bands
            comp_image: ee.Image = ImageCollectionAccessor(ee_coll).medoid(bands=self.specBands)
        else:
            comp_image = getattr(ee_coll, method.name)()

        # populate composite image metadata
        comp_index = ee.String(f'{method.value.upper()}-COMP')
        # set 'properties'->'system:index'
        comp_image = comp_image.set('system:index', comp_index)
        # set root 'id' property
        comp_id = ee.String(self.id + '/') if self.id else ee.String('')
        comp_id = comp_id.cat(comp_index)
        comp_image = comp_image.set('system:id', comp_id)
        # # set composite start-end times
        date_range = ee_coll.reduceColumns(ee.Reducer.minMax(), ['system:time_start'])
        comp_image = comp_image.set('system:time_start', ee.Number(date_range.get('min')))
        comp_image = comp_image.set('system:time_end', ee.Number(date_range.get('max')))
        return comp_image

    def prepareForExport(
        self,
        crs: str | None = None,
        crs_transform: Sequence[float] | None = None,
        shape: tuple[int, int] | None = None,
        region: dict | ee.Geometry | None = None,
        scale: float | None = None,
        resampling: str | ResamplingMethod = ImageAccessor._default_resampling,
        dtype: str | None = None,
        scale_offset: bool | None = False,
        bands: list[str | int] | str | None = None,
    ) -> ee.ImageCollection:
        """
        Prepare the collection for export.

        ..warning::
            The prepared collection images are reprojected and clipped versions of their
            source images. This type of image is `not recommended
            <https://developers.google.com/earth-engine/guides/best_practices>`__ for use in map
            display or further computation.

        :param crs:
            CRS of the prepared images as a well-known authority (e.g. EPSG) or WKT string.
            Defaults to the CRS of the minimum scale band of the first image.
        :param crs_transform:
            Georeferencing transform of the prepared images, as a sequence of 6 numbers.  In
            row-major order: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation].
        :param shape:
            (height, width) dimensions of the prepared images in pixels.
        :param region:
            Region defining the prepared image bounds as a GeoJSON dictionary or ``ee.Geometry``.
            Defaults to the geometry of the first image.  Ignored if ``crs_transform`` is supplied.
        :param scale:
            Pixel scale (m) of the prepared images.  Defaults to the minimum scale of the first
            image's bands.  Ignored if ``crs_transform`` is supplied.
        :param resampling:
            Resampling method to use for reprojecting.  Ignored for images without fixed
            projections e.g. composites.  Composites can be resampled by resampling their
            component images.
        :param dtype:
            Data type of the prepared images (``uint8``, ``int8``, ``uint16``, ``int16``,
            ``uint32``, ``int32``, ``float32`` or ``float64``).  Defaults to the minimum size data
            type able to represent all the first image's bands.
        :param scale_offset:
            Whether to apply any STAC band scales and offsets to the images (e.g. for converting
            digital numbers to physical units).
        :param bands:
            Bands to include in the prepared images as a list of names / indexes, or a regex
            string.  Defaults to all bands of the first image.

        :return:
            Prepared collection.
        """
        # apply the export args to the first image
        first = self._first.prepareForExport(
            crs=crs,
            crs_transform=crs_transform,
            shape=shape,
            region=region,
            scale=scale,
            resampling=resampling,
            dtype=dtype,
            scale_offset=scale_offset,
            bands=bands,
        )
        first = ImageAccessor(first)

        # prepare collection images to have the same grid and bounds as the first image
        def prepare_image(ee_image: ee.Image) -> ee.Image:
            return ImageAccessor(ee_image).prepareForExport(
                crs=first.crs,
                crs_transform=first.transform,
                shape=first.shape,
                resampling=resampling,
                dtype=first.dtype,
                scale_offset=scale_offset,
                bands=first.bandNames,
            )

        return self._ee_coll.map(prepare_image)

    def toGoogleCloud(
        self,
        type: ExportType = ImageAccessor._default_export_type,
        folder: str | None = None,
        wait: bool = True,
        split: str | SplitType = SplitType.bands,
        **kwargs,
    ) -> list[ee.batch.Task]:
        """
        Export the collection as GeoTIFF files to Google Drive, Earth Engine assets or Google Cloud
        Storage using the Earth Engine batch environment.

        Export projection and bounds are defined by the
        :attr:`~geedim.image.ImageAccessor.crs`,
        :attr:`~geedim.image.ImageAccessor.transform` and
        :attr:`~geedim.image.ImageAccessor.shape` properties, and the data type by the
        :attr:`~geedim.image.ImageAccessor.dtype` property of the collection images. All
        bands in the collection should share the same projection, bounds and data type.
        :meth:`prepareForExport` can be called before this method to apply export parameters and
        create an export-ready collection.

        A maximum of 5000 images can be exported.

        :param type:
            Export type.
        :param folder:
            Google Drive folder (when ``type`` is :attr:`~geedim.enums.ExportType.drive`),
            Earth Engine asset project (when ``type`` is :attr:`~geedim.enums.ExportType.asset`),
            or Google Cloud Storage bucket (when ``type`` is
            :attr:`~geedim.enums.ExportType.cloud`).  Can be include sub-folders.  If ``type`` is
            :attr:`~geedim.enums.ExportType.asset` or :attr:`~geedim.enums.ExportType.cloud` then
            ``folder`` is required.
        :param wait:
            Whether to wait for the exports to complete before returning.
        :param split:
            Export a file for each collection band (:attr:`SplitType.bands`), or for each
            collection image (:attr:`SplitType.images`).  Files are named with their band name,
            and file band descriptions are set to the ``system:index`` property of the band's
            source image, when ``split`` is :attr:`SplitType.bands`.  Otherwise, files are named
            with the ``system:index`` property of the file's source image, and file band
            descriptions are set to the image band names, when ``split`` is
            :attr:`SplitType.images`.
        :param kwargs:
            Additional arguments to the ``type`` dependent Earth Engine function:
            ``Export.image.toDrive``, ``Export.image.toAsset`` or ``Export.image.toCloudStorage``.

        :return:
            List of image export tasks, started if ``wait`` is ``False``, or completed if
            ``wait`` is ``True``.
        """
        split = SplitType(split)
        self._raise_image_consistency()
        images = self._split_images(split)

        # start exporting the split images concurrently
        tasks = {}
        for name, image in images.items():
            tasks[name] = image.toGoogleCloud(filename=name, folder=folder, wait=False, **kwargs)

        if wait:
            # wait for tasks to complete
            tqdm_kwargs = utils.get_tqdm_kwargs(desc=self.id or 'Collection', unit=split.value)
            for name, task in tqdm(tasks.items(), **tqdm_kwargs):
                ImageAccessor.monitorTask(task, name)

        return list(tasks.values())

    def toGeoTIFF(
        self,
        dirname: os.PathLike | str | OpenFile,
        overwrite: bool = False,
        split: str | SplitType = SplitType.bands,
        nodata: bool | int | float = True,
        driver: str | Driver = Driver.gtiff,
        max_tile_size: float = Tiler._default_max_tile_size,
        max_tile_dim: int = Tiler._ee_max_tile_dim,
        max_tile_bands: int = Tiler._ee_max_tile_bands,
        max_requests: int = Tiler._max_requests,
        max_cpus: int | None = None,
    ) -> None:
        """
        Export the collection to GeoTIFF files.

        Export projection and bounds are defined by the
        :attr:`~geedim.image.ImageAccessor.crs`,
        :attr:`~geedim.image.ImageAccessor.transform` and
        :attr:`~geedim.image.ImageAccessor.shape` properties, and the data type by the
        :attr:`~geedim.image.ImageAccessor.dtype` property of the collection images. All
        bands in the collection should share the same projection, bounds and data type.
        :meth:`prepareForExport` can be called before this method to apply export parameters and
        create an export-ready collection.

        Images are retrieved as separate tiles which are downloaded and decompressed
        concurrently.  Tile size can be controlled with ``max_tile_size``, ``max_tile_dim`` and
        ``max_tile_bands``, and download / decompress concurrency with ``max_requests`` and
        ``max_cpus``.

        A maximum of 5000 images can be exported.

        :param dirname:
            Destination directory.  Can be a path or URI string, or an
            :class:`~fsspec.core.OpenFile` object.
        :param overwrite:
            Whether to overwrite destination files if they exist.
        :param split:
            Export a file for each collection band (:attr:`SplitType.bands`), or for each
            collection image (:attr:`SplitType.images`).  Files are named with their band name,
            and file band descriptions are set to the ``system:index`` property of the band's
            source image, when ``split`` is :attr:`SplitType.bands`.  Otherwise, files are named
            with the ``system:index`` property of the file's source image, and file band
            descriptions are set to the image band names, when ``split`` is
            :attr:`SplitType.images`.
        :param nodata:
            Set GeoTIFF nodata tags to the shared
            :attr:`~geedim.image.ImageAccessor.nodata` value of the collection images
            (``True``), or leave nodata tags unset (``False``).  If a custom integer or floating
            point value is supplied, nodata tags are set to this value.  Usually, a custom value
            would be supplied when the collection images have been unmasked with
            ``ee.Image.unmask(nodata)``.
        :param driver:
            File format driver.
        :param max_tile_size:
            Maximum tile size (MB).  Should be less than the `Earth Engine size limit
            <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`__ (32 MB).
        :param max_tile_dim:
            Maximum tile width / height (pixels).  Should be less than the `Earth Engine limit
            <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`__ (10000).
        :param max_tile_bands:
            Maximum number of tile bands.  Should be less than the Earth Engine limit (1024).
        :param max_requests:
            Maximum number of concurrent tile downloads.  Should be less than the `max concurrent
            requests quota <https://developers.google.com/earth-engine/guides/usage
            #adjustable_quota_limits>`__.
        :param max_cpus:
            Maximum number of tiles to decompress concurrently.  Defaults to one less than the
            number of CPUs, or one, whichever is greater.  Values larger than the default can
            stall the asynchronous event loop and are not recommended.
        """
        odir = (
            fsspec.open(os.fspath(dirname), 'wb') if not isinstance(dirname, OpenFile) else dirname
        )
        split = SplitType(split)
        driver = Driver(driver)
        self._raise_image_consistency()
        images = self._split_images(split)
        # TODO: use something like accessors_from_images() to get infos for all the images at
        #  once (here and in all export methods)?

        # download the split images sequentially, each into its own file
        tqdm_kwargs = utils.get_tqdm_kwargs(desc=self.id or 'Collection', unit=split.value)
        for name, image in tqdm(images.items(), **tqdm_kwargs):
            joined_path = posixpath.join(odir.path, name + '.tif')
            ofile = OpenFile(odir.fs, joined_path, mode='wb')
            image.toGeoTIFF(
                ofile,
                overwrite=overwrite,
                nodata=nodata,
                driver=driver,
                max_tile_size=max_tile_size,
                max_tile_dim=max_tile_dim,
                max_tile_bands=max_tile_bands,
                max_requests=max_requests,
                max_cpus=max_cpus,
            )

    def toNumPy(
        self,
        masked: bool = False,
        structured: bool = False,
        split: str | SplitType = SplitType.bands,
        max_tile_size: float = Tiler._default_max_tile_size,
        max_tile_dim: int = Tiler._ee_max_tile_dim,
        max_tile_bands: int = Tiler._ee_max_tile_bands,
        max_requests: int = Tiler._max_requests,
        max_cpus: int | None = None,
    ) -> np.ndarray:
        """
        Export the collection to a NumPy array.

        Export projection and bounds are defined by the
        :attr:`~geedim.image.ImageAccessor.crs`,
        :attr:`~geedim.image.ImageAccessor.transform` and
        :attr:`~geedim.image.ImageAccessor.shape` properties, and the data type by the
        :attr:`~geedim.image.ImageAccessor.dtype` property of the collection images. All
        bands in the collection should share the same projection, bounds and data type.
        :meth:`prepareForExport` can be called before this method to apply export parameters and
        create an export-ready collection.

        Images are retrieved as separate tiles which are downloaded and decompressed
        concurrently.  Tile size can be controlled with ``max_tile_size``, ``max_tile_dim`` and
        ``max_tile_bands``, and download / decompress concurrency with ``max_requests`` and
        ``max_cpus``.

        A maximum of 5000 images can be exported.

        :param masked:
            Return a :class:`~numpy.ndarray` with masked pixels set to the shared :attr:`nodata`
            value of the collection images (``False``), or a :class:`~numpy.ma.MaskedArray`
            (``True``).
        :param structured:
            Return a 4D array with a numerical ``dtype`` (``False``), or a 2D array with a
            structured ``dtype`` (``True``).  Array dimension ordering, and structured
            ``dtype`` fields depend on the value of ``split``.
        :param split:
            Return a 4D array with (row, column, band, image) dimensions (
            :attr:`SplitType.bands`), or a 4D array with (row, column, image, band) dimensions (
            :attr:`SplitType.images`), when ``structured`` is ``False``. Otherwise, return a 2D
            array with (row, column) dimensions and a structured ``dtype`` representing images
            nested in bands (:attr:`SplitType.bands`), or a 2D array with (row, column)
            dimensions and a structured ``dtype`` representing bands nested in images (
            :attr:`SplitType.images`), when ``structured`` is ``True``.
        :param max_tile_size:
            Maximum tile size (MB).  Should be less than the `Earth Engine size limit
            <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`__ (32 MB).
        :param max_tile_dim:
            Maximum tile width / height (pixels).  Should be less than the `Earth Engine limit
            <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`__ (10000).
        :param max_tile_bands:
            Maximum number of tile bands.  Should be less than the Earth Engine limit (1024).
        :param max_requests:
            Maximum number of concurrent tile downloads.  Should be less than the `max concurrent
            requests quota <https://developers.google.com/earth-engine/guides/usage
            #adjustable_quota_limits>`__.
        :param max_cpus:
            Maximum number of tiles to decompress concurrently.  Defaults to one less than the
            number of CPUs, or one, whichever is greater.  Values larger than the default can
            stall the asynchronous event loop and are not recommended.

        :returns:
            NumPy array.
        """
        split = SplitType(split)
        self._raise_image_consistency()
        images = self._split_images(split)

        # initialise the destination array
        first = next(iter(images.values()))
        shape = (*first.shape, len(images), first.count)
        dtype = first.dtype
        if masked:
            array = np.ma.zeros(shape, dtype=dtype)
        else:
            array = np.zeros(shape, dtype=dtype)

        # download the split image arrays sequentially, copying into the destination array
        tqdm_kwargs = utils.get_tqdm_kwargs(desc=self.id or 'Collection', unit=split.value)
        for i, image in enumerate(tqdm(images.values(), **tqdm_kwargs)):
            array[:, :, i, :] = image.toNumPy(
                masked=masked,
                max_tile_size=max_tile_size,
                max_tile_dim=max_tile_dim,
                max_tile_bands=max_tile_bands,
                max_requests=max_requests,
                max_cpus=max_cpus,
            )

        if structured:
            # create a structured data dtype to describe the last 2 array dimensions
            band_names = first.bandNames
            image_names = [*images.keys()]

            timestamps = [p.get('system:time_start', None) for p in self.properties.values()]
            if all(timestamps):
                # zip date string 'title's with the corresponding system:index 'name's (allows
                # the time dimension to be indexed by date string or system:index)
                date_strings = [
                    datetime.fromtimestamp(ts / 1000).isoformat(timespec='seconds')
                    for ts in timestamps
                ]
                if split is SplitType.bands:
                    band_names = [*zip(date_strings, band_names)]
                else:
                    image_names = [*zip(date_strings, image_names)]

            # nest the structured data type for a split image's bands (last array dimension) in the
            # structured data type for the split images (second last array dimension)
            band_dtype = np.dtype([*zip(band_names, [array.dtype] * len(band_names))])
            exp_dtype = np.dtype([*zip(image_names, [band_dtype] * len(images))])

            # create a view of the array with the last 2 dimensions as the structured dtype
            array = array.reshape(*array.shape[:2], -1).view(dtype=exp_dtype).squeeze()

        return array

    def toXarray(
        self,
        masked: bool = False,
        split: str | SplitType = SplitType.bands,
        max_tile_size: float = Tiler._default_max_tile_size,
        max_tile_dim: int = Tiler._ee_max_tile_dim,
        max_tile_bands: int = Tiler._ee_max_tile_bands,
        max_requests: int = Tiler._max_requests,
        max_cpus: int | None = None,
    ) -> xarray.Dataset:
        """
        Export the collection to an Xarray Dataset.

        Export projection and bounds are defined by the
        :attr:`~geedim.image.ImageAccessor.crs`,
        :attr:`~geedim.image.ImageAccessor.transform` and
        :attr:`~geedim.image.ImageAccessor.shape` properties, and the data type by the
        :attr:`~geedim.image.ImageAccessor.dtype` property of the collection images. All
        bands in the collection should share the same projection, bounds and data type.
        :meth:`prepareForExport` can be called before this method to apply export parameters and
        create an export-ready collection.

        Images are retrieved as separate tiles which are downloaded and decompressed
        concurrently.  Tile size can be controlled with ``max_tile_size``, ``max_tile_dim`` and
        ``max_tile_bands``, and download / decompress concurrency with ``max_requests`` and
        ``max_cpus``.

        Dataset attributes include the export :attr:`crs`, :attr:`transform` and ``nodata``
        values for compatibility with `rioxarray <https://github.com/corteva/rioxarray>`_,
        as well as ``ee`` and ``stac`` JSON strings corresponding to Earth Engine property and
        STAC dictionaries.

        A maximum of 5000 images can be exported.

        :param masked:
            Set masked pixels in the returned array to the shared :attr:`nodata` value of the
            collection images (``False``), or to NaN (``True``).  If ``True``, the export
            ``dtype`` is integer, and one or more pixels are masked, the returned array is
            converted to a minimal floating point type able to represent the export ``dtype``.
        :param split:
            Return a dataset with bands as variables (:attr:`SplitType.bands`), or a dataset with
            images as variables (:attr:`SplitType.images`).  Variables are named with their band
            name, and time coordinates are converted from the ``system:start_time`` property of
            the images when ``split`` is :attr:`SplitType.bands`.  Variables are named with the
            ``system:index`` property of their image, and band coordinates are set to image band
            names when ``split`` is :attr:`SplitType.images`.
        :param max_tile_size:
            Maximum tile size (MB).  Should be less than the `Earth Engine size limit
            <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`__ (32 MB).
        :param max_tile_dim:
            Maximum tile width / height (pixels).  Should be less than the `Earth Engine limit
            <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`__ (10000).
        :param max_tile_bands:
            Maximum number of tile bands.  Should be less than the Earth Engine limit (1024).
        :param max_requests:
            Maximum number of concurrent tile downloads.  Should be less than the `max concurrent
            requests quota <https://developers.google.com/earth-engine/guides/usage
            #adjustable_quota_limits>`__.
        :param max_cpus:
            Maximum number of tiles to decompress concurrently.  Defaults to one less than the
            number of CPUs, or one, whichever is greater.  Values larger than the default can
            stall the asynchronous event loop and are not recommended.

        :returns:
            Xarray Dataset.
        """
        if not xarray:
            raise ImportError("'toXarray()' requires the 'xarray' package to be installed.")
        split = SplitType(split)
        self._raise_image_consistency()
        images = self._split_images(split)

        # download the split image DataArrays sequentially, storing in a dict
        arrays = {}
        tqdm_kwargs = utils.get_tqdm_kwargs(desc=self.id or 'Collection', unit=split.value)
        for name, image in tqdm(images.items(), **tqdm_kwargs):
            arrays[name] = image.toXarray(
                masked=masked,
                max_tile_size=max_tile_size,
                max_tile_dim=max_tile_dim,
                max_tile_bands=max_tile_bands,
                max_requests=max_requests,
                max_cpus=max_cpus,
            )

        if split is SplitType.bands:
            # change the 'band' coordinate and dimension in each DataArray to 'time'
            timestamps = [p.get('system:time_start', None) for p in self.properties.values()]
            datetimes = to_datetime(timestamps, unit='ms')

            for name, array in arrays.items():
                array = array.rename(band='time')
                array.coords['time'] = datetimes
                arrays[name] = array

        # create attributes dict
        attrs = dict(id=self.id or None)
        # copy rioxarray required attributes from the first DataArray
        for array in arrays.values():
            attrs.update(**{k: array.attrs[k] for k in ['crs', 'transform', 'nodata']})
            break
        # add EE / STAC attributes (use json strings here, then drop all Nones for serialisation
        # compatibility e.g. netcdf)
        attrs['ee'] = json.dumps(self.info['properties']) if 'properties' in self.info else None
        attrs['stac'] = json.dumps(self.stac) if self.stac else None
        attrs = {k: v for k, v in attrs.items() if v is not None}

        # return a Dataset of split image DataArrays
        return xarray.Dataset(arrays, attrs=attrs)


class MaskedCollection(ImageCollectionAccessor):
    def __init__(self, ee_collection: ee.ImageCollection, add_props: list[str] | None = None):
        """
        A class for describing, searching and compositing an Earth Engine image collection,
        with support for cloud/shadow masking.

        :param ee_collection:
            Earth Engine image collection to encapsulate.
        :param add_props:
            Additional Earth Engine image properties to include in :attr:`properties`.
        """
        warnings.warn(
            f"'{self.__class__.__name__}' is deprecated and will be removed in a future release. "
            f"Please use the 'gd' accessor on 'ee.ImageCollection' instead.",
            category=FutureWarning,
            stacklevel=2,
        )
        if not isinstance(ee_collection, ee.ImageCollection):
            raise TypeError('`ee_collection` must be an instance of ee.ImageCollection')
        super().__init__(ee_collection)
        if add_props:
            self.schemaPropertyNames += add_props

    @classmethod
    def from_name(cls, name: str, add_props: list[str] | None = None) -> MaskedCollection:
        """

        :param name:
            Name of the Earth Engine image collection to create.
        :param add_props:
            Additional Earth Engine image properties to include in :attr:`properties`.

        :return:
            MaskedCollection instance.
        """
        # this is separate from __init__ for consistency with MaskedImage.from_id()
        ee_coll = ee.ImageCollection(name)
        gd_coll = cls(ee_coll, add_props=add_props)
        return gd_coll

    @classmethod
    def from_list(
        cls, image_list: list[str | MaskedImage | ee.Image], add_props: list[str] | None = None
    ) -> MaskedCollection:
        """
        Create a MaskedCollection instance from a list of Earth Engine image ID strings,
        ``ee.Image`` instances and/or :class:`~geedim.mask.MaskedImage` instances.  The list may
        include composite images, as created with :meth:`composite`.

        Images from spectrally compatible Landsat collections can be combined i.e. Landsat-4 with
        Landsat-5, and Landsat-8 with Landsat-9.  Otherwise, images should all belong to the same
        collection.

        Any ``ee.Image`` instances in the list (including those encapsulated by
        :class:`~geedim.mask.MaskedImage`) must have ``system:id`` and ``system:time_start``
        properties.  This is always the case for images obtained from the Earth Engine catalog,
        and images returned from :meth:`composite`.  Any other user-created images passed to
        :meth:`from_list` should have these properties set.

        :param image_list:
            List of images to include in the collection (must all be from the same, or compatible
            Earth Engine collections). List items can be ID strings, instances of
            :class:`~geedim.mask.MaskedImage`, or instances of ``ee.Image``.
        :param add_props:
            Additional Earth Engine image properties to include in :attr:`properties`.

        :return:
            A MaskedCollection instance.
        """
        if len(image_list) == 0:
            raise ValueError("'image_list' is empty.")

        images = [
            image.ee_image if isinstance(image, BaseImage) else ee.Image(image)
            for image in image_list
        ]
        ee_coll = cls.fromImages(images)
        return cls(ee_coll, add_props=add_props)

    @property
    def ee_collection(self) -> ee.ImageCollection:
        """Earth Engine image collection."""
        return self._ee_coll

    @property
    def name(self) -> str:
        """Name of the encapsulated Earth Engine image collection."""
        return self.id

    @property
    def image_type(self) -> type:
        """:class:`~geedim.mask.MaskedImage` class for images in :attr:`ee_collection`."""
        return MaskedImage

    @property
    def stats_scale(self) -> float | None:
        """Scale to use for re-projections when finding region statistics."""
        return self._portion_scale

    @property
    def schema_table(self) -> str:
        """:attr:`schema` formatted as a printable table string."""
        return self.schemaTable

    @property
    def properties(self) -> dict[str, dict[str, Any]]:
        coll_schema_props = {}
        for i, im_info in enumerate(self.info.get('features', [])):
            im_id = im_info.get('id', str(i))
            im_props = im_info.get('properties', {})
            im_schema_props = {
                key: im_props[key]
                for key in self.schema.keys()
                if im_props.get(key, None) is not None
            }
            coll_schema_props[im_id] = im_schema_props
        return coll_schema_props

    @property
    def properties_table(self) -> str:
        """:attr:`properties` formatted as a printable table string."""
        return self.propertiesTable

    @property
    def refl_bands(self) -> list[str] | None:
        """List of the collection's spectral / reflectance band names.  ``None`` if there is no
        :attr:`stac` entry, or no spectral / reflectance bands.
        """
        return self.specBands

    def search(self, *args, **kwargs) -> MaskedCollection:
        ee_coll = self.filter(*args, **kwargs)
        gd_coll = MaskedCollection(ee_coll)
        gd_coll.schemaPropertyNames = self.schemaPropertyNames
        return gd_coll

    def composite(self, *args, **kwargs) -> MaskedImage:
        ee_image = super().composite(*args, **kwargs)
        gd_image = MaskedImage(ee_image)
        return gd_image
