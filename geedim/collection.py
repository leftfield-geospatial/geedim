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
import re
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Sequence

import ee
import numpy as np
import tabulate
from tabulate import DataRow, Line, TableFormat
from tqdm.auto import tqdm

from geedim import schema, utils
from geedim.download import BaseImage, BaseImageAccessor
from geedim.enums import CompositeMethod, ResamplingMethod, SplitType
from geedim.errors import InputImageError
from geedim.mask import MaskedImage, _MaskedImage, class_from_id
from geedim.medoid import medoid
from geedim.stac import StacCatalog, StacItem
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
    _max_export_images = 1000

    # TODO: strictly this is an _init_ docstring, not clas docstring.  does moving it (and all
    #  other class docstrings) to _init_ work with sphinx?
    """
    Accessor for describing, searching and compositing an image collection, with support for
    cloud/shadow masking.

    :param ee_coll:
        Image collection to access.
    """

    def __init__(self, ee_coll: ee.ImageCollection):
        self._ee_coll = ee_coll
        self._info = None
        self._schema = None
        self._schema_prop_names = None
        self._properties = None

    @staticmethod
    def fromImages(images: Any) -> ee.ImageCollection:
        """
        Create a image collection with support for cloud/shadow masking, that contains the given
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
        # TODO: previously there was error checking here to see if images were in compatible
        #  collections and all had IDs and dates.  Rather than check here, check/get search and
        #  composite to behave sensibly when this is not the case.
        ee_coll = ee.ImageCollection(images)

        # check the images are from compatible collections
        ids = [
            split_id(im_props.get('id', None))[0]
            for im_props in ee_coll.gd.info.get('features', [])
        ]
        if not _compatible_collections(ids):
            # TODO: test raises an error if any/all names are None
            raise InputImageError(
                'All images must belong to the same, or spectrally compatible, collections.'
            )

        # set the collection ID to enable cloud/shadow masking if supported
        ee_coll = ee_coll.set('system:id', ids[0])
        return ee_coll

    @cached_property
    def _mi(self) -> type[_MaskedImage]:
        """Masking method container."""
        return class_from_id(self.id)

    @cached_property
    def _portion_scale(self) -> float | None:
        """Scale to use for finding mask portions.  ``None`` if there is no STAC entry for this
        collection.
        """
        if not self.stac:
            return None
        gsds = [float(band_dict['gsd']) for band_dict in self.stac.band_props.values()]
        max_gsd = max(gsds)
        min_gsd = min(gsds)
        return min_gsd if (max_gsd > 10 * min_gsd) and (min_gsd > 0) else max_gsd

    @cached_property
    def id(self) -> str | None:
        """Earth Engine ID."""
        # Get the ID from self.properties if it has been cached.  Otherwise get the ID
        # directly rather than retrieving self.properties, which can be time-consuming.
        if self._info:
            return self._info.get('id', None)
        else:
            return self._ee_coll.get('system:id').getInfo()

    @cached_property
    def stac(self) -> StacItem | None:
        """STAC information.  ``None`` if there is no STAC entry for this collection."""
        # TODO: look into refactoring StacItem and/or use raw STAC dictionaries where possible
        return StacCatalog().get_item(self.id)

    @property
    def info(self) -> dict[str, Any]:
        """Earth Engine information as returned by :meth:`ee.ImageCollection.getInfo`,
        but limited to the first 1000 images with band information excluded.
        """
        if not self._info:
            self._info = self._ee_coll.limit(self._max_export_images).getInfo()
        return self._info

    @property
    def schemaPropertyNames(self) -> list[str]:
        """:attr:`schema` property names."""
        if not self._schema_prop_names:
            if self.id in schema.collection_schema:
                self._schema_prop_names = schema.collection_schema[self.id]['prop_schema'].keys()
            else:
                self._schema_prop_names = schema.default_prop_schema.keys()
            self._schema_prop_names = list(self._schema_prop_names)
        return self._schema_prop_names

    @schemaPropertyNames.setter
    def schemaPropertyNames(self, value: list[str]):
        if value != self.schemaPropertyNames:
            self._schema_prop_names = value
            # reset the schema and properties
            self._schema = None
            self._properties = None

    @property
    def schema(self) -> dict[str, dict]:
        """Dictionary of property abbreviations and descriptions used to form
        :attr:`propertiesTable`.
        """
        if not self._schema:
            if self.id in schema.collection_schema:
                coll_schema = schema.collection_schema[self.id]['prop_schema']
            else:
                coll_schema = schema.default_prop_schema

            self._schema = {}
            for prop_name in self.schemaPropertyNames:
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
    def schemaTable(self) -> str:
        """:attr:`schema` formatted as a printable table string."""
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
    def properties(self) -> dict[str, dict[str, Any]]:
        """Dictionary of image properties.  Keys are the image indexes and values the image
        property dictionaries.
        """
        if not self._properties:
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
    def reflBands(self) -> list[str] | None:
        """List of the collection's spectral / reflectance band names.  ``None`` if there is no
        :attr:`stac` entry, or no spectral / reflectance bands.
        """
        if not self.stac:
            return None
        return [
            bname for bname, bdict in self.stac.band_props.items() if 'center_wavelength' in bdict
        ]

    def _prepare_for_composite(
        self,
        method: CompositeMethod | str,
        mask: bool = True,
        resampling: ResamplingMethod | str = BaseImageAccessor._default_resampling,
        date: str | datetime | ee.Date = None,
        region: dict | ee.Geometry = None,
        **kwargs,
    ) -> ee.ImageCollection:
        """Return a collection that has been prepared for compositing."""
        method = CompositeMethod(method)
        resampling = ResamplingMethod(resampling)
        sort_methods = [CompositeMethod.mosaic, CompositeMethod.q_mosaic, CompositeMethod.medoid]

        if (method is CompositeMethod.q_mosaic) and (self.id not in schema.cloud_coll_names):
            raise ValueError(
                f"The 'q-mosaic' method is not supported for this collection: '{self.id}'.  It is"
                f"supported for the {schema.cloud_coll_names} collections only."
            )

        if date and region:
            # TODO: test for this error
            raise ValueError("One of 'date' or 'region' can be specified, but not both.")

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
            return ee_image.gd.resample(resampling)

        ee_coll = self._ee_coll.map(prepare_image)

        if method in sort_methods:
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
                logger.warning(f"'date' is valid for {sort_methods} methods only.")
            elif region:
                logger.warning(f"'region' is valid for {sort_methods} methods only.")

        return ee_coll

    def _raise_image_consistency(self) -> None:
        """Raise an error if the collection image bands are not consistent (i.e. don't have same
        band names, projections or bounds; or don't have fixed projections).
        """
        first_band_names = first_band = None
        band_compare_keys = ['crs', 'crs_transform', 'dimensions']

        # TODO: this test is stricter than it needs to be.  for image splitting, only the min
        #  scale band of each image should have a fixed projection and match all other min scale
        #  band's projection & bounds.  for band splitting, only the first the image should have
        #  all bands with fixed projections, and matching projections and bounds.
        try:
            for im_info in self.info.get('features', []):
                cmp_bands = {bp['id']: bp for bp in im_info.get('bands', [])}

                # test number of bands & names against the first image's bands
                if not first_band_names:
                    first_band_names = cmp_bands.keys()
                elif not cmp_bands.keys() == first_band_names:
                    raise ValueError('Inconsistent number of bands or band names.')

                for band_id, band in cmp_bands.items():
                    cmp_band = {k: band.get(k, None) for k in band_compare_keys}
                    # test band has a fixed projections
                    if not cmp_band['dimensions']:
                        raise ValueError('One or more image bands do not have a fixed projection.')
                    # test band projection & bounds against the first image's first band
                    if not first_band:
                        first_band = cmp_band
                    elif cmp_band != first_band:
                        raise ValueError('Inconsistent band projections or bounds.')

        except ValueError as ex:
            raise ValueError(
                f"Cannot export collection: '{str(ex)}'.  You can call 'prepareForExport()' to "
                f"create an export-ready collection."
            )

    def _split_images(self, split: SplitType) -> dict[str, BaseImageAccessor]:
        """Split the collection into images according to ``split``."""
        indexes = [*self.properties.keys()]

        if split is SplitType.bands:
            # split collection into an image per band (i.e. the same band from every collection
            # image form the bands of a new 'band' image)
            first_info = next(iter(self.info.get('features', [])))
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

        # return a dictionary of image name keys, and BaseImageAccessor values
        return {k: BaseImageAccessor(ee.Image(im_list.get(i))) for i, k in enumerate(im_names)}

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
        return medoid(self._ee_coll, bands=bands or self.reflBands)

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
        """
        Search the collection for images that satisfy date, region, filled/cloudless portion,
        and custom criteria.

        Filled and cloudless portions are only calculated and included in collection
        :attr:`properties` when one or both of ``fill_portion`` / ``cloudless_portion`` are
        provided.

        Search speed can be increased by specifying ``custom_filter``, and or by omitting
        ``fill_portion`` / ``cloudless_portion``.

        :param start_date:
            Start date, in ISO format if a string.
        :param end_date:
            End date, in ISO format if a string.  Defaults to a day after ``start_date``, if
            ``end_date`` is ``None``, and ``start_date`` is supplied.
        :param region:
            Region that images should intersect as a GeoJSON dictionary or ``ee.Geometry``.
        :param fill_portion:
            Lower limit on the portion of region that contains filled/valid image pixels (%).
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
        # TODO: refactor error classes and what gets raised where.
        # TODO: test for these errors
        # TODO: refactor as e.g. filterCoverage / filterClouds?  rename as filter?
        # TODO: if the collection has been filtered elsewhere, we should not enforce start_date
        #  or region
        if not start_date and not region:
            raise ValueError("At least one of 'start_date' or 'region' should be specified")
        if (fill_portion is not None or cloudless_portion is not None) and not region:
            raise ValueError(
                "'region' is required when 'fill_portion' or 'cloudless_portion' are specified."
            )
        if not start_date or not region:
            logger.warning(
                "Specifying 'start_date' together with 'region' will help limit the search."
            )

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
        resampling: ResamplingMethod | str = BaseImageAccessor._default_resampling,
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
            comp_image: ee.Image = ee_coll.gd.medoid(bands=self.reflBands)
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
        crs: str = None,
        crs_transform: Sequence[float] = None,
        shape: tuple[int, int] = None,
        region: dict | ee.Geometry = None,
        scale: float = None,
        resampling: str | ResamplingMethod = BaseImageAccessor._default_resampling,
        dtype: str = None,
        scale_offset: bool = False,
        bands: list[str | int] | str = None,
    ) -> ee.ImageCollection:
        """
        Prepare the collection for export.

        Bounds and resolution of the collection images can be specified with ``region`` and
        ``scale`` / ``shape``, or ``crs_transform`` and ``shape``.  Bounds default to those of
        the first image when they are not specified (with either ``region``, or ``crs_transform``
        & ``shape``).

        All images in the prepared collection share a common pixel grid and bounds.

        When ``crs``, ``scale``, ``crs_transform`` & ``shape`` are not provided, the pixel grids
        of the prepared images and first source image will match (i.e. if all source images share
        a pixel grid, the pixel grid of all prepared and source images will match).

        ..warning::
            The prepared collection images are reprojected and clipped versions of their
            source images. This type of image is `not recommended
            <https://developers.google.com/earth-engine/guides/best_practices>`__ for use in map
            display or further computation.

        :param crs:
            CRS of the prepared images as an EPSG or WKT string.  All image bands are
            re-projected to this CRS.  Defaults to the CRS of the minimum scale band of the first
            image.
        :param crs_transform:
            Geo-referencing transform of the prepared images, as a sequence of 6 numbers.  In
            row-major order: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation].
            All image bands are re-projected to this transform.
        :param shape:
            (height, width) dimensions of the prepared images in pixels.
        :param region:
            Region defining the prepared image bounds as a GeoJSON dictionary or ``ee.Geometry``.
            Defaults to the geometry of the first image.
        :param scale:
            Pixel scale (m) of the prepared images.  All image bands are re-projected to this
            scale. Ignored if ``crs`` and ``crs_transform`` are provided.  Defaults to the
            minimum scale of the first image's bands.
        :param resampling:
            Resampling method to use for reprojecting.  Ignored for images without fixed
            projections e.g. composites.  Composites can be resampled by resampling their
            component images.
        :param dtype:
            Data type of the prepared images (``uint8``, ``int8``, ``uint16``, ``int16``,
            ``uint32``, ``int32``, ``float32`` or ``float64``).  Defaults to the minimum size
            data type able to represent all of the first image's bands.
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
        first = BaseImageAccessor(self._ee_coll.first()).prepareForExport(
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
        first = BaseImageAccessor(first)

        # prepare collection images to have the same grid and bounds as the first image
        def prepare_image(ee_image: ee.Image) -> ee.Image:
            return BaseImageAccessor(ee_image).prepareForExport(
                crs=first.crs,
                crs_transform=first.transform,
                shape=first.shape,
                resampling=resampling,
                dtype=first.dtype,
                scale_offset=scale_offset,
                bands=first.bandNames,
            )

        return self._ee_coll.map(prepare_image)

    def toGeoTIFF(
        self,
        dirname: os.PathLike | str,
        overwrite: bool = False,
        split: str | SplitType = SplitType.bands,
        max_tile_size: float = BaseImageAccessor._ee_max_tile_size,
        max_tile_dim: int = BaseImageAccessor._ee_max_tile_dim,
        max_tile_bands: int = BaseImageAccessor._ee_max_tile_bands,
        max_requests: int = BaseImageAccessor._max_requests,
        max_cpus: int = None,
    ) -> None:
        """Export the collection to GeoTIFF files."""
        # TODO: document limitation on number of images.
        # TODO: can the max_* args be passed and documented as kwargs?
        split = SplitType(split)
        self._raise_image_consistency()
        images = self._split_images(split)

        dirname = Path(dirname)
        dirname.mkdir(exist_ok=True)
        tqdm_kwargs = utils.get_tqdm_kwargs(desc=self.id or 'Collection', unit=split)

        # download the split images sequentially, each into its own file
        for name, image in tqdm(images.items(), **tqdm_kwargs):
            filename = dirname.joinpath(str(name) + '.tif')
            image.toGeoTIFF(
                filename,
                overwrite=overwrite,
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
        max_tile_size: float = BaseImageAccessor._ee_max_tile_size,
        max_tile_dim: int = BaseImageAccessor._ee_max_tile_dim,
        max_tile_bands: int = BaseImageAccessor._ee_max_tile_bands,
        max_requests: int = BaseImageAccessor._max_requests,
        max_cpus: int = None,
    ) -> np.ndarray:
        """Export the collection to a NumPy array."""
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
        tqdm_kwargs = utils.get_tqdm_kwargs(desc=self.id, unit=split.value)
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

    def toXArray(
        self,
        masked: bool = False,
        split: str | SplitType = SplitType.bands,
        max_tile_size: float = BaseImageAccessor._ee_max_tile_size,
        max_tile_dim: int = BaseImageAccessor._ee_max_tile_dim,
        max_tile_bands: int = BaseImageAccessor._ee_max_tile_bands,
        max_requests: int = BaseImageAccessor._max_requests,
        max_cpus: int = None,
    ) -> xarray.Dataset:
        """Export the collection to an XArray DataSet."""
        if not xarray:
            raise ImportError("'toXArray()' requires the 'xarray' package to be installed.")
        split = SplitType(split)
        self._raise_image_consistency()
        images = self._split_images(split)

        # download the split image DataArrays sequentially, storing in a dict
        arrays = {}
        tqdm_kwargs = utils.get_tqdm_kwargs(desc=self.id, unit=split.value)
        for name, image in tqdm(images.items(), **tqdm_kwargs):
            arrays[name] = image.toXArray(
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
        attrs['stac'] = json.dumps(self.stac._item_dict) if self.stac else None
        attrs = {k: v for k, v in attrs.items() if v is not None}

        # return a DataSet of split image DataArrays
        return xarray.Dataset(arrays, attrs=attrs)


class MaskedCollection(ImageCollectionAccessor):
    def __init__(self, ee_collection: ee.ImageCollection, add_props: list[str] = None):
        """
        A class for describing, searching and compositing an Earth Engine image collection,
        with support for cloud/shadow masking.

        :param ee_collection:
            Earth Engine image collection to encapsulate.
        :param add_props:
            Additional Earth Engine image properties to include in :attr:`properties`.
        """
        if not isinstance(ee_collection, ee.ImageCollection):
            raise TypeError('`ee_collection` must be an instance of ee.ImageCollection')
        super().__init__(ee_collection)
        if add_props:
            self.schemaPropertyNames += add_props

    @classmethod
    def from_name(cls, name: str, add_props: list[str] = None) -> MaskedCollection:
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
        cls, image_list: list[str | MaskedImage | ee.Image], add_props: list[str] = None
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
        return self.reflBands

    def search(self, *args, **kwargs) -> MaskedCollection:
        ee_coll = self._ee_coll.gd.search(*args, **kwargs)
        gd_coll = MaskedCollection(ee_coll)
        gd_coll.schemaPropertyNames = self.schemaPropertyNames
        return gd_coll

    def composite(self, *args, **kwargs) -> MaskedImage:
        ee_image = self._ee_coll.gd.composite(*args, **kwargs)
        gd_image = MaskedImage(ee_image)
        return gd_image
