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
##
from datetime import datetime, timedelta

import ee
import pandas
import pandas as pd

from geedim import info, medoid
from geedim.enums import ResamplingMethod, CompositeMethod
from geedim.image import BaseImage, split_id
from geedim.masked_image import MaskedImage, class_from_id

logger = logging.getLogger(__name__)


##
class MaskedCollection:
    """
    Class for encapsulating, searching and compositing an Earth Engine image collection, with support for
    cloud/shadow masking.
    """
    _default_comp_method = CompositeMethod.q_mosaic

    def __init__(self, ee_coll_name):
        """
        Create a MaskedCollection instance.

        Parameters
        ----------
        ee_coll_name : str
            The ID of EE image collection encapsulate.
        """
        self._ee_coll_name = ee_coll_name
        if ee_coll_name in info.collection_info:
            self._collection_info = info.collection_info[ee_coll_name]
        else:
            self._collection_info = info.collection_info['*']
        self._ee_collection = ee.ImageCollection(ee_coll_name)

        self._summary_key_df = pd.DataFrame(self._collection_info['properties'])  # key to metadata summary
        self._summary_df = None  # summary of the image metadata

        self._image_class = class_from_id(ee_coll_name)
        self._ee_collection = ee.ImageCollection(ee_coll_name)

    @classmethod
    def from_list(cls, image_list):
        """
        Create a MaskedCollection instance from a list of EE image IDs, ee.Image's and/or MaskedImage's

        Parameters
        ----------
        image_list : List[Union[str, ee.Image, MaskedImage], ]
            A list of images to include in the collection (must all be from the same EE collection).

        Returns
        -------
        collection: MaskedCollection
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
                raise TypeError(f'Unsupported image object type: {type(image_obj)}')

        # check the images all come from the same collection
        ee_coll_name = split_id(ee_id_list[0])[0]
        id_check = [split_id(im_id)[0] == ee_coll_name for im_id in ee_id_list[1:]]
        if not all(id_check):
            # TODO: allow images from compatible landsat collections
            raise ValueError('All images must belong to the same collection')

        # create the collection object
        gd_collection = cls(ee_coll_name)
        gd_collection._ee_collection = ee.ImageCollection(ee.List(ee_image_list))
        return gd_collection

    @property
    def ee_collection(self) -> ee.ImageCollection:
        """The encapsulated ee.ImageCollection."""
        return self._ee_collection

    @property
    def summary_key_df(self) -> pandas.DataFrame:
        """
        A key to MaskedCollection.summary_df (pandas.DataFrame with ABBREV and DESCRIPTION columns, and rows
        corresponding to columns in summary_df).
        """
        return self._summary_key_df

    @property
    def summary_df(self) -> pandas.DataFrame:
        """Summary of collection image properties with a row for each image."""
        if self._summary_df is None:
            self._summary_df = self._get_summary_df(self._ee_collection)
        return self._summary_df

    @property
    def summary_key(self) -> str:
        """Formatted string of MaskedCollection.summary_key_df."""
        return self._summary_key_df[['ABBREV', 'DESCRIPTION']].to_string(index=False, justify='right')

    @property
    def summary(self) -> str:
        """Formatted string of MaskedCollection.summary_df."""
        # TODO: allow this to be called before search & refactor all these methods
        return self._get_summary_str(self._summary_df)

    def _get_summary_str(self, summary_df) -> str:
        """Get a formatted/printable string for a given summary DataFrame."""
        return summary_df.to_string(
            float_format='{:.2f}'.format, formatters={'DATE': lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M')},
            columns=self._summary_key_df.ABBREV, index=False, justify='center'
        )

    def _get_summary_df(self, ee_collection) -> pandas.DataFrame:
        """Retrieve a summary of the collection image metadata."""

        if ee_collection is None:
            return pd.DataFrame([], columns=self._summary_key_df.ABBREV)  # return empty dataframe

        # server side aggregation of relevant properties of ee_collection images
        init_list = ee.List([])

        def aggregrate_props(ee_image, prop_list):
            all_props = ee_image.propertyNames()
            _prop_dict = ee.Dictionary()
            for prop_key in self._summary_key_df.PROPERTY.values:
                _prop_dict = _prop_dict.set(
                    prop_key, ee.Algorithms.If(
                        all_props.contains(prop_key), ee_image.get(prop_key), ee.String('None')
                    )
                )
            return ee.List(prop_list).add(_prop_dict)

        # retrieve list of dicts of collection image properties (the only call to getInfo() in MaskedCollection)
        im_prop_list = ee.List(ee_collection.iterate(aggregrate_props, init_list)).getInfo()

        if len(im_prop_list) == 0:
            return pd.DataFrame([], columns=self._summary_key_df.ABBREV)  # return empty dataframe

        # Convert ee.Date to python datetime
        start_time_key = 'system:time_start'
        for i, prop_dict in enumerate(im_prop_list):
            if start_time_key in prop_dict:
                prop_dict[start_time_key] = datetime.utcfromtimestamp(prop_dict[start_time_key] / 1000)

        # convert property list to DataFrame
        im_prop_df = pd.DataFrame(im_prop_list, columns=im_prop_list[0].keys())
        im_prop_df = im_prop_df.sort_values(by=start_time_key).reset_index(drop=True)  # sort by acquisition time
        im_prop_df = im_prop_df.reset_index(drop=True)
        # abbreviate column names
        im_prop_df = im_prop_df.rename(
            columns=dict(zip(self._summary_key_df.PROPERTY, self._summary_key_df.ABBREV))
        )
        im_prop_df = im_prop_df[self._summary_key_df.ABBREV.to_list()]  # reorder columns

        return im_prop_df

    def search(self, start_date, end_date, region, cloudless_portion=0, **kwargs):
        """
        Search for images based on date, region etc criteria

        Parameters
        ----------
        start_date : datetime.datetime
            Start image capture date.
        end_date : datetime.datetime
            End image capture date (if None, then set to start_date + 1 day).
        region : dict, geojson, ee.Geometry
            Polygon in WGS84 specifying a region that images should intersect.
        cloudless_portion: int, optional
            Minimum portion (%) of image pixels that should be cloud/shadow free.
        kwargs: optional
            Cloud/shadow masking parameters - see geedim.MaskedImage.__init__() for details.

        Returns
        -------
        results_df: pandas.DataFrame
            Dataframe specifying image properties that match the search criteria.
        """
        # TODO: make a reset method to unfilter the collection
        # Initialise
        if end_date is None:
            end_date = start_date + timedelta(days=1)
        if end_date <= start_date:
            raise ValueError('`end_date` must be at least a day later than `start_date`')

        def set_region_stats(ee_image):
            gd_image = self._image_class(ee_image, **kwargs)
            gd_image.set_region_stats(region)
            return gd_image.ee_image

        try:
            # filter the image collection, finding cloud/shadow masks, and region stats
            self._ee_collection = (
                self._ee_collection.filterDate(start_date, end_date).
                    filterBounds(region).
                    map(set_region_stats).
                    filter(ee.Filter.gte('CLOUDLESS_PORTION', cloudless_portion))
            )
        finally:
            # update summary_df with image metadata from the filtered collection
            self._summary_df = self._get_summary_df(self._ee_collection)

        return self._summary_df

    def composite(
            self, method=_default_comp_method, mask=True, resampling=BaseImage._default_resampling, date=None,
            region=None, **kwargs
    ):
        """
        Create a composite image from the encapsulated collection.

        Parameters
        ----------
        method: CompositeMethod, optional
            The comppositing method to use.  One of:
                `q_mosiac`: Select each composite pixel from the collection image with the highest quality (cloud
                    distance). When more than one image shares the highest quality value, the first of the competing
                    images is used. Valid for cloud/shadow maskable image collections only (Sentinel-2 TOA and SR, and
                    Landsat4-9 level 2 collection 2).
                `mosaic`: Select each composite pixel from the first unmasked collection image.
                `medoid`: Select each composite pixel as the the image pixel having the minimum summed diff (across
                    bands) from the median of all collection images.  Maintains the original relationship between
                    bands.  See https://www.mdpi.com/2072-4292/5/12/6481 for detail.
                `median`: Median of the collection images.
                `mode`: Mode of the collection images.
                `mean`: Mean of the collection images.
        mask: bool, optional
            Whether to cloud/shadow mask images before compositing  [default: True].
        resampling: ResamplingMethod, optional
            The resampling method to use on collection images prior to compositing.  If 'near', no resampling is done
            [default: 'near'].
        date: datetime.datetime, optional
            Sort collection images by their absolute difference in time from this date.  Useful for
            prioritising pixels from images closest to this date.  Valid for the `q-mosaic`
            and `mosaic` methods only.  If None, time difference sorting is not done. [default: None].
        region: dict, geojson, optional
            Sort collection images by their cloudless portion inside this region (only if `date` is not
            specified).  This is useful to prioritise pixels from the least cloudy image(s).  If `date` and `region`
            are not specified, collection images are sorted by their capture date.  Valid for the `q-mosaic` and
            `mosaic` methods.
        kwargs: optional
            Cloud/shadow masking parameters - see geedim.MaskedImage.__init__() for details.

        Returns
        -------
        comp_image: MaskedImage
            The composite image.
        """
        # TODO: test composite of resampled images and resampled composite
        method = CompositeMethod(method)
        resampling = ResamplingMethod(resampling)
        if (method == CompositeMethod.q_mosaic) and (self._image_class == MaskedImage):
            # TODO get a list of supported collections, report this in CLI help too
            raise ValueError(f'The `q-mosaic` method is not supported for the {self._ee_coll_name} collection.')

        def prepare_image(ee_image):
            gd_image = self._image_class(ee_image, **kwargs)
            if method in ['mosaic', 'q_mosaic']:
                if date:
                    date_dist = ee.Number(gd_image.ee_image.get('system:time_start')).subtract(
                        ee.Date(date).millis()
                    ).abs()
                    gd_image.ee_image = gd_image.ee_image.set('DATE_DIST', date_dist)
                elif region:
                    gd_image.set_region_stats(region)
            if mask:
                gd_image.mask_clouds()
            if resampling != BaseImage._default_resampling:
                # TODO: what does resampling do to non SR bands - should they be excluded?  should we not do this
                #  after masking?
                gd_image.ee_image = gd_image.ee_image.resample(resampling.value)
            return gd_image.ee_image

        ee_collection = self._ee_collection.map(prepare_image)

        if method in [CompositeMethod.mosaic, CompositeMethod.q_mosaic]:
            if date:
                # sort the collection by time difference to `date`, so that *mosaic uses the closest in time pixels
                ee_collection = ee_collection.sort('DATE_DIST', opt_ascending=False)
            elif region:
                # sort the collection by cloud/shadow free portion, so that *mosaic favours pixels from the least
                # cloudy image
                ee_collection = ee_collection.sort('CLOUDLESS_PORTION')
            else:
                # sort the collection by capture date
                ee_collection = ee_collection.sort('system:time_start')

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
            sr_bands = [band_dict['id'] for band_dict in self._collection_info['bands']]
            comp_image = medoid.medoid(ee_collection, bands=sr_bands)
        elif method == CompositeMethod.mode:
            comp_image = ee_collection.mode()
        elif method == CompositeMethod.mean:
            comp_image = ee_collection.mean()
        else:
            raise ValueError(f'Unsupported composite method: {method}')

        # populate image metadata with info on component images
        summary_df = self._get_summary_df(ee_collection)
        summary_str = self._get_summary_str(summary_df)
        comp_image = comp_image.set('COMPONENT_IMAGES', '\n' + summary_str)

        # construct an ID for the composite
        # TODO: get summary_df for ee_collection, not self._ee_collection.  We want to leave collection unchanged,
        #  in case there are repeat composites/searches.  Which should also be tested.
        start_date = summary_df.DATE.min().strftime('%Y_%m_%d')
        end_date = summary_df.DATE.max().strftime('%Y_%m_%d')

        method_str = method.value.upper()
        if method in [CompositeMethod.mosaic, CompositeMethod.q_mosaic] and date:
            method_str += '-' + date.strftime('%Y_%m_%d')

        comp_id = f'{self._ee_coll_name}/{start_date}-{end_date}-{method_str}-COMP'
        comp_image = comp_image.set('system:id', comp_id)
        comp_image = comp_image.set('system:index', comp_id)
        comp_image = comp_image.set('system:time_start', summary_df.DATE.iloc[0].timestamp() * 1000)
        gd_comp_image = self._image_class(comp_image)

        return gd_comp_image
