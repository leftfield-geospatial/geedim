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
# Functionality for searching and compositing EE image collections
from datetime import datetime, timedelta

import ee
import pandas as pd

from geedim import info, medoid
from geedim.image import BaseImage, split_id
from geedim.masked_image import MaskedImage, class_from_id

logger = logging.getLogger(__name__)


##
class MaskedCollection:
    _composite_methods = ["q_mosaic", "mosaic", "median", "medoid"]  # supported composite methods
    _default_comp_method = 'q_mosaic'

    def __init__(self, ee_coll_name):
        """
        Class for searching and compositing an EE image collection

        Parameters
        ----------
        ee_coll_name : str
                       EE image collection ID
        """
        self._ee_coll_name = ee_coll_name
        if ee_coll_name in info.collection_info:
            self._collection_info = info.collection_info[ee_coll_name]
        else:
            self._collection_info = info.collection_info['*']
        self._ee_collection = ee.ImageCollection(ee_coll_name)

        self._summary_key_df = pd.DataFrame(self._collection_info["properties"])  # key to metadata summary
        self._summary_df = None  # summary of the image metadata

        self._image_class = class_from_id(ee_coll_name)  # geedim.masked_image.*Image class for this collection
        self._ee_collection = ee.ImageCollection(
            ee_coll_name)  # self._image_class.ee_collection(self._ee_coll_name)  # the wrapped ee.ImageCollection

    @classmethod
    def from_list(cls, image_list):
        """
        Create collection from a list of image IDs, ee.Image's and MaskedImage's

        Parameters
        ----------
        image_list : list(Union[str, ee.Image, MaskedImage])
                    A list of images (must all be from the same collection)

        Returns
        -------
        collection: cls
        """
        ee_image_list = []
        ee_id_list = []
        # cloud_masked = []  # TODO: we should be able to remove this logic when we get rid of BaseCollection, maybe get rid of this factory entirely and just use MaskedCollection.from_list()
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
        ee_coll_name = split_id(ee_id_list[0])[0]
        id_check = [split_id(im_id)[0] == ee_coll_name for im_id in ee_id_list[1:]]
        if not all(id_check):
            # TODO: allow images from compatible landsat collections
            raise ValueError("All images must belong to the same collection")

        # create the collection object
        gd_collection = cls(ee_coll_name)
        gd_collection._ee_collection = ee.ImageCollection(ee.List(ee_image_list))
        return gd_collection

    @property
    def ee_collection(self):
        """ee.ImageCollection : Returns the wrapped ee.ImageCollection"""
        return self._ee_collection

    @property
    def summary_key_df(self):
        """pandas.DataFrame : A key to MaskedCollection.summary_df
        (pandas.DataFrame with ABBREV and DESCRIPTION columns, and rows corresponding columns in summary_df)"""
        return self._summary_key_df

    @property
    def summary_df(self):
        """pandas.DataFrame : Summary of collection image properties with a row for each image"""
        if self._summary_df is None:
            self._summary_df = self._get_summary_df(self._ee_collection)
        return self._summary_df

    @property
    def summary_key(self):
        """str :  Formatted string of MaskedCollection.summary_key_df"""
        return self._summary_key_df[["ABBREV", "DESCRIPTION"]].to_string(index=False, justify="right")

    @property
    def summary(self):
        """str : Formatted string of MaskedCollection.summary_df"""
        # TODO: allow this to be called before search, refactor all these methods somehow to make better sense
        return self._get_summary_str(self._summary_df)

    def _get_summary_str(self, summary_df):
        return summary_df.to_string(
            float_format="{:.2f}".format,
            formatters={"DATE": lambda x: datetime.strftime(x, "%Y-%m-%d %H:%M")},
            columns=self._summary_key_df.ABBREV,
            index=False,
            justify="center",
        )

    def _get_summary_df(self, ee_collection):
        """
        Retrieve a summary of collection image metadata.

        Parameters
        ----------
        ee_collection : ee.ImageCollection
                        Filtered image collection whose image metadata to retrieve

        Returns
        -------
        : pandas.DataFrame
        pandas.DataFrame with a row of metadata for each image)
        """

        if ee_collection is None:
            return pd.DataFrame([], columns=self._summary_key_df.ABBREV)  # return empty dataframe

        # server side aggregation of relevant properties of ee_collection images
        init_list = ee.List([])

        def aggregrate_props(ee_image, prop_list):
            all_props = ee_image.propertyNames()
            prop_dict = ee.Dictionary()
            for prop_key in self._summary_key_df.PROPERTY.values:
                prop_dict = prop_dict.set(
                    prop_key, ee.Algorithms.If(
                        all_props.contains(prop_key), ee_image.get(prop_key), ee.String("None"))
                )
            return ee.List(prop_list).add(prop_dict)

        # retrieve list of dicts of collection image properties (the only call to getInfo() in MaskedCollection)
        im_prop_list = ee.List(ee_collection.iterate(aggregrate_props, init_list)).getInfo()

        if len(im_prop_list) == 0:
            return pd.DataFrame([], columns=self._summary_key_df.ABBREV)  # return empty dataframe

        # Convert ee.Date to python datetime
        start_time_key = "system:time_start"
        for i, prop_dict in enumerate(im_prop_list):
            if start_time_key in prop_dict:
                prop_dict[start_time_key] = datetime.utcfromtimestamp(prop_dict[start_time_key] / 1000)

        # convert property list to DataFrame
        im_prop_df = pd.DataFrame(im_prop_list, columns=im_prop_list[0].keys())
        im_prop_df = im_prop_df.sort_values(by=start_time_key).reset_index(drop=True)  # sort by acquisition time
        im_prop_df = im_prop_df.reset_index(drop=True)
        im_prop_df = im_prop_df.rename(
            columns=dict(zip(self._summary_key_df.PROPERTY, self._summary_key_df.ABBREV))
        )  # abbreviate column names
        im_prop_df = im_prop_df[self._summary_key_df.ABBREV.to_list()]  # reorder columns

        return im_prop_df

    # TODO: we need to pass cloud masking kwargs somehow.  it may be better to do this in a sort of environment, or globally?
    def search(self, start_date, end_date, region, valid_portion=0, **kwargs):
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
        valid_portion: int, optional
                       Minimum portion (%) of image pixels that should be valid (not cloud/shadow).

        Returns
        -------
        pandas.DataFrame
            Dataframe specifying image properties that match the search criteria
        """
        # TODO: error message for q_mosaic with generic collection...  Or allow to spec a band?
        # TODO: make a reset method to unfilter the collection
        # Initialise
        if end_date is None:
            end_date = start_date + timedelta(days=1)
        if end_date <= start_date:
            raise ValueError("`end_date` must be at least a day later than `start_date`")

        def set_region_stats(ee_image):
            gd_image = self._image_class(ee_image, **kwargs)
            gd_image.set_region_stats(region)
            return gd_image.ee_image

        try:
            # filter the image collection, finding cloud/shadow masks, and region stats
            self._ee_collection = (
                self._ee_collection
                    .filterDate(start_date, end_date)
                    .filterBounds(region)
                    .map(set_region_stats)
                    .filter(ee.Filter.gte("CLOUDLESS_PORTION", valid_portion))
            )
        finally:
            # update summary_df with image metadata from the filtered collection
            self._summary_df = self._get_summary_df(self._ee_collection)

        return self._summary_df

    # TODO: expose region and date to CLI
    def composite(self, method=_default_comp_method, mask=True, resampling=BaseImage._default_resampling,
                  region=None, date=None):
        """
        Create a cloud/shadow free composite.

        Note: composite() can be called on a filtered collection created by search(..), or on a collection created with
              fromIds(...)
              The `mask` parameter MaskedCollection.fromIds(...) affects the composite and should generally be
              True so that cloud/shadow pixels are excluded.

        Parameters
        ----------
        method : str, optional
                 Compositing method to use (q_mosaic|mosaic|median|medoid).  (Default: q_mosaic).
        resampling : str, optional
               Resampling method for compositing and re-projecting: ("near"|"bilinear"|"bicubic") (default: "near")

        Returns
        -------
        comp_image: MaskedImage
          The composite image, composite image ID
        """
        method = str(method).lower()

        def prepare_image(ee_image):
            gd_image = self._image_class(ee_image)
            if method in ['mosaic', 'q_mosaic']:
                if date:
                    date_dist = ee.Number(gd_image.ee_image.get("system:time_start")).subtract(
                        ee.Date(date).millis()).abs()
                    gd_image.ee_image = gd_image.ee_image.set('DATE_DIST', date_dist)
                else:
                    gd_image.set_region_stats(region)
            if mask:
                gd_image.mask_clouds()
            if resampling != BaseImage._default_resampling:
                # TODO: what does resampling do to non SR bands - should they be excluded?  should we not do this after masking?
                gd_image.ee_image = gd_image.ee_image.resample(resampling)
            return gd_image.ee_image

        ee_collection = self._ee_collection.map(prepare_image)

        if method in ['mosaic', 'q_mosaic']:
            if date:
                # sort the collection by time difference to `date`, so that *mosaic uses the closest in time pixels
                ee_collection = ee_collection.sort('DATE_DIST', opt_ascending=False)
            else:
                # sort the collection by cloud/shadow free portion, so that *mosaic favours pixels from the least
                # cloudy image
                ee_collection = ee_collection.sort('CLOUDLESS_PORTION')

        if method == "q_mosaic":
            comp_image = ee_collection.qualityMosaic("CLOUD_DIST")
        elif method == "mosaic":
            comp_image = ee_collection.mosaic()
        elif method == "median":
            comp_image = ee_collection.median()
            # median creates float images, so re-apply any type conversion
        elif method == "medoid":
            # limit medoid to surface reflectance bands
            # TODO: as we are losing collection_info, we will need another way to get sr_bands
            sr_bands = [band_dict["id"] for band_dict in self._collection_info["bands"]]
            comp_image = medoid.medoid(ee_collection, bands=sr_bands)
        else:
            raise ValueError(f"Unsupported composite method: {method}")

        # populate image metadata with info on component images
        summary_df = self._get_summary_df(ee_collection)
        summary_str = self._get_summary_str(summary_df)
        comp_image = comp_image.set("COMPONENT_IMAGES", '\n' + summary_str)

        # construct an ID for the composite
        # TODO: get summary_df for ee_collection, not self._ee_collection.  We want to leave collection unchanged,
        #  in case there are repeat composites/searches.  Which should also be tested.
        start_date = summary_df.DATE.min().strftime("%Y_%m_%d")
        end_date = summary_df.DATE.max().strftime("%Y_%m_%d")

        method_str = method.upper()
        if method in ['mosaic', 'q_mosaic'] and date:
            method_str += '-' + date.strftime("%Y_%m_%d")

        comp_id = f"{self._ee_coll_name}/{start_date}-{end_date}-{method_str}-COMP"
        comp_image = comp_image.set("system:id", comp_id)
        comp_image = comp_image.set("system:index", comp_id)
        comp_image = comp_image.set("system:time_start", self.summary_df.DATE.iloc[0].timestamp() * 1000)
        # TODO: do the QA, mask and score bands mosaic correctly?
        #  would re-calculating the masks and score on the mosaics QA bands work?
        # TODO: leave out the median method entirely?
        if method == 'median':
            gd_image = MaskedImage(comp_image)
        else:
            gd_image = self._image_class(comp_image)
        return gd_image
