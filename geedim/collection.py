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
# Classes for searching GEE image collections
from datetime import datetime, timedelta
import ee
import pandas as pd
import click
import json

import geedim.image
from geedim import export, image, info, medoid, root_path

##
class Collection(object):
    def __init__(self, gd_coll_name):
        """
        Earth engine image collection related functions

        Parameters
        ----------
        gd_coll_name : str
                     Earth engine image collection name:
                     (possible values are: landsat7_c2_l2|landsat8_c2_l2|sentinel2_toa|sentinel2_sr|modis_nbar)
        """
        if not gd_coll_name in info.collection_info:
            raise ValueError(f'Unsupported collection: {gd_coll_name}')
        self._collection_info = info.collection_info[gd_coll_name]
        self.band_df = pd.DataFrame.from_dict(self._collection_info['bands'])
        self._im_props = pd.DataFrame(self._collection_info['properties'])  # TODO: rather get this from image class? as it is the thing providing the properties

        self._gd_image_cls =  image.coll_to_cls_map[gd_coll_name]
        self._ee_mapped_coll = None

    @classmethod
    def from_ids(cls, image_ids, mask=False, scale_refl=False):
        ee_coll_name = image.ee_split(image_ids[0])[0]
        if not ee_coll_name in info.ee_to_gd_map:
            raise ValueError(f'Unsupported collection: {ee_coll_name}')

        id_check = [image.ee_split(im_id)[0] == ee_coll_name for im_id in image_ids[1:]]
        if not all(id_check):
            raise ValueError(f'All images must belong to the same collection')

        gd_coll_name = info.ee_to_gd_map[ee_coll_name]
        gd_collection = cls(gd_coll_name)

        # build an ee.ImageCollection of processed (masked and scored) images
        im_list = ee.List([])
        for im_id in image_ids:
            gd_image = gd_collection._gd_image_cls.from_id(im_id, mask=mask, scale_refl=scale_refl)
            im_list = im_list.add(gd_image.ee_image)

        gd_collection._ee_mapped_coll = ee.ImageCollection(im_list)
        return gd_collection

    @property
    def ee_collection(self):
        return self._ee_mapped_coll

    @property
    def metadata(self):
        if self._ee_mapped_coll is None:
            return pd.DataFrame([], columns=self._im_props.ABBREV)

        return self._get_collection_df(self._ee_mapped_coll)

    def search(self, start_date, end_date, region, valid_portion=0, mask=False, scale_refl=False):
        """
        Search for images based on date, region etc criteria

        Parameters
        collection : geedim.collection.ImCollection
        start_date : datetime.datetime
                     Python datetime specifying the start image capture date
        end_date : datetime.datetime
                   Python datetime specifying the end image capture date (if None, then set to start_date)
        region : dict, geojson, ee.Geometry
                 Polygon in WGS84 specifying a region that images should intersect
        valid_portion: int, optional
                       Minimum portion (%) of image pixels that should be valid (not cloud)

        Returns
        -------
        image_df : pandas.DataFrame
        Dataframe specifying image properties that match the search criteria
        """
        # Initialise
        if (end_date is None):
            end_date = start_date + timedelta(days=1)
        if (end_date <= start_date):
            raise Exception('`end_date` must be at least a day later than `start_date`')

        click.echo(f'\nSearching for {self._collection_info["ee_coll_name"]} images between '
                   f'{start_date.strftime("%Y-%m-%d")} and {end_date.strftime("%Y-%m-%d")}...')

        def set_valid_portion(ee_image):
            max_scale = geedim.image.get_projection(ee_image, min=False).nominalScale()
            gd_image = self._gd_image_cls(ee_image, mask=mask, scale_refl=scale_refl)

            valid_portion = (gd_image.masks['valid_mask'].
                             unmask().
                             multiply(100).
                             reduceRegion(reducer='mean', geometry=region, scale=max_scale).
                             rename(['VALID_MASK'], ['VALID_PORTION']))

            return gd_image.ee_image.set(valid_portion)

        # filter the image collection
        self._ee_mapped_coll = (self._gd_image_cls.ee_collection().
                                filterDate(start_date, end_date).
                                filterBounds(region).
                                map(set_valid_portion).
                                filter(ee.Filter.gt('VALID_PORTION', valid_portion)))

        # convert and print search results
        return self._get_collection_df(self._ee_mapped_coll, do_print=True)

    def composite(self, method='q_mosaic'):
        # qualityMosaic will prefer clear pixels based on SCORE and irrespective of mask, for other methods, the mask
        # is needed to avoid including cloudy pixels
        method = str(method).lower()

        if method == 'q_mosaic':
            comp_image = self._ee_mapped_coll.qualityMosaic('SCORE')
        elif method == 'mosaic':
            comp_image = self._ee_mapped_coll.mosaic()
        elif method == 'median':
            comp_image = self._ee_mapped_coll.median()  # TODO this finds median of mask, q bands which may not be meaningful, find median of sr bands only as in medoid, if appropriate
            comp_image = self._gd_image_cls._im_transform(comp_image)
        elif method == 'medoid':
            bands = [band_dict['id'] for band_dict in self._collection_info['bands']]
            comp_image = medoid.medoid(self._ee_mapped_coll, bands=bands)
        else:
            raise ValueError(f'Unsupported composite method: {method}')

        # populate image metadata with info on component images
        im_prop_df = self._get_collection_df(self._ee_mapped_coll, do_print=False)
        comp_str = im_prop_df.to_string(
            float_format='%.2f',
            formatters={'DATE': lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M')},
            columns=self._im_props.ABBREV,
            # header=property_df.ABBREV,
            index=False,
            justify='center')

        comp_image = comp_image.set('COMPOSITE_IMAGES', comp_str)

        # name the composite
        start_date = im_prop_df.DATE.iloc[0].strftime('%Y_%m_%d')
        end_date = im_prop_df.DATE.iloc[-1].strftime('%Y_%m_%d')
        ee_coll_name = image.ee_split(im_prop_df.ID.values[0])[0]
        comp_name = f'{ee_coll_name}/{start_date}-{end_date}-{method.upper()}_COMP'
        comp_image = comp_image.set('system:id', comp_name)

        return comp_image, comp_name

    # TODO: get this once per collection, and make a property?  update on each search, once only
    def _get_collection_df(self, ee_collection, do_print=True):
        """
        Convert a filtered image collection to a pandas dataframe of images and their properties

        Parameters
        ----------
        ee_collection : ee.ImageCollection
                        Filtered image collection
        do_print : bool, optional
                   Print the dataframe

        Returns
        -------
        : pandas.DataFrame
        Dataframe of ee.Image objects and their properties
        """

        init_list = ee.List([])

        # aggregate relevant properties of ee_collection images
        def aggregrate_props(image, prop_list):
            prop = ee.Dictionary()
            for prop_key in self._im_props.PROPERTY.values:
                prop = prop.set(prop_key, ee.Algorithms.If(image.get(prop_key), image.get(prop_key), ee.String('None')))
            return ee.List(prop_list).add(prop)

        # retrieve list of dicts of collection image properties (the only call to getInfo in *ImSearch)
        im_prop_list = ee.List(ee_collection.iterate(aggregrate_props, init_list)).getInfo()

        if len(im_prop_list) == 0:
            click.echo('No images found')   # TODO: put this in CLI
            return pd.DataFrame([], columns=self._im_props.ABBREV)

        # im_list = ee_collection.toList(ee_collection.size())  # image objects

        # add EE image objects and convert ee.Date to python datetime
        for i, prop_dict in enumerate(im_prop_list):
            if 'system:time_start' in prop_dict:
                prop_dict['system:time_start'] = datetime.utcfromtimestamp(prop_dict['system:time_start'] / 1000)
            # prop_dict['IMAGE'] = ee.Image(im_list.get(i))

        # convert to DataFrame
        im_prop_df = pd.DataFrame(im_prop_list, columns=im_prop_list[0].keys())
        im_prop_df = im_prop_df.sort_values(by='system:time_start').reset_index(drop=True)
        im_prop_df = im_prop_df.rename(columns=dict(zip(self._im_props.PROPERTY, self._im_props.ABBREV)))  # rename cols to abbrev
        im_prop_df = im_prop_df[self._im_props.ABBREV.to_list()] #+ ['IMAGE']]     # reorder columns

        if do_print:
            click.echo(f'{len(im_prop_list)} images found')
            click.echo('\nImage property descriptions:\n\n' +
                        self._im_props[['ABBREV', 'DESCRIPTION']].to_string(index=False, justify='right'))

            click.echo('\nSearch Results:\n\n' + im_prop_df.to_string(
                float_format='%.2f',
                formatters={'DATE': lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M')},
                columns=self._im_props.ABBREV,
                # header=property_df.ABBREV,
                index=False,
                justify='center'))

        return im_prop_df
