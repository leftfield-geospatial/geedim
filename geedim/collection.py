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
from datetime import datetime
import ee
import pandas as pd
import click

from geedim import export, search

##
class ImCollection:
    def __init__(self, collection):
        """
        Earth engine image collection related functions

        Parameters
        ----------
        collection : str
                     Earth engine image collection name:
                     (possible values are: landsat7_c2_l2|landsat8_c2_l2|sentinel2_toa|sentinel2_sr|modis_nbar)
        """
        collection_info = search.load_collection_info()
        if collection not in collection_info:
            raise ValueError(f'Unknown collection: {collection}')
        self.collection_info = collection_info[collection]

        self._im_props = pd.DataFrame(self.collection_info['properties'])  # list of image properties of interest
        self._im_transform = lambda image: image

    def get_ee_collection(self):
        """
        Returns the unfiltered earth engine image collection

        Returns
        -------
        : ee.ImageCollection
        """
        return ee.ImageCollection(self.collection_info['ee_collection'])

    def get_image(self, image_id, apply_mask=False, add_aux_bands=False, scale_refl=False):
        """
        Retrieve an ee.Image object, optionally adding metadata and auxillary bands

        Parameters
        ----------
        image_id : str
             Earth engine image ID e.g. 'LANDSAT/LC08/C02/T1_L2/LC08_182037_20190118 2019-01-18'
        apply_mask : bool, optional
                     Apply any validity mask to the image by setting nodata (default: False)
        add_aux_bands: bool, optional
                       Add auxiliary bands (cloud, shadow, fill & validity masks, and quality score) (default: False)
        scale_refl : bool, optional
                     Scale reflectance values from 0-10000 if they are not in that range already (default: True)

        Returns
        -------
        : ee.Image
          The processed image
        """
        if '/'.join(image_id.split('/')[:-1]) != self.collection_info['ee_collection']:
            raise ValueError(f'{image_id} is not a valid earth engine id for {self.__class__}')

        image = ee.Image(image_id)
        masks = self._get_image_masks(image)

        if add_aux_bands:
            score = self.get_image_score(image, masks=masks)
            image = image.addBands(ee.Image(list(masks.values())))
            image = image.addBands(score)

        if apply_mask:  # mask before adding aux bands
            image = image.updateMask(masks['valid_mask'])

        return self._im_transform(image)


    def _get_image_masks(self, image):
        """
        Derive cloud, shadow, fill and validity masks for an image

        Parameters
        ----------
        image : ee.Image
                Derive masks for this image

        Returns
        -------
        masks : dict
                A dictionary with cloud_mask, shadow_mask, fill_mask and valid_mask keys, and corresponding ee.Image
                values
        """
        masks = dict(cloud_mask=ee.Image(0).rename('CLOUD_MASK'),
                     shadow_mask=ee.Image(0).rename('SHADOW_MASK'),
                     fill_mask=ee.Image(1).rename('FILL_MASK'),
                     valid_mask=ee.Image(1).rename('VALID_MASK'))

        return masks

    def set_image_valid_portion(self, image, region=None, masks=None):
        """
        Find the portion of valid image pixels for a given region

        Parameters
        ----------
        image : ee.Image
                Find valid portion for this image
        region : dict, geojson, ee.Geometry, optional.
                 Polygon in WGS84 specifying the region. If none, uses the image granule if it exists.

        Returns
        -------
        : ee.Image
        Image with the 'VALID_PORTION' property set
        """
        if masks is None:
            masks = self._get_image_masks(image)

        max_scale = export.get_projection(image, min=False).nominalScale()
        if region is None:
            region = image.geometry()

        valid_portion = (masks['valid_mask'].
                         unmask().
                         multiply(100).
                         reduceRegion(reducer='mean', geometry=region, scale=max_scale).
                         rename(['VALID_MASK'], ['VALID_PORTION']))

        return image.set(valid_portion)

    def get_image_score(self, image, cloud_dist=2000, masks=None):
        """
        Get the cloud distance quality score for this image

        Parameters
        ----------
        image : ee.Image
                Find the score for this image
        cloud_dist : int, oprtional
                     The neighbourhood (in meters) in which to search for clouds
        masks : dict, optional
                Masks returned by _get_image_masks(...)
        Returns
        -------
        : ee.Image
        The cloud distance score as a single band image
        """
        radius = 1.5
        min_proj = export.get_projection(image)
        cloud_pix = ee.Number(cloud_dist).divide(min_proj.nominalScale()).toInt()

        if masks is None:
            masks = self._get_image_masks(image)

        cloud_shadow_mask = masks['cloud_mask'].Or(masks['shadow_mask'])
        cloud_shadow_mask = cloud_shadow_mask.focal_min(radius=radius).focal_max(radius=radius)

        score = cloud_shadow_mask.fastDistanceTransform(neighborhood=cloud_pix, units='pixels',
                                                      metric='squared_euclidean').sqrt().rename('SCORE')

        return score.unmask().where(masks['fill_mask'].unmask().Not(), 0)

    def _get_collection_df(self, im_collection, do_print=True):
        """
        Convert a filtered image collection to a pandas dataframe of images and their properties

        Parameters
        ----------
        im_collection : ee.ImageCollection
                        Filtered image collection
        do_print : bool, optional
                   Print the dataframe

        Returns
        -------
        : pandas.DataFrame
        Dataframe of ee.Image objects and their properties
        """

        init_list = ee.List([])

        # aggregate relevant properties of im_collection images
        def aggregrate_props(image, prop_list):
            prop = ee.Dictionary()
            for prop_key in self._im_props.PROPERTY.values:
                prop = prop.set(prop_key, ee.Algorithms.If(image.get(prop_key), image.get(prop_key), ee.String('None')))
            return ee.List(prop_list).add(prop)

        # retrieve list of dicts of collection image properties (the only call to getInfo in *ImSearch)
        im_prop_list = ee.List(im_collection.iterate(aggregrate_props, init_list)).getInfo()

        if len(im_prop_list) == 0:
            click.echo('No images found')
            return pd.DataFrame([], columns=self._im_props.ABBREV)

        im_list = im_collection.toList(im_collection.size())  # image objects

        # add EE image objects and convert ee.Date to python datetime
        for i, prop_dict in enumerate(im_prop_list):
            if 'system:time_start' in prop_dict:
                prop_dict['system:time_start'] = datetime.utcfromtimestamp(prop_dict['system:time_start'] / 1000)
            prop_dict['IMAGE'] = ee.Image(im_list.get(i))

        # convert to DataFrame
        im_prop_df = pd.DataFrame(im_prop_list, columns=im_prop_list[0].keys())
        im_prop_df = im_prop_df.sort_values(by='system:time_start').reset_index(drop=True)
        im_prop_df = im_prop_df.rename(columns=dict(zip(self._im_props.PROPERTY, self._im_props.ABBREV)))  # rename cols to abbrev
        im_prop_df = im_prop_df[self._im_props.ABBREV.to_list() + ['IMAGE']]     # reorder columns

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


class LandsatImCollection(ImCollection):
    def __init__(self, collection='landsat8_c2_l2'):
        """
        Class for Landsat 7-8 earth engine image collections

        Parameters
        ----------
        collection : str, optional
                     'landsat7_c2_l2' or 'landsat8_c2_l2' (default)
        """
        ImCollection.__init__(self, collection=collection)

        # TODO: add support for landsat 4-5 collection 2 when they are available
        if collection not in ['landsat8_c2_l2', 'landsat7_c2_l2']:
            raise ValueError(f'Unsupported landsat collection: {collection}')

        self._im_transform = ee.Image.toUint16

    def _get_image_masks(self, image):
        """
        Derive cloud, shadow, fill and validity masks for an image

        Parameters
        ----------
        image : ee.Image
                Derive masks for this image

        Returns
        -------
        masks : dict
                A dictionary with cloud_mask, shadow_mask, fill_mask and valid_mask keys, and corresponding ee.Image
                values
        """

        masks = ImCollection._get_image_masks(self, image)

        # get cloud, shadow and fill masks from QA_PIXEL
        qa_pixel = image.select('QA_PIXEL')
        cloud_mask = qa_pixel.bitwiseAnd((1 << 1) | (1 << 2) | (1 << 3)).neq(0).rename('CLOUD_MASK')
        shadow_mask = qa_pixel.bitwiseAnd(1 << 4).neq(0).rename('SHADOW_MASK')
        fill_mask = qa_pixel.bitwiseAnd(1).eq(0).rename('FILL_MASK')

        # extract cloud etc quality scores (2 bits)
        # cloud_conf = qa_pixel.rightShift(8).bitwiseAnd(3).rename('CLOUD_CONF')
        # cloud_shadow_conf = qa_pixel.rightShift(10).bitwiseAnd(3).rename('CLOUD_SHADOW_CONF')
        # cirrus_conf = qa_pixel.rightShift(14).bitwiseAnd(3).rename('CIRRUS_CONF')

        if self.collection_info['ee_collection'] == 'LANDSAT/LC08/C02/T1_L2':  # landsat8_c2_l2
            # TODO: is SR_QA_AEROSOL helpful? (Looks suspect for GEF region images)
            # Add SR_QA_AEROSOL to cloud mask
            # bits 6-7 of SR_QA_AEROSOL, are aerosol level where 3 = high, 2=medium, 1=low
            sr_qa_aerosol = image.select('SR_QA_AEROSOL')
            aerosol_prob = sr_qa_aerosol.rightShift(6).bitwiseAnd(3)
            aerosol_mask = aerosol_prob.gt(2).rename('AEROSOL_MASK')
            cloud_mask = cloud_mask.Or(aerosol_mask)

        valid_mask = ((cloud_mask.Or(shadow_mask)).Not()).And(fill_mask).rename('VALID_MASK')
        masks.update(cloud_mask=cloud_mask, shadow_mask=shadow_mask, fill_mask=fill_mask, valid_mask=valid_mask)
        return masks

    # change this to process image, and allow passing an ee.Image or an ID
    def get_image(self, image_id, apply_mask=False, add_aux_bands=False, scale_refl=False):
        """
        Retrieve an ee.Image object, optionally adding metadata and auxillary bands

        Parameters
        ----------
        image_id : str
             Earth engine image ID e.g. 'LANDSAT/LC08/C02/T1_L2/LC08_182037_20190118 2019-01-18'
        apply_mask : bool, optional
                     Apply any validity mask to the image by setting nodata (default: False)
        add_aux_bands: bool, optional
                       Add auxiliary bands (cloud, shadow, fill & validity masks, and quality score) (default: False)
        scale_refl : bool, optional
                     Scale reflectance values from 0-10000 if they are not in that range already (default: True)

        Returns
        -------
        : ee.Image
          The processed image
        """
        image = ImCollection.get_image(self, image_id, apply_mask=apply_mask, add_aux_bands=add_aux_bands)

        if scale_refl:
            image = LandsatImCollection.scale_to_reflectance(image)

        return self._im_transform(image)


    @staticmethod
    def scale_to_reflectance(image):
        """
        Scale and offset landsat pixels in SR bands to surface reflectance (0-10000)
        Uses hard coded ranges

        Parameters
        ----------
        image : ee.Image
                image to scale and offset

        Returns
        -------
        : ee.Image
        Image with SR bands in range 0-10000
        """
        # retrieve the names of SR bands
        all_bands = image.bandNames()
        init_bands = ee.List([])

        def add_refl_bands(band, refl_bands):
            refl_bands = ee.Algorithms.If(ee.String(band).rindex('SR_B').eq(0), ee.List(refl_bands).add(band),
                                          refl_bands)
            return refl_bands

        sr_bands = ee.List(all_bands.iterate(add_refl_bands, init_bands))
        non_sr_bands = all_bands.removeAll(sr_bands)  # all the other non-SR bands

        # scale to new range
        # low/high values from https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2?hl=en
        low = 0.2/2.75e-05
        high = low + 1/2.75e-05
        calib_image = image.select(sr_bands).unitScale(low=low, high=high).multiply(10000.0)
        calib_image = calib_image.addBands(image.select(non_sr_bands))
        calib_image = calib_image.updateMask(image.mask())  # apply any existing mask to refl image

        for key in ['system:index', 'system:id', 'id']:   # copy id
            calib_image = calib_image.set(key, ee.String(image.get(key)))

        return ee.Image(calib_image.copyProperties(image))

class Landsat8ImCollection(LandsatImCollection):
    def __init__(self):
        LandsatImCollection.__init__(self, collection='landsat8_c2_l2')

class Landsat7ImCollection(LandsatImCollection):
    def __init__(self):
        LandsatImCollection.__init__(self, collection='landsat7_c2_l2')

class Sentinel2ImCollection(ImCollection):

    def __init__(self, collection='sentinel2_toa'):
        """
        Class for Sentinel-2 TOA and SR earth engine image collections

        Parameters
        ----------
        collection : str, optional
                     'sentinel_toa' (top of atmosphere - default) or 'sentinel_sr' (surface reflectance)
        """
        ImCollection.__init__(self, collection=collection)

        self._im_transform = ee.Image.toUint16

    def _get_image_masks(self, image):
        """
        Derive cloud, shadow, fill and validity masks for an image

        Parameters
        ----------
        image : ee.Image
                Derive masks for this image

        Returns
        -------
        masks : dict
                A dictionary with cloud_mask, shadow_mask, fill_mask and valid_mask keys, and corresponding ee.Image
                values
        """

        masks = ImCollection._get_image_masks(self, image)
        qa = image.select('QA60')   # bits 10 and 11 are opaque and cirrus clouds respectively
        cloud_mask = qa.bitwiseAnd((1 << 11) | (1 << 10)).neq(0).rename('CLOUD_MASK')
        valid_mask = cloud_mask.Not().rename('VALID_MASK')
        masks.update(cloud_mask=cloud_mask, valid_mask=valid_mask)
        # masks.update(cloud_mask=cloud_mask, valid_mask=valid_mask)
        return masks

class Sentinel2SrImCollection(Sentinel2ImCollection):
    def __init__(self):
        Sentinel2ImCollection.__init__(self, collection='sentinel2_sr')

class Sentinel2ToaImCollection(Sentinel2ImCollection):
    def __init__(self):
        Sentinel2ImCollection.__init__(self, collection='sentinel2_toa')


class Sentinel2ClImCollection(ImCollection):
    def __init__(self, collection='sentinel2_toa'):
        """
        Class for Sentinel-2 TOA and SR earth engine image collections. Uses cloud-probability for masking
        and quality scoring.

        Parameters
        ----------
        collection : str, optional
                     'sentinel_toa' (top of atmosphere - default) or 'sentinel_sr' (surface reflectance)
        """
        ImCollection.__init__(self, collection=collection)
        self._im_transform = ee.Image.toUint16

        self._cloud_filter = 60  # Maximum image cloud cover percent allowed in image collection
        self._cloud_prob_thresh = 35  # Cloud probability (%); values greater than are considered cloud
        # self._nir_drk_thresh = 0.15# Near-infrared reflectance; values less than are considered potential cloud shadow
        self._cloud_proj_dist = 1  # Maximum distance (km) to search for cloud shadows from cloud edges
        self._buffer = 100  # Distance (m) to dilate the edge of cloud-identified objects


    def _get_image_masks(self, image):
        """
        Derive cloud, shadow, fill and validity masks for an image

        Parameters
        ----------
        image : ee.Image
                Derive masks for this image

        Returns
        -------
        masks : dict
                A dictionary with cloud_mask, shadow_mask, fill_mask and valid_mask keys, and corresponding ee.Image
                values
        """
        masks = ImCollection._get_image_masks(self, image)
        # qa = image.select('QA60')   # bits 10 and 11 are opaque and cirrus clouds respectively
        # cloud_mask = qa.bitwiseAnd((1 << 11) | (1 << 10)).neq(0).rename('VALID_MASK')

        # convert cloud probability in 0-100 quality score
        cloud_prob = ee.Image(image.get('s2cloudless')).select('probability')
        cloud_mask = cloud_prob.gt(self._cloud_prob_thresh).rename('CLOUD_MASK')

        # TODO: dilate valid_mask by _buffer ?
        # TODO: does below work in N hemisphere?
        # See https://en.wikipedia.org/wiki/Solar_azimuth_angle
        shadow_azimuth = ee.Number(-90).add(ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
        min_scale = export.get_projection(image).nominalScale()

        # project the the cloud mask in the direction of shadows
        proj_dist_px = ee.Number(self._cloud_proj_dist * 1000).divide(min_scale)
        proj_cloud_mask = (cloud_mask.directionalDistanceTransform(shadow_azimuth, proj_dist_px).
                           select('distance').mask().rename('PROJ_CLOUD_MASK'))     # mask converts to boolean?
        # .reproject(**{'crs': image.select(0).projection(), 'scale': 100})

        if self.collection_info['ee_collection'] == 'COPERNICUS/S2_SR':  # use SCL to reduce shadow_mask
            # Note: SCL does not classify cloud shadows well, they are often labelled "dark".  Instead of using only
            # cloud shadow areas from this band, we combine it with the projected dark and shadow areas from s2cloudless
            scl = image.select('SCL')
            dark_shadow_mask = (scl.eq(3).Or(scl.eq(2)).
                                focal_min(self._buffer, 'circle', 'meters').
                                focal_max(self._buffer, 'circle', 'meters'))
            shadow_mask = proj_cloud_mask.And(dark_shadow_mask).rename('SHADOW_MASK')
        else:
            shadow_mask = proj_cloud_mask.rename('SHADOW_MASK')  # mask all areas that could be cloud shadow

        # combine cloud and shadow masks
        valid_mask = (cloud_mask.Or(shadow_mask)).Not().rename('VALID_MASK')
        masks.update(cloud_mask=cloud_mask, shadow_mask=shadow_mask, valid_mask=valid_mask)
        return masks


    def get_ee_collection(self):
        """
        Return the unfiltered earth engine image collection (a join of the sentinel-2 reflectance and cloud probability
        collections)

        Returns
        -------
        : ee.ImageCollection
        """

        # adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless

        s2_sr_toa_col = ee.ImageCollection(self.collection_info['ee_collection'])
                         #.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self._cloud_filter))) # TODO: add this back?

        s2_cloudless_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

        # join filtered s2cloudless collection to the SR/TOA collection by the 'system:index' property.
        return (ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(
            primary=s2_sr_toa_col, secondary=s2_cloudless_col,
            condition=ee.Filter.equals(leftField='system:index', rightField='system:index'))))

    def get_image(self, image_id, apply_mask=False, add_aux_bands=False, scale_refl=False):
        """
        Retrieve an ee.Image object, optionally adding metadata and auxillary bands

        Parameters
        ----------
        image_id : str
             Earth engine image ID e.g. 'LANDSAT/LC08/C02/T1_L2/LC08_182037_20190118 2019-01-18'
        apply_mask : bool, optional
                     Apply any validity mask to the image by setting nodata (default: False)
        add_aux_bands: bool, optional
                       Add auxiliary bands (cloud, shadow, fill & validity masks, and quality score) (default: False)
        scale_refl : bool, optional
                     Scale reflectance values from 0-10000 if they are not in that range already (default: True)

        Returns
        -------
        : ee.Image
          The processed image
        """
        index = image_id.split('/')[-1]
        collection = '/'.join(image_id.split('/')[:-1])

        if collection != self.collection_info['ee_collection']:
            raise ValueError(f'{image_id} is not a valid earth engine id for {self.__class__}')

        cloud_prob = ee.Image(f'COPERNICUS/S2_CLOUD_PROBABILITY/{index}')
        image = ee.Image(image_id).set('s2cloudless', cloud_prob)

        masks = self._get_image_masks(image)

        if add_aux_bands:
            score = self.get_image_score(image, masks=masks)
            # cloud_prob = ee.Image(image.get('s2cloudless')).select('probability').rename('CLOUD_PROB')
            image = image.addBands(ee.Image(list(masks.values()) + [cloud_prob, score]))
            # image = image.set('s2cloudless', None)

        if apply_mask:
            image = image.updateMask(masks['valid_mask'])


        return self._im_transform(image)

class Sentinel2SrClImCollection(Sentinel2ClImCollection):
    def __init__(self):
        Sentinel2ClImCollection.__init__(self, collection='sentinel2_sr')

class Sentinel2ToaClImCollection(Sentinel2ClImCollection):
    def __init__(self):
        Sentinel2ClImCollection.__init__(self, collection='sentinel2_toa')

class ModisNbarImCollection(ImCollection):
    def __init__(self):
        """
        Class for the MODIS daily NBAR earth engine image collection
        """
        ImCollection.__init__(self, collection='modis_nbar')
        self._im_transform = ee.Image.toUint16

##

