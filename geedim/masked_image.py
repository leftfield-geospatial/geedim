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

# Functionality for wrapping, cloud/shadow masking and scoring Earth Engine images
import collections
import logging
from typing import Union

import ee
import numpy as np

from geedim.image import BaseImage, split_id

logger = logging.getLogger(__name__)


##


class MaskedImage(BaseImage):
    _default_mask = False
    _default_cloud_dist = 5000
    _supported_collection_ids = []

    # TODO: all the cloud mask params need to passed, but do they belong in __init__?  how will this combine with e.g. masking for search and masking for download
    def __init__(self, ee_image, is_composite=False, **kwargs):
        """
        Class to cloud/shadow mask and quality score Earth engine images from supported collections.

        Parameters
        ----------
        ee_image : ee.Image
            Earth engine image to wrap.
        """
        # construct the cloud/shadow masks and cloudless score
        if not is_composite:
            ee_image = self._add_aux_bands(ee_image, **kwargs)
        BaseImage.__init__(self, ee_image)

    @classmethod
    def _from_id(cls, image_id, mask=_default_mask, cloud_dist=_default_cloud_dist, region=None):
        """ Internal method for creating an image with region statistics. """
        gd_image = cls.from_id(image_id, mask=mask, cloud_dist=cloud_dist)
        if region is not None:
            gd_image._ee_image = cls.set_region_stats(gd_image, region)
        return gd_image

    @classmethod
    def ee_collection(cls, ee_coll_name):
        """
        Returns the ee.ImageCollection corresponding to this image.

        Returns
        -------
        ee.ImageCollection
        """
        # TODO: lose the ee_coll_name parameter being passed?  or this method entirely?
        if not ee_coll_name in cls._supported_collection_ids:
            raise ValueError(f"Unsupported collection: {ee_coll_name}.  {cls.__name__} supports images from "
                             "{cls._supported_collection_ids}")
        return ee.ImageCollection(ee_coll_name)

    @staticmethod
    def _add_aux_bands(ee_image, **kwargs):
        fill_mask = ee_image.mask().reduce(ee.Reducer.allNonZero()).rename('FILL_MASK')
        return ee_image.addBands(fill_mask, overwrite=True)

    def set_region_stats(self, region):
        """
        Set VALID_PORTION and AVG_SCORE statistics for a specified region in an image object.

        Parameters
        ----------
        region : dict, geojson, ee.Geometry
                 Region inside of which to find statistics

        Returns
        -------
         : ee.Image
            EE image with VALID_PORTION and AVG_SCORE properties set.
        """
        proj = get_projection(self._ee_image, min_scale=False)
        stats_image = ee.Image([self._ee_image.select('FILL_MASK').rename('FILL_PORTION'),
                                ee.Image(1).rename('REGION_SUM')])

        # sum stats_image bands over region
        sums_dict = stats_image.reduceRegion(reducer="sum", geometry=region, crs=proj.crs(), scale=proj.nominalScale(),
                                             bestEffort=True, maxPixels=1e6)

        # find average VALID_MASK and SCORE over region (not the same as image if image does not cover region)
        def region_percentage(key, value):
            return ee.Number(value).multiply(100).divide(ee.Number(sums_dict.get("REGION_SUM")))

        means = sums_dict.select(['FILL_PORTION']).map(region_percentage)
        self._ee_image = self._ee_image.set(means)
        self.ee_image = self._ee_image.set('CLOUDLESS_PORTION', means.get('FILL_PORTION'))
        return self._ee_image

    def mask_clouds(self):
        logger.warning(f'Cloud/shadow masking is not supported for this image')
        self._ee_image = self._ee_image.updateMask(self._ee_image.select('FILL_MASK'))


class SrMaskedImage(MaskedImage):
    def set_region_stats(self, region):
        """
        Set VALID_PORTION and AVG_SCORE statistics for a specified region in an image object.

        Parameters
        ----------
        region : dict, geojson, ee.Geometry
                 Region inside of which to find statistics

        Returns
        -------
         : ee.Image
            EE image with VALID_PORTION and AVG_SCORE properties set.
        """
        proj = get_projection(self._ee_image, min_scale=False)
        stats_image = ee.Image([self._ee_image.select(['FILL_MASK', 'CLOUDLESS_MASK']),
                                ee.Image(1).rename('REGION_SUM')])

        # sum stats_image bands over region
        sums = (
            stats_image.reduceRegion(reducer="sum", geometry=region, crs=proj.crs(), scale=proj.nominalScale(),
                                     bestEffort=True, maxPixels=1e6).
                rename(['FILL_MASK', 'CLOUDLESS_MASK'], ['FILL_PORTION', 'CLOUDLESS_PORTION'])
        )

        # find average VALID_MASK and SCORE over region (not the same as image if image does not cover region)
        def region_percentage(key, value):
            return ee.Number(value).multiply(100).divide(ee.Number(sums.get("REGION_SUM")))

        means = sums.select(['FILL_PORTION', 'CLOUDLESS_PORTION']).map(region_percentage)
        self.ee_image = self._ee_image.set(means)
        return self._ee_image

    def _cloud_dist(self, max_cloud_dist=5000):
        """
        Get the cloud/shadow distance quality score for this image.

        Returns
        -------
        ee.Image
            The cloud/shadow distance score (m) as a single band image.
        """
        ee_image = self._ee_image
        radius = 1.5  # morphological pixel radius
        proj = get_projection(ee_image, min_scale=False)  # use maximum scale projection to save processing time

        # combine cloud and shadow masks and morphologically open to remove small isolated patches
        cloud_shadow_mask = ee_image.select('CLOUD_MASK').Or(ee_image.select['SHADOW_MASK'])    # TODO just use cloudless_mask?
        cloud_shadow_mask = cloud_shadow_mask.focal_min(radius=radius).focal_max(radius=radius) # TODO necessary?
        cloud_pix = ee.Number(max_cloud_dist).divide(proj.nominalScale()).round()  # cloud_dist in pixels

        # distance to nearest cloud/shadow (m)
        cloud_dist = (
            cloud_shadow_mask.fastDistanceTransform(neighborhood=cloud_pix, units="pixels", metric="squared_euclidean")
                .sqrt()
                .multiply(proj.nominalScale())
                .rename("CLOUD_DIST")
                .reproject(crs=proj.crs(), scale=proj.nominalScale())  # reproject to force calculation at correct scale
        )

        # clip score to cloud_dist and set to 0 in unfilled areas
        cloud_dist = (cloud_dist.
                 where(cloud_dist.gt(ee.Image(self._cloud_dist)), self._cloud_dist).
                 where(ee_image.select('FILL_MASK').Not(), 0))    # TODO: I don't think this where is necessary when we don't unmask to start with
        return cloud_dist


class LandsatImage(SrMaskedImage):
    """ Base class for cloud/shadow masking and quality scoring landsat images """
    _supported_collection_ids = ['LANDSAT/LT04/C02/T1_L2', 'LANDSAT/LT05/C02/T1_L2', 'LANDSAT/LE07/C02/T1_L2',
                                 'LANDSAT/LC08/C02/T1_L2', 'LANDSAT/LC09/C02/T1_L2']

    @staticmethod
    def _add_aux_bands(ee_image, mask_shadows=True, mask_cirrus=True, **kwargs):
        # TODO: add warning for unsupported args?
        # get cloud, shadow and fill masks from QA_PIXEL
        qa_pixel = ee_image.select("QA_PIXEL")

        # incorporate the existing mask (for zero SR pixels) into the shadow mask
        ee_mask = ee_image.select('SR_B.*').mask().reduce(ee.Reducer.allNonZero())
        fill_mask = qa_pixel.bitwiseAnd(1).eq(0).And(ee_mask).rename("FILL_MASK")
        shadow_mask = qa_pixel.bitwiseAnd(0b10000).neq(0).rename("SHADOW_MASK")
        if mask_cirrus:
            cloud_mask = qa_pixel.bitwiseAnd(0b1100).neq(0).rename("CLOUD_MASK")    # TODO: is this bit always zero for landsat 4-7
        else:
            cloud_mask = qa_pixel.bitwiseAnd(0b1000).neq(0).rename("CLOUD_MASK")

        # combine cloud, shadow and fill masks into cloudless mask
        cloudless_mask = (cloud_mask.Or(shadow_mask)).Not() if mask_shadows else cloud_mask.Not()
        cloudless_mask = cloudless_mask.And(fill_mask).rename("CLOUDLESS_MASK")
        return ee_image.addBands([fill_mask, cloud_mask, shadow_mask, cloudless_mask])

    def _cloud_dist(self, max_cloud_dist=5000):
        return self._ee_image.select('ST_CDIST')


class Sentinel2ClImage(SrMaskedImage):
    """
    Base class for cloud/shadow masking and quality scoring sentinel2_sr and sentinel2_toa images.

    (Uses cloud probability to improve cloud/shadow masking).
    """
    _supported_collection_ids = []

    # TODO: provide CLI access to these kwargs, and document them here
    def __init__(self, ee_image, **kwargs):
        """
        Class to cloud/shadow mask and quality score GEE Sentinel-2 images.

        Parameters
        ----------
        ee_image : ee.Image
            Earth engine Sentinel-2 image to wrap.  This image must have a `CLOUD_PROB` band containing the
            corresponding image from the `COPERNICUS/S2_CLOUD_PROBABILITY` collection.
        """
        SrMaskedImage.__init__(self, ee_image, **kwargs)

    @classmethod
    def from_id(cls, image_id, **kwargs):
        # check image_id
        ee_coll_name = split_id(image_id)[0]
        if ee_coll_name not in cls._supported_collection_ids:
            raise ValueError(f"Unsupported collection: {ee_coll_name}.  "
                             f"{cls.__name__} only supports images from {cls._supported_collection_ids}")

        ee_image = ee.Image(image_id)

        # get cloud probability for ee_image and add as a band
        cloud_prob = ee.Image(f"COPERNICUS/S2_CLOUD_PROBABILITY/{split_id(image_id)[1]}").rename('CLOUD_PROB')
        ee_image = ee_image.addBands(cloud_prob, overwrite=True)
        gd_image = cls(ee_image, **kwargs)
        gd_image._id = image_id
        return gd_image

    @staticmethod
    def _add_aux_bands(ee_image, s2_toa=False, method='cloud_prob', mask_cirrus=True, mask_shadows=True, prob=60,
                       dark=0.15, cloud_dist=1000, buffer=250, cdi=None):
        """
        Derive cloud, shadow and validity masks for an image, using the additional cloud probability band.

        Adapted from https://github.com/r-earthengine/ee_extra, under Apache 2.0 license

        Parameters
        ----------
        ee_image : ee.Image
                   Derive masks for this image
        s2_toa : bool, optional
            S2 TOA/SR collection.  Set to True if this image is from COPERNICUS/S2, or False if it is from
            COPERNICUS/S2_SR.
        method : str, optional
            Method used to mask clouds.
            Available options:
                - 'cloud_prob' : Use cloud probability.
                - 'qa' : Use Quality Assessment band.
        prob : float, optional
            Cloud probability threshold. Valid just for method = 'cloud_prob'.
        dark : float, optional
            NIR threshold [0-1]. NIR values below this threshold are potential cloud shadows.
        cloud_dist : int, optional
            Maximum distance in meters (m) to look for cloud shadows from cloud edges.
        buffer : int, optional
            Distance in meters (m) to dilate cloud and cloud shadows objects.
        cdi : float, optional
            Cloud Displacement Index threshold. Values below this threshold are considered potential clouds.
            A cdi = None means that the index is not used.
            For more info see 'Frantz, D., HaS, E., Uhl, A., Stoffels, J., Hill, J. 2018. Improvement of the Fmask
            algorithm for Sentinel-2 images: Separating clouds from bright surfaces based on parallax effects. Remote
            Sensing of Environment 2015: 471-481'.

        Returns
        -------
        dict
            A dictionary of ee.Image objects for each of the fill, cloud, shadow and validity masks.
        """
        proj_scale = 60
        # maskCirrus : Whether to mask cirrus clouds. Valid just for method = 'qa'. This parameter is ignored for Landsat products.
        # maskShadows : Whether to mask cloud shadows. For more info see 'Braaten, J. 2020. Sentinel-2 Cloud Masking with s2cloudless. Google Earth Engine, Community Tutorials'.
        # scaledImage : Whether the pixel values are scaled to the range [0,1] (reflectance values). This parameter is ignored for Landsat products.

        if method == 'cloud_prob':
            cloud_mask = ee_image.select('CLOUD_PROB').gte(prob).rename('CLOUD_MASK')
        else:
            qa = ee_image.select('QA60')
            cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)
            if mask_cirrus:
                cloud_mask = cloud_mask.And(qa.bitwiseAnd(1 << 11).eq(0))

        if cdi is not None:
            if s2_toa:
                s2_toa_image = ee_image
            else:
                idx = ee_image.get("system:index")
                s2_toa_image = (ee.ImageCollection("COPERNICUS/S2").
                                filter(ee.Filter.eq("system:index", idx)).
                                first())
            cdi_image = ee.Algorithms.Sentinel2.CDI(s2_toa_image)
            cloud_mask_cdi = cdi_image.lt(cdi).rename("CLOUD_MASK_CDI")

        if mask_shadows:
            dark_mask = ee_image.select("B8").lt(dark * 1e4)
            if not s2_toa:
                dark_mask = ee_image.select("SCL").neq(6).And(dark_mask)

            shadow_azimuth = ee.Number(90).subtract(ee.Number(ee_image.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
            proj_cloud_mask = (cloud_mask.directionalDistanceTransform(shadow_azimuth, int(cloud_dist / proj_scale))
                               .reproject(crs=ee_image.select(0).projection(), scale=60)
                               .select('distance')
                               .mask())
            shadow_mask = proj_cloud_mask.And(dark_mask).rename("SHADOW_MASK")

        cloud_shadow_mask = cloud_mask

        if cdi:
            cloud_shadow_mask = cloud_shadow_mask.And(cloud_mask_cdi)
        if mask_shadows:
            cloud_shadow_mask = cloud_shadow_mask.Or(shadow_mask)    # TODO: or?
        cloud_shadow_mask = cloud_shadow_mask.focal_min(20, units="meters").focal_max(buffer * 2 / 10, units="meters")

        fill_mask = ee_image.select('B.*').mask().reduce(ee.Reducer.allNonZero()).rename('FILL_MASK')
        fill_mask = fill_mask.clip(ee_image.geometry())

        cloudless_mask = (cloud_shadow_mask.Not()).And(fill_mask).rename('CLOUDLESS_MASK')
        return ee_image.addBands([fill_mask, cloud_mask, shadow_mask, cloudless_mask])


    @classmethod
    def ee_collection(cls, ee_coll_name):
        """
        Returns an augmented ee.ImageCollection with cloud probability bands added to multi-spectral images.

        Returns
        -------
        ee.ImageCollection
        """
        if not ee_coll_name in cls._supported_collection_ids:
            raise ValueError(f"Unsupported collection: {ee_coll_name}.  {cls.__name__} supports images from "
                             f"{cls._supported_collection_ids}")
        s2_sr_toa_col = ee.ImageCollection(ee_coll_name)
        s2_cloudless_col = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")

        # create a collection of index-matched images from the SR/TOA and cloud probability collections
        filt = ee.Filter.equals(leftField="system:index", rightField="system:index")
        inner_join = ee.ImageCollection(ee.Join.inner().apply(s2_sr_toa_col, s2_cloudless_col, filt))

        # re-configure the collection so that cloud probability is added as a band to the SR/TOA image
        def map(feature):
            """ Server-side function to concatenate images """
            return ee.Image.cat(feature.get("primary"), ee.Image(feature.get("secondary")).rename('CLOUD_PROB'))

        return inner_join.map(map)


class Sentinel2SrClImage(Sentinel2ClImage):
    """
    Base class for cloud/shadow masking and quality scoring 'COPERNICUS/S2_SR' images.

    (Uses cloud probability to improve cloud/shadow masking).
    """
    _supported_collection_ids = ['COPERNICUS/S2_SR']
    @staticmethod
    def _add_aux_bands(ee_image, s2_toa=False, **kwargs):
        return Sentinel2ClImage._add_aux_bands(ee_image, s2_toa=False, **kwargs)

class Sentinel2ToaClImage(Sentinel2ClImage):
    """
    Base class for cloud/shadow masking and quality scoring 'COPERNICUS/S2' images.

    (Uses cloud probability to improve cloud/shadow masking).
    """
    _supported_collection_ids = ['COPERNICUS/S2']

    @staticmethod
    def _add_aux_bands(ee_image, s2_toa=False, **kwargs):
        return Sentinel2ClImage._add_aux_bands(ee_image, s2_toa=True, **kwargs)


class ModisNbarImage(MaskedImage):
    """
    Class for wrapping modis_nbar images.

    (These images are already cloud/shadow free composites, so no further processing is done on them, and
    constant cloud, shadow etc masks are used).
    """
    _supported_collection_ids = ['MODIS/006/MCD43A4']

    @staticmethod
    def _im_transform(ee_image):
        return ee.Image.toUint16(ee_image)


def class_from_id(image_id: str) -> Union[BaseImage, MaskedImage]:
    """Return the *Image class that corresponds to the provided EE image/collection ID."""

    masked_image_dict = {
        'LANDSAT/LT04/C02/T1_L2': LandsatImage,
        'LANDSAT/LT05/C02/T1_L2': LandsatImage,
        'LANDSAT/LE07/C02/T1_L2': LandsatImage,
        'LANDSAT/LC08/C02/T1_L2': LandsatImage,
        'LANDSAT/LC09/C02/T1_L2': LandsatImage,
        'COPERNICUS/S2': Sentinel2ToaClImage,
        'COPERNICUS/S2_SR': Sentinel2SrClImage,
        'MODIS/006/MCD43A4': ModisNbarImage
    }
    ee_coll_name, _ = split_id(image_id)
    if image_id in masked_image_dict:
        return masked_image_dict[image_id]
    elif ee_coll_name in masked_image_dict:
        return masked_image_dict[ee_coll_name]
    else:
        return BaseImage


def image_from_id(image_id: str, **kwargs) -> BaseImage:
    """Return a *Image instance for a given EE image ID."""
    return class_from_id(image_id).from_id(image_id, **kwargs)


def get_projection(image, min_scale=True):
    """
    Get the min/max scale projection of image bands.  Server side - no calls to getInfo().
    Adapted from from https://github.com/gee-community/gee_tools, MIT license.

    Parameters
    ----------
    image : ee.Image, geedim.image.BaseImage
            The image whose min/max projection to retrieve.
    min: bool, optional
         Retrieve the projection corresponding to the band with the minimum (True) or maximum (False) scale.
         (default: True)

    Returns
    -------
    ee.Projection
        The requested projection.
    """
    if isinstance(image, BaseImage):
        image = image.ee_image

    bands = image.bandNames()

    transform = np.array([1, 0, 0, 0, 1, 0])
    if min_scale:
        compare = ee.Number.lte
        init_proj = ee.Projection('EPSG:4326', list(1e100 * transform))
    else:
        compare = ee.Number.gte
        init_proj = ee.Projection('EPSG:4326', list(1e-100 * transform))

    def compare_scale(name, prev_proj):
        """ Server side comparison of band scales"""
        prev_proj = ee.Projection(prev_proj)
        prev_scale = prev_proj.nominalScale()

        curr_proj = image.select([name]).projection()
        curr_scale = ee.Number(curr_proj.nominalScale())

        # compare scales, excluding WGS84 bands (constant or composite bands)
        condition = (
            compare(curr_scale, prev_scale).And(curr_proj.crs().compareTo(ee.String("EPSG:4326"))).neq(ee.Number(0))
        )
        comp_proj = ee.Algorithms.If(condition, curr_proj, prev_proj)
        return ee.Projection(comp_proj)

    return ee.Projection(bands.iterate(compare_scale, init_proj))

def _get_projection(image, min_scale=True):
    """
    Get the min/max scale projection of image bands.  Server side - no calls to getInfo().

    Parameters
    ----------
    image : ee.Image, geedim.image.BaseImage
            The image whose min/max projection to retrieve.
    min: bool, optional
         Retrieve the projection corresponding to the band with the minimum (True) or maximum (False) scale.
         (default: True)

    Returns
    -------
    ee.Projection
        The requested projection.
    """
    if isinstance(image, BaseImage):
        image = image.ee_image

    bands = image.bandNames()
    def band_crs_scale(band_name):
        band = image.select([band_name])
        projection = band.projection()
        crs = projection.crs()
        scale = projection.nominalScale()
        return ee.Feature(None, dict(scale=scale, crs=crs, projection=projection))

    crs_scale_fc = ee.FeatureCollection(bands.map(band_crs_scale))
    # crs_scale_fc.getInfo()
    crs_scale_fc = crs_scale_fc.filter(ee.Filter.neq('crs', 'EPSG:4326'))

    def gather_scale_list(feature, _scale_list):
        return ee.List(_scale_list).add(ee.Number(feature.get('scale')))

    scale_array = ee.Array(crs_scale_fc.iterate(gather_scale_list, ee.List([])))
    # scale_list.getInfo()
    feat_idx = scale_array.multiply(-1).argmax() if min_scale else scale_array.argmax()
    feature = ee.Feature(crs_scale_fc.toList(bands.size()).get(feat_idx.get(0)))
    return ee.Projection(feature.get('projection'))
