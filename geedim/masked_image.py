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

    def __init__(self, ee_image, mask=_default_mask, cloud_dist=_default_cloud_dist):
        """
        Class to cloud/shadow mask and quality score Earth engine images from supported collections.

        Parameters
        ----------
        ee_image : ee.Image
            Earth engine image to wrap.
        mask : bool, optional
            Apply a validity (cloud & shadow) mask to the image (default: False).
        cloud_dist : int, optional
            The radius (m) to search for cloud/shadow for quality scoring (default: 5000).
        """
        # prevent instantiation of base class(es)
        if len(self._supported_collection_ids) == 0:
            raise NotImplementedError("This base class cannot be instantiated, use a sub-class")

        # construct the cloud/shadow masks and cloudless score
        self._cloud_dist = cloud_dist
        ee_image = ee_image.unmask()
        ee_image = self._process_image(ee_image, mask=mask)
        BaseImage.__init__(self, ee_image)

    @classmethod
    def from_masked_image(cls, ee_image):
        """Wrap an already masked and scored ee.Image (e.g a composite) in MaskedImage."""
        gd_image = cls(ee.Image(0))
        gd_image._ee_image = ee_image
        return gd_image

    @classmethod
    def _from_id(cls, image_id, mask=_default_mask, cloud_dist=_default_cloud_dist, region=None):
        """ Internal method for creating an image with region statistics. """
        gd_image = cls.from_id(image_id, mask=mask, cloud_dist=cloud_dist)
        if region is not None:
            gd_image._ee_image = cls.set_region_stats(gd_image, region)
        return gd_image

    @staticmethod
    def _im_transform(ee_image):
        """ Optional data type conversion to run after masking and scoring. """
        return ee_image

    @property
    def ee_image(self):
        """ ee.Image: The wrapped image. """
        return self._ee_image

    @classmethod
    def ee_collection(cls, ee_coll_name):
        """
        Returns the ee.ImageCollection corresponding to this image.

        Returns
        -------
        ee.ImageCollection
        """
        if not ee_coll_name in cls._supported_collection_ids:
            raise ValueError(f"Unsupported collection: {ee_coll_name}.  {cls.__name__} supports images from "
                             "{cls._supported_collection_ids}")
        return ee.ImageCollection(ee_coll_name)

    @classmethod
    def set_region_stats(cls, image_obj, region, mask=_default_mask):
        """
        Set VALID_PORTION and AVG_SCORE statistics for a specified region in an image object.

        Parameters
        ----------
        image_obj: ee.Image, geedim.image.BaseImage
                    Image object whose region statistics to find and set
        region : dict, geojson, ee.Geometry
                 Region inside of which to find statistics
        mask : bool, optional
               Apply the validity (cloud & shadow) mask to the image (default: False)

        Returns
        -------
         : ee.Image
            EE image with VALID_PORTION and AVG_SCORE properties set.
        """
        if isinstance(image_obj, ee.Image):
            gd_image = cls(image_obj, mask=mask)
        elif isinstance(image_obj, cls):
            gd_image = image_obj
        else:
            raise TypeError(f'Unexpected image_obj type: {type(image_obj)}')

        stats_image = ee.Image([gd_image.ee_image.select(['FILL_MASK', 'VALID_MASK']), ee.Image(1).rename('REGION')])
        proj = get_projection(gd_image.ee_image, min_scale=False)

        # sum stats_image bands over region
        sums = (
            stats_image
                .reduceRegion(reducer="sum", geometry=region, crs=proj.crs(), scale=proj.nominalScale(),
                              bestEffort=True, maxPixels=1e6)
                .rename(['FILL_MASK', 'VALID_MASK', 'REGION'], ['FILL_PORTION', 'VALID_PORTION', 'REGION_SUM'])
        )

        # find average VALID_MASK and SCORE over region (not the same as image if image does not cover region)
        def region_percentage(key, value):
            return ee.Number(value).multiply(100).divide(ee.Number(sums.get("REGION_SUM")))

        means = sums.select(['FILL_PORTION', 'VALID_PORTION']).map(region_percentage)
        return gd_image.ee_image.set(means)

    def _get_image_masks(self, ee_image):
        """
        Derive cloud, shadow, fill and validity masks for an image.

        Parameters
        ----------
        ee_image : ee.Image
                   Derive masks for this image.

        Returns
        -------
        dict
            A dictionary of ee.Image objects for each of the fill, cloud, shadow and validity masks.
        """
        # create constant masks for this base class
        masks = dict(
            cloud_mask=ee.Image(0).rename("CLOUD_MASK"),
            shadow_mask=ee.Image(0).rename("SHADOW_MASK"),
            fill_mask=ee.Image(1).rename("FILL_MASK"),
            valid_mask=ee.Image(1).rename("VALID_MASK"),
        )

        return masks

    def _get_image_score(self, ee_image, masks=None):
        """
        Get the cloud/shadow distance quality score for this image.

        Parameters
        ----------
        ee_image : ee.Image
                   Find the score for this image.
        masks : dict, optional
                Existing masks as returned by _get_image_masks(...) (default: calculate the masks).
        Returns
        -------
        ee.Image
            The cloud/shadow distance score (m) as a single band image.
        """
        radius = 1.5  # morphological pixel radius
        proj = get_projection(ee_image, min_scale=False)  # use maximum scale projection to save processing time
        if masks is None:
            masks = self._get_image_masks(ee_image)

        # combine cloud and shadow masks and morphologically open to remove small isolated patches
        cloud_shadow_mask = masks["cloud_mask"].Or(masks["shadow_mask"])
        cloud_shadow_mask = cloud_shadow_mask.focal_min(radius=radius).focal_max(radius=radius)
        cloud_pix = ee.Number(self._cloud_dist).divide(proj.nominalScale()).round()  # cloud_dist in pixels

        # distance to nearest cloud/shadow (m)
        score = (
            cloud_shadow_mask.fastDistanceTransform(neighborhood=cloud_pix, units="pixels", metric="squared_euclidean")
                .sqrt()
                .multiply(proj.nominalScale())
                .rename("SCORE")
                .reproject(crs=proj.crs(), scale=proj.nominalScale())  # reproject to force calculation at correct scale
        )

        # clip score to cloud_dist and set to 0 in unfilled areas
        score = (score.
                 where(score.gt(ee.Image(self._cloud_dist)), self._cloud_dist).
                 where(masks["fill_mask"].Not(), 0))
        return score

    def _process_image(self, ee_image, mask=_default_mask, masks=None, score=None):
        """
        Create, and add, mask and score bands to a an Earth Engine image.

        Parameters
        ----------
        ee_image : ee.Image
                   Earth engine image to add bands to.
        mask : bool, optional
               Apply any validity mask to the image by setting nodata (default: False).

        Returns
        -------
        ee.Image
            The processed image with added mask and score bands.
        """
        if masks is None:
            masks = self._get_image_masks(ee_image)
        if score is None:
            score = self._get_image_score(ee_image, masks=masks)

        ee_image = ee_image.addBands(ee.Image(list(masks.values())), overwrite=True)
        ee_image = ee_image.addBands(score, overwrite=True)

        # TODO: can we omit masking here and leave it to composite and download?  Maybe have a mask method that
        #  for MaskedImage applies VALID_MASK, and for BaseImage just calls mask()?
        if mask:  # apply the validity mask to all bands (i.e. set those areas to nodata)
            ee_image = ee_image.mask(masks["valid_mask"])

        return self._im_transform(ee_image)


class LandsatImage(MaskedImage):
    """ Base class for cloud/shadow masking and quality scoring landsat images """
    _supported_collection_ids = ['LANDSAT/LT04/C02/T1_L2', 'LANDSAT/LT05/C02/T1_L2', 'LANDSAT/LE07/C02/T1_L2',
                                 'LANDSAT/LC08/C02/T1_L2', 'LANDSAT/LC09/C02/T1_L2']

    # TODO: remove these dtype conversions here and leave it up to download.
    @staticmethod
    def _im_transform(ee_image):
        return ee.Image.toUint16(ee_image)

    @staticmethod
    def _split_band_names(ee_image):
        """Get SR and non-SR band names"""
        all_bands = ee_image.bandNames()
        init_bands = ee.List([])

        def add_refl_bands(band, refl_bands):
            """ Server side function to add SR band names to a list """
            refl_bands = ee.Algorithms.If(
                ee.String(band).rindex("SR_B").eq(0), ee.List(refl_bands).add(band), refl_bands
            )
            return refl_bands

        sr_bands = ee.List(all_bands.iterate(add_refl_bands, init_bands))
        non_sr_bands = all_bands.removeAll(sr_bands)
        split_band_names = collections.namedtuple("SplitBandNames", ["sr", "non_sr"])
        return split_band_names(sr_bands, non_sr_bands)

    def _get_image_masks(self, ee_image):
        # get cloud, shadow and fill masks from QA_PIXEL
        qa_pixel = ee_image.select("QA_PIXEL")

        # incorporate the existing mask (for zero SR pixels) into the shadow mask
        sr_bands, non_sr_bands = LandsatImage._split_band_names(ee_image)
        ee_mask = ee_image.select(sr_bands).reduce(ee.Reducer.allNonZero())
        fill_mask = qa_pixel.bitwiseAnd(1).eq(0).And(ee_mask).rename("FILL_MASK")

        # TODO: include Landsat 8 SR_QA_AEROSOL in cloud mask? it has lots of false positives which skews valid portion
        cloud_mask = qa_pixel.bitwiseAnd((1 << 1) | (1 << 2) | (1 << 3)).neq(0).rename("CLOUD_MASK")
        shadow_mask = qa_pixel.bitwiseAnd(1 << 4).neq(0)
        shadow_mask = shadow_mask.rename("SHADOW_MASK")

        # combine cloud, shadow and fill masks into validity mask
        valid_mask = ((cloud_mask.Or(shadow_mask)).Not()).And(fill_mask).rename("VALID_MASK")

        return dict(cloud_mask=cloud_mask, shadow_mask=shadow_mask, fill_mask=fill_mask, valid_mask=valid_mask)


class Sentinel2Image(MaskedImage):  # pragma: no cover
    """
    Base class for cloud masking and quality scoring sentinel2_sr and sentinel2_toa images

    (Does not use cloud probability).
    """
    _supported_collection_ids = ['COPERNICUS/S2', 'COPERNICUS/S2_SR']

    @staticmethod
    def _im_transform(ee_image):
        return ee.Image.toUint16(ee_image)

    def _get_image_masks(self, ee_image):
        masks = MaskedImage._get_image_masks(self, ee_image)  # get constant masks

        # derive cloud mask (only)
        qa = ee_image.select("QA60")  # bits 10 and 11 are opaque and cirrus clouds respectively
        cloud_mask = qa.bitwiseAnd((1 << 11) | (1 << 10)).neq(0).rename("CLOUD_MASK")

        # update validity and cloud masks
        valid_mask = cloud_mask.Not().rename("VALID_MASK")
        masks.update(cloud_mask=cloud_mask, valid_mask=valid_mask)
        return masks


class Sentinel2ClImage(MaskedImage):
    """
    Base class for cloud/shadow masking and quality scoring sentinel2_sr and sentinel2_toa images.

    (Uses cloud probability to improve cloud/shadow masking).
    """
    _supported_collection_ids = ['COPERNICUS/S2', 'COPERNICUS/S2_SR']

    def __init__(self, ee_image, mask=MaskedImage._default_mask, cloud_dist=MaskedImage._default_cloud_dist):
        """
        Class to cloud/shadow mask and quality score GEE Sentinel-2 images.

        Parameters
        ----------
        ee_image : ee.Image
            Earth engine Sentinel-2 image to wrap.  This image must have a `CLOUD_PROB` band containing the
            corresponding image from the `COPERNICUS/S2_CLOUD_PROBABILITY` collection.
        mask : bool, optional
            Apply a validity (cloud & shadow) mask to the image (default: False).
        cloud_dist : int, optional
            The radius (m) to search for cloud/shadow for quality scoring (default: 5000).
        """

        # TODO: provide CLI access to these attributes
        # set attributes before their use in __init__ below
        self._cloud_prob_thresh = 35  # Cloud probability (%); values greater than are considered cloud
        self._cloud_proj_dist = 1  # Maximum distance (km) to search for cloud shadows from cloud edges
        self._buffer = 100  # Distance (m) to dilate the edge of cloud-identified objects
        MaskedImage.__init__(self, ee_image, mask=mask, cloud_dist=cloud_dist)

    @staticmethod
    def _im_transform(ee_image):
        return ee.Image.toUint16(ee_image)

    @classmethod
    def from_id(cls, image_id, mask=MaskedImage._default_mask, cloud_dist=MaskedImage._default_cloud_dist):
        # check image_id
        ee_coll_name = split_id(image_id)[0]
        if ee_coll_name not in cls._supported_collection_ids:
            raise ValueError(f"Unsupported collection: {ee_coll_name}.  "
                             f"{cls.__name__} only supports images from {cls._supported_collection_ids}")

        ee_image = ee.Image(image_id)

        # get cloud probability for ee_image and add as a band
        cloud_prob = ee.Image(f"COPERNICUS/S2_CLOUD_PROBABILITY/{split_id(image_id)[1]}").rename('CLOUD_PROB')
        ee_image = ee_image.addBands(cloud_prob, overwrite=True)
        gd_image = cls(ee_image, mask=mask, cloud_dist=cloud_dist)
        gd_image._id = image_id
        return gd_image

    def _get_image_masks(self, ee_image):
        """
        Derive cloud, shadow and validity masks for an image, using the additional cloud probability band.

        Adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless

        Parameters
        ----------
        ee_image : ee.Image
                   Derive masks for this image

        Returns
        -------
        dict
            A dictionary of ee.Image objects for each of the fill, cloud, shadow and validity masks.
        """

        masks = MaskedImage._get_image_masks(self, ee_image)  # get constant masks from base class
        proj = get_projection(ee_image, min_scale=False)  # use maximum scale projection to save processing time

        # threshold the added cloud probability to get the initial cloud mask
        cloud_prob = ee_image.select("CLOUD_PROB")
        cloud_mask = cloud_prob.gt(self._cloud_prob_thresh).rename("CLOUD_MASK")

        # TODO: dilate valid_mask by _buffer ?
        # See https://en.wikipedia.org/wiki/Solar_azimuth_angle
        # get solar azimuth
        shadow_azimuth = ee.Number(-90).add(ee.Number(ee_image.get("MEAN_SOLAR_AZIMUTH_ANGLE")))

        # remove small clouds
        cloud_mask_open = (
            cloud_mask.focal_min(self._buffer, "circle", "meters").focal_max(self._buffer, "circle", "meters")
        )

        # project the opened cloud mask in the direction of sun's rays (i.e. shadows)
        proj_dist_pix = ee.Number(self._cloud_proj_dist * 1000).divide(proj.nominalScale()).round()
        proj_cloud_mask = (
            cloud_mask_open.directionalDistanceTransform(shadow_azimuth, proj_dist_pix)
                .select("distance")
                .mask()
                .rename("PROJ_CLOUD_MASK")
                .reproject(crs=proj.crs(), scale=proj.nominalScale())  # force calculation at correct scale
        )

        # if this is an S2_SR image, use SCL to find the shadow_mask, else just use proj_cloud_mask.
        # note that while GEE recommends against If statements, but the below does not seem to impact speed.
        ee_coll_name = ee.String(ee_image.get('system:id')).split('/').slice(0, -1).join('/')
        shadow_mask = ee.Image(
            ee.Algorithms.If(
                ee_coll_name.equals('COPERNICUS/S2_SR'),
                proj_cloud_mask.And(
                    ee_image.select("SCL").eq(3).
                        Or(ee_image.select("SCL").eq(2)).
                        focal_min(self._buffer, "circle", "meters").
                        focal_max(2 * self._buffer, "circle", "meters")
                ).rename("SHADOW_MASK"),
                proj_cloud_mask.rename("SHADOW_MASK")
            )).rename("SHADOW_MASK")
        # shadow_mask = proj_cloud_mask.And(
        #             ee_image.select("SCL").eq(3).
        #                 Or(ee_image.select("SCL").eq(2)).
        #                 focal_min(self._buffer, "circle", "meters").
        #                 focal_max(2 * self._buffer, "circle", "meters")
        #         ).rename("SHADOW_MASK")

        # incorporate the existing mask (for zero SR pixels) into the shadow mask
        fill_mask = ee_image.mask().reduce(ee.Reducer.allNonZero()).rename('FILL_MASK')
        # shadow_mask = shadow_mask.Or(zero_sr_mask).rename("SHADOW_MASK")

        # combine cloud and shadow masks
        valid_mask = ((cloud_mask.Or(shadow_mask)).Not()).And(fill_mask).rename("VALID_MASK")
        masks.update(cloud_mask=cloud_mask, shadow_mask=shadow_mask, valid_mask=valid_mask, fill_mask=fill_mask)
        return masks

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
        'COPERNICUS/S2': Sentinel2ClImage,
        'COPERNICUS/S2_SR': Sentinel2ClImage,
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
