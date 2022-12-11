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
from typing import Dict

import ee
import geedim.schema
from geedim.download import BaseImage
from geedim.enums import CloudMaskMethod
from geedim.utils import split_id, get_projection

logger = logging.getLogger(__name__)

##


class MaskedImage(BaseImage):
    _default_mask = False

    def __init__(self, ee_image: ee.Image, mask: bool = _default_mask, region: dict = None, **kwargs):
        """
        A class for describing, masking and downloading an Earth Engine image.

        Parameters
        ----------
        ee_image: ee.Image
            Earth Engine image to encapsulate.
        mask: bool, optional
            Whether to mask the image.
        region: dict, optional
            A geojson polygon inside of which to find statistics for the image.  These values are stored in the image
            properties. If None, statistics are not found (the default).

        **kwargs
            Cloud/shadow masking parameters - see below:
        mask_cirrus: bool, optional
            Whether to mask cirrus clouds.  Valid for Landsat 8-9 images, and for Sentinel-2 images with
            the `qa` ``mask_method``.
        mask_shadows: bool, optional
            Whether to mask cloud shadows.
        mask_method: CloudMaskMethod, str, optional
            Method used to mask clouds.  Valid for Sentinel-2 images.  See :class:`~geedim.enums.CloudMaskMethod` for
            details.
        prob: float, optional
            Cloud probability threshold (%). Valid for Sentinel-2 images with the `cloud-prob` ``mask-method``.
        dark: float, optional
            NIR threshold [0-1]. NIR values below this threshold are potential cloud shadows.  Valid for Sentinel-2
            images.
        shadow_dist: int, optional
            Maximum distance (m) to look for cloud shadows from cloud edges.  Valid for Sentinel-2 images.
        buffer: int, optional
            Distance (m) to dilate cloud/shadow.  Valid for Sentinel-2 images.
        cdi_thresh: float, optional
            Cloud Displacement Index threshold. Values below this threshold are considered potential clouds.
            If this parameter is not specified (=None), the index is not used.  Valid for Sentinel-2 images.
            See https://developers.google.com/earth-engine/apidocs/ee-algorithms-sentinel2-cdi for details.
        max_cloud_dist: int, optional
            Maximum distance (m) to look for clouds when forming the 'cloud distance' band.  Valid for
            Sentinel-2 images.
        """
        # TODO: consider adding proj_scale parameter here, rather than in _set_region_stats, then it can be re-used in
        #  S2 cloud masking and distance
        BaseImage.__init__(self, ee_image)
        self._add_aux_bands(**kwargs)  # add any mask and cloud distance bands
        if region:
            self._set_region_stats(region)
        if mask:
            self.mask_clouds()

    @staticmethod
    def from_id(image_id: str, **kwargs) -> 'MaskedImage':
        """
        Create a MaskedImage, or sub-class, instance from an Earth Engine image ID.

        Parameters
        ----------
        image_id: str
            ID of the Earth Engine image to encapsulate.
        **kwargs
            Optional keyword arguments to pass to :meth:`__init__`.

        Returns
        -------
        MaskedImage
            A MaskedImage, or sub-class instance.
        """

        cls = class_from_id(image_id)
        ee_image = ee.Image(image_id)
        gd_image = cls(ee_image, **kwargs)
        gd_image._id = image_id  # set the id attribute (avoids a call to getInfo() for .id property)
        return gd_image

    def _aux_image(self, **kwargs) -> ee.Image:
        """
        Retrieve the auxiliary image (MaskedImage provides an image with a FILL_MASK band only).

        Derived classes should override this method and return an image with FILL_MASK and any other auxiliary bands
        they support.
        """
        return self.ee_image.mask().reduce(ee.Reducer.allNonZero()).rename('FILL_MASK')

    def _add_aux_bands(self, **kwargs):
        """
        Add auxiliary bands to the encapsulated image. Existing auxiliary bands are overwritten, unless the image has
        no fixed projection and existing auxiliary bands (i.e. is likely a MaskedCollection composite image).
        """
        aux_image = self._aux_image(**kwargs)
        proj = self.ee_image.select(0).projection()
        has_fixed_scale = proj.nominalScale().toInt64().neq(111319)  # 1 deg in meters
        has_no_aux_bands = ee.Number(self.ee_image.bandNames().indexOf('FILL_MASK').lt(0))
        # overwrite unless it is a composite image with existing aux bands
        overwrite = has_no_aux_bands.Or(has_fixed_scale)
        self.ee_image = ee.Image(
            ee.Algorithms.If(overwrite, self.ee_image.addBands(aux_image, overwrite=True), self.ee_image)
        )

    def _set_region_stats(self, region: Dict = None, scale: float = None):
        """
        Set FILL_PORTION and CLOUDLESS_PORTION on the encapsulated image for the specified region.  Derived classes
        should override this method and set CLOUDLESS_PORTION, and/or other statistics they support.

        Parameters
        ----------
        region : dict, ee.Geometry, optional
            Region inside of which to find statistics.  If not specified, the image footprint is used.
        scale: float, optional
            Re-project to this scale when finding statistics.
        """
        if not region:
            region = self.ee_image.geometry()  # use the image footprint

        proj = get_projection(self.ee_image, min_scale=False)  # get projection of minimum scale band
        # If _proj_scale is set, use that as the scale, otherwise use the proj.nomimalScale().  For non-composite images
        # these should be the same value.  For composite images, there is no `fixed` projection, hence the
        # need for _proj_scale.
        scale = scale or proj.nominalScale()

        # Find the fill portion as the (sum over the region of FILL_MASK) divided by (sum over the region of a constant
        # image (==1)).  We take this approach rather than using a mean reducer, as this does not find the mean over
        # the region, but the mean over the part of the region covered by the image.
        stats_image = ee.Image(
            [self.ee_image.select('FILL_MASK').unmask(), ee.Image(1).rename('REGION_SUM')]
        )  # yapf: disable
        # Note: sometimes proj has no EPSG in crs(), hence use crs=proj and not crs=proj.crs() below
        sums = stats_image.reduceRegion(
            reducer="sum", geometry=region, crs=proj, scale=scale, bestEffort=True, maxPixels=1e6
        )

        fill_portion = (
            ee.Number(sums.get('FILL_MASK')).divide(ee.Number(sums.get('REGION_SUM'))).multiply(100)
        )

        # set the encapsulated image properties
        self.ee_image = self.ee_image.set('FILL_PORTION', fill_portion)
        # set CLOUDLESS_PORTION=100 for the generic case, where cloud/shadow masking is not supported
        self.ee_image = self.ee_image.set('CLOUDLESS_PORTION', 100.)

    def mask_clouds(self):
        """ Apply the cloud/shadow mask if supported, otherwise apply the fill mask. """
        self.ee_image = self.ee_image.updateMask(self.ee_image.select('FILL_MASK'))


class CloudMaskedImage(MaskedImage):
    """ A base class for encapsulating cloud/shadow masked images. """

    def _cloud_dist(self, cloudless_mask: ee.Image = None, max_cloud_dist: float = 5000) -> ee.Image:
        """ Find the cloud/shadow distance in units of 10m. """
        if not cloudless_mask:
            cloudless_mask = self.ee_image.select('CLOUDLESS_MASK')
        proj = get_projection(self.ee_image, min_scale=False)  # use maximum scale projection to save processing time

        # Note that initial *MASK bands before any call to mask_clouds(), are themselves masked, so this cloud/shadow
        # mask excludes (i.e. masks) already masked pixels.  This avoids finding distance to e.g. scanline errors in
        # Landsat-7.
        cloud_shadow_mask = cloudless_mask.Not()
        cloud_pix = ee.Number(max_cloud_dist).divide(proj.nominalScale()).round()  # cloud_dist in pixels

        # Find distance to nearest cloud/shadow (units of 10m).
        cloud_dist = cloud_shadow_mask.fastDistanceTransform(
            neighborhood=cloud_pix, units='pixels', metric='squared_euclidean'
        ).sqrt().multiply(proj.nominalScale().divide(10))

        # Reproject to force calculation at correct scale.
        cloud_dist = cloud_dist.reproject(crs=proj, scale=proj.nominalScale()).rename('CLOUD_DIST')

        # Clip cloud_dist to max_cloud_dist.
        cloud_dist = cloud_dist.where(cloud_dist.gt(ee.Image(max_cloud_dist / 10)), max_cloud_dist / 10)

        # cloud_dist is float64 by default, so convert to Uint16 here to avoid forcing the whole image to float64 on
        # download.
        return cloud_dist.toUint16().rename('CLOUD_DIST')

    def _set_region_stats(self, region: Dict = None, scale: float = None):
        """
        Set FILL_PORTION and CLOUDLESS_PORTION on the encapsulated image for the specified region.

        Parameters
        ----------
        region : dict, ee.Geometry, optional
            Region inside of which to find statistics.  If not specified, the image footprint is used.
        scale: float, optional
            Re-project to this scale when finding statistics.
        """
        if not region:
            region = self.ee_image.geometry()  # use the image footprint

        proj = get_projection(self.ee_image, min_scale=False)  # get projection of minimum scale band
        # If _proj_scale is set, use that as the scale, otherwise use the proj.nomimalScale().  For non-composite images
        # these should be the same value.  For composite images, there is no `fixed` projection, hence the
        # need for _proj_scale.
        scale = scale or proj.nominalScale()

        # Find the fill portion as the (sum over the region of FILL_MASK) divided by (sum over the region of a constant
        # image (==1)).  We take this approach rather than using a mean reducer, as this does not find the mean over
        # the region, but the mean over the part of the region covered by the image.  Then we find cloudless portion
        # as portion of fill that is cloudless.
        stats_image = ee.Image(
            [self.ee_image.select(['FILL_MASK', 'CLOUDLESS_MASK']).unmask(), ee.Image(1).rename('REGION_SUM')]
        )  # yapf: disable
        sums = stats_image.reduceRegion(
            reducer="sum", geometry=region, crs=proj, scale=scale, bestEffort=True, maxPixels=1e6
        ).rename(['FILL_MASK', 'CLOUDLESS_MASK'], ['FILL_PORTION', 'CLOUDLESS_PORTION'])
        fill_portion = (
            ee.Number(sums.get('FILL_PORTION')).divide(ee.Number(sums.get('REGION_SUM'))).multiply(100)
        )
        cloudless_portion = (
            ee.Number(sums.get('CLOUDLESS_PORTION')).divide(ee.Number(sums.get('FILL_PORTION'))).multiply(100)
        )

        # set the encapsulated image properties
        region_stats = ee.Dictionary(dict(FILL_PORTION=fill_portion, CLOUDLESS_PORTION=cloudless_portion))
        self.ee_image = self.ee_image.set(region_stats)

    def _aux_image(self, **kwargs) -> ee.Image:
        """
        Retrieve the auxiliary image containing cloud/shadow masks and cloud distance.  The returned image should
        contain at least FILL_MASK, CLOUDLESS_MASK and CLOUD_DIST bands.

        See :meth:`LandsatImage._aux_image` for an example.
        """
        raise NotImplementedError('This virtual method should be overridden by derived classes.')

    def mask_clouds(self):
        """ Apply the cloud/shadow mask. """
        self.ee_image = self.ee_image.updateMask(self.ee_image.select('CLOUDLESS_MASK'))


class LandsatImage(CloudMaskedImage):
    """
    Class for cloud/shadow masking of Landsat level 2, collection 2 images.

    Supports images from:
    * LANDSAT/LT04/C02/T1_L2
    * LANDSAT/LT05/C02/T1_L2
    * LANDSAT/LE07/C02/T1_L2
    * LANDSAT/LC08/C02/T1_L2
    * LANDSAT/LC09/C02/T1_L2
    """

    def _aux_image(self, mask_shadows: bool = True, mask_cirrus: bool = True, max_cloud_dist: int = 5000) -> ee.Image:
        """
        Retrieve the auxiliary image containing cloud/shadow masks and cloud distance.

        Parameters
        ----------
        mask_shadows: bool, optional
            Whether to mask cloud shadows.
        mask_cirrus: bool, optional
            Whether to mask cirrus clouds.  Valid for Landsat 8-9 images.
        max_cloud_dist: int, optional
            Maximum distance (m) to look for clouds when forming the 'cloud distance' band.  Valid for
            Sentinel-2 images.

        Returns
        -------
        ee.Image
            An Earth Engine image containing *_MASK and CLOUD_DIST bands.
        """
        ee_image = self._ee_image
        qa_pixel = ee_image.select('QA_PIXEL')

        # construct fill mask from Earth Engine mask and QA_PIXEL
        ee_mask = ee_image.select('SR_B.*').mask().reduce(ee.Reducer.allNonZero())
        fill_mask = qa_pixel.bitwiseAnd(1).eq(0).And(ee_mask).rename('FILL_MASK')

        shadow_mask = qa_pixel.bitwiseAnd(0b10000).neq(0).rename('SHADOW_MASK')
        if mask_cirrus:
            cloud_mask = qa_pixel.bitwiseAnd(0b1100).neq(0).rename('CLOUD_MASK')
        else:
            cloud_mask = qa_pixel.bitwiseAnd(0b1000).neq(0).rename('CLOUD_MASK')

        # combine cloud, shadow and fill masks into cloudless mask
        cloud_shadow_mask = (cloud_mask.Or(shadow_mask)) if mask_shadows else cloud_mask
        cloudless_mask = cloud_shadow_mask.Not().And(fill_mask).rename('CLOUDLESS_MASK')

        # copy cloud distance from existing ST_CDIST band (in 10m units), and clip to max_cloud_dist
        cloud_dist = ee_image.select('ST_CDIST').rename('CLOUD_DIST').toUint16()
        cloud_dist = cloud_dist.where(cloud_dist.gt(ee.Image(max_cloud_dist / 10)), max_cloud_dist / 10)

        return ee.Image([fill_mask, cloud_mask, shadow_mask, cloudless_mask, cloud_dist])


class Sentinel2ClImage(CloudMaskedImage):
    """ Base class for cloud/shadow masking of Sentinel-2 TOA and SR images. """

    def _aux_image(
        self, s2_toa: bool = False, mask_cirrus: bool = True, mask_shadows: bool = True,
        mask_method: CloudMaskMethod = CloudMaskMethod.cloud_prob, prob: float = 60, dark: float = 0.15,
        shadow_dist: int = 1000, buffer: int = 50, cdi_thresh: float = None, max_cloud_dist: int = 5000
    ) -> ee.Image:
        """
        Derive cloud, shadow and validity masks for the encapsulated image.

        Adapted from https://github.com/r-earthengine/ee_extra, under Apache 2.0 license.

        Parameters
        ----------
        s2_toa : bool, optional
            S2 TOA/SR collection.  Set to True if this image is from COPERNICUS/S2, or False if it is from
            COPERNICUS/S2_SR.
        mask_cirrus: bool, optional
            Whether to mask cirrus clouds.  Valid for Landsat 8-9 images, and for Sentinel-2 images with
            the `qa` ``mask_method``.
        mask_shadows: bool, optional
            Whether to mask cloud shadows.
        mask_method: CloudMaskMethod, str, optional
            Method used to mask clouds.  Valid for Sentinel-2 images.  See :class:`~geedim.enums.CloudMaskMethod` for
            details.
        prob: float, optional
            Cloud probability threshold (%). Valid for Sentinel-2 images with the `cloud-prob` ``mask-method``.
        dark: float, optional
            NIR threshold [0-1]. NIR values below this threshold are potential cloud shadows.  Valid for Sentinel-2
            images.
        shadow_dist: int, optional
            Maximum distance (m) to look for cloud shadows from cloud edges.  Valid for Sentinel-2 images.
        buffer: int, optional
            Distance (m) to dilate cloud/shadow.  Valid for Sentinel-2 images.
        cdi_thresh: float, optional
            Cloud Displacement Index threshold. Values below this threshold are considered potential clouds.
            If this parameter is not specified (=None), the index is not used.  Valid for Sentinel-2 images.
            See https://developers.google.com/earth-engine/apidocs/ee-algorithms-sentinel2-cdi for details.
        max_cloud_dist: int, optional
            Maximum distance (m) to look for clouds when forming the 'cloud distance' band.  Valid for
            Sentinel-2 images.

        Returns
        -------
        ee.Image
            An Earth Engine image containing *_MASK and CLOUD_DIST bands.
        """
        mask_method = CloudMaskMethod(mask_method)

        def get_cloud_prob(ee_im):
            """Get the cloud probability image from COPERNICUS/S2_CLOUD_PROBABILITY that corresponds to `ee_im`."""
            filt = ee.Filter.eq('system:index', ee_im.get('system:index'))
            cloud_prob = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filter(filt).first()
            return cloud_prob.rename('CLOUD_PROB')

        def get_cloud_mask(ee_im, cloud_prob=None):
            """Get the cloud mask for ee_im"""
            if mask_method == CloudMaskMethod.cloud_prob:
                if not cloud_prob:
                    cloud_prob = get_cloud_prob(ee_im)
                cloud_mask = cloud_prob.gte(prob)
            else:
                qa = ee_im.select('QA60')
                cloud_mask = qa.bitwiseAnd(1 << 10).neq(0)
                if mask_cirrus:
                    cloud_mask = cloud_mask.Or(qa.bitwiseAnd(1 << 11).neq(0))
            return cloud_mask.rename('CLOUD_MASK')

        def get_cdi_cloud_mask(ee_im):
            """
            Get a CDI cloud mask for ee_im.
            See https://developers.google.com/earth-engine/apidocs/ee-algorithms-sentinel2-cdi for more detail.
            """
            if s2_toa:
                s2_toa_image = ee_im
            else:
                # get the Sentinel-2 TOA image that corresponds to ee_im
                idx = ee_im.get('system:index')
                s2_toa_image = (ee.ImageCollection('COPERNICUS/S2').filter(ee.Filter.eq('system:index', idx)).first())
            cdi_image = ee.Algorithms.Sentinel2.CDI(s2_toa_image)
            return cdi_image.lt(cdi_thresh).rename('CDI_CLOUD_MASK')

        def get_shadow_mask(ee_im, cloud_mask):
            """Given a cloud mask, get a shadow mask for ee_im."""
            dark_mask = ee_im.select('B8').lt(dark * 1e4)
            if not s2_toa:
                dark_mask = ee_im.select('SCL').neq(6).And(dark_mask)

            proj = get_projection(ee_im, min_scale=False)
            # Note:
            # S2 MEAN_SOLAR_AZIMUTH_ANGLE (SAA) appears to be measured clockwise with 0 at N (i.e. shadow goes in the
            # opposite direction), directionalDistanceTransform() angle appears to be measured clockwise with 0 at W.
            # So we need to add/subtract 180 to SAA to get shadow angle in S2 convention, then add 90 to get
            # directionalDistanceTransform() convention i.e. we need to add -180 + 90 = -90 to the SAA.  This is not
            # the same as in the EE tutorial which is 90-SAA
            # (https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless).
            shadow_azimuth = ee.Number(-90).add(ee.Number(ee_im.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
            proj_pixels = ee.Number(shadow_dist).divide(proj.nominalScale()).round()
            # Project the cloud mask in the direction of the shadows it will cast.
            cloud_cast_proj = cloud_mask.directionalDistanceTransform(shadow_azimuth, proj_pixels).select('distance')
            # The reproject is necessary to force calculation at the correct scale - the coarse proj scale is used to
            # improve processing times.
            cloud_cast_mask = cloud_cast_proj.mask().reproject(crs=proj.crs(), scale=proj.nominalScale())
            return cloud_cast_mask.And(dark_mask).rename('SHADOW_MASK')

        # gather and combine the various masks
        ee_image = self.ee_image
        cloud_prob = get_cloud_prob(ee_image) if mask_method == CloudMaskMethod.cloud_prob else None
        cloud_mask = get_cloud_mask(ee_image, cloud_prob=cloud_prob)
        if cdi_thresh is not None:
            cloud_mask = cloud_mask.And(get_cdi_cloud_mask(ee_image))
        if mask_shadows:
            shadow_mask = get_shadow_mask(ee_image, cloud_mask)
            cloud_shadow_mask = cloud_mask.Or(shadow_mask)
        else:
            cloud_shadow_mask = cloud_mask

        # do a morphological opening type operation that removes small (20m) blobs from the mask and then dilates
        cloud_shadow_mask = cloud_shadow_mask.focal_min(20, units='meters').focal_max(buffer, units='meters')
        # derive a fill mask from the Earth Engine mask for the surface reflectance bands
        fill_mask = ee_image.select('B.*').mask().reduce(ee.Reducer.allNonZero())
        # Clip this mask to the image footprint.  (Without this step we get memory limit errors on download.)
        fill_mask = fill_mask.clip(ee_image.geometry()).rename('FILL_MASK')

        # combine all masks into cloudless_mask
        cloudless_mask = (cloud_shadow_mask.Not()).And(fill_mask).rename('CLOUDLESS_MASK')

        # construct and return the auxiliary image
        aux_bands = [fill_mask, cloud_mask, cloudless_mask]
        if mask_shadows:
            aux_bands.append(shadow_mask)
        if mask_method == CloudMaskMethod.cloud_prob:
            aux_bands.append(cloud_prob)

        cloud_dist = self._cloud_dist(cloudless_mask=cloudless_mask, max_cloud_dist=max_cloud_dist)
        return ee.Image(aux_bands + [cloud_dist])


class Sentinel2SrClImage(Sentinel2ClImage):
    """ Class for cloud/shadow masking of Sentinel-2 SR (COPERNICUS/S2_SR) images. """

    def _aux_image(self, s2_toa: bool = False, **kwargs) -> ee.Image:
        return Sentinel2ClImage._aux_image(self, s2_toa=False, **kwargs)


class Sentinel2ToaClImage(Sentinel2ClImage):
    """ Class for cloud/shadow masking of Sentinel-2 TOA (COPERNICUS/S2) images. """

    def _aux_image(self, s2_toa: bool = False, **kwargs) -> ee.Image:
        return Sentinel2ClImage._aux_image(self, s2_toa=True, **kwargs)


def class_from_id(image_id: str) -> type:
    """ Return the *Image class that corresponds to the provided Earth Engine image/collection ID. """
    ee_coll_name, _ = split_id(image_id)
    if image_id in geedim.schema.collection_schema:
        return geedim.schema.collection_schema[image_id]['image_type']
    elif ee_coll_name in geedim.schema.collection_schema:
        return geedim.schema.collection_schema[ee_coll_name]['image_type']
    else:
        return MaskedImage
