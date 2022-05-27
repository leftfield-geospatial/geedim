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

import ee

from geedim.download import BaseImage
from geedim.enums import CloudMaskMethod
from geedim.utils import split_id, get_projection

logger = logging.getLogger(__name__)


##


class MaskedImage(BaseImage):
    _default_mask = False
    _supported_collection_ids = ['*']
    _proj_scale = None

    # TODO: for images w/o fixed projections, nominalScale() is 1deg~100km.  Can we get this from STAC?

    #  STAC w/o overheads for e.g. mapping over collections

    def __init__(self, ee_image, mask=_default_mask, region=None, **kwargs):
        """
        Base class for encapsulating and masking any Earth Engine image.

        Parameters
        ----------
        ee_image: ee.Image
            The Earth Engine image to encapsulate.
        mask: bool, optional
            Whether to mask the image [default: False].
        region: dict, optional
            A geojson polygon inside of which to find statistics for the image.  These values are stored in the image
            properties [default: don't find statistics].
        kwargs: optional
            Any cloud/shadow masking parameters supported for the encapsulated image.

            mask_cirrus: Whether to mask cirrus clouds.  Valid for Landsat 8-9 images, and,for Sentinel-2 images
                with method=`qa`.
            mask_shadows: Whether to mask cloud shadows.
            mask_method : CloudMaskMethod, optional
                Method used to mask clouds.  Valid for Sentinel-2 images.
                Available options:
                    - 'cloud-prob' : Use S2 cloud probability.
                    - 'qa' : Use Quality Assessment band.
            prob : float, optional
                Cloud probability threshold (%). Valid for Sentinel-2 images with the `cloud-prob` mask-method.
            dark : float, optional
                NIR threshold [0-1]. NIR values below this threshold are potential cloud shadows.  Valid for Sentinel-2
                images.
            shadow_dist : int, optional
                Maximum distance (m) to look for cloud shadows from cloud edges.  Valid for Sentinel-2 images.
            buffer : int, optional
                Distance (m) to dilate cloud/shadow.  Valid for Sentinel-2 images.
            cdi_thresh : float, optional
                Cloud Displacement Index threshold. Values below this threshold are considered potential clouds.
                If this parameter is not specified (=None), the index is not used.  Valid for Sentinel-2 images.
                See https://developers.google.com/earth-engine/apidocs/ee-algorithms-sentinel2-cdi for details.
            max_cloud_dist: int, optional
                Maximum distance (m) to look for clouds when forming the `cloud distance` band.  Valid for
                Sentinel-2 images.
        """
        BaseImage.__init__(self, ee_image)
        self._add_aux_bands(**kwargs)  # add any mask and cloud distance bands
        if region:
            self.set_region_stats(region)
        if mask:
            self.mask_clouds()

    @staticmethod
    def from_id(image_id: str, **kwargs) -> 'MaskedImage':
        """
        Given an Earth Engine image ID, create an instance of MaskedImage, or the appropriate sub-class

        Parameters
        ----------
        image_id: str
            The ID of the Earth Engine image to encapsulate.
        kwargs: optional
            Any arguments to pass through to the class __init__() method.
            See the MaskedImage.__init__() documentation for more detail.

        Returns
        -------
        gd_image: MaskedImage
            A MaskedImage, or sub-class instance.
        """

        cls = class_from_id(image_id)
        ee_image = ee.Image(image_id)
        gd_image = cls(ee_image, **kwargs)
        gd_image._id = image_id  # set the id attribute (avoids a call to getInfo() for .id property)
        return gd_image

    def _aux_image(self, **kwargs) -> ee.Image:
        """
        Retrieve the auxiliary image (MaskedImage provides FILL_MASK only). Derived classes should override this
        method and return whatever additional mask etc bands they support.
        """
        return self.ee_image.mask().reduce(ee.Reducer.allNonZero()).rename('FILL_MASK')

    def _add_aux_bands(self, **kwargs):
        """Add auxiliary bands to the encapsulated image, if they are not present already."""
        aux_image = self._aux_image(**kwargs)
        cond = ee.Number(self.ee_image.bandNames().contains('FILL_MASK'))
        self.ee_image = ee.Image(ee.Algorithms.If(cond, self.ee_image, self.ee_image.addBands(aux_image)))

    def set_region_stats(self, region=None):
        """
        Set FILL_PORTION on the encapsulated image for the specified region.  Derived classes should override this
        method and add a true CLOUDLESS_PORTION and/or other statistics they support.

        Parameters
        ----------
        region : dict, ee.Geometry, optional
            Region inside of which to find statistics.  If not specified, the image footprint is used.
        """
        if not region:
            region = self.ee_image.geometry()  # use the image footprint

        # TODO: min_scale=False works for Sentinel-2, but not e.g. for Sentinel-1 SAR where one band is 12km and the
        #  others 10m.  So we need something that works for all cases which apparently include cases where one band's
        #  scale is very different to the others.  And the case of composite images.  Currently if it is a supported
        #  cloud-masked image, it will have a meaningful _proj_scale, so that it ok for those images.  Otherwise
        #  perhaps it is best to use STAC and use min scale, but there could be cases where min scale is extreme,
        #  so we should check that across STAC too.  Some sort of generic algorithm like mode(scales) might be
        #  better?
        proj = get_projection(self.ee_image, min_scale=True)  # get projection of minimum scale band
        # If _proj_scale is set, use that as the scale, otherwise use the proj.nomimalScale().  For non-composite images
        # these should be the same value.  For composite images, there is no `fixed` projection, hence the
        # need for _proj_scale.
        scale = self._proj_scale or proj.nominalScale()

        # Find the fill portion as the (sum over the region of FILL_MASK) divided by (sum over the region of a constant
        # image (==1)).  We take this approach rather than using a mean reducer, as this does not find the mean over
        # the region, but the mean over the part of the region covered by the image.
        stats_image = ee.Image(
            [self.ee_image.select('FILL_MASK').rename('FILL_PORTION').unmask(), ee.Image(1).rename('REGION_SUM')]
        )
        # Note: sometimes proj has no EPSG in crs(), hence use crs=proj and not crs=proj.crs() below
        sums_dict = stats_image.reduceRegion(
            reducer="sum", geometry=region, crs=proj, scale=scale, bestEffort=True, maxPixels=1e6
        )

        def region_percentage(key, value):
            return ee.Number(value).multiply(100).divide(ee.Number(sums_dict.get("REGION_SUM")))

        means = sums_dict.select(['FILL_PORTION']).map(region_percentage)

        # set the encapsulated image properties
        self.ee_image = self.ee_image.set(means)
        # set CLOUDLESS_PORTION=FILL_PORTION for the generic case, where cloud/shadow masking is not supported
        self.ee_image = self.ee_image.set('CLOUDLESS_PORTION', means.get('FILL_PORTION'))

    def mask_clouds(self):
        """
        Mask cloud/shadow in the encapsulated image.  For MaskedImage, cloud/shadow masking is not supported,
        so this just applies the Earth Engine derived FILL_MASK.
        """
        self.ee_image = self.ee_image.updateMask(self.ee_image.select('FILL_MASK'))


class CloudMaskedImage(MaskedImage):
    """
    Base class for encapsulating supported cloud/shadow masked images.
    """
    _supported_collection_ids = []  # abstract base class

    def _cloud_dist(self, cloudless_mask=None, max_cloud_dist=5000) -> ee.Image:
        """Get the cloud/shadow distance for encapsulated image."""
        if not cloudless_mask:
            cloudless_mask = self.ee_image.select('CLOUDLESS_MASK')
        proj = get_projection(self.ee_image, min_scale=False)  # use maximum scale projection to save processing time

        # Note that initial *MASK bands before any call to mask_clouds(), are themselves masked, so this cloud/shadow
        # mask excludes (i.e. masks) already masked pixels.  This avoids finding distance to e.g. scanline errors in
        # Landsat7.
        cloud_shadow_mask = cloudless_mask.Not()
        cloud_pix = ee.Number(max_cloud_dist).divide(proj.nominalScale()).round()  # cloud_dist in pixels

        # Find distance to nearest cloud/shadow (m).  Reproject is necessary to force calculation at correct scale.
        cloud_dist = (
            cloud_shadow_mask.fastDistanceTransform(
                neighborhood=cloud_pix, units='pixels', metric='squared_euclidean'
            ).sqrt().multiply(proj.nominalScale()).rename('CLOUD_DIST').reproject(
                crs=proj, scale=proj.nominalScale()
            )
        )

        # Clip cloud_dist to max_cloud_dist.
        cloud_dist = cloud_dist.where(cloud_dist.gt(ee.Image(max_cloud_dist)), max_cloud_dist)

        # cloud_dist is float64 by default, so convert to Uint32 here to avoid forcing the whole image to float64 on
        # download.
        return cloud_dist.toUint32().rename('CLOUD_DIST')

    def set_region_stats(self, region=None):
        """
        Set FILL_ and CLOUDLESS_PORTION on the encapsulated image for the specified region.

        Parameters
        ----------
        region : dict, ee.Geometry, optional
            Region inside of which to find statistics.  If not specified, the image footprint is used.
        """
        if not region:
            region = self.ee_image.geometry()  # use the image footprint

        proj = get_projection(self.ee_image, min_scale=False)  # get projection of minimum scale band
        # If _proj_scale is set, use that as the scale, otherwise use the proj.nomimalScale().  For non-composite images
        # these should be the same value.  For composite images, there is no `fixed` projection, hence the
        # need for _proj_scale.
        scale = self._proj_scale or proj.nominalScale()

        # Find the fill portion as the (sum over the region of FILL_MASK) divided by (sum over the region of a constant
        # image (==1)).  We take this approach rather than using a mean reducer, as this does not find the mean over
        # the region, but the mean over the part of the region covered by the image.
        stats_image = ee.Image(
            [self.ee_image.select(['FILL_MASK', 'CLOUDLESS_MASK']).unmask(), ee.Image(1).rename('REGION_SUM')]
        )
        sums = stats_image.reduceRegion(
            reducer="sum", geometry=region, crs=proj, scale=scale, bestEffort=True, maxPixels=1e6
        ).rename(['FILL_MASK', 'CLOUDLESS_MASK'], ['FILL_PORTION', 'CLOUDLESS_PORTION'])

        def region_percentage(key, value):
            return ee.Number(value).multiply(100).divide(ee.Number(sums.get("REGION_SUM")))

        means = sums.select(['FILL_PORTION', 'CLOUDLESS_PORTION']).map(region_percentage)
        # set the encapsulated image properties
        self.ee_image = self.ee_image.set(means)

    def mask_clouds(self):
        """
        Mask cloud/shadow in the encapsulated image.
        """
        self.ee_image = self.ee_image.updateMask(self.ee_image.select('CLOUDLESS_MASK'))


class LandsatImage(CloudMaskedImage):
    """ Class for cloud/shadow masking of Landsat level 2, collection 2 images """
    _supported_collection_ids = ['LANDSAT/LT04/C02/T1_L2', 'LANDSAT/LT05/C02/T1_L2', 'LANDSAT/LE07/C02/T1_L2',
                                 'LANDSAT/LC08/C02/T1_L2', 'LANDSAT/LC09/C02/T1_L2']
    _proj_scale = 30

    def _aux_image(self, mask_shadows=True, mask_cirrus=True) -> ee.Image:
        """
        Retrieve the auxiliary image containing cloud/shadow masks and cloud distance.

        Parameters
        ----------
        mask_shadows: bool, optional
            Whether to mask cloud shadows.
        mask_cirrus: bool, optional
            Whether to mask cirrus clouds.  Valid for Landsat 8-9 images.

        Returns
        -------
        aux_image: ee.Image
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
        cloudless_mask = (cloud_mask.Or(shadow_mask)).Not() if mask_shadows else cloud_mask.Not()
        cloudless_mask = cloudless_mask.And(fill_mask).rename('CLOUDLESS_MASK')
        # copy cloud distance from existing ST_CDIST band
        cloud_dist = ee_image.select('ST_CDIST').rename('CLOUD_DIST')
        return ee.Image([fill_mask, cloud_mask, shadow_mask, cloudless_mask, cloud_dist])


class Sentinel2ClImage(CloudMaskedImage):
    """Base class for cloud/shadow masking of Sentinel-2 TOA and SR (surface reflectance) images."""
    _supported_collection_ids = []
    _proj_scale = 60

    def _aux_image(
        self, s2_toa=False, mask_cirrus=True, mask_shadows=True, mask_method=CloudMaskMethod.cloud_prob, prob=60,
        dark=0.15, shadow_dist=1000, buffer=50, cdi_thresh=None, max_cloud_dist=5000
    ):
        """
        Derive cloud, shadow and validity masks for the encapsulated image.

        Adapted from https://github.com/r-earthengine/ee_extra, under Apache 2.0 license

        Parameters
        ----------
        s2_toa : bool, optional
            S2 TOA/SR collection.  Set to True if this image is from COPERNICUS/S2, or False if it is from
            COPERNICUS/S2_SR.
        mask_cirrus: Whether to mask cirrus clouds.  Valid for Landsat 8-9 images, and,for Sentinel-2 images
            with method=`qa`.
        mask_shadows: Whether to mask cloud shadows.
        mask_method : CloudMaskMethod, optional
            Method used to mask clouds.  Valid for Sentinel-2 images.
            Available options:
                - 'cloud-prob' : Use S2 cloud probability.
                - 'qa' : Use Quality Assessment band.
        prob : float, optional
            Cloud probability threshold (%). Valid for Sentinel-2 images with the `cloud-prob` mask-method.
        dark : float, optional
            NIR threshold [0-1]. NIR values below this threshold are potential cloud shadows.  Valid for Sentinel-2
            images.
        shadow_dist : int, optional
            Maximum distance (m) to look for cloud shadows from cloud edges.  Valid for Sentinel-2 images.
        buffer : int, optional
            Distance (m) to dilate cloud/shadow.  Valid for Sentinel-2 images.
        cdi_thresh : float, optional
            Cloud Displacement Index threshold. Values below this threshold are considered potential clouds.
            If this parameter is not specified (=None), the index is not used.  Valid for Sentinel-2 images.
            See https://developers.google.com/earth-engine/apidocs/ee-algorithms-sentinel2-cdi for details.
        max_cloud_dist: int, optional
            Maximum distance (m) to look for clouds when forming the `cloud distance` band.  Valid for
            Sentinel-2 images.

        Returns
        -------
        aux_image: ee.Image
            An Earth Engine image containing *_MASK and CLOUD_DIST bands.
        """
        mask_method = CloudMaskMethod(mask_method)

        def get_cloud_prob(ee_im):
            """Get the cloud probability image from COPERNICUS/S2_CLOUD_PROBABILITY that corresponds to `ee_im`."""
            idx = ee_im.get('system:index')
            return ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filter(
                ee.Filter.eq('system:index', idx)
            ).first().rename('CLOUD_PROB')

        def get_cloud_mask(ee_im, cloud_prob=None):
            """Get the cloud mask for ee_im"""
            if mask_method == CloudMaskMethod.cloud_prob:
                if not cloud_prob:
                    cloud_prob = get_cloud_prob(ee_im)
                cloud_mask = cloud_prob.gte(prob).rename('CLOUD_MASK')
            else:
                qa = ee_im.select('QA60')
                cloud_mask = qa.bitwiseAnd(1 << 10).neq(0)
                if mask_cirrus:
                    cloud_mask = cloud_mask.Or(qa.bitwiseAnd(1 << 11).neq(0))
            return cloud_mask

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

            shadow_azimuth = ee.Number(90).subtract(ee.Number(ee_im.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
            # Project the cloud mask in the direction of the shadows it will cast.
            # The reproject is necessary to force calculation at the correct scale - the coarse _proj_scale is used to
            # improve processing times.
            proj_cloud_mask = (cloud_mask.directionalDistanceTransform(
                shadow_azimuth, int(shadow_dist / self._proj_scale)
            ).reproject(crs=ee_im.select(0).projection(), scale=self._proj_scale).select('distance').mask())
            return proj_cloud_mask.And(dark_mask).rename('SHADOW_MASK')

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
    """Class for cloud/shadow masking of Sentinel-2 SR (COPERNICUS/S2_SR) images."""
    _supported_collection_ids = ['COPERNICUS/S2_SR']

    def _aux_image(self, s2_toa=False, **kwargs):
        return Sentinel2ClImage._aux_image(self, s2_toa=False, **kwargs)


class Sentinel2ToaClImage(Sentinel2ClImage):
    """Class for cloud/shadow masking of Sentinel-2 TOA (COPERNICUS/S2) images."""
    _supported_collection_ids = ['COPERNICUS/S2']

    def _aux_image(self, s2_toa=False, **kwargs):
        return Sentinel2ClImage._aux_image(self, s2_toa=True, **kwargs)


def class_from_id(image_id: str) -> type:
    """Return the *Image class that corresponds to the provided EE image/collection ID."""

    masked_image_dict = {
        'LANDSAT/LT04/C02/T1_L2': LandsatImage,
        'LANDSAT/LT05/C02/T1_L2': LandsatImage,
        'LANDSAT/LE07/C02/T1_L2': LandsatImage,
        'LANDSAT/LC08/C02/T1_L2': LandsatImage,
        'LANDSAT/LC09/C02/T1_L2': LandsatImage,
        'COPERNICUS/S2': Sentinel2ToaClImage,
        'COPERNICUS/S2_SR': Sentinel2SrClImage,
    }
    ee_coll_name, _ = split_id(image_id)
    if image_id in masked_image_dict:
        return masked_image_dict[image_id]
    elif ee_coll_name in masked_image_dict:
        return masked_image_dict[ee_coll_name]
    else:
        return MaskedImage
