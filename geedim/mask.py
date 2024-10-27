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

import logging
import warnings

import ee

import geedim.schema
from geedim.download import BaseImage
from geedim.enums import CloudMaskMethod, CloudScoreBand
from geedim.utils import get_projection, split_id

logger = logging.getLogger(__name__)

##


class MaskedImage(BaseImage):
    _default_mask = False

    def __init__(self, ee_image: ee.Image, mask: bool = _default_mask, region: dict | ee.Geometry = None, **kwargs):
        """
        A class for describing, masking and downloading an Earth Engine image.

        Parameters
        ----------
        ee_image: ee.Image
            Earth Engine image to encapsulate.
        mask: bool, optional
            Whether to mask the image.
        region: dict, ee.Geometry, optional
            Region in which to find statistics for the image, as a GeoJSON dictionary or ``ee.Geometry``.  Statistics
            are stored in the image properties. If None, statistics are not found (the default).

        **kwargs
            Cloud/shadow masking parameters - see below:
        mask_cirrus: bool, optional
            Whether to mask cirrus clouds.  Valid for Landsat 8-9 images, and for Sentinel-2 images with
            the `qa` ``mask_method``.
        mask_shadows: bool, optional
            Whether to mask cloud shadows.  Valid for Landsat images, and for Sentinel-2 images with
            the `qa` or `cloud-prob` ``mask_method``.
        mask_method: CloudMaskMethod, str, optional
            Method used to mask clouds.  Valid for Sentinel-2 images.  See :class:`~geedim.enums.CloudMaskMethod` for
            details.
        prob: float, optional
            Cloud probability threshold (%). Valid for Sentinel-2 images with the `cloud-prob` ``mask_method``.
        dark: float, optional
            NIR threshold [0-1]. NIR values below this threshold are potential cloud shadows.  Valid for Sentinel-2
            images with the `qa` or `cloud-prob` ``mask_method``.
        shadow_dist: int, optional
            Maximum distance (m) to look for cloud shadows from cloud edges.  Valid for Sentinel-2 images with the `qa`
            or `cloud-prob` ``mask_method``.
        buffer: int, optional
            Distance (m) to dilate cloud/shadow.  Valid for Sentinel-2 images with the `qa` or `cloud-prob`
            ``mask_method``.
        cdi_thresh: float, optional
            Cloud Displacement Index threshold. Values below this threshold are considered potential clouds.
            If this parameter is not specified (=None), the index is not used.  Valid for Sentinel-2 images with the
            `qa` or `cloud-prob` ``mask_method``.  See
            https://developers.google.com/earth-engine/apidocs/ee-algorithms-sentinel2-cdi for details.
        max_cloud_dist: int, optional
            Maximum distance (m) to look for clouds when forming the 'cloud distance' band.  Valid for
            Sentinel-2 images.
        score: float, optional
            Cloud Score+ threshold.  Valid for Sentinel-2 images with the `cloud-score` ``mask_method``.
        cs_band: CloudScoreBand, str, optional
            Cloud Score+ band to threshold.  Valid for Sentinel-2 images with the `cloud-score` ``mask_method``.
        """
        BaseImage.__init__(self, ee_image)
        self._ee_projection = None
        self._add_aux_bands(**kwargs)  # add any mask and cloud distance bands
        if region:
            self._set_region_stats(region)
        if mask:
            self.mask_clouds()

    @property
    def _ee_proj(self) -> ee.Projection:
        """Projection to use for mask calculations and statistics."""
        if self._ee_projection is None:
            # use the minimum scale projection for the base class / generic case (excludes non-fixed projections with
            # 1 deg. scales)
            self._ee_projection = get_projection(self._ee_image, min_scale=True)
        return self._ee_projection

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

    def _set_region_stats(self, region: dict | ee.Geometry = None, scale: float = None):
        """
        Set FILL_PORTION and CLOUDLESS_PORTION on the encapsulated image for the specified region.  Derived classes
        should override this method and set CLOUDLESS_PORTION, and/or other statistics they support.

        Parameters
        ----------
        region: dict, ee.Geometry, optional
            Region in which to find statistics for the image, as a GeoJSON dictionary or ``ee.Geometry``.  If not
            specified, the image footprint is used.
        scale: float, optional
            Re-project to this scale when finding statistics.  Defaults to the scale of
            :attr:`~MaskedImage._ee_proj`.  Should be provided if the encapsulated image is a composite / without a
            fixed projection.
        """
        if not region:
            region = self.ee_image.geometry()  # use the image footprint

        # composite images have no fixed projection (scale = 1deg) and need the scale kwarg
        scale = scale or self._ee_proj.nominalScale()

        # Find the fill portion as the (sum over the region of FILL_MASK) divided by (sum over the region of a constant
        # image (==1)).  Note that a mean reducer, does not find the mean over the region, but the mean over the part
        # of the region covered by the image.
        stats_image = ee.Image([self.ee_image.select('FILL_MASK').unmask(), ee.Image(1).rename('REGION_SUM')])
        # Note: sometimes proj has no EPSG in crs(), hence use crs=proj and not crs=proj.crs() below
        sums = stats_image.reduceRegion(
            reducer="sum", geometry=region, crs=self._ee_proj, scale=scale, bestEffort=True, maxPixels=1e6
        )

        fill_portion = ee.Number(sums.get('FILL_MASK')).divide(ee.Number(sums.get('REGION_SUM'))).multiply(100)

        # set the encapsulated image properties
        self.ee_image = self.ee_image.set('FILL_PORTION', fill_portion)
        # set CLOUDLESS_PORTION=100 for the generic case, where cloud/shadow masking is not supported
        self.ee_image = self.ee_image.set('CLOUDLESS_PORTION', 100.0)

    def mask_clouds(self):
        """Apply the cloud/shadow mask if supported, otherwise apply the fill mask."""
        self.ee_image = self.ee_image.updateMask(self.ee_image.select('FILL_MASK'))


class CloudMaskedImage(MaskedImage):
    """A base class for encapsulating cloud/shadow masked images."""

    def _cloud_dist(self, cloudless_mask: ee.Image = None, max_cloud_dist: float = 5000) -> ee.Image:
        """Find the cloud/shadow distance in units of 10m."""
        if not cloudless_mask:
            cloudless_mask = self.ee_image.select('CLOUDLESS_MASK')

        cloud_shadow_mask = cloudless_mask.Not()
        cloud_pix = ee.Number(max_cloud_dist).divide(self._ee_proj.nominalScale()).round()  # cloud_dist in pixels

        # Find distance to nearest cloud/shadow (units of 10m).  Distances are found for all pixels, including masked /
        # invalid pixels, which are treated as 0 (non cloud/shadow).
        cloud_dist = (
            cloud_shadow_mask.fastDistanceTransform(neighborhood=cloud_pix, units='pixels', metric='squared_euclidean')
            .sqrt()
            .multiply(self._ee_proj.nominalScale().divide(10))
        )

        # Reproject to force calculation at correct scale.
        cloud_dist = cloud_dist.reproject(crs=self._ee_proj, scale=self._ee_proj.nominalScale()).rename('CLOUD_DIST')

        # Prevent use of invalid pixels.
        cloud_dist = cloud_dist.updateMask(cloudless_mask.mask())

        # Clip cloud_dist to max_cloud_dist.
        cloud_dist = cloud_dist.where(cloud_dist.gt(ee.Image(max_cloud_dist / 10)), max_cloud_dist / 10)

        # cloud_dist is float64 by default, so convert to Uint16 here to avoid forcing the whole image to float64 on
        # download.
        return cloud_dist.toUint16().rename('CLOUD_DIST')

    def _set_region_stats(self, region: dict | ee.Geometry = None, scale: float = None):
        """
        Set FILL_PORTION and CLOUDLESS_PORTION on the encapsulated image for the specified region.

        Parameters
        ----------
        region: dict, ee.Geometry, optional
            Region in which to find statistics for the image, as a GeoJSON dictionary or ``ee.Geometry``.  If not
            specified, the image footprint is used.
        scale: float, optional
            Re-project to this scale when finding statistics.  Defaults to the scale of
            :attr:`~MaskedImage._ee_proj`.  Should be provided if the encapsulated image is a composite / without a
            fixed projection.
        """
        if not region:
            region = self.ee_image.geometry()  # use the image footprint

        # composite images have no fixed projection (scale = 1deg) and need the scale kwarg
        scale = scale or self._ee_proj.nominalScale()

        # Find the fill portion as the (sum over the region of FILL_MASK) divided by (sum over the region of a
        # constant image (==1)).  Then find cloudless portion as portion of fill that is cloudless.  Note that a mean
        # reducer, does not find the mean over the region, but the mean over the part of the region covered by the
        # image.
        stats_image = ee.Image(
            [self.ee_image.select(['FILL_MASK', 'CLOUDLESS_MASK']).unmask(), ee.Image(1).rename('REGION_SUM')]
        )
        sums = stats_image.reduceRegion(
            reducer="sum", geometry=region, crs=self._ee_proj, scale=scale, bestEffort=True, maxPixels=1e6
        ).rename(['FILL_MASK', 'CLOUDLESS_MASK'], ['FILL_PORTION', 'CLOUDLESS_PORTION'])
        fill_portion = ee.Number(sums.get('FILL_PORTION')).divide(ee.Number(sums.get('REGION_SUM'))).multiply(100)
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
        """Apply the cloud/shadow mask."""
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

    @property
    def _ee_proj(self) -> ee.Projection:
        if self._ee_projection is None:
            # use the default projection (all landsat bands have same scale)
            self._ee_projection = self._ee_image.projection()
        return self._ee_projection

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
    """Base class for cloud/shadow masking of Sentinel-2 TOA and SR images."""

    @property
    def _ee_proj(self) -> ee.Projection:
        if self._ee_projection is None:
            # use the B1 projection with maximum scale (60m) to reduce processing times (some S2 images have empty QA
            # bands with no fixed projection, so utils.get_projection(min_scale=False) should not be used here).
            self._ee_projection = self._ee_image.select(0).projection()
        return self._ee_projection

    def _aux_image(
        self,
        s2_toa: bool = False,
        mask_cirrus: bool = True,
        mask_shadows: bool = True,
        mask_method: CloudMaskMethod = CloudMaskMethod.cloud_score,
        prob: float = 60,
        dark: float = 0.15,
        shadow_dist: int = 1000,
        buffer: int = 50,
        cdi_thresh: float = None,
        max_cloud_dist: int = 5000,
        score: float = 0.6,
        cs_band: CloudScoreBand = CloudScoreBand.cs,
    ) -> ee.Image:
        """Derive cloud, shadow and validity masks for the encapsulated image.

        Parts adapted from https://github.com/r-earthengine/ee_extra, under Apache 2.0 license.
        """
        mask_method = CloudMaskMethod(mask_method)
        if mask_method is not CloudMaskMethod.cloud_score:
            warnings.warn(
                f"The '{mask_method}' mask method is deprecated and will be removed in a future release.  Please "
                f"switch to 'cloud-score'.",
                category=FutureWarning,
            )

        def match_image(ee_image: ee.Image, collection: str, band: str, match_prop: str = 'system:index') -> ee.Image:
            """Return an image from ``collection`` matching ``ee_image`` with single ``band`` selected, or a fully
            masked image if no match is found.
            """
            # default fully masked image
            default = ee.Image().updateMask(0)
            default = default.rename(band)

            # find matching image
            filt = ee.Filter.eq(match_prop, ee_image.get(match_prop))
            match = ee.ImageCollection(collection).filter(filt).first()

            # revert to default if no match found
            match = ee.Image(ee.List([match, default]).reduce(ee.Reducer.firstNonNull()))
            return match.select(band)

        def cloud_cast_shadow_mask(ee_image: ee.Image, cloud_mask: ee.Image) -> ee.Image:
            """Create & return a shadow mask for ``ee_image`` by projecting shadows from ``cloud_mask``.

            Adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless.
            """
            dark_mask = ee_image.select('B8').lt(dark * 1e4)
            if not s2_toa:
                # exclude water
                dark_mask = ee_image.select('SCL').neq(6).And(dark_mask)

            # Find shadow direction.  Note that the angle convention depends on the reproject args below.
            shadow_azimuth = ee.Number(90).subtract(ee.Number(ee_image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

            # Project the cloud mask in the shadow direction (can project the mask into invalid areas).
            proj_pixels = ee.Number(shadow_dist).divide(self._ee_proj.nominalScale()).round()
            cloud_cast_proj = cloud_mask.directionalDistanceTransform(shadow_azimuth, proj_pixels).select('distance')

            # Reproject to force calculation at the correct scale.
            cloud_cast_mask = cloud_cast_proj.mask().reproject(crs=self._ee_proj, scale=self._ee_proj.nominalScale())

            # Remove any projections in invalid areas.
            cloud_cast_mask = cloud_cast_mask.updateMask(cloud_mask.mask())

            # Find shadow mask as intersection between projected clouds and dark areas.
            shadow_mask = cloud_cast_mask.And(dark_mask)

            return shadow_mask.rename('SHADOW_MASK')

        def qa_cloud_mask(ee_image: ee.Image) -> ee.Image:
            """Create & return a cloud mask for ``ee_image`` using the QA60 band."""
            # mask QA60 if it is between Feb 2022 - Feb 2024 to invalidate the cloud mask (QA* bands are not populated
            # over this period and may be masked / transparent or zero).
            qa_valid = (
                ee_image.date()
                .difference(ee.Date('2022-02-01'), 'days')
                .lt(0)
                .Or(ee_image.date().difference(ee.Date('2024-02-01'), 'days').gt(0))
            )
            qa = ee_image.select('QA60').updateMask(qa_valid)

            cloud_mask = qa.bitwiseAnd(1 << 10).neq(0)
            if mask_cirrus:
                cloud_mask = cloud_mask.Or(qa.bitwiseAnd(1 << 11).neq(0))

            return cloud_mask.rename('CLOUD_MASK')

        def cloud_prob_cloud_mask(ee_image: ee.Image) -> tuple[ee.Image, ee.Image]:
            """Return the cloud probability thresholded cloud mask, and cloud probability image for ``ee_image``."""
            cloud_prob = match_image(ee_image, 'COPERNICUS/S2_CLOUD_PROBABILITY', 'probability')
            cloud_mask = cloud_prob.gte(prob)
            return cloud_mask.rename('CLOUD_MASK'), cloud_prob.rename('CLOUD_PROB')

        def cloud_score_cloud_shadow_mask(ee_image: ee.Image) -> tuple[ee.Image, ee.Image]:
            """Return the cloud score thresholded cloud mask, and cloud score image for ``ee_image``."""
            cloud_score = match_image(ee_image, 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED', cs_band.name)
            cloud_shadow_mask = cloud_score.lte(score)
            return cloud_shadow_mask.rename('CLOUD_SHADOW_MASK'), cloud_score.rename('CLOUD_SCORE')

        def cdi_cloud_mask(ee_image: ee.Image) -> ee.Image:
            """Return the CDI cloud mask for ``ee_image``.
            See https://developers.google.com/earth-engine/apidocs/ee-algorithms-sentinel2-cdi for detail.
            """
            if s2_toa:
                s2_toa_image = ee_image
            else:
                # get the Sentinel-2 TOA image that corresponds to ``ee_image``
                idx = ee_image.get('system:index')
                s2_toa_image = ee.ImageCollection('COPERNICUS/S2').filter(ee.Filter.eq('system:index', idx)).first()
            cdi_image = ee.Algorithms.Sentinel2.CDI(s2_toa_image)
            return cdi_image.lt(cdi_thresh).rename('CDI_CLOUD_MASK')

        def cloud_shadow_masks(ee_image: ee.Image) -> dict[str, ee.Image]:
            """Return a dictionary of cloud/shadow masks & related images for ``ee_image``."""
            aux_bands = {}
            if mask_method is CloudMaskMethod.cloud_score:
                aux_bands['cloud_shadow'], aux_bands['score'] = cloud_score_cloud_shadow_mask(ee_image)
            else:
                if mask_method is CloudMaskMethod.qa:
                    aux_bands['cloud'] = qa_cloud_mask(ee_image)
                else:
                    aux_bands['cloud'], aux_bands['prob'] = cloud_prob_cloud_mask(ee_image)

                if cdi_thresh is not None:
                    aux_bands['cloud'] = aux_bands['cloud'].And(cdi_cloud_mask(ee_image))

                aux_bands['shadow'] = cloud_cast_shadow_mask(ee_image, aux_bands['cloud'])

                if mask_shadows:
                    aux_bands['cloud_shadow'] = aux_bands['cloud'].Or(aux_bands['shadow'])
                else:
                    aux_bands['cloud_shadow'] = aux_bands['cloud']
                    aux_bands.pop('shadow', None)

                # do a morphological opening that removes small (20m) blobs from the mask and then dilates
                aux_bands['cloud_shadow'] = (
                    aux_bands['cloud_shadow'].focal_min(20, units='meters').focal_max(buffer, units='meters')
                )
            return aux_bands

        # get cloud/shadow etc bands
        aux_bands = cloud_shadow_masks(self.ee_image)

        # derive a fill mask from the Earth Engine mask for the surface reflectance bands
        aux_bands['fill'] = self.ee_image.select('B.*').mask().reduce(ee.Reducer.allNonZero())

        # clip fill mask to the image footprint (without this step we get memory limit errors on download)
        aux_bands['fill'] = aux_bands['fill'].clip(self.ee_image.geometry()).rename('FILL_MASK')

        # combine masks into cloudless_mask
        aux_bands['cloudless'] = (aux_bands['cloud_shadow'].Not()).And(aux_bands['fill']).rename('CLOUDLESS_MASK')
        aux_bands.pop('cloud_shadow')

        # construct and return the auxiliary image
        aux_bands['dist'] = self._cloud_dist(cloudless_mask=aux_bands['cloudless'], max_cloud_dist=max_cloud_dist)
        return ee.Image(list(aux_bands.values()))


class Sentinel2SrClImage(Sentinel2ClImage):
    """Class for cloud/shadow masking of Sentinel-2 SR (COPERNICUS/S2_SR & COPERNICUS/S2_SR_HARMONIZED) images."""

    def _aux_image(self, s2_toa: bool = False, **kwargs) -> ee.Image:
        return Sentinel2ClImage._aux_image(self, s2_toa=False, **kwargs)


class Sentinel2ToaClImage(Sentinel2ClImage):
    """Class for cloud/shadow masking of Sentinel-2 TOA (COPERNICUS/S2 & COPERNICUS/S2_HARMONIZED) images."""

    def _aux_image(self, s2_toa: bool = False, **kwargs) -> ee.Image:
        return Sentinel2ClImage._aux_image(self, s2_toa=True, **kwargs)


def class_from_id(image_id: str) -> type:
    """Return the *Image class that corresponds to the provided Earth Engine image/collection ID."""
    ee_coll_name, _ = split_id(image_id)
    if image_id in geedim.schema.collection_schema:
        return geedim.schema.collection_schema[image_id]['image_type']
    elif ee_coll_name in geedim.schema.collection_schema:
        return geedim.schema.collection_schema[ee_coll_name]['image_type']
    else:
        return MaskedImage
