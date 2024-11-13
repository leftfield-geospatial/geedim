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
from abc import ABCMeta
from functools import cached_property

import ee

from geedim import schema
from geedim.download import BaseImage
from geedim.enums import CloudMaskMethod, CloudScoreBand, ResamplingMethod
from geedim.utils import split_id, register_accessor

logger = logging.getLogger(__name__)




class _MaskedImage(ABCMeta):
    @staticmethod
    def _get_mask_bands(ee_image: ee.Image) -> dict[str, ee.Image]:
        # TODO: this (and wherever FILL_MASK is found) uses the projection of the first band.  which
        #  could be non-fixed, or max/min/? scale.
        fill_mask = ee_image.mask().reduce(ee.Reducer.allNonZero()).rename('FILL_MASK')
        return dict(fill=fill_mask)

    @classmethod
    def add_mask_bands(cls, ee_image: ee.Image, **kwargs) -> ee.Image:
        # TODO: this (and wherever FILL_MASK is found) uses the projection of the first band.  which
        #  could be non-fixed, or max/min/? scale.
        # TODO: should this (and all other **kwargs fns) take *args and **kwargs in case kwargs are
        #  passed as args?
        mask_bands = cls._get_mask_bands(ee_image, **kwargs)
        no_mask_bands = ee.Number(ee_image.bandNames().indexOf(ee.String('FILL_MASK')).lt(0))
        # overwrite unless it is a composite image with existing aux bands
        # TODO: add without overwrite then select original + aux bands to remove any added
        overwrite = no_mask_bands.Or(ee_image.gd.fixed())
        return ee.Image(
            ee.Algorithms.If(
                overwrite, ee_image.addBands(list(mask_bands.values()), overwrite=True), ee_image
            )
        )

    @staticmethod
    def mask_clouds(ee_image: ee.Image) -> ee.Image:
        return ee_image.updateMask(ee_image.select('FILL_MASK'))

    @staticmethod
    def set_mask_portions(
        ee_image: ee.Image, region: dict | ee.Geometry = None, scale: float | ee.Number = None
    ) -> ee.Image:
        portions = ee_image.select('FILL_MASK').gd.maskCoverRegion(
            region=region, scale=scale, maxPixels=1e6
        )
        return ee_image.set('FILL_PORTION', portions.get('FILL_MASK'), 'CLOUDLESS_PORTION', 100)


class _CloudlessImage(_MaskedImage):
    # TODO: make this class / _get_mask_bands method abstract

    @staticmethod
    def mask_clouds(ee_image: ee.Image) -> ee.Image:
        return ee_image.updateMask(ee_image.select('CLOUDLESS_MASK'))

    @staticmethod
    def set_mask_portions(
        ee_image: ee.Image, region: dict | ee.Geometry = None, scale: float | ee.Number = None
    ) -> ee.Image:
        portions = ee_image.select(['FILL_MASK', 'CLOUDLESS_MASK']).gd.maskCoverRegion(
            region=region, scale=scale, maxPixels=1e6
        )
        fill_portion = ee.Number(portions.get('FILL_MASK'))
        cl_portion = ee.Number(portions.get('CLOUDLESS_MASK')).divide(fill_portion).multiply(100)
        return ee_image.set('FILL_PORTION', fill_portion, 'CLOUDLESS_PORTION', cl_portion)


class _LandsatImage(_CloudlessImage):
    @staticmethod
    def _get_mask_bands(
        ee_image: ee.Image,
        mask_shadows: bool = True,
        mask_cirrus: bool = True,
        max_cloud_dist: float = 5000,
    ) -> dict[str, ee.Image]:
        """Return an image of cloud, shadow and validity masks for the given Landsat C2 SR image."""
        # TODO: add support for landsat TOA images
        qa_pixel = ee_image.select('QA_PIXEL')

        # construct fill mask from Earth Engine mask and QA_PIXEL
        ee_mask = ee_image.select('SR_B.*').mask().reduce(ee.Reducer.allNonZero())
        fill_mask = qa_pixel.bitwiseAnd(1).eq(0).And(ee_mask).rename('FILL_MASK')

        # find cloud & shadow masks from QA_PIXEL band
        shadow_mask = qa_pixel.bitwiseAnd(0b10000).neq(0).rename('SHADOW_MASK')
        if mask_cirrus:
            cloud_mask = qa_pixel.bitwiseAnd(0b1100).neq(0).rename('CLOUD_MASK')
        else:
            cloud_mask = qa_pixel.bitwiseAnd(0b1000).neq(0).rename('CLOUD_MASK')

        # combine cloud, shadow and fill masks into cloudless mask
        cloud_shadow_mask = (cloud_mask.Or(shadow_mask)) if mask_shadows else cloud_mask
        cloudless_mask = cloud_shadow_mask.Not().And(fill_mask).rename('CLOUDLESS_MASK')

        # convert cloud distance from existing ST_CDIST band (10m units) to meters, and clamp to
        # max_cloud_dist
        cloud_dist = ee_image.select('ST_CDIST').multiply(10).rename('CLOUD_DIST')
        cloud_dist = cloud_dist.clamp(0, max_cloud_dist).toUint16()

        # return ee.Image([fill_mask, cloud_mask, shadow_mask, cloudless_mask, cloud_dist])
        return dict(
            fill=fill_mask,
            cloud=cloud_mask,
            shadow=shadow_mask,
            cloudless=cloudless_mask,
            dist=cloud_dist,
        )


class _Sentinel2Image(_CloudlessImage):
    @staticmethod
    def _get_cloud_dist(cloudless_mask: ee.Image, max_cloud_dist: float = 5000) -> ee.Image:
        """Return a cloud/shadow distance image (m) for the given cloudless mask."""
        # TODO: previously this used a 60m scale for S2 - does S2 q-mosaic compositing with cloud-prob
        #  mask method work ok?  the shadow mask should be at 60m, as that is reprojected,
        #  i don't the reproject here will interfere.
        # use the projection & scale of cloudless_mask for the distance image
        proj = cloudless_mask.projection()

        # max_cloud_dist in pixels
        max_cloud_pix = ee.Number(max_cloud_dist).divide(proj.nominalScale()).round()

        # Find distance to nearest cloud/shadow (m).  Distances are found for all pixels, including
        # masked / invalid pixels, which are treated as 0 (non cloud/shadow).
        cloud_shadow_mask = cloudless_mask.Not()
        cloud_dist = (
            cloud_shadow_mask.fastDistanceTransform(
                neighborhood=max_cloud_pix, units='pixels', metric='squared_euclidean'
            )
            .sqrt()
            .multiply(proj.nominalScale())
        )

        # reproject to force calculation at correct scale
        cloud_dist = cloud_dist.reproject(crs=proj, scale=proj.nominalScale()).rename('CLOUD_DIST')

        # prevent use of invalid pixels
        cloud_dist = cloud_dist.updateMask(cloudless_mask.mask())

        # clamp cloud_dist to max_cloud_dist
        cloud_dist = cloud_dist.clamp(0, max_cloud_dist)

        # cloud_dist is float64 by default, so convert to Uint16 here to avoid forcing the whole
        # image to float64 on download.
        return cloud_dist.toUint16().rename('CLOUD_DIST')

    @staticmethod
    def _get_mask_bands(
        ee_image: ee.Image,
        s2_toa: bool = False,
        mask_cirrus: bool = True,
        mask_shadows: bool = True,
        mask_method: str | CloudMaskMethod = CloudMaskMethod.cloud_score,
        prob: float = 60,
        dark: float = 0.15,
        shadow_dist: float = 1000,
        buffer: int = 50,
        cdi_thresh: float = None,
        max_cloud_dist: float = 5000,
        score: float = 0.6,
        cs_band: str | CloudScoreBand = CloudScoreBand.cs,
    ) -> dict[str, ee.Image]:
        """Return an image of cloud, shadow and validity masks for the given Sentinel-2 image.

        Parts adapted from https://github.com/r-earthengine/ee_extra, under Apache 2.0 license.
        """
        mask_method = CloudMaskMethod(mask_method)
        if mask_method is not CloudMaskMethod.cloud_score:
            warnings.warn(
                f"The '{mask_method}' mask method is deprecated and will be removed in a future "
                f"release.  Please switch to 'cloud-score'.",
                category=FutureWarning,
            )
        cs_band = CloudScoreBand(cs_band)

        def match_image(
            ee_image: ee.Image, collection: str, band: str, match_prop: str = 'system:index'
        ) -> ee.Image:
            """Return an image from ``collection`` matching ``ee_image`` with single ``band``
            selected, or a fully masked image if no match is found.
            """
            # TODO: would it be possible (and faster) to use ee.ImageCollection.linkCollection,
            #  for the ImageCollection accessor rather than linking per image?
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

            Adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2
            -s2cloudless.
            """
            # use the 60m B1 projection for shadow mask
            proj = ee_image.select('B1').projection()

            dark_mask = ee_image.select('B8').lt(dark * 1e4)
            if not s2_toa:
                # exclude water
                dark_mask = ee_image.select('SCL').neq(6).And(dark_mask)

            # Find shadow direction.  Note that the angle convention depends on the reproject args
            # below.
            shadow_azimuth = ee.Number(90).subtract(
                ee.Number(ee_image.get('MEAN_SOLAR_AZIMUTH_ANGLE'))
            )

            # Project the cloud mask in the shadow direction (can project the mask into invalid areas).
            proj_pixels = ee.Number(shadow_dist).divide(proj.nominalScale()).round()
            cloud_cast_proj = cloud_mask.directionalDistanceTransform(
                shadow_azimuth, proj_pixels
            ).select('distance')

            # Reproject to force calculation at the correct scale.
            cloud_cast_mask = cloud_cast_proj.mask().reproject(crs=proj, scale=proj.nominalScale())

            # Remove any projections in invalid areas.
            cloud_cast_mask = cloud_cast_mask.updateMask(cloud_mask.mask())

            # Find shadow mask as intersection between projected clouds and dark areas.
            shadow_mask = cloud_cast_mask.And(dark_mask)

            return shadow_mask.rename('SHADOW_MASK')

        def qa_cloud_mask(ee_image: ee.Image) -> ee.Image:
            """Create & return a cloud mask for ``ee_image`` using the QA60 band."""
            # mask QA60 if it is between Feb 2022 - Feb 2024 to invalidate the cloud mask (QA* bands
            # are not populated over this period and may be masked / transparent or zero).
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
            """Return the cloud probability thresholded cloud mask, and cloud probability image for
            ``ee_image``.
            """
            cloud_prob = match_image(ee_image, 'COPERNICUS/S2_CLOUD_PROBABILITY', 'probability')
            cloud_mask = cloud_prob.gte(prob)
            return cloud_mask.rename('CLOUD_MASK'), cloud_prob.rename('CLOUD_PROB')

        def cloud_score_cloud_shadow_mask(ee_image: ee.Image) -> tuple[ee.Image, ee.Image]:
            """Return the cloud score thresholded cloud mask, and cloud score image for ``ee_image``."""
            cloud_score = match_image(
                ee_image, 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED', cs_band.name
            )
            cloud_shadow_mask = cloud_score.lte(score)
            return cloud_shadow_mask.rename('CLOUD_SHADOW_MASK'), cloud_score.rename('CLOUD_SCORE')

        def cdi_cloud_mask(ee_image: ee.Image) -> ee.Image:
            """Return the CDI cloud mask for ``ee_image``.

            See https://developers.google.com/earth-engine/apidocs/ee-algorithms-sentinel2-cdi for
            detail.
            """
            if s2_toa:
                s2_toa_image = ee_image
            else:
                # get the Sentinel-2 TOA image that corresponds to ``ee_image``
                idx = ee_image.get('system:index')
                s2_toa_image = (
                    ee.ImageCollection('COPERNICUS/S2')
                    .filter(ee.Filter.eq('system:index', idx))
                    .first()
                )
            cdi_image = ee.Algorithms.Sentinel2.CDI(s2_toa_image)
            return cdi_image.lt(cdi_thresh).rename('CDI_CLOUD_MASK')

        def cloud_shadow_masks(ee_image: ee.Image) -> dict[str, ee.Image]:
            """Return a dictionary of cloud/shadow masks & related images for ``ee_image``."""
            aux_bands = {}
            if mask_method is CloudMaskMethod.cloud_score:
                aux_bands['cloud_shadow'], aux_bands['score'] = cloud_score_cloud_shadow_mask(
                    ee_image
                )
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

                # do a morphological opening that removes small (20m) blobs from the mask and then
                # dilates
                aux_bands['cloud_shadow'] = (
                    aux_bands['cloud_shadow']
                    .focal_min(20, units='meters')
                    .focal_max(buffer, units='meters')
                )
            return aux_bands

        # get cloud/shadow etc bands
        aux_bands = cloud_shadow_masks(ee_image)

        # derive a fill mask from the Earth Engine mask for the surface reflectance bands
        aux_bands['fill'] = (
            ee_image.select('B.*').mask().reduce(ee.Reducer.allNonZero()).rename('FILL_MASK')
        )

        # clip fill mask to the image footprint (without this step we get memory limit errors on
        # download)
        # TODO: is clipping really necessary - should be avoided if possible (does not work on
        #  composites and is not best practice)
        # aux_bands['fill'] = aux_bands['fill'].clip(ee_image.geometry()).rename('FILL_MASK')

        # combine masks into cloudless_mask
        aux_bands['cloudless'] = (
            (aux_bands['cloud_shadow'].Not()).And(aux_bands['fill']).rename('CLOUDLESS_MASK')
        )
        aux_bands.pop('cloud_shadow')

        # construct and return the auxiliary image
        aux_bands['dist'] = _Sentinel2Image._get_cloud_dist(
            cloudless_mask=aux_bands['cloudless'], max_cloud_dist=max_cloud_dist
        )
        return aux_bands


class _Sentinel2ToaImage(_CloudlessImage):
    @staticmethod
    def _get_mask_bands(ee_image: ee.Image, **kwargs):
        return _Sentinel2Image._get_mask_bands(ee_image, s2_toa=True, **kwargs)


class _Sentinel2SrImage(_CloudlessImage):
    @staticmethod
    def _get_mask_bands(ee_image: ee.Image, **kwargs):
        return _Sentinel2Image._get_mask_bands(ee_image, s2_toa=False, **kwargs)

def class_from_id(image_id: str) -> type[_MaskedImage]:
    """Return the *Image class that corresponds to the provided Earth Engine image/collection ID."""
    ee_coll_name, _ = split_id(image_id)
    if image_id in schema.collection_schema:
        return schema.collection_schema[image_id]['image_type']
    elif ee_coll_name in schema.collection_schema:
        return schema.collection_schema[ee_coll_name]['image_type']
    else:
        return _MaskedImage



@register_accessor('gd', ee.Image)
class ImageAccessor(BaseImage):
    def __init__(self, ee_image: ee.Image):
        super().__init__(ee_image)
        # TODO: can _id be copied from ee_coll.gd here? Or this is called from descriptor
        #  creation in register_accessor decorator and accessing .gd would be recursive?  What
        #  about a decorator that would

    @staticmethod
    def load(image_id: str) -> ee.Image:
        # TODO: fromId?
        ee_image = ee.Image(image_id)
        ee_image.gd._id = image_id
        return ee_image

    @cached_property
    def _mi(self) -> type[_MaskedImage]:
        return class_from_id(self.id)

    def projection(self, min_scale: bool = True) -> ee.Projection:
        # TODO: minProjection?
        bands = self.ee_image.bandNames()
        scales = bands.map(
            lambda band: self._ee_image.select(ee.String(band)).projection().nominalScale()
        )
        projs = bands.map(lambda band: self.ee_image.select(ee.String(band)).projection())
        projs = projs.sort(scales)
        return ee.Projection(projs.get(0) if min_scale else projs.get(-1))

    def fixed(self) -> ee.Number:
        proj = self.projection()
        # cannot use ee.String.compareTo or ee.String.equals with proj.crs() when it is null
        not_wgs84 = ee.List([proj.crs()]).indexOf(ee.String('EPSG:4326')).eq(-1)
        not_degree_scale = proj.nominalScale().toInt64().neq(111319)
        return not_wgs84.Or(not_degree_scale)

    def resample(self, method: ResamplingMethod | str) -> ee.Image:
        method = ResamplingMethod(method)
        if method == ResamplingMethod.near:
            return self._ee_image

        # resample the image, if it has a fixed projection
        proj = self.projection(min_scale=True)

        def _resample(ee_image: ee.Image) -> ee.Image:
            """Resample the given image, allowing for additional 'average' method."""
            if method == ResamplingMethod.average:
                # set the default projection to the minimum scale projection (required for e.g.
                # S2 images that have non-fixed projection bands)
                # TODO: test this works for different res S2 bands
                ee_image = ee_image.setDefaultProjection(proj)
                return ee_image.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
            else:
                return ee_image.resample(method.value)

        # TODO: refactor without If.  Perhaps map has_fixed_proj with dropNulls=True, or map
        #  has_fixed_proj and filter.
        return ee.Image(ee.Algorithms.If(self.fixed(), _resample(self._ee_image), self._ee_image))

    def maskCoverRegion(
        self,
        region: dict | ee.Geometry,
        scale: float | ee.Number = None,
        bands: str | ee.String | list[str] | ee.List = None,
        **kwargs,
    ) -> ee.Dictionary:
        # TODO: add crs kwarg for consistency?
        # TODO: drop bands kwarg?
        mask_image = self.ee_image.select(bands) if bands else self.ee_image
        region = region or self.ee_image.geometry()  # use the image footprint

        # composite images have no fixed projection (scale = 1deg) and need the scale kwarg
        # TODO: default crs/scale to max scale band of ee_image/mask_image?  If the image includes
        #  non-fixed bands, then I think FILL_MASK will also be non-fixed.  With S2, which can have
        #  non-fixed bands, the FILL_MASK is formed from SR bands only, so it is OK.  But with other
        #  images, FILL_MASK is made from all bands.  Another option is to put scales in schema,
        #  or use STAC.  There's not a strong case for using STAC over getInfo though I think,
        #  both require network requests, only that S2 non-fixed / masked bands have proper scales
        #  in STAC.

        proj = mask_image.gd.projection(min_scale=True)
        scale = scale or proj.nominalScale()  # mask_image.gd.projection().nominalScale()

        # Find the fill portion as the (sum over the region of FILL_MASK) divided by (sum over the region of a constant
        # image (==1)).  Note that a mean reducer, does not find the mean over the region, but the mean over the part
        # of the region covered by the image.
        # TODO: can this be made to work on non-mask bands by taking .mask().unmask()?
        stats_image = ee.Image([mask_image.unmask(), ee.Image(1).rename('ttl')])
        # Note: sometimes proj has no EPSG in crs(), hence use crs=proj and not crs=proj.crs() below
        sums = stats_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            crs=proj,  # TODO: test omitting on composite
            scale=scale,
            bestEffort=True,
            # maxPixels=1e6,
            **kwargs,
        )

        ttl = ee.Number(sums.values().get(-1))
        sums = sums.remove([ee.String('ttl')])

        def get_portion(key: ee.String, value: ee.Number) -> ee.Number:
            return ee.Number(value).divide(ttl).multiply(100)

        return sums.map(get_portion)

    def addMaskBands(self, **kwargs) -> ee.Image:
        # TODO: this will keep adding extra aux bands by default.  Is this compatible with old
        #  MaskedImage expected behaviour?  For a composite with existing aux bands, this will
        #  add extra (unusable) aux bands with overwrite=False, but perhaps that doesn't matter
        #  as the original aux bands will get used in maskClouds, and the added aux bands would
        #  be excluded from a further composite.  If the added aux bands for a composite generate
        #  errors, they should not be added of course, but I don't think this is the case.
        ee_image = self._mi.add_mask_bands(self.ee_image, **kwargs)
        ee_image.gd._id = self._id
        return ee_image

    def maskClouds(self, **kwargs) -> ee.Image:
        # TODO: is it more efficient to only get mask, excluding cloud distance etc
        # TODO: do we want to mask with FILL_MASK when there is no CLOUDLESS_MASK,
        #  and incorporate it into the CLOUDLESS_MASK when there is?  FILL_MASK is used for
        #  search filtering on fill portion property, but is it useful to mask with it over EE
        #  mask? If FILL_MASK masking is abandoned, we should test the e.g. Landsat-7 mask
        #  defines the invalid areas.
        # TODO: in some cases (e.g S2) images have fully masked bands which means a default
        #  FILL_MASK is also fully masked.
        # TODO: neither this nor addAuxBands can be mapped over an ImageCollection as they
        #  require getInfo
        ee_image = self._mi.mask_clouds(self.ee_image)
        ee_image.gd._id = self._id
        return ee_image


class MaskedImage(ImageAccessor):
    _default_mask = False

    def __init__(
        self,
        ee_image: ee.Image,
        mask: bool = _default_mask,
        region: dict | ee.Geometry = None,
        **kwargs,
    ):
        """
        A class for describing, masking and downloading an Earth Engine image.

        Parameters
        ----------
        ee_image: ee.Image
            Earth Engine image to encapsulate.
        mask: bool, optional
            Whether to mask the image.
        region: dict, ee.Geometry, optional
            Region in which to find statistics for the image, as a GeoJSON dictionary or
            ``ee.Geometry``.  Statistics are stored in the image properties. If None, statistics
            are not found (the default).

        **kwargs
            Cloud/shadow masking parameters - see below:
        mask_cirrus: bool, optional
            Whether to mask cirrus clouds.  Valid for Landsat 8-9 images, and for Sentinel-2
            images with the `qa` ``mask_method``.
        mask_shadows: bool, optional
            Whether to mask cloud shadows.  Valid for Landsat images, and for Sentinel-2 images
            with the `qa` or `cloud-prob` ``mask_method``.
        mask_method: CloudMaskMethod, str, optional
            Method used to mask clouds.  Valid for Sentinel-2 images.  See
            :class:`~geedim.enums.CloudMaskMethod` for details.
        prob: float, optional
            Cloud probability threshold (%). Valid for Sentinel-2 images with the `cloud-prob`
            ``mask_method``.
        dark: float, optional
            NIR threshold [0-1]. NIR values below this threshold are potential cloud shadows.
            Valid for Sentinel-2 images with the `qa` or `cloud-prob` ``mask_method``.
        shadow_dist: int, optional
            Maximum distance (m) to look for cloud shadows from cloud edges.  Valid for Sentinel-2
            images with the `qa` or `cloud-prob` ``mask_method``.
        buffer: int, optional
            Distance (m) to dilate cloud/shadow.  Valid for Sentinel-2 images with the `qa` or
            `cloud-prob` ``mask_method``.
        cdi_thresh: float, optional
            Cloud Displacement Index threshold. Values below this threshold are considered
            potential clouds. If this parameter is not specified (=None), the index is not used.
            Valid for Sentinel-2 images with the `qa` or `cloud-prob` ``mask_method``.  See
            https://developers.google.com/earth-engine/apidocs/ee-algorithms-sentinel2-cdi for
            details.
        max_cloud_dist: int, optional
            Maximum distance (m) to look for clouds when forming the 'cloud distance' band.  Valid
            for Sentinel-2 images.
        score: float, optional
            Cloud Score+ threshold.  Valid for Sentinel-2 images with the `cloud-score`
            ``mask_method``.
        cs_band: CloudScoreBand, str, optional
            Cloud Score+ band to threshold.  Valid for Sentinel-2 images with the `cloud-score`
            ``mask_method``.
        """
        super().__init__(ee_image)
        self._cloud_kwargs = kwargs
        self.ee_image = self.addMaskBands(**kwargs)
        if region:
            # TODO: is reusing aux_bands above faster? e.g.:
            #  aux_bands = self._get_aux_bands(**kwargs)
            #  self.ee_image = self.ee_image.addBands(list(aux_bands.values()), overwrite=True)
            #  if region:
            #   self.ee_image = set_fill_and_cloudless_portions(
            #     self.ee_image, aux_bands=aux_bands, region=region
            #   )
            self._set_region_stats(region)
        if mask:
            self.mask_clouds()

    @property
    def _ee_proj(self) -> ee.Projection:
        """Projection to use for mask calculations and statistics."""
        # TODO: remove when tests updated + other internal APIs
        return self.ee_image.select(0).projection()

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
        ee_image = ee.Image(image_id)
        gd_image = MaskedImage(ee_image, **kwargs)
        # TODO: the id has already been retrieved and used in __init__
        # set the id attribute (avoids a call to getInfo() for .id property)
        gd_image._id = image_id
        return gd_image

    def _set_region_stats(self, region: dict | ee.Geometry = None, scale: float = None):
        """
        Set FILL_PORTION and CLOUDLESS_PORTION on the encapsulated image for the specified
        region.  Derived classes should override this method and set CLOUDLESS_PORTION,
        and/or other statistics they support.

        Parameters
        ----------
        region: dict, ee.Geometry, optional
            Region in which to find statistics for the image, as a GeoJSON dictionary or
            ``ee.Geometry``.  If not specified, the image footprint is used.
        scale: float, optional
            Re-project to this scale when finding statistics.  Defaults to the scale of
            :attr:`~MaskedImage._ee_proj`.  Should be provided if the encapsulated image is a
            composite / without a fixed projection.
        """
        # TODO: remove
        self.ee_image = self._mi.set_mask_portions(self.ee_image, region=region, scale=scale)

    def mask_clouds(self):
        """Apply the cloud/shadow mask if supported, otherwise apply the fill mask."""
        # TODO: every time self.ee_image is assigned to, the ee_info & id is invalidated and will
        #  have to be retrieved on the next mask_clouds.  is it worth keeping the id property
        #  separate from ee_info and the assoc from_id method?  Or should we set it whenever we
        #  make a change to ee_image that doesn't change the ID i.e. mask_clouds and add_aux_bands
        self.ee_image = self._mi.mask_clouds(self.ee_image)

