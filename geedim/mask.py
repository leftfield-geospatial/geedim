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
from abc import abstractmethod

import ee

from geedim import schema
from geedim.download import BaseImage
from geedim.enums import CloudMaskMethod, CloudScoreBand
from geedim.image import ImageAccessor
from geedim.utils import split_id

logger = logging.getLogger(__name__)


class _MaskedImage:
    """Masking method container for images without cloud/shadow support."""

    @staticmethod
    def _get_mask_bands(ee_image: ee.Image) -> dict[str, ee.Image]:
        """Return a dictionary of masks & related images for the given image."""
        # note that fill_mask uses the projection of the first ee_image band
        fill_mask = ee_image.mask().reduce(ee.Reducer.allNonZero()).rename('FILL_MASK')
        return dict(fill=fill_mask)

    @classmethod
    def add_mask_bands(cls, ee_image: ee.Image, **kwargs) -> ee.Image:
        """Return the given image with cloud/shadow masks and related bands added when supported,
        otherwise with fill (validity) mask added. Mask bands are overwritten if they exist,
        except on images without fixed projections, in which case no bands are added or overwritten.
        """
        mask_bands = cls._get_mask_bands(ee_image, **kwargs)
        # add/overwrite mask bands unless the image has no fixed projection
        add = ImageAccessor(ee_image).fixed()
        return ee.Image(
            ee.Algorithms.If(
                add, ee_image.addBands(list(mask_bands.values()), overwrite=True), ee_image
            )
        )

    @staticmethod
    def mask_clouds(ee_image: ee.Image) -> ee.Image:
        """Return the given image with cloud/shadow masks applied when supported, otherwise with
        fill (validity) mask applied.  Mask bands should be added with :meth:`add_mask_bands`
        before calling this method.
        """
        return ee_image.updateMask(ee_image.select('FILL_MASK'))

    @staticmethod
    def set_mask_portions(
        ee_image: ee.Image, region: dict | ee.Geometry = None, scale: float | ee.Number = None
    ) -> ee.Image:
        """Return the given image with the ``FILL_PORTION`` property set to the filled percentage
        of the given region, and the ``CLOUDLESS_PORTION`` property set to the cloudless
        percentage of ``FILL_PORTION``.  ``CLOUDLESS_PORTION`` is set to ``100`` if cloud/shadow
        masking is not supported.
        """
        portions = ImageAccessor(ee_image.select('FILL_MASK')).regionCoverage(
            region=region, scale=scale, maxPixels=1e6, bestEffort=True
        )
        return ee_image.set('FILL_PORTION', portions.get('FILL_MASK'), 'CLOUDLESS_PORTION', 100)


class _CloudlessImage(_MaskedImage):
    """Abstract masking method container for images with cloud/shadow support."""

    @staticmethod
    @abstractmethod
    def _get_mask_bands(ee_image: ee.Image) -> dict[str, ee.Image]:
        pass

    @staticmethod
    def mask_clouds(ee_image: ee.Image) -> ee.Image:
        return ee_image.updateMask(ee_image.select('CLOUDLESS_MASK'))

    @staticmethod
    def set_mask_portions(
        ee_image: ee.Image, region: dict | ee.Geometry = None, scale: float | ee.Number = None
    ) -> ee.Image:
        # TODO: is this maxPixels value ok? it results in bestEffort using lower scales for the
        #  test region_10000ha
        portions = ImageAccessor(ee_image.select(['FILL_MASK', 'CLOUDLESS_MASK'])).regionCoverage(
            region=region, scale=scale, maxPixels=1e6, bestEffort=True
        )
        fill_portion = ee.Number(portions.get('FILL_MASK'))
        cl_portion = ee.Number(portions.get('CLOUDLESS_MASK')).divide(fill_portion).multiply(100)
        return ee_image.set('FILL_PORTION', fill_portion, 'CLOUDLESS_PORTION', cl_portion)


class _LandsatImage(_CloudlessImage):
    """Masking method container for Landsat level 2 collection 2 images."""

    @staticmethod
    def _get_mask_bands(
        ee_image: ee.Image,
        mask_shadows: bool = True,
        mask_cirrus: bool = True,
        max_cloud_dist: float = 5000,
    ) -> dict[str, ee.Image]:
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

        return dict(
            fill=fill_mask,
            cloud=cloud_mask,
            shadow=shadow_mask,
            cloudless=cloudless_mask,
            dist=cloud_dist,
        )


class _Sentinel2Image(_CloudlessImage):
    """Abstract masking method container for Sentinel-2 TOA / SR images."""

    @staticmethod
    def _get_cloud_dist(
        cloudless_mask: ee.Image, proj: ee.Projection = None, max_cloud_dist: float = 5000
    ) -> ee.Image:
        """Return a cloud/shadow distance (m) image for the given cloudless mask."""
        # TODO: previously this used a 60m scale for S2 - does S2 q-mosaic compositing with
        #  cloud-prob mask method work ok?
        # projection & scale of the distance image
        proj = proj or cloudless_mask.projection()
        scale = proj.nominalScale()

        # max_cloud_dist in pixels
        max_cloud_pix = ee.Number(max_cloud_dist).divide(scale).round()

        # Find distance to nearest cloud/shadow (m).  Distances are found for all pixels, including
        # masked / invalid pixels, which are treated as 0 (non cloud/shadow).
        cloud_shadow_mask = cloudless_mask.Not()
        cloud_dist = (
            cloud_shadow_mask.fastDistanceTransform(
                neighborhood=max_cloud_pix, units='pixels', metric='squared_euclidean'
            )
            .sqrt()
            .multiply(scale)
        )

        # reproject to force calculation at correct scale
        cloud_dist = cloud_dist.reproject(crs=proj).rename('CLOUD_DIST')

        # prevent use of invalid pixels
        cloud_dist = cloud_dist.updateMask(cloudless_mask.mask())

        # clamp cloud_dist to max_cloud_dist
        cloud_dist = cloud_dist.clamp(0, max_cloud_dist)

        # cloud_dist is float64 by default, so convert to Uint16 here to avoid forcing the whole
        # image to float64 on export.
        return cloud_dist.toUint16().rename('CLOUD_DIST')

    @staticmethod
    @abstractmethod
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
        cdi_thresh: float | None = None,
        max_cloud_dist: float = 5000,
        score: float = 0.6,
        cs_band: str | CloudScoreBand = CloudScoreBand.cs,
    ) -> dict[str, ee.Image]:
        """Return a dictionary of mask bands and related images for the given Sentinel-2 image.
        Parts adapted from https://github.com/r-earthengine/ee_extra, under Apache 2.0 licence.
        """
        mask_method = CloudMaskMethod(mask_method)
        if mask_method is not CloudMaskMethod.cloud_score:
            warnings.warn(
                f"The '{mask_method}' mask method is deprecated and will be removed in a future "
                f"release.  Please use the 'cloud-score' method instead.",
                category=FutureWarning,
                stacklevel=2,
            )
        cs_band = CloudScoreBand(cs_band)

        def match_image(
            ee_image: ee.Image, collection: str, band: str, match_prop: str = 'system:index'
        ) -> ee.Image:
            """Return an image from ``collection`` matching ``ee_image`` with a single ``band``
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
            """Return a shadow mask for ``ee_image`` based on the intersection of dark areas and
            projected shadows from ``cloud_mask``. Adapted from
            https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless.
            """
            # use the 60m B1 projection for shadow mask to save some computation
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

            # Project the cloud mask in the shadow direction (can project the mask into invalid
            # areas).
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
            """Return a cloud mask for ``ee_image`` derived from the QA60 band."""
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
            """Return the cloud score thresholded cloud mask, and cloud score image for
            ``ee_image``.
            """
            cloud_score = match_image(
                ee_image, 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED', cs_band.name
            )
            cloud_shadow_mask = cloud_score.lte(score)
            cloud_score = cloud_score.toFloat()
            return cloud_shadow_mask.rename('CLOUD_SHADOW_MASK'), cloud_score.rename('CLOUD_SCORE')

        def cdi_cloud_mask(ee_image: ee.Image) -> ee.Image:
            """Return the CDI cloud mask for ``ee_image``. See
            https://developers.google.com/earth-engine/apidocs/ee-algorithms-sentinel2-cdi for
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

        # TODO: check if clipping is necessary - should be avoided if possible
        # clip fill mask to the image footprint (without this step we get memory limit errors on
        # export)
        # aux_bands['fill'] = aux_bands['fill'].clip(ee_image.geometry()).rename('FILL_MASK')

        # combine masks into cloudless_mask
        aux_bands['cloudless'] = (
            (aux_bands['cloud_shadow'].Not()).And(aux_bands['fill']).rename('CLOUDLESS_MASK')
        )
        aux_bands.pop('cloud_shadow')

        # find cloud distance from cloudless mask
        aux_bands['dist'] = _Sentinel2Image._get_cloud_dist(
            aux_bands['cloudless'],
            # use 60m B1 projection for the cloud distance to save some computation
            proj=ee_image.select('B1').projection(),
            max_cloud_dist=max_cloud_dist,
        )
        return aux_bands


class _Sentinel2ToaImage(_CloudlessImage):
    """Masking method container for Sentinel-2 TOA images."""

    @staticmethod
    def _get_mask_bands(ee_image: ee.Image, **kwargs):
        return _Sentinel2Image._get_mask_bands(ee_image, s2_toa=True, **kwargs)


class _Sentinel2SrImage(_CloudlessImage):
    """Masking method container for Sentinel-2 SR images."""

    @staticmethod
    def _get_mask_bands(ee_image: ee.Image, **kwargs):
        return _Sentinel2Image._get_mask_bands(ee_image, s2_toa=False, **kwargs)


def _get_class_for_id(image_id: str) -> type[_MaskedImage]:
    """Return the masking class for the given Earth Engine image/collection ID."""
    ee_coll_name, _ = split_id(image_id)
    if image_id in schema.collection_schema:
        return schema.collection_schema[image_id]['image_type']
    elif ee_coll_name in schema.collection_schema:
        return schema.collection_schema[ee_coll_name]['image_type']
    else:
        return _MaskedImage


class MaskedImage(BaseImage):
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

        :param ee_image:
            Earth Engine image to encapsulate.
        :param mask:
            Whether to mask the image.
        :param region:
            Region over which to find filled and cloudless percentages for the image,
            as a GeoJSON dictionary or ``ee.Geometry``.  Percentages are stored in the image
            properties. If ``None``, statistics are not found (the default).
        :param kwargs:
            Cloud/shadow masking parameters - see :meth:`ImageAccessor.addMaskBands` for details.
        """
        super().__init__(ee_image)
        self._cloud_kwargs = kwargs
        self.ee_image = self.addMaskBands(**kwargs)
        if region:
            self._set_region_stats(region)
        if mask:
            self.mask_clouds()

    @property
    def _ee_proj(self) -> ee.Projection:
        """Projection to use for mask calculations and statistics."""
        # TODO: remove when tests updated + any other internal APIs
        return self._ee_image.select(0).projection()

    @staticmethod
    def from_id(image_id: str, **kwargs) -> MaskedImage:
        """
        Create a MaskedImage instance from an Earth Engine image ID.

        :param image_id:
            ID of the Earth Engine image to encapsulate.
        :param kwargs:
            Optional keyword arguments to pass to :meth:`__init__`.

        :return:
            A MaskedImage instance.
        """
        return MaskedImage(ee.Image(image_id), **kwargs)

    def _set_region_stats(self, region: dict | ee.Geometry = None, scale: float | None = None):
        """Set FILL_PORTION and CLOUDLESS_PORTION on the encapsulated image for the specified
        region.
        """
        # TODO: remove when tests updated
        self.ee_image = self._mi.set_mask_portions(self._ee_image, region=region, scale=scale)

    def mask_clouds(self):
        """Apply the cloud/shadow mask if supported, otherwise apply the fill mask."""
        self.ee_image = self.maskClouds()
