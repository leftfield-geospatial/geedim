# Copyright The Geedim Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

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
        """Return the given image with cloud/shadow masks and related bands added
        when supported, otherwise with fill (validity) mask added. Mask bands are
        overwritten if they exist, except on images without fixed projections,
        in which case no bands are added or overwritten.
        """
        mask_bands = cls._get_mask_bands(ee_image, **kwargs)
        # add/overwrite mask bands unless the image has no fixed projection
        add = ImageAccessor(ee_image).fixed()
        return ee.Image(
            ee.Algorithms.If(
                add,
                ee_image.addBands(list(mask_bands.values()), overwrite=True),
                ee_image,
            )
        )

    @staticmethod
    def mask_clouds(ee_image: ee.Image) -> ee.Image:
        """Return the given image with cloud/shadow masks applied when supported,
        otherwise return the given image unaltered.  Mask bands should be added with
        :meth:`add_mask_bands` before calling this method.
        """
        return ee_image

    @staticmethod
    def set_mask_portions(
        ee_image: ee.Image,
        region: dict | ee.Geometry | None = None,
        scale: float | ee.Number | None = None,
    ) -> ee.Image:
        """Return the given image with the ``FILL_PORTION`` property set to the
        filled percentage of the given region, and the ``CLOUDLESS_PORTION`` property
        set to the cloudless percentage of ``FILL_PORTION``.  ``CLOUDLESS_PORTION``
        is set to ``100`` if cloud/shadow masking is not supported.
        """
        portions = ImageAccessor(ee_image.select('FILL_MASK')).regionCoverage(
            region=region, scale=scale, maxPixels=1e6, bestEffort=True
        )
        return ee_image.set(
            'FILL_PORTION', portions.get('FILL_MASK'), 'CLOUDLESS_PORTION', 100
        )


class _CloudlessImage(_MaskedImage):
    """Abstract masking method container for images with cloud/shadow support."""

    @staticmethod
    def _get_cloud_dist(
        cloudless_mask: ee.Image,
        proj: ee.Projection = None,
        max_cloud_dist: float = 5000,
    ) -> ee.Image:
        """Return a cloud/shadow distance (m) image for the given cloudless mask."""
        # projection & scale of the distance image
        proj = proj or cloudless_mask.projection()
        scale = proj.nominalScale()

        # max_cloud_dist in pixels
        max_cloud_pix = ee.Number(max_cloud_dist).divide(scale).round()

        # Find distance to nearest cloud/shadow (m).  Distances are found for all
        # pixels, including masked / invalid pixels, which are treated as 0 (non
        # cloud/shadow).
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

        # cloud_dist is float64 by default, so convert to Uint16 here to avoid
        # forcing the whole image to float64 on export.
        return cloud_dist.toUint16().rename('CLOUD_DIST')

    @staticmethod
    @abstractmethod
    def _get_mask_bands(ee_image: ee.Image) -> dict[str, ee.Image]:
        pass

    @staticmethod
    def mask_clouds(ee_image: ee.Image) -> ee.Image:
        return ee_image.updateMask(ee_image.select('CLOUDLESS_MASK'))

    @staticmethod
    def set_mask_portions(
        ee_image: ee.Image,
        region: dict | ee.Geometry = None,
        scale: float | ee.Number = None,
    ) -> ee.Image:
        # use maxPixels=1e6 to speed up computations for large regions
        portions = ImageAccessor(
            ee_image.select(['FILL_MASK', 'CLOUDLESS_MASK'])
        ).regionCoverage(region=region, scale=scale, maxPixels=1e6, bestEffort=True)
        fill_portion = ee.Number(portions.get('FILL_MASK'))
        cl_portion = (
            ee.Number(portions.get('CLOUDLESS_MASK')).divide(fill_portion).multiply(100)
        )
        return ee_image.set(
            'FILL_PORTION', fill_portion, 'CLOUDLESS_PORTION', cl_portion
        )


class _LandsatToaRawImage(_CloudlessImage):
    """Masking method container for Landsat collection 2 TOA reflectance and at
    sensor radiance images.
    """

    @staticmethod
    def _get_mask_bands(
        ee_image: ee.Image,
        mask_shadows: bool = True,
        mask_cirrus: bool = True,
        mask_saturation: bool = False,
        mask_nonphysical: bool = False,
        max_cloud_dist: float = 5000,
    ) -> dict[str, ee.Image]:
        # Adapted from https://gis.stackexchange.com/a/473652.  With
        # mask_nonphysical=True and mask_aerosols=True in _LandsatSrAerosolImage,
        # the cloudless mask will be the same as
        # https://gis.stackexchange.com/a/473652, excepting for snow/ice, which is
        # included.
        aux_bands = {}

        # construct fill mask from Earth Engine mask
        refl_mask = ee_image.select('B.*|SR_B.*').mask()
        aux_bands['fill'] = refl_mask.reduce(ee.Reducer.allNonZero()).rename(
            'FILL_MASK'
        )

        # find cloud mask from QA_PIXEL band
        qa_pixel = ee_image.select('QA_PIXEL')
        mid_cloud_mask = qa_pixel.bitwiseAnd(1 << 9).eq(1 << 9)
        dilated_cloud_mask = qa_pixel.bitwiseAnd(1 << 1).eq(1 << 1)
        aux_bands['cloud'] = mid_cloud_mask.Or(dilated_cloud_mask).rename('CLOUD_MASK')
        if mask_cirrus:
            cirrus_mask = qa_pixel.bitwiseAnd(1 << 15).eq(1 << 15)
            aux_bands['cloud'] = aux_bands['cloud'].Or(cirrus_mask)
        combined_mask = aux_bands['cloud']

        if mask_shadows:
            # find & incorporate QA_PIXEL shadow mask
            aux_bands['shadow'] = (
                qa_pixel.bitwiseAnd(1 << 11).eq(1 << 11).rename('SHADOW_MASK')
            )
            combined_mask = combined_mask.Or(aux_bands['shadow'])

        if mask_saturation:
            # find & incorporate QA_RADSAT saturation mask
            qa_radsat = ee_image.select('QA_RADSAT')
            aux_bands['saturation'] = qa_radsat.neq(0).rename('SATURATION_MASK')
            combined_mask = combined_mask.Or(aux_bands['saturation'])

        # find cloudless mask
        aux_bands['cloudless'] = (
            combined_mask.Not().And(aux_bands['fill']).rename('CLOUDLESS_MASK')
        )

        # find cloud distance from cloudless mask
        aux_bands['dist'] = _CloudlessImage._get_cloud_dist(
            aux_bands['cloudless'],
            # use 30m B1|SR_B1 projection for the cloud distance
            proj=ee_image.select(0).projection(),
            max_cloud_dist=max_cloud_dist,
        )

        return aux_bands


class _LandsatSrImage(_LandsatToaRawImage):
    """Masking method container for Landsat collection 2 surface reflectance images."""

    @staticmethod
    def _get_mask_bands(
        ee_image: ee.Image, mask_nonphysical: bool = False, **kwargs
    ) -> dict[str, ee.Image]:
        aux_bands = _LandsatToaRawImage._get_mask_bands(ee_image, **kwargs)
        if mask_nonphysical:
            # find & incorporate non-physical reflectance mask
            lims = [(v + 0.2) / 0.0000275 for v in [0.0, 1.0]]
            refl_image = ee_image.select('SR_B.*')
            aux_bands['nonphysical'] = (
                (refl_image.reduce('min').lt(lims[0]))
                .Or(refl_image.reduce('max').gt(lims[1]))
                .rename('NONPHYSICAL_MASK')
            )
            aux_bands['cloudless'] = aux_bands['cloudless'].And(
                aux_bands['nonphysical'].Not()
            )

        return aux_bands


class _LandsatSrAerosolImage(_LandsatSrImage):
    """Masking method container for Landsat collection 2 surface reflectance images
    with the SR_QA_AEROSOL band.
    """

    @staticmethod
    def _get_mask_bands(
        ee_image: ee.Image, mask_aerosols: bool = False, **kwargs
    ) -> dict[str, ee.Image]:
        aux_bands = _LandsatSrImage._get_mask_bands(ee_image, **kwargs)
        if mask_aerosols:
            # find & incorporate high aerosol level mask
            sr_qa_aerosol = ee_image.select('SR_QA_AEROSOL')
            aux_bands['aerosol'] = sr_qa_aerosol.bitwiseAnd(3 << 6).eq(3 << 6)
            aux_bands['aerosol'] = aux_bands['aerosol'].rename('AEROSOL_MASK')
            aux_bands['cloudless'] = aux_bands['cloudless'].And(
                aux_bands['aerosol'].Not()
            )
        return aux_bands


class _Sentinel2Image(_CloudlessImage):
    """Abstract masking method container for Sentinel-2 TOA / SR images."""

    _cloud_score_coll_id = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
    _cloud_prob_coll_id = 'COPERNICUS/S2_CLOUD_PROBABILITY'

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
        mask_nonphysical: bool = False,
    ) -> dict[str, ee.Image]:
        """Return a dictionary of mask bands and related images for the given
        Sentinel-2 image. Parts adapted from
        https://github.com/r-earthengine/ee_extra, under Apache 2.0 licence.
        """
        mask_method = CloudMaskMethod(mask_method)
        if mask_method is not CloudMaskMethod.cloud_score:
            warnings.warn(
                f"The '{mask_method}' mask method is deprecated and will be removed "
                f"in a future release.  Please use the 'cloud-score' method instead.",
                category=FutureWarning,
                stacklevel=2,
            )
        cs_band = CloudScoreBand(cs_band)

        def match_image(
            ee_image: ee.Image,
            collection: str,
            band: str,
            match_prop: str = 'system:index',
        ) -> ee.Image:
            """Return an image from ``collection`` matching ``ee_image`` with a
            single ``band`` selected, or a fully masked image if no match is found.
            """
            # TODO: would it be possible (and faster) to use
            #  ee.ImageCollection.linkCollection, for the ImageCollection accessor
            #  rather than linking per image?  I think linkCollection is a shortcut
            #  for ee.Join.saveFirst
            # default fully masked image
            default = ee.Image().updateMask(0)
            default = default.rename(band)

            # find matching image
            filt = ee.Filter.eq(match_prop, ee_image.get(match_prop))
            match = ee.ImageCollection(collection).filter(filt).first()

            # revert to default if no match found
            match = ee.Image(
                ee.List([match, default]).reduce(ee.Reducer.firstNonNull())
            )
            return match.select(band)

        def cloud_cast_shadow_mask(
            ee_image: ee.Image, cloud_mask: ee.Image
        ) -> ee.Image:
            """Return a shadow mask for ``ee_image`` based on the intersection of
            dark areas and projected shadows from ``cloud_mask``.

            Adapted from https://developers.google.com/earth-engine/tutorials
            /community/sentinel-2-s2cloudless>.
            """
            # use the 60m B1 projection for shadow mask to save some computation
            proj = ee_image.select('B1').projection()

            dark_mask = ee_image.select('B8').lt(dark * 1e4)
            if not s2_toa:
                # exclude water
                dark_mask = ee_image.select('SCL').neq(6).And(dark_mask)

            # Find shadow direction.  Note that the angle convention depends on the
            # reproject args below.
            shadow_azimuth = ee.Number(90).subtract(
                ee.Number(ee_image.get('MEAN_SOLAR_AZIMUTH_ANGLE'))
            )

            # Project the cloud mask in the shadow direction (can project the mask
            # into invalid areas).
            proj_pixels = ee.Number(shadow_dist).divide(proj.nominalScale()).round()
            cloud_cast_proj = cloud_mask.directionalDistanceTransform(
                shadow_azimuth, proj_pixels
            ).select('distance')

            # Reproject to force calculation at the correct scale.
            cloud_cast_mask = cloud_cast_proj.mask().reproject(
                crs=proj, scale=proj.nominalScale()
            )

            # Remove any projections in invalid areas.
            cloud_cast_mask = cloud_cast_mask.updateMask(cloud_mask.mask())

            # Find shadow mask as intersection between projected clouds and dark areas.
            shadow_mask = cloud_cast_mask.And(dark_mask)

            return shadow_mask.rename('SHADOW_MASK')

        def qa_cloud_mask(ee_image: ee.Image) -> ee.Image:
            """Return a cloud mask for ``ee_image`` derived from the QA60 band."""
            # mask QA60 if it is between Feb 2022 - Feb 2024 to invalidate the cloud
            # mask (QA* bands are not populated over this period and may be masked /
            # transparent or zero).
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
            """Return the cloud probability thresholded cloud mask, and cloud
            probability image for ``ee_image``.
            """
            cloud_prob = match_image(
                ee_image, _Sentinel2Image._cloud_prob_coll_id, 'probability'
            )
            cloud_mask = cloud_prob.gte(prob)
            return cloud_mask.rename('CLOUD_MASK'), cloud_prob.rename('CLOUD_PROB')

        def cloud_score_cloud_shadow_mask(
            ee_image: ee.Image,
        ) -> tuple[ee.Image, ee.Image]:
            """Return the cloud score thresholded cloud mask, and cloud score image for
            ``ee_image``.
            """
            cloud_score = match_image(
                ee_image, _Sentinel2Image._cloud_score_coll_id, cs_band.name
            )
            cloud_shadow_mask = cloud_score.lte(score)
            cloud_score = cloud_score.toFloat()
            return cloud_shadow_mask.rename('CLOUD_SHADOW_MASK'), cloud_score.rename(
                'CLOUD_SCORE'
            )

        def cdi_cloud_mask(ee_image: ee.Image) -> ee.Image:
            """Return the CDI cloud mask for ``ee_image``. See
            https://developers.google.com/earth-engine/apidocs/ee-algorithms
            -sentinel2-cdi for detail.
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
            """Return a dictionary of cloud/shadow masks & related images for
            ``ee_image``.
            """
            aux_bands = {}
            if mask_method is CloudMaskMethod.cloud_score:
                aux_bands['cloud_shadow'], aux_bands['score'] = (
                    cloud_score_cloud_shadow_mask(ee_image)
                )
            else:
                if mask_method is CloudMaskMethod.qa:
                    aux_bands['cloud'] = qa_cloud_mask(ee_image)
                else:
                    aux_bands['cloud'], aux_bands['prob'] = cloud_prob_cloud_mask(
                        ee_image
                    )

                if cdi_thresh is not None:
                    aux_bands['cloud'] = aux_bands['cloud'].And(
                        cdi_cloud_mask(ee_image)
                    )

                aux_bands['shadow'] = cloud_cast_shadow_mask(
                    ee_image, aux_bands['cloud']
                )

                if mask_shadows:
                    aux_bands['cloud_shadow'] = aux_bands['cloud'].Or(
                        aux_bands['shadow']
                    )
                else:
                    aux_bands['cloud_shadow'] = aux_bands['cloud']
                    aux_bands.pop('shadow', None)

                # do a morphological opening that removes small (20m) blobs from the
                # mask and then dilates
                aux_bands['cloud_shadow'] = (
                    aux_bands['cloud_shadow']
                    .focal_min(20, units='meters')
                    .focal_max(buffer, units='meters')
                )

            if mask_nonphysical:
                # mask nonphysical reflectance (reflectance<0 is clipped to 0 and
                # masked by default)
                aux_bands['nonphysical'] = (
                    ee_image.select('B.*')
                    .reduce('max')
                    .gt(10000)
                    .rename('NONPHYSICAL_MASK')
                )
                aux_bands['cloud_shadow'] = aux_bands['cloud_shadow'].Or(
                    aux_bands['nonphysical']
                )

            return aux_bands

        # get cloud/shadow etc bands
        aux_bands = cloud_shadow_masks(ee_image)

        # derive a fill mask from the Earth Engine mask for the surface reflectance
        # bands
        aux_bands['fill'] = (
            ee_image.select('B.*')
            .mask()
            .reduce(ee.Reducer.allNonZero())
            .rename('FILL_MASK')
        )

        # combine masks into cloudless_mask
        aux_bands['cloudless'] = (
            (aux_bands['cloud_shadow'].Not())
            .And(aux_bands['fill'])
            .rename('CLOUDLESS_MASK')
        )
        aux_bands.pop('cloud_shadow')

        # find cloud distance from cloudless mask
        aux_bands['dist'] = _CloudlessImage._get_cloud_dist(
            aux_bands['cloudless'],
            # use 60m B1 projection for the cloud distance to save some computation
            proj=ee_image.select('B1').projection(),
            max_cloud_dist=max_cloud_dist,
        )
        return aux_bands


class _Sentinel2ToaImage(_Sentinel2Image):
    """Masking method container for Sentinel-2 TOA images."""

    @staticmethod
    def _get_mask_bands(ee_image: ee.Image, **kwargs):
        return _Sentinel2Image._get_mask_bands(ee_image, s2_toa=True, **kwargs)


class _Sentinel2SrImage(_Sentinel2Image):
    """Masking method container for Sentinel-2 SR images."""

    @staticmethod
    def _get_mask_bands(ee_image: ee.Image, **kwargs):
        return _Sentinel2Image._get_mask_bands(ee_image, s2_toa=False, **kwargs)


def _get_class_for_id(image_id: str) -> type[_MaskedImage]:
    """Return the masking class for the given Earth Engine image/collection ID."""
    coll_id, _ = split_id(image_id)
    if image_id in schema.collection_schema:
        return schema.collection_schema[image_id]['image_type']
    elif coll_id in schema.collection_schema:
        return schema.collection_schema[coll_id]['image_type']
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

        .. deprecated:: 2.0.0
            Please use the :class:`gd <geedim.image.ImageAccessor>` accessor on
            :class:`ee.Image` instead.

        :param ee_image:
            Earth Engine image to encapsulate.
        :param mask:
            Whether to mask the image.
        :param region:
            Region over which to find filled and cloudless percentages for the image,
            as a GeoJSON dictionary or ``ee.Geometry``.  Percentages are stored in
            the image properties. If ``None``, statistics are not found (the default).
        :param kwargs:
            Cloud/shadow masking parameters - see
            :meth:`~geedim.image.ImageAccessor.addMaskBands` for details.
        """
        super().__init__(ee_image)
        # copy the _MaskedImage class to avoid another getInfo() if region is supplied
        mi = self._mi
        self.ee_image = mi.add_mask_bands(self._ee_image, **kwargs)
        if region:
            self.ee_image = mi.set_mask_portions(self._ee_image, region=region)
        if mask:
            self.mask_clouds()

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

    def mask_clouds(self):
        """Apply the cloud/shadow mask if supported, otherwise apply the fill mask."""
        self._ee_image = self.maskClouds()
