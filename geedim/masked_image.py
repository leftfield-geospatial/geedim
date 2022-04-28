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
import logging

import ee

from geedim.image import BaseImage, split_id

logger = logging.getLogger(__name__)


##


class MaskedImage(BaseImage):
    _default_mask = False
    _default_cloud_dist = 5000
    _supported_collection_ids = ['*']
    _proj_scale = None  # TODO: might we get this from STAC?

    # TODO: all the cloud mask params need to passed, but do they belong in __init__?  how will this combine with e.g. masking for search and masking for download
    # TODO: rename has_aux_bands to something like
    def __init__(self, ee_image, mask=_default_mask, region=None, **kwargs):
        """
        Class to cloud/shadow mask and quality score Earth engine images from supported collections.

        Parameters
        ----------
        ee_image : ee.Image
            Earth engine image to wrap.
        """
        # construct the cloud/shadow masks and cloudless score
        BaseImage.__init__(self, ee_image)
        # if not has_aux_bands:
        self._add_aux_bands(**kwargs)
        if region:
            self.set_region_stats(region)
        if mask:
            self.mask_clouds()

    @staticmethod
    def from_id(image_id: str, **kwargs) -> 'MaskedImage':
        """Return a *Image instance for a given EE image ID."""
        cls = class_from_id(image_id)  # .from_id(image_id, **kwargs)
        ee_image = ee.Image(image_id)
        gd_image = cls(ee_image, **kwargs)
        gd_image._id = image_id  # set the id attribute from image_id (avoids a call to getInfo() for .id property)
        return gd_image

    def _aux_image(self, **kwargs):
        return self.ee_image.mask().reduce(ee.Reducer.allNonZero()).rename('FILL_MASK')

    def _add_aux_bands(self, **kwargs):
        aux_image = self._aux_image(**kwargs)
        # add aux bands if they are not already there
        cond = ee.Number(self.ee_image.bandNames().contains('FILL_MASK'))
        self.ee_image = ee.Image(ee.Algorithms.If(cond, self.ee_image, self.ee_image.addBands(aux_image)))

    def set_region_stats(self, region=None):
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
        # TODO: as it stands, calling set_region_stats after calling mask_clouds will give the masked region stats
        #   which is probably not what we want.  We don't want to enforce some order in which the API methods must be
        #   called.  It would be nice to only ever mask the non mask bands..., but this
        #   creates disparities with S2 images where the mask is apparently resampled internally by GEE.  We could live
        #   with that if necessary.  We could keep the aux bands outside the ee_image as we did before so that they are
        #   not masked.  But then we need some logic to add them for q_mosaic, and before download which is fiddly.  Or,
        #   we can unmask the stats image below before finding region stats.  This means that unfilled areas will no
        #   longer be masked.  Which is probably what we actually want for FILL_PORTION.  For CLOUDLESS_PORTION, I think
        #   it is also ok, unfilled pixels will then be 0, which is really what we want.
        #
        #   But, unmasked MASK bands will be problematic for mosaic compositing i.e. the composited MASK bands will not
        #   consistt of the same mix of images as the SR (masked) bands i.e. the MASK bands will not apply to these
        #   bands.
        if not region:
            region = self.ee_image.geometry()


        proj = get_projection(self.ee_image, min_scale=False)
        scale = self._proj_scale or proj.nominalScale()
        stats_image = ee.Image([self.ee_image.select('FILL_MASK').rename('FILL_PORTION').unmask(),
                                ee.Image(1).rename('REGION_SUM')])

        # sum stats_image bands over region
        sums_dict = stats_image.reduceRegion(reducer="sum", geometry=region, crs=proj.crs(), scale=scale,
                                             bestEffort=True, maxPixels=1e6)

        # find average VALID_MASK and SCORE over region (not the same as image if image does not cover region)
        def region_percentage(key, value):
            return ee.Number(value).multiply(100).divide(ee.Number(sums_dict.get("REGION_SUM")))

        means = sums_dict.select(['FILL_PORTION']).map(region_percentage)
        self.ee_image = self.ee_image.set(means)
        self.ee_image = self.ee_image.set('CLOUDLESS_PORTION', means.get('FILL_PORTION'))

    def mask_clouds(self):
        # logger.warning(f'Cloud/shadow masking is not supported for this image')
        self.ee_image = self.ee_image.updateMask(self.ee_image.select('FILL_MASK'))


class CloudMaskedImage(MaskedImage):
    _supported_collection_ids = []  # abstract base class

    def _cloud_dist(self, cloudless_mask=None, max_cloud_dist=5000):
        """
        Get the cloud/shadow distance quality score for this image.

        Returns
        -------
        ee.Image
            The cloud/shadow distance score (m) as a single band image.
        """
        if not cloudless_mask:
            cloudless_mask = self.ee_image.select('CLOUDLESS_MASK')
        proj = get_projection(self.ee_image, min_scale=False)  # use maximum scale projection to save processing time

        # the mask of cloud and shadows (and unfilled pixels which are often cloud shadow, or other deep shadow)
        # note that initial *MASK bands are not themselves masked with fill mask
        cloud_shadow_mask = cloudless_mask.Not()
        cloud_pix = ee.Number(max_cloud_dist).divide(proj.nominalScale()).round()  # cloud_dist in pixels

        # distance to nearest cloud/shadow (m)
        cloud_dist = (
            cloud_shadow_mask.fastDistanceTransform(neighborhood=cloud_pix, units="pixels", metric="squared_euclidean")
                .sqrt()
                .multiply(proj.nominalScale())
                .rename("CLOUD_DIST")
                .reproject(crs=proj.crs(), scale=proj.nominalScale())  # reproject to force calculation at correct scale
        )

        # clip score to max_cloud_dist and set to 0 in unfilled areas
        cloud_dist = (cloud_dist.
                      where(cloud_dist.gt(ee.Image(max_cloud_dist)),
                            max_cloud_dist))  # TODO: I don't think this where is necessary when we don't unmask to start with
        return cloud_dist.toUint32().rename('CLOUD_DIST')

    def mask_clouds(self):
        self.ee_image = self.ee_image.updateMask(self.ee_image.select('CLOUDLESS_MASK'))

    # TODO: this method is only/mainly used on ee.Image's from *Colleciton.  So make it static or something like before - that is cleaner.
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
        # TODO: get_projection does not work if the wrapped image is composite
        proj = get_projection(self.ee_image, min_scale=False)
        scale = self._proj_scale or proj.nominalScale()
        stats_image = ee.Image([self.ee_image.select(['FILL_MASK', 'CLOUDLESS_MASK']).unmask(),
                                ee.Image(1).rename('REGION_SUM')])

        # sum stats_image bands over region
        sums = (
            stats_image.reduceRegion(reducer="sum", geometry=region, crs=proj.crs(), scale=scale, bestEffort=True,
                                     maxPixels=1e6).
                rename(['FILL_MASK', 'CLOUDLESS_MASK'], ['FILL_PORTION', 'CLOUDLESS_PORTION'])
        )

        # find average VALID_MASK and SCORE over region (not the same as image if image does not cover region)
        def region_percentage(key, value):
            return ee.Number(value).multiply(100).divide(ee.Number(sums.get("REGION_SUM")))

        means = sums.select(['FILL_PORTION', 'CLOUDLESS_PORTION']).map(region_percentage)
        self.ee_image = self.ee_image.set(means)


class LandsatImage(CloudMaskedImage):
    """ Base class for cloud/shadow masking and quality scoring landsat images """
    _supported_collection_ids = ['LANDSAT/LT04/C02/T1_L2', 'LANDSAT/LT05/C02/T1_L2', 'LANDSAT/LE07/C02/T1_L2',
                                 'LANDSAT/LC08/C02/T1_L2', 'LANDSAT/LC09/C02/T1_L2']
    _proj_scale = 30

    def _aux_image(self, mask_shadows=True, mask_cirrus=True, **kwargs):
        # TODO: add warning for unsupported args?
        ee_image = self._ee_image
        # get cloud, shadow and fill masks from QA_PIXEL
        qa_pixel = ee_image.select("QA_PIXEL")

        # incorporate the existing mask (for zero SR pixels) into the shadow mask
        ee_mask = ee_image.select('SR_B.*').mask().reduce(ee.Reducer.allNonZero())
        fill_mask = qa_pixel.bitwiseAnd(1).eq(0).And(ee_mask).rename("FILL_MASK")
        shadow_mask = qa_pixel.bitwiseAnd(0b10000).neq(0).rename("SHADOW_MASK")
        if mask_cirrus:
            cloud_mask = qa_pixel.bitwiseAnd(0b1100).neq(0).rename(
                "CLOUD_MASK")  # TODO: is this bit always zero for landsat 4-7
        else:
            cloud_mask = qa_pixel.bitwiseAnd(0b1000).neq(0).rename("CLOUD_MASK")

        # combine cloud, shadow and fill masks into cloudless mask
        cloudless_mask = (cloud_mask.Or(shadow_mask)).Not() if mask_shadows else cloud_mask.Not()
        cloudless_mask = cloudless_mask.And(fill_mask).rename("CLOUDLESS_MASK")
        cloud_dist = self._cloud_dist()  # TODO work around band naming so we don't need to re-add this
        # self.ee_image = ee_image.addBands([fill_mask, cloud_mask, shadow_mask, cloudless_mask, cloud_dist],
        #                                   overwrite=True)
        return ee.Image([fill_mask, cloud_mask, shadow_mask, cloudless_mask, cloud_dist])

    def _cloud_dist(self, max_cloud_dist=5000):
        return self._ee_image.select('ST_CDIST').rename('CLOUD_DIST')


class Sentinel2ClImage(CloudMaskedImage):
    """
    Base class for cloud/shadow masking and quality scoring sentinel2_sr and sentinel2_toa images.

    (Uses cloud probability to improve cloud/shadow masking).
    """
    _supported_collection_ids = []
    _proj_scale = 60

    # TODO: provide CLI access to these kwargs, and document them here
    def _aux_image(self, s2_toa=False, method='cloud_prob', mask_cirrus=True, mask_shadows=True, prob=60,
                   dark=0.15, shadow_dist=1000, buffer=250, cdi_thresh=None, max_cloud_dist=5000):
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
        shadow_dist : int, optional
            Maximum distance in meters (m) to look for cloud shadows from cloud edges.
        buffer : int, optional
            Distance in meters (m) to dilate cloud and cloud shadows objects.
        cdi_thresh : float, optional
            Cloud Displacement Index threshold. Values below this threshold are considered potential clouds.
            A cdi_thresh = None means that the index is not used.
            For more info see 'Frantz, D., HaS, E., Uhl, A., Stoffels, J., Hill, J. 2018. Improvement of the Fmask
            algorithm for Sentinel-2 images: Separating clouds from bright surfaces based on parallax effects. Remote
            Sensing of Environment 2015: 471-481'.
        max_cloud_dist: int, optional
            Maximum distance in meters (m) to look for clouds for the `cloud distance` band.
        Returns
        -------
        dict
            A dictionary of ee.Image objects for each of the fill, cloud, shadow and validity masks.
        """

        # maskCirrus : Whether to mask cirrus clouds. Valid just for method = 'qa'. This parameter is ignored for Landsat products.
        # maskShadows : Whether to mask cloud shadows. For more info see 'Braaten, J. 2020. Sentinel-2 Cloud Masking with s2cloudless. Google Earth Engine, Community Tutorials'.
        # scaledImage : Whether the pixel values are scaled to the range [0,1] (reflectance values). This parameter is ignored for Landsat products.

        def get_cloud_prob(ee_im):
            s2_sr_toa_col = ee.ImageCollection(ee_im)
            s2_cloudless_col = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")

            # create a collection of index-matched images from the SR/TOA and cloud probability collections
            filt = ee.Filter.equals(leftField="system:index", rightField="system:index")
            inner_join = ee.ImageCollection(ee.Join.inner().apply(s2_sr_toa_col, s2_cloudless_col, filt))
            return ee.Image(inner_join.first().get('secondary')).rename('CLOUD_PROB')

        def get_cloud_mask(ee_im, cloud_prob=None):
            if method == 'cloud_prob':
                if not cloud_prob:
                    cloud_prob = get_cloud_prob(ee_im)
                cloud_mask = cloud_prob.gte(prob).rename('CLOUD_MASK')
            else:
                qa = ee_im.select('QA60')
                cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)
                if mask_cirrus:
                    cloud_mask = cloud_mask.And(qa.bitwiseAnd(1 << 11).eq(0))
            return cloud_mask

        def get_cdi_cloud_mask(ee_im):
            if s2_toa:
                s2_toa_image = ee_im
            else:
                idx = ee_im.get("system:index")
                s2_toa_image = (ee.ImageCollection("COPERNICUS/S2").
                                filter(ee.Filter.eq("system:index", idx)).
                                first())
            cdi_image = ee.Algorithms.Sentinel2.CDI(s2_toa_image)
            return cdi_image.lt(cdi_thresh).rename("CDI_CLOUD_MASK")

        def get_shadow_mask(ee_im, cloud_mask):
            dark_mask = ee_im.select("B8").lt(dark * 1e4)
            if not s2_toa:
                dark_mask = ee_im.select("SCL").neq(6).And(dark_mask)

            shadow_azimuth = ee.Number(90).subtract(ee.Number(ee_im.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
            proj_cloud_mask = (cloud_mask.directionalDistanceTransform(shadow_azimuth,
                                                                       int(shadow_dist / self._proj_scale))
                               .reproject(crs=ee_im.select(0).projection(), scale=60)
                               .select('distance')
                               .mask())
            return proj_cloud_mask.And(dark_mask).rename("SHADOW_MASK")

        ee_image = self.ee_image
        cloud_prob = get_cloud_prob(ee_image)
        cloud_mask = get_cloud_mask(ee_image, cloud_prob=cloud_prob)
        cloud_shadow_mask = cloud_mask
        if cdi_thresh is not None:
            cloud_shadow_mask = cloud_shadow_mask.And(get_cdi_cloud_mask(ee_image))
        if mask_shadows:
            shadow_mask = get_shadow_mask(ee_image, cloud_mask)
            cloud_shadow_mask = cloud_shadow_mask.Or(shadow_mask)  # TODO: or / add

        cloud_shadow_mask = cloud_shadow_mask.focal_min(20, units="meters").focal_max(buffer * 2 / 10, units="meters")

        fill_mask = ee_image.select('B.*').mask().reduce(ee.Reducer.allNonZero())
        fill_mask = fill_mask.clip(ee_image.geometry()).rename('FILL_MASK')

        cloudless_mask = (cloud_shadow_mask.Not()).And(fill_mask).rename('CLOUDLESS_MASK')

        aux_bands = [cloud_prob, fill_mask, cloud_mask, cloudless_mask]
        if mask_shadows:
            aux_bands.append(shadow_mask)

        cloud_dist = self._cloud_dist(cloudless_mask=cloudless_mask, max_cloud_dist=max_cloud_dist)
        return ee.Image(aux_bands + [cloud_dist])


class Sentinel2SrClImage(Sentinel2ClImage):
    """
    Base class for cloud/shadow masking and quality scoring 'COPERNICUS/S2_SR' images.

    (Uses cloud probability to improve cloud/shadow masking).
    """
    _supported_collection_ids = ['COPERNICUS/S2_SR']

    def _aux_image(self, s2_toa=False, **kwargs):
        return Sentinel2ClImage._aux_image(self, s2_toa=False, **kwargs)


class Sentinel2ToaClImage(Sentinel2ClImage):
    """
    Base class for cloud/shadow masking and quality scoring 'COPERNICUS/S2' images.

    (Uses cloud probability to improve cloud/shadow masking).
    """
    _supported_collection_ids = ['COPERNICUS/S2']

    def _aux_image(self, s2_toa=False, **kwargs):
        return Sentinel2ClImage._aux_image(self, s2_toa=True, **kwargs)


class ModisNbarImage(MaskedImage):
    """
    Class for wrapping modis_nbar images.

    (These images are already cloud/shadow free composites, so no further processing is done on them, and
    constant cloud, shadow etc masks are used).
    """
    _supported_collection_ids = ['MODIS/006/MCD43A4']


def class_from_id(image_id: str) -> MaskedImage:
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
        return MaskedImage


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

    compare = ee.Number.lte if min_scale else ee.Number.gte
    init_proj = image.select(0).projection()

    def compare_scale(name, prev_proj):
        """ Server side comparison of band scales"""
        prev_proj = ee.Projection(prev_proj)
        prev_scale = prev_proj.nominalScale()

        curr_proj = image.select([name]).projection()
        curr_scale = ee.Number(curr_proj.nominalScale())

        # compare scales, excluding WGS84 bands (constant or composite bands)
        condition = (
            # compare(curr_scale, prev_scale).And(curr_proj.crs().compareTo(ee.String("EPSG:4326"))).neq(ee.Number(0))
            compare(curr_scale, prev_scale)
        )
        comp_proj = ee.Algorithms.If(condition, curr_proj, prev_proj)
        return ee.Projection(comp_proj)

    return ee.Projection(bands.iterate(compare_scale, init_proj))
