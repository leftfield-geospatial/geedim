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
import importlib.util
import logging

import ee
import numpy as np
import pandas as pd
from geedim import info


##
def split_id(image_id):
    """
    Split Earth Engine image ID into collection and index components.

    Parameters
    ----------
    image_id: str
              Earth engine image ID.

    Returns
    -------
    tuple
        A tuple of strings: (collection name, image index).
    """
    index = image_id.split("/")[-1]
    ee_coll_name = "/".join(image_id.split("/")[:-1])
    return ee_coll_name, index


def get_info(ee_image, min=True):
    """
    Retrieve Earth Engine image metadata

    Parameters
    ----------
    ee_image : ee.Image
               The image whose information to retrieve.
    min : bool, optional
          Retrieve the crs & scale corresponding to the band with the minimum (True) or maximum (False) scale.
          (default: True)

    Returns
    -------
    dict
        Dictionary of image information with 'id', 'properties', 'bands', 'crs' and 'scale' keys.
    """
    gd_info = dict(id=None, properties={}, bands=[], crs=None, scale=None)
    ee_info = ee_image.getInfo()  # retrieve image info from cloud

    if "id" in ee_info:
        gd_info["id"] = ee_info["id"]

    if "properties" in ee_info:
        gd_info["properties"] = ee_info["properties"]

    if "bands" in ee_info:
        # get scale & crs corresponding to min/max scale band (exclude 'EPSG:4326' (composite/constant) bands)
        band_df = pd.DataFrame(ee_info["bands"])
        scales = pd.DataFrame(band_df["crs_transform"].tolist())[0].abs().astype(float)
        band_df["scale"] = scales
        filt_band_df = band_df[(band_df.crs != "EPSG:4326") & (band_df.scale != 1)]
        if filt_band_df.shape[0] > 0:
            idx = filt_band_df.scale.idxmin() if min else filt_band_df.scale.idxmax()
            gd_info["crs"], gd_info["scale"] = filt_band_df.loc[idx, ["crs", "scale"]]

        # populate band metadata
        ee_coll_name = split_id(str(gd_info["id"]))[0]
        if ee_coll_name in info.ee_to_gd:  # include SR band metadata if it exists
            # use DataFrame to concat SR band metadata from collection_info with band IDs from the image
            gd_info["bands"] = info.collection_info[info.ee_to_gd[ee_coll_name]]["bands"].copy()
            sr_band_df = pd.DataFrame.from_dict(gd_info["bands"])
            band_df.index = band_df.id
            band_df = band_df.drop(index=sr_band_df.id)
            gd_info["bands"] += band_df[["id"]].to_dict("records")
        else:  # just use the image band IDs
            gd_info["bands"] = band_df[["id"]].to_dict("records")

    return gd_info


def get_projection(image, min=True):
    """
    Get the min/max scale projection of image bands.  Server side - no calls to getInfo().
    Adapted from from https://github.com/gee-community/gee_tools, MIT license.

    Parameters
    ----------
    image : ee.Image, geedim.image.Image
            The image whose min/max projection to retrieve.
    min: bool, optional
         Retrieve the projection corresponding to the band with the minimum (True) or maximum (False) scale.
         (default: True)

    Returns
    -------
    ee.Projection
        The requested projection.
    """
    if isinstance(image, Image):
        image = image.ee_image

    bands = image.bandNames()

    transform = np.array([1, 0, 0, 0, 1, 0])
    if min:
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


if importlib.util.find_spec("rasterio"):  # if rasterio is installed
    import rasterio as rio
    from rasterio.warp import transform_geom


    def get_bounds(filename, expand=5):  # pragma coverage
        """
        Get a geojson polygon representing the bounds of an image.

        Parameters
        ----------
        filename :  str, pathlib.Path
                    Path of the image file whose bounds to find.
        expand :    int
                    Percentage (0-100) by which to expand the bounds (default: 5).

        Returns
        -------
        bounds : dict
                 Geojson polygon.
        crs : str
              Image CRS as EPSG string.
        """
        try:
            # GEE sets tif colorinterp tags incorrectly, suppress rasterio warning relating to this:
            # 'Sum of Photometric type-related color channels and ExtraSamples doesn't match SamplesPerPixel'
            logging.getLogger("rasterio").setLevel(logging.ERROR)
            with rio.open(filename) as im:
                bbox = im.bounds
                if (im.crs.linear_units == "metre") and (expand > 0):  # expand the bounding box
                    expand_x = (bbox.right - bbox.left) * expand / 100.0
                    expand_y = (bbox.top - bbox.bottom) * expand / 100.0
                    bbox_expand = rio.coords.BoundingBox(
                        bbox.left - expand_x,
                        bbox.bottom - expand_y,
                        bbox.right + expand_x,
                        bbox.top + expand_y,
                    )
                else:
                    bbox_expand = bbox

                coordinates = [
                    [bbox_expand.right, bbox_expand.bottom],
                    [bbox_expand.right, bbox_expand.top],
                    [bbox_expand.left, bbox_expand.top],
                    [bbox_expand.left, bbox_expand.bottom],
                    [bbox_expand.right, bbox_expand.bottom],
                ]

                bbox_expand_dict = dict(type="Polygon", coordinates=[coordinates])
                src_bbox_wgs84 = transform_geom(im.crs, "WGS84", bbox_expand_dict)  # convert to WGS84 geojson
        finally:
            logging.getLogger("rasterio").setLevel(logging.WARNING)

        image_bounds = collections.namedtuple('ImageBounds', ['bounds', 'crs'])
        return image_bounds(src_bbox_wgs84, im.crs.to_epsg())


##
# Image classes
class Image(object):
    def __init__(self, ee_image):
        """
        Base class to wrap any ee.Image and provide access to metadata.

        Parameters
        ----------
        ee_image : ee.Image
                   Image to wrap.
        """
        self._ee_image = ee_image
        self._info = None

    @property
    def ee_image(self):
        """ ee.Image: The wrapped image. """
        return self._ee_image

    @property
    def info(self):
        """ dict: Image information as from get_info(). """
        if self._info is None:
            self._info = get_info(self._ee_image)
        return self._info

    @property
    def id(self):
        """ str: Earth Engine image ID. """
        return self.info["id"]

    @property
    def crs(self):
        """ str, None: Image CRS corresponding to minimum scale band, as EPSG string. None if all bands are in
        EPSG:4326. """
        return self.info["crs"]

    @property
    def scale(self):
        """ float, None: Scale (m) corresponding to minimum scale band.  None if all bands are in EPSG:4326. """
        return self.info["scale"]


class MaskedImage(Image):
    def __init__(self, ee_image, mask=False):
        """
        Class to cloud/shadow mask and quality score Earth engine images from supported collections.

        Parameters
        ----------
        ee_image : ee.Image
                   Earth engine image to wrap
        mask : bool, optional
               Apply a validity (cloud & shadow) mask to the image (default: False)
        """
        # prevent instantiation of base class(es)
        if self.gd_coll_name not in info.collection_info:
            raise NotImplementedError("This base class cannot be instantiated, use a sub-class")

        # construct the cloud/shadow masks and cloudless score
        ee_image = ee_image.unmask()
        self._masks = self._get_image_masks(ee_image)
        self._score = self._get_image_score(ee_image)
        self._ee_image = self._process_image(ee_image, mask=mask, masks=self._masks, score=self._score)
        self._info = None
        self._projection = None

    @classmethod
    def from_id(cls, image_id, mask=False):
        """
        Earth engine image wrapper for cloud/shadow masking and quality scoring.

        Parameters
        ----------
        image_id : str
                   ID of earth engine image to wrap.
        mask : bool, optional
               Apply a validity (cloud & shadow) mask to the image (default: False).

        Returns
        -------
        geedim.image.MaskedImage
            The image object.
        """
        # check image is from a supported collection
        ee_coll_name = split_id(image_id)[0]
        if ee_coll_name not in info.ee_to_gd:
            raise ValueError(f"Unsupported collection: {ee_coll_name}")

        # check this class supports the image's collection
        gd_coll_name = info.ee_to_gd[ee_coll_name]
        if gd_coll_name != cls._gd_coll_name:
            raise ValueError(f"{cls.__name__} only supports images from {info.gd_to_ee[cls._gd_coll_name]}")

        ee_image = ee.Image(image_id)
        return cls(ee_image, mask=mask)

    @classmethod
    def _from_id(cls, image_id, mask=False, region=None):
        """ Internal method for creating an image with region statistics. """
        gd_image = cls.from_id(image_id, mask=mask)
        if region is not None:
            gd_image._ee_image = cls.set_region_stats(gd_image, region)
        return gd_image

    _gd_coll_name = ""  # geedim image collection name

    @staticmethod
    def _im_transform(ee_image):
        """ Optional data type conversion to run after masking and scoring. """
        return ee_image

    @property
    def nodata(self):
        return 0

    @property
    def gd_coll_name(self):
        """ str: geedim collection name (landsat7_c2_l2|landsat8_c2_l2|sentinel2_toa|sentinel2_sr|modis_nbar). """
        return self._gd_coll_name

    @property
    def masks(self):
        """ dict: A dictionary of ee.Image objects for each of the fill, cloud, shadow and validity masks. """
        return self._masks

    @property
    def score(self):
        """ ee.Image: Pixel quality score (distance to nearest cloud/shadow (m)). """
        return self._score

    @classmethod
    def ee_collection(cls):
        """
        Returns the ee.ImageCollection corresponding to this image.

        Returns
        -------
        ee.ImageCollection
        """
        return ee.ImageCollection(info.gd_to_ee[cls._gd_coll_name])

    @classmethod
    def set_region_stats(cls, image_obj, region, mask=False):
        """
        Set VALID_PORTION and AVG_SCORE statistics for a specified region in an image object.

        Parameters
        ----------
        image_obj: ee.Image, geedim.image.Image
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

        stats_image = ee.Image([gd_image.masks["valid_mask"], gd_image.score])
        proj = get_projection(stats_image, min=False)

        # sum number of image pixels over the region
        region_sum = (
            ee.Image(1)
            .reduceRegion(reducer="sum", geometry=region, crs=proj.crs(), scale=proj.nominalScale(),
                          bestEffort=True)
        )

        # sum VALID_MASK and SCORE over image
        stats = (
            stats_image
            .reduceRegion(reducer="sum", geometry=region, crs=proj.crs(), scale=proj.nominalScale(),
                          bestEffort=True)
            .rename(["VALID_MASK", "SCORE"], ["VALID_PORTION", "AVG_SCORE"])
        )

        # find average VALID_MASK and SCORE over region (not the same as image if image does not cover region)
        def region_mean(key, value):
            return ee.Number(value).divide(ee.Number(region_sum.get("constant")))

        stats = stats.map(region_mean)
        stats = stats.set("VALID_PORTION", ee.Number(stats.get("VALID_PORTION")).multiply(100))
        return gd_image.ee_image.set(stats)

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

    # TODO: provide CLI access to cloud_dist
    def _get_image_score(self, ee_image, cloud_dist=5000, masks=None):
        """
        Get the cloud/shadow distance quality score for this image.

        Parameters
        ----------
        ee_image : ee.Image
                   Find the score for this image.
        cloud_dist : int, optional
                     The neighbourhood (m) in which to search for clouds (default: 5000).
        masks : dict, optional
                Existing masks as returned by _get_image_masks(...) (default: calculate the masks).
        Returns
        -------
        ee.Image
            The cloud/shadow distance score (m) as a single band image.
        """
        radius = 1.5  # morphological pixel radius
        proj = get_projection(ee_image, min=False)  # use maximum scale projection to save processing time
        if masks is None:
            masks = self._get_image_masks(ee_image)

        # combine cloud and shadow masks and morphologically open to remove small isolated patches
        cloud_shadow_mask = masks["cloud_mask"].Or(masks["shadow_mask"])
        cloud_shadow_mask = cloud_shadow_mask.focal_min(radius=radius).focal_max(radius=radius)
        cloud_pix = ee.Number(cloud_dist).divide(proj.nominalScale()).round()  # cloud_dist in pixels

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
                 where(score.gt(ee.Image(cloud_dist)), cloud_dist).
                 where(masks["fill_mask"].Not(), 0))
        return score

    def _process_image(self, ee_image, mask=False, masks=None, score=None):
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

        if mask:  # apply the validity mask to all bands (i.e. set those areas to nodata)
            ee_image = ee_image.mask(self._masks["valid_mask"])
        # else:  # sentinel and landsat come with default mask on SR==0, so force unmask
        #     ee_image = ee_image.unmask()

        return self._im_transform(ee_image)


class LandsatImage(MaskedImage):
    """ Base class for cloud/shadow masking and quality scoring landsat8_c2_l2 and landsat7_c2_l2 images """

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


class Landsat8Image(LandsatImage):
    """ Class for cloud/shadow masking and quality scoring landsat8_c2_l2 images """
    _gd_coll_name = "landsat8_c2_l2"


class Landsat7Image(LandsatImage):
    """ Class for cloud/shadow masking and quality scoring landsat7_c2_l2 images """
    _gd_coll_name = "landsat7_c2_l2"


class Landsat5Image(LandsatImage):
    """ Class for cloud/shadow masking and quality scoring landsat5_c2_l2 images """
    _gd_coll_name = "landsat5_c2_l2"


class Landsat4Image(LandsatImage):
    """ Class for cloud/shadow masking and quality scoring landsat4_c2_l2 images """
    _gd_coll_name = "landsat4_c2_l2"


class Sentinel2Image(MaskedImage):  # pragma: no cover
    """
    Base class for cloud masking and quality scoring sentinel2_sr and sentinel2_toa images

    (Does not use cloud probability).
    """

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


class Sentinel2SrImage(Sentinel2Image):  # pragma: no cover
    """
    Class for cloud masking and quality scoring sentinel2_sr images

    (Does not use cloud probability).
    """
    _gd_coll_name = "sentinel2_sr"


class Sentinel2ToaImage(Sentinel2Image):  # pragma: no cover
    """
    Class for cloud masking and quality scoring sentinel2_toa images

    (Does not use cloud probability).
    """
    _gd_coll_name = "sentinel2_toa"


class Sentinel2ClImage(MaskedImage):
    """
    Base class for cloud/shadow masking and quality scoring sentinel2_sr and sentinel2_toa images.

    (Uses cloud probability to improve cloud/shadow masking).
    """

    def __init__(self, ee_image, mask=False):
        # TODO: provide CLI access to these attributes

        # set attributes before their use in __init__ below
        self._cloud_prob_thresh = 35  # Cloud probability (%); values greater than are considered cloud
        self._cloud_proj_dist = 1  # Maximum distance (km) to search for cloud shadows from cloud edges
        self._buffer = 100  # Distance (m) to dilate the edge of cloud-identified objects

        MaskedImage.__init__(self, ee_image, mask=mask)

    @staticmethod
    def _im_transform(ee_image):
        return ee.Image.toUint16(ee_image)

    @classmethod
    def from_id(cls, image_id, mask=False):
        # check image_id
        ee_coll_name = split_id(image_id)[0]
        if ee_coll_name not in info.ee_to_gd:
            raise ValueError(f"Unsupported collection: {ee_coll_name}")

        gd_coll_name = info.ee_to_gd[ee_coll_name]
        if gd_coll_name != cls._gd_coll_name:
            raise ValueError(
                f"{cls.__name__} only supports images from the {info.gd_to_ee[cls._gd_coll_name]} collection"
            )

        ee_image = ee.Image(image_id)

        # get cloud probability for ee_image and add as a band
        cloud_prob = ee.Image(f"COPERNICUS/S2_CLOUD_PROBABILITY/{split_id(image_id)[1]}")
        ee_image = ee_image.addBands(cloud_prob, overwrite=True)
        return cls(ee_image, mask=mask)

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
        proj = get_projection(ee_image, min=False)  # use maximum scale projection to save processing time

        # threshold the added cloud probability to get the initial cloud mask
        cloud_prob = ee_image.select("probability")
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

        if self.gd_coll_name == "sentinel2_sr":  # use SCL band to reduce shadow_mask
            # Get the shadow mask from the SCL band and perform morphological opening to remove small isolated blobs
            scl = ee_image.select("SCL")
            dark_shadow_mask = (
                scl.eq(3)
                .Or(scl.eq(2))
                .focal_min(self._buffer, "circle", "meters")
                .focal_max(2 * self._buffer, "circle", "meters")
            )
            # improve the shadow mask by combining it with the projected cloud mask
            shadow_mask = proj_cloud_mask.And(dark_shadow_mask).rename("SHADOW_MASK")
        else:
            shadow_mask = proj_cloud_mask.rename("SHADOW_MASK")  # mask all areas that could be cloud shadow

        # incorporate the existing mask (for zero SR pixels) into the shadow mask
        zero_sr_mask = ee_image.mask().reduce(ee.Reducer.allNonZero()).Not()
        shadow_mask = shadow_mask.Or(zero_sr_mask).rename("SHADOW_MASK")

        # combine cloud and shadow masks
        valid_mask = (cloud_mask.Or(shadow_mask)).Not().rename("VALID_MASK")
        masks.update(cloud_mask=cloud_mask, shadow_mask=shadow_mask, valid_mask=valid_mask)
        return masks

    @classmethod
    def ee_collection(cls):
        """
        Returns an augmented ee.ImageCollection with cloud probability bands added to multi-spectral images.

        Returns
        -------
        ee.ImageCollection
        """
        s2_sr_toa_col = ee.ImageCollection(info.gd_to_ee[cls._gd_coll_name])
        s2_cloudless_col = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")

        # create a collection of index-matched images from the SR/TOA and cloud probability collections
        filt = ee.Filter.equals(leftField="system:index", rightField="system:index")
        inner_join = ee.ImageCollection(ee.Join.inner().apply(s2_sr_toa_col, s2_cloudless_col, filt))

        # re-configure the collection so that cloud probability is added as a band to the SR/TOA image
        def map(feature):
            """ Server-side function to concatenate images """
            return ee.Image.cat(feature.get("primary"), feature.get("secondary"))

        return inner_join.map(map)


class Sentinel2SrClImage(Sentinel2ClImage):
    """
    Class for cloud/shadow masking and quality scoring sentinel2_sr images.

    (Uses cloud probability to improve cloud/shadow masking).
    """
    _gd_coll_name = "sentinel2_sr"


class Sentinel2ToaClImage(Sentinel2ClImage):
    """
    Class for cloud/shadow masking and quality scoring sentinel2_toa images.

    (Uses cloud probability to improve cloud/shadow masking).
    """
    _gd_coll_name = "sentinel2_toa"


class ModisNbarImage(MaskedImage):
    """
    Class for wrapping modis_nbar images.

    (These images are already cloud/shadow free composites, so no further processing is done on them, and
    constant cloud, shadow etc masks are used).
    """

    @staticmethod
    def _im_transform(ee_image):
        return ee.Image.toUint16(ee_image)

    _gd_coll_name = "modis_nbar"


##


def get_class(coll_name):
    """
    Get the ProcImage subclass for wrapping image from a specified collection.

    Parameters
    ----------
    coll_name : str
                geedim or Earth Engine collection name to get class for.
                (landsat7_c2_l2|landsat8_c2_l2|sentinel2_toa|sentinel2_sr|modis_nbar) or
                (LANDSAT/LE07/C02/T1_L2|LANDSAT/LC08/C02/T1_L2|COPERNICUS/S2|COPERNICUS/S2_SR|MODIS/006/MCD43A4).

    Returns
    -------
    geedim.image.ProcImage
        The class corresponding to coll_name.
    """
    # TODO: populate this list by traversing the class hierarchy
    # TODO: allow coll_name = full image id
    # import inspect
    # from geedim import image
    # def find_subclasses():
    #     image_classes = {cls._gd_coll_name: cls for name, cls in inspect.getmembers(image)
    #                      if inspect.isclass(cls) and issubclass(cls, image.Image) and not cls is image.Image}
    #
    #     return image_classes

    gd_coll_name_map = dict(
        landsat4_c2_l2=Landsat4Image,
        landsat5_c2_l2=Landsat5Image,
        landsat7_c2_l2=Landsat7Image,
        landsat8_c2_l2=Landsat8Image,
        sentinel2_toa=Sentinel2ToaClImage,
        sentinel2_sr=Sentinel2SrClImage,
        modis_nbar=ModisNbarImage,
    )

    if split_id(coll_name)[0] in info.ee_to_gd:
        coll_name = split_id(coll_name)[0]

    if coll_name in gd_coll_name_map:
        return gd_coll_name_map[coll_name]
    elif coll_name in info.ee_to_gd:
        return gd_coll_name_map[info.ee_to_gd[coll_name]]
    else:
        raise ValueError(f"Unknown collection name: {coll_name}")
