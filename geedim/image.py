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

##
# Classes wrapping, cloud/shadow masking and scoring Earth Engine images
import ee
import pandas as pd
import logging
import importlib.util
from geedim import info
import inspect

## Image classes
class Image(object):

    def __init__(self, ee_image, mask=False, scale_refl=False):
        """
        Earth engine image wrapper for cloud/shadow masking and quality scoring

        Parameters
        ----------
        ee_image : ee.Image
                   Earth engine image to wrap
        mask : bool, optional
               Apply a validity (cloud & shadow) mask to the image (default: False)
        scale_refl : bool, optional
                     Scale reflectance bands 0-10000 if they are not in that range already (default: False)
        """
        # prevent instantiation of this (non-abstract) base class
        if not self.gd_coll_name in info.collection_info:
            raise Exception('This base class cannot be instantiated, use a derived class')

        self._collection_info = info.collection_info[self.gd_coll_name]
        self.band_df = pd.DataFrame.from_dict(self._collection_info['bands'])

        self._masks = self._get_image_masks(ee_image)
        self._score = self._get_image_score(ee_image)
        self._ee_image = self._process_image(ee_image,
                                             mask=mask,
                                             scale_refl=scale_refl,
                                             masks=self._masks,
                                             score=self._score)
        self._info = None


    @classmethod
    def from_id(cls, image_id, mask=False, scale_refl=False):
        """
        Earth engine image wrapper for cloud/shadow masking and quality scoring

        Parameters
        ----------
        image_id : str
                   ID of earth engine image to wrap
        mask : bool, optional
               Apply a validity (cloud & shadow) mask to the image (default: False)
        scale_refl : bool, optional
                     Scale reflectance bands 0-10000 if they are not in that range already (default: False)
        """
        ee_coll_name = ee_split(image_id)[0]
        if ee_coll_name not in info.ee_to_gd:
            raise ValueError(f'Unsupported collection: {ee_coll_name}')

        gd_coll_name = info.ee_to_gd[ee_coll_name]
        if gd_coll_name != cls._gd_coll_name:
            raise ValueError(f'{cls.__name__} only supports images from {info.gd_to_ee[cls._gd_coll_name]}')

        ee_image = ee.Image(image_id)
        return cls(ee_image, mask=mask, scale_refl=scale_refl)

    _gd_coll_name = ''

    @staticmethod
    def _im_transform(ee_image):
        return ee_image

    @property
    def gd_coll_name(self):
        return self._gd_coll_name

    @property
    def ee_image(self):
        return self._ee_image

    @property
    def masks(self):
        return self._masks

    @property
    def score(self):
        return self._score

    @property
    def info(self):
        if self._info is None:
            self._info = self._ee_image.getInfo()
        return self._info

    @classmethod
    def ee_collection(cls):
        return ee.ImageCollection(info.gd_to_ee[cls._gd_coll_name])

    def _scale_refl(self, ee_image):
        return ee_image

    def _get_image_masks(self, ee_image):
        """
        Derive cloud, shadow, fill and validity masks for an image

        Parameters
        ----------
        ee_image : ee.Image
                Derive masks for this image

        Returns
        -------
        masks : dict
                A dictionary with cloud_mask, shadow_mask, fill_mask and valid_mask keys, and corresponding ee.Image
                values
        """
        masks = dict(cloud_mask=ee.Image(0).rename('CLOUD_MASK'),
                     shadow_mask=ee.Image(0).rename('SHADOW_MASK'),
                     fill_mask=ee.Image(1).rename('FILL_MASK'),
                     valid_mask=ee.Image(1).rename('VALID_MASK'))

        return masks

    # TODO: provide cli access to cloud_dist
    def _get_image_score(self, ee_image, cloud_dist=5000, masks=None):
        """
        Get the cloud distance quality score for this image

        Parameters
        ----------
        ee_image : ee.Image
                Find the score for this image
        cloud_dist : int, oprtional
                     The neighbourhood (in meters) in which to search for clouds
        masks : dict, optional
                Masks returned by _get_image_masks(...)
        Returns
        -------
        : ee.Image
        The cloud distance score as a single band image
        """
        radius = 1.5
        min_proj = get_projection(ee_image)
        cloud_pix = ee.Number(cloud_dist).divide(min_proj.nominalScale()).toInt()

        if masks is None:
            masks = self._get_image_masks(ee_image)

        cloud_shadow_mask = masks['cloud_mask'].Or(masks['shadow_mask'])
        cloud_shadow_mask = cloud_shadow_mask.focal_min(radius=radius).focal_max(radius=radius)

        score = (cloud_shadow_mask.
                 fastDistanceTransform(neighborhood=cloud_pix, units='pixels', metric='squared_euclidean').
                 sqrt().
                 multiply(min_proj.nominalScale()).
                 rename('SCORE'))

        return (score.unmask().
                where(score.gt(ee.Image(cloud_dist)), cloud_dist).
                where(masks['fill_mask'].unmask().Not(), 0))

    def _process_image(self, ee_image, mask=False, scale_refl=False, masks=None, score=None):
        """
        Adds mask and score bands to a raw Earth Engine image

        Parameters
        ----------
        ee_image : ee.Image
                    Earth engine image to add bands to
        mask : bool, optional
                     Apply any validity mask to the image by setting nodata (default: False)
        scale_refl : bool, optional
                     Scale reflectance values from 0-10000 if they are not in that range already (default: False)

        Returns
        -------
        : ee.Image
          The processed image
        """
        if masks is None:
            masks = self._get_image_masks(ee_image)
        if score is None:
            score = self._get_image_score(ee_image, masks=masks)

        ee_image = ee_image.addBands(ee.Image(list(masks.values())), overwrite=False)
        ee_image = ee_image.addBands(score, overwrite=False)

        if mask:  # mask before adding aux bands
            ee_image = ee_image.updateMask(self._masks['valid_mask'])

        if scale_refl:
            ee_image = self._scale_refl(ee_image)

        return self._im_transform(ee_image)


class LandsatImage(Image):

    @staticmethod
    def _im_transform(ee_image):
        return ee.Image.toUint16(ee_image)

    def _get_image_masks(self, ee_image):
        """
        Derive cloud, shadow, fill and validity masks for an image

        Parameters
        ----------
        ee_image : ee.Image
                Derive masks for this image

        Returns
        -------
        masks : dict
                A dictionary with cloud_mask, shadow_mask, fill_mask and valid_mask keys, and corresponding ee.Image
                values
        """

        # get cloud, shadow and fill masks from QA_PIXEL
        qa_pixel = ee_image.select('QA_PIXEL')
        cloud_mask = qa_pixel.bitwiseAnd((1 << 1) | (1 << 2) | (1 << 3)).neq(0).rename('CLOUD_MASK')
        shadow_mask = qa_pixel.bitwiseAnd(1 << 4).neq(0).rename('SHADOW_MASK')
        fill_mask = qa_pixel.bitwiseAnd(1).eq(0).rename('FILL_MASK')

        # extract cloud etc quality scores (2 bits)
        # cloud_conf = qa_pixel.rightShift(8).bitwiseAnd(3).rename('CLOUD_CONF')
        # cloud_shadow_conf = qa_pixel.rightShift(10).bitwiseAnd(3).rename('CLOUD_SHADOW_CONF')
        # cirrus_conf = qa_pixel.rightShift(14).bitwiseAnd(3).rename('CIRRUS_CONF')

        if self.gd_coll_name == 'landsat8_c2_l2':
            # TODO: is SR_QA_AEROSOL helpful? (Looks suspect for GEF region images)
            # Add SR_QA_AEROSOL to cloud mask
            # bits 6-7 of SR_QA_AEROSOL, are aerosol level where 3 = high, 2=medium, 1=low
            sr_qa_aerosol = ee_image.select('SR_QA_AEROSOL')
            aerosol_prob = sr_qa_aerosol.rightShift(6).bitwiseAnd(3)
            aerosol_mask = aerosol_prob.gt(2).rename('AEROSOL_MASK')
            cloud_mask = cloud_mask.Or(aerosol_mask)

        valid_mask = ((cloud_mask.Or(shadow_mask)).Not()).And(fill_mask).rename('VALID_MASK')
        masks = dict(cloud_mask=cloud_mask, shadow_mask=shadow_mask, fill_mask=fill_mask, valid_mask=valid_mask)

        return masks

    def _scale_refl(self, ee_image):
        """
        Scale and offset landsat pixels in SR bands to surface reflectance (0-10000)
        Uses hard coded ranges

        Parameters
        ----------
        ee_image : ee.Image
                image to scale and offset

        Returns
        -------
        : ee.Image
        Image with SR bands in range 0-10000
        """
        # retrieve the names of SR bands
        all_bands = ee_image.bandNames()
        init_bands = ee.List([])

        def add_refl_bands(band, refl_bands):
            refl_bands = ee.Algorithms.If(ee.String(band).rindex('SR_B').eq(0), ee.List(refl_bands).add(band),
                                          refl_bands)
            return refl_bands

        sr_bands = ee.List(all_bands.iterate(add_refl_bands, init_bands))
        non_sr_bands = all_bands.removeAll(sr_bands)  # all the other non-SR bands

        # scale to new range
        # low/high values from https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2?hl=en
        low = 0.2/2.75e-05
        high = low + 1/2.75e-05
        calib_image = ee_image.select(sr_bands).unitScale(low=low, high=high).multiply(10000.0)
        calib_image = calib_image.addBands(ee_image.select(non_sr_bands))
        calib_image = calib_image.updateMask(ee_image.mask())  # apply any existing mask to refl image

        # copy required system properties that are not copied in copyProperties below
        for key in ['system:index', 'system:id', 'id', 'system:time_start', 'system:time_end']:
            calib_image = calib_image.set(key, ee.String(ee_image.get(key)))

        return ee.Image(calib_image.copyProperties(ee_image))


class Landsat8Image(LandsatImage):
    _gd_coll_name = 'landsat8_c2_l2'


class Landsat7Image(LandsatImage):
    _gd_coll_name = 'landsat7_c2_l2'


class Sentinel2Image(Image):

    @staticmethod
    def _im_transform(ee_image):
        return ee.Image.toUint16(ee_image)

    def _get_image_masks(self, ee_image):
        """
        Derive cloud, shadow, fill and validity masks for an image

        Parameters
        ----------
        ee_image : ee.Image
                Derive masks for this image

        Returns
        -------
        masks : dict
                A dictionary with cloud_mask, shadow_mask, fill_mask and valid_mask keys, and corresponding ee.Image
                values
        """

        masks = Image._get_image_masks(self, ee_image)
        qa = ee_image.select('QA60')   # bits 10 and 11 are opaque and cirrus clouds respectively
        cloud_mask = qa.bitwiseAnd((1 << 11) | (1 << 10)).neq(0).rename('CLOUD_MASK')
        valid_mask = cloud_mask.Not().rename('VALID_MASK')
        masks.update(cloud_mask=cloud_mask, valid_mask=valid_mask)
        return masks


class Sentinel2SrImage(Sentinel2Image):
    _gd_coll_name = 'sentinel2_sr'


class Sentinel2ToaImage(Sentinel2Image):
    _gd_coll_name = 'sentinel2_toa'


class Sentinel2ClImage(Image):
    def __init__(self, ee_image, mask=False, scale_refl=False):
        """
        Earth engine image wrapper for cloud/shadow masking and quality scoring

        Parameters
        ----------
        ee_image : ee.Image
                   Earth engine image to wrap
        mask : bool, optional
               Apply a validity (cloud & shadow) mask to the image (default: False)
        scale_refl : bool, optional
                     Scale reflectance bands 0-10000 if they are not in that range already (default: False)
        """
        # TODO: provide cli access to these
        self._cloud_filter = 60  # Maximum image cloud cover percent allowed in image collection
        self._cloud_prob_thresh = 35  # Cloud probability (%); values greater than are considered cloud
        # self._nir_drk_thresh = 0.15# Near-infrared reflectance; values less than are considered potential cloud shadow
        self._cloud_proj_dist = 1  # Maximum distance (km) to search for cloud shadows from cloud edges
        self._buffer = 100  # Distance (m) to dilate the edge of cloud-identified objects

        Image.__init__(self, ee_image, mask=mask, scale_refl=scale_refl)


    @staticmethod
    def _im_transform(ee_image):
        return ee.Image.toUint16(ee_image)


    @classmethod
    def from_id(cls, image_id, mask=False, scale_refl=False):
        """
        Earth engine image wrapper for cloud/shadow masking and quality scoring

        Parameters
        ----------
        image_id : str
                   ID of earth engine image to wrap
        mask : bool, optional
               Apply a validity (cloud & shadow) mask to the image (default: False)
        scale_refl : bool, optional
                     Scale reflectance bands 0-10000 if they are not in that range already (default: False)
        """
        ee_coll_name = ee_split(image_id)[0]
        if ee_coll_name not in info.ee_to_gd:
            raise ValueError(f'Unsupported collection: {ee_coll_name}')

        gd_coll_name = info.ee_to_gd[ee_coll_name]
        if gd_coll_name != cls._gd_coll_name:
            raise ValueError(f'{cls.__name__} only supports images from the {info.gd_to_ee[cls._gd_coll_name]} collection')

        ee_image = ee.Image(image_id)

        # add cloud probability to the image
        cloud_prob = ee.Image(f'COPERNICUS/S2_CLOUD_PROBABILITY/{ee_split(image_id)[1]}')
        ee_image = ee_image.addBands(cloud_prob, overwrite=True)

        return cls(ee_image, mask=mask, scale_refl=scale_refl)

    def _get_image_masks(self, ee_image):
        """
        Derive cloud, shadow, fill and validity masks for an image

        Parameters
        ----------
        ee_image : ee.Image
                Derive masks for this image

        Returns
        -------
        masks : dict
                A dictionary with cloud_mask, shadow_mask, fill_mask and valid_mask keys, and corresponding ee.Image
                values
        """
        masks = Image._get_image_masks(self, ee_image)
        # qa = ee_image.select('QA60')   # bits 10 and 11 are opaque and cirrus clouds respectively
        # cloud_mask = qa.bitwiseAnd((1 << 11) | (1 << 10)).neq(0).rename('VALID_MASK')

        # convert cloud probability in 0-100 quality score
        # cloud_prob = ee.Image(ee_image.get('s2cloudless')).select('probability')
        # cloud_prob_id = ee.String('COPERNICUS/S2_CLOUD_PROBABILITY/').cat(ee_image.get('system:index'))
        cloud_prob = ee_image.select('probability')
        cloud_mask = cloud_prob.gt(self._cloud_prob_thresh).rename('CLOUD_MASK')

        # TODO: dilate valid_mask by _buffer ?
        # TODO: does below work in N hemisphere?
        # See https://en.wikipedia.org/wiki/Solar_azimuth_angle
        shadow_azimuth = ee.Number(-90).add(ee.Number(ee_image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
        min_scale = get_projection(ee_image).nominalScale()

        # project the the cloud mask in the direction of shadows
        proj_dist_px = ee.Number(self._cloud_proj_dist * 1000).divide(min_scale)
        proj_cloud_mask = (cloud_mask.directionalDistanceTransform(shadow_azimuth, proj_dist_px).
                           select('distance').mask().rename('PROJ_CLOUD_MASK'))     # mask converts to boolean?
        # .reproject(**{'crs': ee_image.select(0).projection(), 'scale': 100})

        if self.gd_coll_name == 'sentinel2_sr':  # use SCL to reduce shadow_mask
            # Note: SCL does not classify cloud shadows well, they are often labelled "dark".  Instead of using only
            # cloud shadow areas from this band, we combine it with the projected dark and shadow areas from s2cloudless
            scl = ee_image.select('SCL')
            dark_shadow_mask = (scl.eq(3).Or(scl.eq(2)).
                                focal_min(self._buffer, 'circle', 'meters').
                                focal_max(self._buffer, 'circle', 'meters'))
            shadow_mask = proj_cloud_mask.And(dark_shadow_mask).rename('SHADOW_MASK')
        else:
            shadow_mask = proj_cloud_mask.rename('SHADOW_MASK')  # mask all areas that could be cloud shadow

        # combine cloud and shadow masks
        valid_mask = (cloud_mask.Or(shadow_mask)).Not().rename('VALID_MASK')
        masks.update(cloud_mask=cloud_mask, shadow_mask=shadow_mask, valid_mask=valid_mask)
        return masks

    @classmethod
    def ee_collection(cls):
        """
        Return the unfiltered earth engine image collection (a join of the sentinel-2 reflectance and cloud probability
        collections)

        Returns
        -------
        : ee.ImageCollection
        """

        # adapted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless

        s2_sr_toa_col = ee.ImageCollection(info.gd_to_ee[cls._gd_coll_name])
                         #.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self._cloud_filter))) # TODO: add this back?
        s2_cloudless_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

        # join filtered s2cloudless collection to the SR/TOA collection by the 'system:index' property.
        filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
        inner_join = ee.ImageCollection(ee.Join.inner().apply(s2_sr_toa_col, s2_cloudless_col, filter))
        def map(feature):
            return ee.Image.cat(feature.get('primary'), feature.get('secondary'))
        return inner_join.map(map)


class Sentinel2SrClImage(Sentinel2ClImage):
    _gd_coll_name = 'sentinel2_sr'

class Sentinel2ToaClImage(Sentinel2ClImage):
    _gd_coll_name = 'sentinel2_toa'

class ModisNbarImage(Image):
    @staticmethod
    def _im_transform(ee_image):
        return ee.Image.toUint16(ee_image)
    _gd_coll_name = 'modis_nbar'

##

def get_class(coll_name):
    #TODO: populate this list by traversing the class heirarchy
    # import inspect
    # from geedim import image
    # def find_subclasses():
    #     image_classes = {cls._gd_coll_name: cls for name, cls in inspect.getmembers(image)
    #                      if inspect.isclass(cls) and issubclass(cls, image.Image) and not cls is image.Image}
    #
    #     return image_classes

    gd_coll_name_map = dict(
        landsat7_c2_l2=Landsat7Image,
        landsat8_c2_l2=Landsat8Image,
        sentinel2_toa=Sentinel2ToaClImage,
        sentinel2_sr=Sentinel2SrClImage,
        modis_nbar=ModisNbarImage
    )
    if coll_name in gd_coll_name_map:
        return gd_coll_name_map[coll_name]
    elif coll_name in info.ee_to_gd:
        return gd_coll_name_map[info.ee_to_gd[coll_name]]
    else:
        raise ValueError(f'Unknown collection name: {coll_name}')


if importlib.util.find_spec('rasterio'):
    import rasterio as rio
    from rasterio.warp import transform_geom
    def get_image_bounds(filename, expand=5):
        """
        Get a WGS84 geojson polygon representing the optionally expanded bounds of an image

        Parameters
        ----------
        filename :  str, pathlib.Path
                    name of the image file whose bounds to find
        expand :    int
                    percentage (0-100) by which to expand the bounds (default: 5)

        Returns
        -------
        bounds : geojson
                 polygon of bounds in WGS84
        crs: str
             WKT CRS string of image file
        """
        try:
            # GEE sets tif colorinterp tags incorrectly, suppress rasterio warning relating to this:
            # 'Sum of Photometric type-related color channels and ExtraSamples doesn't match SamplesPerPixel'
            logging.getLogger("rasterio").setLevel(logging.ERROR)
            with rio.open(filename) as im:
                bbox = im.bounds
                if (im.crs.linear_units == 'metre') and (expand > 0):  # expand the bounding box
                    expand_x = (bbox.right - bbox.left) * expand / 100.
                    expand_y = (bbox.top - bbox.bottom) * expand / 100.
                    bbox_expand = rio.coords.BoundingBox(bbox.left - expand_x, bbox.bottom - expand_y,
                                                         bbox.right + expand_x, bbox.top + expand_y)
                else:
                    bbox_expand = bbox

                coordinates = [[bbox_expand.right, bbox_expand.bottom],
                               [bbox_expand.right, bbox_expand.top],
                               [bbox_expand.left, bbox_expand.top],
                               [bbox_expand.left, bbox_expand.bottom],
                               [bbox_expand.right, bbox_expand.bottom]]

                bbox_expand_dict = dict(type='Polygon', coordinates=[coordinates])
                src_bbox_wgs84 = transform_geom(im.crs, 'WGS84', bbox_expand_dict)  # convert to WGS84 geojson
        finally:
            logging.getLogger("rasterio").setLevel(logging.WARNING)

        return src_bbox_wgs84, im.crs.to_wkt()


def ee_split(image_id):
    """ Split Earth Engine image ID to collection and index components """
    index = image_id.split('/')[-1]
    coll = '/'.join(image_id.split('/')[:-1])
    return coll, index


def get_image_info(image):
    """
    Retrieve image info, and create a pandas DataFrame of band properties

    Parameters
    ----------
    image : ee.Image

    Returns
    -------
    im_info_dict : dict
                   Image properties
    band_info_df : pandas.DataFrame
                   Band properties including scale
    """
    im_info_dict = image.getInfo()

    band_info_df = pd.DataFrame(im_info_dict['bands'])
    crs_transforms = band_info_df['crs_transform'].values
    scales = [abs(float(crs_transform[0])) for crs_transform in crs_transforms]
    band_info_df['scale'] = scales

    return im_info_dict, band_info_df

# Adapted from from https://github.com/gee-community/gee_tools, MIT license
def get_projection(image, min=True):
    """
    Server side operations to find the min/max scale projection from image bands.  No calls to getInfo().

    Parameters
    ----------
    image : ee.Image
            The image whose min/max projection to retrieve
    min: bool, optional
         Retrieve the projection corresponding to the band with the minimum (True) or maximum (False) scale
         [default: True]

    Returns
    -------
    : ee.Projection
      The projection with the smallest scale
    """

    bands = image.bandNames()
    init_proj = image.select(0).projection()

    if min:
        compare = ee.Number.lte
    else:
        compare = ee.Number.gte

    def compare_scale(name, prev_proj):
        prev_proj = ee.Projection(prev_proj)
        prev_scale = prev_proj.nominalScale()

        curr_proj = image.select([name]).projection()
        curr_scale = ee.Number(curr_proj.nominalScale())

        # exclude WGS84 bands (constant or composite bands)
        # (curr_scale <= / >= prev_scale) and (curr_proj.crs != EPSG:4326 )
        condition = (compare(curr_scale, prev_scale).
                     And(curr_proj.crs().compareTo(ee.String('EPSG:4326')))
                     .neq(ee.Number(0)))

        comp_proj = ee.Algorithms.If(condition, curr_proj, prev_proj)
        return ee.Projection(comp_proj)

    return ee.Projection(bands.iterate(compare_scale, init_proj))

