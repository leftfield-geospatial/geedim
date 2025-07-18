from pathlib import Path

# [initialise]
import ee

import geedim  # noqa: F401

ee.Initialize()
# [end initialise]

# [cloud support]
im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')

print(im.gd.cloudSupport)
# True
# [end cloud support]

# [mask]
# add mask bands created with a threshold of 0.7 on the 'cs_cdf' Cloud Score+ band
im = im.gd.addMaskBands(score=0.7, cs_band='cs_cdf')

print(im.gd.bandNames)
# ['B1', 'B2', 'B3', ..., 'CLOUD_SCORE', 'FILL_MASK', 'CLOUDLESS_MASK', 'CLOUD_DIST']

im = im.gd.maskClouds()
# [end mask]

# [filter]
region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

# filter by date range, region bounds, and a lower limit of 60% on the cloud-free
# portion of region
filt_coll = coll.gd.filter(
    '2021-10-01', '2022-04-01', region=region, cloudless_portion=60
)
# [end filter]

# [tables]
# include the VEGETATION_PERCENTAGE property in schemaTable & propertiesTable
filt_coll.gd.schemaPropertyNames += ('VEGETATION_PERCENTAGE',)

print(filt_coll.gd.schemaTable)
# ABBREV     NAME                             DESCRIPTION
# ---------  -------------------------------  ------------------------------------------------
# INDEX      system:index                     Earth Engine image index
# DATE       system:time_start                Image capture date/time (UTC)
# FILL       FILL_PORTION                     Portion of region pixels that are valid (%)
# CLOUDLESS  CLOUDLESS_PORTION                Portion of filled pixels that are cloud-free (%)
# ...        ...                              ...
# VP         VEGETATION_PERCENTAGE            Percentage of pixels classified as vegetation
print(filt_coll.gd.propertiesTable)
# INDEX                                  DATE               FILL CLOUDLESS ...      VP
# -------------------------------------- ---------------- ------ --------- ----- -----
# 20211006T075809_20211006T082043_T35HKC 2021-10-06 08:29 100.00     99.33 ...   22.25
# 20211021T075951_20211021T082750_T35HKC 2021-10-21 08:29 100.00    100.00 ...   14.52
# ...                                    ...              ...       ...    ...   ...
# 20220330T075611_20220330T082727_T35HKC 2022-03-30 08:29 100.00    100.00 ...   21.63
# [end tables]

# [composite]
comp_im = filt_coll.gd.composite(method='median')
# [end composite]

# [image grid]
im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')

print(im.gd.crs)
# EPSG:32735
print(im.gd.transform)
# (10, 0, 199980, 0, -10, 6300040)
print(im.gd.shape)
# (10980, 10980)
print(im.gd.dtype)
# uint32
# [end image grid]

# [image prepare for export]
region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
prep_im = im.gd.prepareForExport(
    crs='EPSG:3857', region=region, scale=30, dtype='uint16'
)
# [end image prepare for export]

Path('s2.tif').unlink(missing_ok=True)
# [image geotiff]
# create and prepare image
im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')
region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
prep_im = im.gd.prepareForExport(region=region, scale=30, dtype='uint16')

# export
prep_im.gd.toGeoTIFF('s2.tif')
# [end image geotiff]

# [image geotiff tags]
import rasterio as rio

with rio.open('s2.tif') as ds:
    # default namespace tags
    print(ds.tags())
    # {'AOT_RETRIEVAL_ACCURACY': '0', 'CLOUDY_PIXEL_PERCENTAGE': '7.464998', ...

    # band 1 tags
    print(ds.tags(bidx=1))
    # {'center_wavelength': '0.4439', 'description': 'Aerosols', 'gee-scale': '0.0001', ...
# [end image geotiff tags]

exp_path = Path('s2')
for f in exp_path.glob('*.tif'):
    f.unlink()
if exp_path.exists():
    exp_path.rmdir()
# fmt: off
# [coll geotiff]
from pathlib import Path

# create and prepare a collection (prepared collection has two images and three bands)
region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
coll = coll.filterBounds(region).limit(2)
prep_coll = coll.gd.prepareForExport(
    region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
)

# create export directory
dirname = Path('s2')
dirname.mkdir()

# export (one file for each collection band)
prep_coll.gd.toGeoTIFF(dirname, split='bands')

# print exported files
print(*dirname.glob('*.tif'))
# s2/B2.tif s2/B3.tif s2/B4.tif
# [end coll geotiff]
# fmt: on

Path('s2_nodata.tif').unlink(missing_ok=True)
# [geotiff nodata]
# set masked pixels to a new nodata value
nodata = 65535
prep_im = prep_im.unmask(nodata)

# export, setting the nodata tag to a custom value
prep_im.gd.toGeoTIFF('s2_nodata.tif', nodata=nodata)

# print file nodata
with rio.open('s2_nodata.tif') as ds:
    print(ds.nodata)
    # 65535.0
# [end geotiff nodata]

# [image numpy]
# create and prepare image
im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')
region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
prep_im = im.gd.prepareForExport(
    region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
)

# export (3D array with bands along the third dimension)
array = prep_im.gd.toNumPy()

# print array format
print(type(array))
# <class 'numpy.ndarray'>
print(array.shape)
# (379, 320, 3)
print(array.dtype)
# uint16
# [end image numpy]

# [coll numpy]
# create and prepare a collection (prepared collection has two images and three bands)
region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
coll = coll.filterBounds(region).limit(2)
prep_coll = coll.gd.prepareForExport(
    region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
)

# export (4D array with bands along the third, and images along the fourth dimension)
array = prep_coll.gd.toNumPy(split='bands')

# print array format
print(type(array))
# <class 'numpy.ndarray'>
print(array.shape)
# (379, 320, 3, 2)
print(array.dtype)
# uint16
# [end coll numpy]

# [numpy masked structured]
# export (2D masked array with a structured dtype representing the bands)
array = prep_im.gd.toNumPy(masked=True, structured=True)

# print array format
print(type(array))
# <class 'numpy.ma.MaskedArray'>
print(array.shape)
# (379, 320)
print(array.dtype)
# [('B4', '<u2'), ('B3', '<u2'), ('B2', '<u2')]
# [end numpy masked structured]

# [image xarray]
# create and prepare a cloud masked image
im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')
im = im.gd.addMaskBands().gd.maskClouds()
region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
prep_im = im.gd.prepareForExport(
    region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
)

# export (3D DataArray)
da = prep_im.gd.toXarray()

# examine data array
print(da)
# <xarray.DataArray (y: 379, x: 320, band: 3)> Size: 728kB
# array([[[ 427,  450,  343],
#          ...,
#         [1033,  996,  797]]], shape=(379, 320, 3), dtype=uint16)
# Coordinates:
#   * y        (y) float64 3kB 6.274e+06 6.274e+06 ... 6.262e+06 6.262e+06
#   * x        (x) float64 3kB 2.542e+05 2.543e+05 ... 2.638e+05 2.638e+05
#   * band     (band) <U2 24B 'B4' 'B3' 'B2'
# Attributes:
#     id:         COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T...
#     date:       2021-12-20T08:29:42.907+00:00
#     crs:        EPSG:32735
#     transform:  (30.0, 0.0, 254220.0, 0.0, -30.0, 6273760.0)
#     nodata:     0
#     ee:         {"system:footprint": {"geodesic": false, "crs": {"type": "nam...
#     stac:       {"description": "After 2022-01-25, Sentinel-2 scenes with PRO...
# [end image xarray]

# [coll xarray]
# create and prepare a collection (prepared collection has two images and three bands)
region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
coll = coll.filterBounds(region).limit(2)
prep_coll = coll.gd.prepareForExport(
    region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
)

# export (Dataset with bands as variables)
ds = prep_coll.gd.toXarray(split='bands')

# examine dataset
print(ds)
# <xarray.Dataset> Size: 1MB
# Dimensions:  (y: 379, x: 320, time: 2)
# Coordinates:
#   * y        (y) float64 3kB 6.274e+06 6.274e+06 ... 6.262e+06 6.262e+06
#   * x        (x) float64 3kB 2.542e+05 2.543e+05 ... 2.638e+05 2.638e+05
#   * time     (time) datetime64[ns] 16B 2018-05-10T08:29:44.389000 2018-12-16T...
# Data variables:
#     B4       (y, x, time) uint16 485kB 193 455 752 1041 357 ... 963 561 1192 617
#     B3       (y, x, time) uint16 485kB 232 515 645 919 418 ... 862 482 989 538
#     B2       (y, x, time) uint16 485kB 39 329 373 615 202 ... 617 330 713 369
# Attributes:
#     id:         COPERNICUS/S2_SR_HARMONIZED
#     crs:        EPSG:32735
#     transform:  (30, 0, 254220, 0, -30, 6273760)
#     nodata:     0
#     ee:         {"date_range": [1490659200000, 1647907200000], "period": 0, "...
#     stac:       {"description": "After 2022-01-25, Sentinel-2 scenes with PRO...>
# [end coll xarray]

# [xarray masked]
# export setting masked pixels to NaN
da = prep_im.gd.toXarray(masked=True)

print(da.isnull().any())
# <xarray.DataArray ()> Size: 1B
# array(True)
print(da.dtype)
# float32
# [end xarray masked]

# TODO: change folder='geedim' to folder='<project name>'?
# [image google cloud]
# create and prepare image
im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')
region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
prep_im = im.gd.prepareForExport(region=region, scale=30, dtype='uint16')

# export to Earth Engine asset 's2' in the 'geedim' project, waiting for completion
prep_im.gd.toGoogleCloud('s2', type='asset', folder='geedim', wait=True)

# print asset image info
print(ee.Image('projects/geedim/assets/s2').getInfo())
# {'type': 'Image', 'bands': [{'id': 'B1', 'data_type': {'type': 'PixelType', ...
# [end image google cloud]
ee.data.deleteAsset('projects/geedim/assets/s2')

# [coll google cloud]
# create and prepare a collection (prepared collection has two images and three bands)
region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
coll = coll.filterBounds(region).limit(2)
prep_coll = coll.gd.prepareForExport(
    region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
)

# export to earth engine assets in the 'geedim' project, waiting for completion
# (one asset for each collection band)
prep_coll.gd.toGoogleCloud(type='asset', folder='geedim', wait=True, split='bands')

# print info of the first asset image
print(ee.Image('projects/geedim/assets/B4').getInfo())
# {'type': 'Image', 'bands': [{'id': 'B_20180510T075611_20180510T082300_T35HKC', ...
# [end coll google cloud]
for bn in ['B4', 'B3', 'B2']:
    try:
        ee.data.deleteAsset(f'projects/geedim/assets/{bn}')
    except:
        pass

# [google cloud kwargs]
# export to Google Drive using the TFRecord format
prep_im.gd.toGoogleCloud(
    's2',
    type='drive',
    folder='geedim',
    fileFormat='TFRecord',
    formatOptions={'patchDimensions': [256, 256], 'compressed': True},
)
# [end google cloud kwargs]

# [mem limit]
# create and prepare an image
im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')
region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
prep_im = im.gd.prepareForExport(region=region, scale=30, dtype='uint16')

# export to Earth Engine asset 's2' in the 'geedim' project
prep_im.gd.toGoogleCloud('s2', type='asset', folder='geedim', wait=True)

# export the asset to a NumPy array
array = ee.Image('projects/geedim/assets/s2').gd.toNumPy()
# [end mem limit]
ee.data.deleteAsset('projects/geedim/assets/s2')
