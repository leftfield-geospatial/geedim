[comment]: <> ([![Publish]&#40;https://github.com/dugalh/geedim/actions/workflows/publish-pypi.yml/badge.svg&#41;]&#40;https://github.com/dugalh/geedim/actions/workflows/publish-pypi.yml&#41;)
[![Tests](https://github.com/dugalh/geedim/actions/workflows/run-unit-tests.yml/badge.svg)](https://github.com/dugalh/geedim/actions/workflows/run-unit-tests.yml)
[![PyPI version](https://badge.fury.io/py/geedim.svg)](https://badge.fury.io/py/geedim)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/geedim/badges/version.svg)](https://anaconda.org/conda-forge/geedim)
[![codecov](https://codecov.io/gh/dugalh/geedim/branch/main/graph/badge.svg?token=69GZNQ3TI3)](https://codecov.io/gh/dugalh/geedim)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# `geedim`
Search, composite, and download [Google Earth Engine](https://earthengine.google.com/) imagery, without size limits. 

## Description
`geedim` provides a command line interface and API for searching, compositing and downloading satellite imagery from Google Earth Engine (EE).  It optionally performs cloud/shadow masking, and cloud/shadow-free compositing on supported collections.  Images and composites can be downloaded, or exported to Google Drive.  Images larger than the EE size limit are split and downloaded as separate tiles, then re-assembled into a single GeoTIFF.   

### Cloud/shadow masking collections
`geedim` supports cloud/shadow masking on the following surface/TOA reflectance image collections:

`geedim` name | EE name| Description
---------|-----------|------------
landsat4_c2_l2 | [LANDSAT/LT04/C02/T1_L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT04_C02_T1_L2) | Landsat 4, collection 2, tier 1, level 2 surface reflectance 
landsat5_c2_l2 | [LANDSAT/LT05/C02/T1_L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2) | Landsat 5, collection 2, tier 1, level 2 surface reflectance 
landsat7_c2_l2 | [LANDSAT/LE07/C02/T1_L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2) | Landsat 7, collection 2, tier 1, level 2 surface reflectance 
landsat8_c2_l2 | [LANDSAT/LC08/C02/T1_L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2) | Landsat 8, collection 2, tier 1, level 2 surface reflectance 
landsat9_c2_l2 | [LANDSAT/LC09/C02/T1_L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2) | Landsat 9, collection 2, tier 1, level 2 surface reflectance 
sentinel2_toa | [COPERNICUS/S2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2) | Sentinel-2, level 1C, top of atmosphere reflectance 
sentinel2_sr | [COPERNICUS/S2_SR](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR) | Sentinel-2, level 2A, surface reflectance

## Requirements
`geedim` is a python 3 library, and requires users to be registered with [Google Earth Engine](https://signup.earthengine.google.com).

## Installation
`geedim` is available via `pip` and `conda`.  Under Windows, using `conda` is the easiest way to resolve binary dependencies. 
### conda
The [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installation provides a minimal `conda`.
```shell
conda install -c conda-forge geedim
```
### pip
```shell
pip install geedim
```

Alternatively, the repository can be cloned and linked into your python environment with:
```shell
git clone https://github.com/dugalh/geedim.git
pip install -e geedim
```
Following installation, Earth Engine must be authenticated:
```shell
earthengine authenticate
```
## Quick Start
Search for Landsat 8 images.
```shell
geedim search -c landsat8_c2_l2 -s 2021-06-01 -e 2021-07-01 --bbox 24 -33 24.1 -33.1
```
Download Landsat 8 image 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20210610' with cloud/shadow mask.
```shell
geedim download -i LANDSAT/LC08/C02/T1_L2/LC08_172083_20210610 --bbox 24 -33 24.1 -33.1 --mask
```
Composite the results of a search, then download with specified CRS and scale (pixel size).
```shell
geedim search -c landsat8_c2_l2 -s 2021-06-01 -e 2021-07-01 --bbox 24 -33 24.1 -33.1 composite download --crs EPSG:32634 --scale 30
```

## Usage
### Command line interface
`geedim` command line functionality is accessed through sub-commands: `search`, `composite`, `download` and `export`.  Sub-commands can be chained.
```
geedim --help
```
```
Usage: geedim [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

Options:
  -v, --verbose  Increase verbosity.
  -q, --quiet    Decrease verbosity.
  --version      Show the version and exit.
  --help         Show this message and exit.

Commands:
  composite  Create a cloud-free composite image.
  config     Configure cloud/shadow masking.
  download   Download image(s).
  export     Export image(s) to Google Drive.
  search     Search for images.
```

### Search
Search for images with filtering by image collection, date, region and portion of cloud/shadow free pixels (in the specified region).  Image metadata of interest is included in the results. 
```shell
geedim search --help
```
```
Usage: geedim search [OPTIONS]

  Search for images.

Options:
  -c, --collection TEXT           Earth Engine image collection to search.  [l
                                  andsat4_c2_l2|landsat5_c2_l2|landsat7_c2_l2|
                                  landsat8_c2_l2|landsat9_c2_l2|sentinel2_toa|
                                  sentinel2_sr|modis_nbar], or any valid Earth
                                  Engine image collection ID.  [default:
                                  landsat8_c2_l2]
  -s, --start-date [%Y-%m-%d|%Y-%m-%dT%H:%M:%S|%Y-%m-%d %H:%M:%S]
                                  Start date (UTC).  [required]
  -e, --end-date [%Y-%m-%d|%Y-%m-%dT%H:%M:%S|%Y-%m-%d %H:%M:%S]
                                  End date (UTC).
                                  [default: start_date + 1 day]
  -b, --bbox FLOAT...             Region defined by bounding box co-ordinates
                                  in WGS84 (xmin, ymin, xmax, ymax).
  -r, --region FILE               Region defined by geojson or raster file.
                                  Use "-" to read geojson from stdin.  [One of
                                  --bbox or --region is required]
  -cp, --cloudless-portion FLOAT RANGE
                                  Lower limit on the cloud/shadow free portion
                                  of the region (%).  [default: 0; 0<=x<=100]
  -o, --output FILE               Write search results to this json file
  --help                          Show this message and exit.
```
#### Example
```shell
geedim search -c landsat8_c2_l2 -b 23.9 -33.6 24 -33.5 -s 2019-01-01 -e 2019-02-01
```

### Download / Export
Download or export image(s) by specifying their ID(s).  Search result image(s) can be downloaded / exported by chaining the `search` and `download` / `export` sub-commands.  Images exceeding EE download limits are split and downloaded in smaller tiles, which are then re-assembled on the client into a GeoTIFF file.  The following auxiliary bands are included in images downloaded / exported from supported [cloud/shadow masking collections](#cloudshadow-masking-collections):

- FILL_MASK: Filled / captured pixels 
- CLOUD_MASK: Cloudy pixels 
- SHADOW_MASK: Shadow pixels
- CLOUDLESS_MASK: Filled and cloud / shadow-free pixels
- CLOUD_DIST: Distance to nearest cloud or shadow (m) 

#### Download
```shell
geedim download --help
```
```
Usage: geedim download [OPTIONS]

  Download image(s).

Options:
  -i, --id TEXT                   Earth engine image ID(s).
  -b, --bbox FLOAT...             Region defined by bounding box co-ordinates
                                  in WGS84 (xmin, ymin, xmax, ymax).
  -r, --region FILE               Region defined by geojson or raster file.
                                  Use "-" to read geojson from stdin.
  -dd, --download-dir DIRECTORY   Download image file(s) to this directory.
                                  [default: cwd]
  -c, --crs TEXT                  Reproject image(s) to this CRS (EPSG string
                                  or path to text file containing WKT).
                                  [default: source CRS]
  -s, --scale FLOAT               Resample image bands to this pixel
                                  resolution (m).
                                  [default: minimum of the source band
                                  resolutions]
  -dt, --dtype [int8|uint8|uint16|int16|uint32|int32|float32|float64]
                                  Convert image(s) to this data type.
  -m, --mask / -nm, --no-mask     Do/don't apply (cloud and shadow) nodata
                                  mask(s).  [default: --no-mask]
  -rs, --resampling [near|bilinear|bicubic]
                                  Resampling method.  [default: near]
  -o, --overwrite                 Overwrite the destination file if it exists.
                                  [default: don't overwrite]
  --help                          Show this message and exit.
```

#### Export
```shell
geedim export --help
```
```
Usage: geedim export [OPTIONS]

  Export image(s) to Google Drive.

Options:
  -i, --id TEXT                   Earth engine image ID(s).
  -b, --bbox FLOAT...             Region defined by bounding box co-ordinates
                                  in WGS84 (xmin, ymin, xmax, ymax).
  -r, --region FILE               Region defined by geojson or raster file.
                                  Use "-" to read geojson from stdin.
  -df, --drive-folder TEXT        Export image(s) to this Google Drive folder.
                                  [default: root]
  -c, --crs TEXT                  Reproject image(s) to this CRS (EPSG string
                                  or path to text file containing WKT).
                                  [default: source CRS]
  -s, --scale FLOAT               Resample image bands to this pixel
                                  resolution (m).
                                  [default: minimum of the source band
                                  resolutions]
  -dt, --dtype [int8|uint8|uint16|int16|uint32|int32|float32|float64]
                                  Convert image(s) to this data type.
  -m, --mask / -nm, --no-mask     Do/don't apply (cloud and shadow) nodata
                                  mask(s).  [default: --no-mask]
  -rs, --resampling [near|bilinear|bicubic]
                                  Resampling method.  [default: near]
  -w, --wait / -nw, --no-wait     Wait / don't wait for export to complete.
                                  [default: --wait]
  --help                          Show this message and exit.
```

#### Examples
```shell
geedim download -i LANDSAT/LC08/C02/T1_L2/LC08_172083_20190128 -b 23.9 -33.6 24 -33.5 --resampling bilinear --mask
```
```shell
geedim export -i LANDSAT/LC08/C02/T1_L2/LC08_172083_20190128 -b 23.9 -33.6 24 -33.5 -df geedim_test --mask
```

### Composite
Form a single composite image from a specified stack, using one of the following methods:
- `q_mosaic`: Use the pixel with the highest score (i.e. distance to cloud / shadow).
- `mosaic`: Use the first unmasked pixel.
- `median`: Use the median of the unmasked pixels.
- `medoid`: Use the [medoid](https://www.mdpi.com/2072-4292/5/12/6481) of the unmasked pixels.


The `composite` sub-command must be chained with one of `download` / `export` to get the resulting composite image.  It can also be chained with `search` to form a composite of the search results.  


```shell
geedim composite --help
```
```
Usage: geedim composite [OPTIONS]

  Create a cloud-free composite image.

Options:
  -i, --id TEXT                   Earth engine image ID(s).
  -cm, --method [q-mosaic|mosaic|medoid|median|mode|mean]
                                  Compositing method to use.  [default:
                                  q-mosaic]
  -m, --mask / -nm, --no-mask     Do/don't apply (cloud and shadow) nodata
                                  mask(s) before compositing.  [default:
                                  --mask]
  -rs, --resampling [near|bilinear|bicubic]
                                  Resampling method.  [default: near]
  -b, --bbox FLOAT...             Region defined by bounding box co-ordinates
                                  in WGS84 (xmin, ymin, xmax, ymax).
  -r, --region FILE               Region defined by geojson or raster file.
                                  Use "-" to read geojson from stdin.
  -d, --date [%Y-%m-%d|%Y-%m-%dT%H:%M:%S|%Y-%m-%d %H:%M:%S]
                                  Give preference to images closest to this
                                  date (UTC).  [Supported by `mosaic` and
                                  `q-mosaic` methods.]
  --help                          Show this message and exit.
```
#### Example
Composite the results of a search and download the result.
```shell
geedim search -c landsat8_c2_l2 -s 2019-02-01 -e 2019-03-01 --bbox 23 -33 23.2 -33.2 composite -cm q-mosaic --mask download --scale 30 --crs EPSG:3857
```

### Config
Configure cloud/shadow masking for [supported collections](#cloudshadow-masking-collections).  `config` can be chained with any of the above sub-commands to configure cloud/shadow masking for that, and subsequent sub-command(s).  

```shell
geedim config --help
```
```
Usage: geedim config [OPTIONS]

  Configure cloud/shadow masking.

Options:
  -mc, --mask-cirrus / -nmc, --no-mask-cirrus
                                  Whether to mask cirrus clouds. For sentinel2
                                  collections this is valid just for method =
                                  `qa`.  [default: --mask-cirrus]
  -ms, --mask-shadows / -nms, --no-mask-shadows
                                  Whether to mask cloud shadows. [default:
                                  --mask-shadows]
  -mm, --mask-method [cloud-prob|qa]
                                  Method used to cloud mask Sentinel-2 images.
                                  [default: cloud-prob]
  -p, --prob FLOAT RANGE          Cloud probability threshold. Valid just for
                                  --mask-method `cloud-prob`. (%).  [default:
                                  60; 0<=x<=100]
  -d, --dark FLOAT RANGE          NIR reflectance threshold [0-1] for shadow
                                  masking Sentinel-2 images. NIR values below
                                  this threshold are potential cloud shadows.
                                  [default: 0.15; 0<=x<=1]
  -sd, --shadow-dist INTEGER      Maximum distance in meters (m) to look for
                                  cloud shadows from cloud edges.  Valid for
                                  Sentinel-2 images.  [default: 1000]
  -b, --buffer INTEGER            Distance in meters (m) to dilate cloud and
                                  cloud shadows objects.  Valid for Sentinel-2
                                  images.  [default: 250]
  -cdi, --cdi-thresh FLOAT RANGE  Cloud Displacement Index threshold. Values
                                  below this threshold are considered
                                  potential clouds.  A cdi-thresh = None means
                                  that the index is not used.  Valid for
                                  Sentinel-2 images.  [default: None]
                                  [-1<=x<=1]
  -mcd, --max-cloud-dist INTEGER  Maximum distance in meters (m) to look for
                                  clouds.  Used to form the cloud distance
                                  band for `q-mosaic` compositing. Valid for
                                  Sentinel-2 images.  [default: 5000]
  --help                          Show this message and exit.
```

## API
### Example

```python
import ee
from geedim import MaskedImage, MaskedCollection

ee.Initialize()  # initialise earth engine

# geojson region to search / download
region = {
    "type": "Polygon",
    "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]
}

# make collection and search
gd_collection = MaskedCollection.from_name('COPERNICUS/S2_SR')
gd_collection = gd_collection.search('2019-01-10', '2019-01-21', region)
print(gd_collection.key_table)
print(gd_collection.properties_table)

# create and download an image
im = MaskedImage.from_id('COPERNICUS/S2_SR/20190115T080251_20190115T082230_T35HKC')
im.download('s2_image.tif', region=region)

# composite search results and download
comp_image = gd_collection.composite()
comp_image.download('s2_comp_image.tif', region=region, crs='EPSG:32735', scale=30)
```

## Known limitations
- There is a GEE bug exporting MODIS in its native CRS, this can be worked around by re-projecting to another CRS with the `--crs` `download`/`export` option.  Please star [the issue](https://issuetracker.google.com/issues/194561313) if it affects you.

## License
This project is licensed under the terms of the [Apache-2.0 License](LICENSE).

## Contributing
Contributions are welcome.  Report bugs or contact me with questions [here](https://github.com/dugalh/geedim/issues).

## Credits
- Tiled downloading was inspired by and adapted from [GEES2Downloader](https://github.com/cordmaur/GEES2Downloader) under terms of the [MIT license](https://github.com/cordmaur/GEES2Downloader/blob/main/LICENSE). 
- Sentinel-2 cloud/shadow masking was adapted from [ee_extra](https://github.com/r-earthengine/ee_extra) under terms of the [Apache-2.0 license](https://github.com/r-earthengine/ee_extra/blob/master/LICENSE)
- Medoid compositing was adapted from [gee_tools](https://github.com/gee-community/gee_tools) under the terms of the [MIT license](https://github.com/gee-community/gee_tools/blob/master/LICENSE).
- The CLI design was informed by [landsatxplore](https://github.com/yannforget/landsatxplore).

## Author
**Dugal Harris** - [dugalh@gmail.com](mailto:dugalh@gmail.com)

