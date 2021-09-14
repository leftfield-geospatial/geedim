[![Publish](https://github.com/dugalh/geedim/actions/workflows/publish.yml/badge.svg)](https://github.com/dugalh/geedim/actions/workflows/publish.yml)
[![Tests](https://github.com/dugalh/geedim/actions/workflows/test.yml/badge.svg)](https://github.com/dugalh/geedim/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/dugalh/geedim/branch/main/graph/badge.svg?token=69GZNQ3TI3)](https://codecov.io/gh/dugalh/geedim)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# geedim
Searching, compositing and downloading of satellite imagery from [Google Earth Engine](https://earthengine.google.com/) (EE). 
## Description
geedim provides a command line interface (CLI) and API that allow searching by date, region, and cloud/shadow statistics.  It optionally performs cloud/shadow masking, and cloud-free compositing.  Images and composites (including metadata) can be downloaded, or exported to Google Drive.

It supports access to the following surface reflectance image collections:

geedim name | EE name| Description
---------|-----------|------------
landsat7_c2_l2 | [LANDSAT/LE07/C02/T1_L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2) | Landsat 7, collection 2, tier 1, level 2 surface reflectance 
landsat8_c2_l2 | [LANDSAT/LC08/C02/T1_L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2) | Landsat 8, collection 2, tier 1, level 2 surface reflectance 
sentinel2_toa | [COPERNICUS/S2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2) | Sentinel-2, level 1C, top of atmosphere reflectance 
sentinel2_sr | [COPERNICUS/S2_SR](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR) | Sentinel-2, level 2A, surface reflectance
modis_nbar | [MODIS/006/MCD43A4](https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD43A4) | MODIS nadir BRDF adjusted reflectance

## Requirements
geedim is a python 3 library, and requires users to be registered with [Google Earth Engine](https://signup.earthengine.google.com).

## Installation
It can be installed from [PyPI](https://pypi.org) with:
```
pip install geedim
````
Alternatively, the repository can be cloned and linked into your python environment with:
```
git clone https://github.com/dugalh/geedim.git
pip install -e geedim
```
Following installation, Earth Engine must be authenticated:
```
earthengine authenticate
```
## Quick Start
Search for Landsat 8 images
```
geedim search -c landsat8_c2_l2 -s 2021-06-01 -e 2021-07-01 --bbox 24 -33 24.1 -33.1
```
Download Landsat 8 image 'LANDSAT/LC08/C02/T1_L2/LC08_172083_20210610' with cloud/shadow mask
```
geedim download -i LANDSAT/LC08/C02/T1_L2/LC08_172083_20210610 --bbox 24 -33 24.1 -33.1 --mask
```
Composite the results of the above search and download, specifying destination CRS and pixel size
```
geedim search -c landsat8_c2_l2 -s 2021-06-01 -e 2021-07-01 --bbox 24 -33 24.1 -33.1 composite download --crs EPSG:32634 --scale 30
```

## Usage
### Command line interface
geedim command line functionality is accessed through sub-commands: `search`, `composite`, `download` and `export`.  
```
geedim --help
```
```
Usage: geedim [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

Options:
  --help  Show this message and exit.

Commands:
  composite  Create a cloud-free composite image
  download   Download image(s), with cloud and shadow masking
  export     Export image(s) to Google Drive, with cloud and shadow masking
  search     Search for images
```
### Search
Searching is filtered by date, region and optionally, the portion of cloud/shadow free pixels in the specified region.  Other image metadata of interest is included in the results. 
```
geedim search --help
```
```
Usage: geedim search [OPTIONS]

  Search for images

Options:
  -c, --collection [landsat7_c2_l2|landsat8_c2_l2|sentinel2_toa|sentinel2_sr|modis_nbar]
                                  Earth Engine image collection to search.
                                  [default: landsat8_c2_l2]

  -s, --start-date [%Y-%m-%d|%Y-%m-%dT%H:%M:%S|%Y-%m-%d %H:%M:%S]
                                  Start date (UTC).  [required]
  -e, --end-date [%Y-%m-%d|%Y-%m-%dT%H:%M:%S|%Y-%m-%d %H:%M:%S]
                                  End date (UTC).   [default: start_date + 1
                                  day]

  -b, --bbox FLOAT...             Region defined by bounding box co-ordinates
                                  in WGS84 (xmin, ymin, xmax, ymax).  [One of
                                  --bbox or --region is required.]

  -r, --region FILE               Region defined by geojson or raster file.
                                  [One of --bbox or --region is required.]

  -vp, --valid-portion FLOAT RANGE
                                  Lower limit of the portion of valid (cloud
                                  and shadow free) pixels (%).  [default: 0]

  -o, --output FILE               Write results to this filename, file type
                                  inferred from extension: [.csv|.json]

  -m, --mask / -nm, --no-mask     Do/don't apply (cloud and shadow) nodata
                                  mask(s).  [default: --no-mask]

  -sr, --scale-refl / -nsr, --no-scale-refl
                                  Scale reflectance bands from 0-10000.
                                  [default: --no-scale-refl]

  --help                          Show this message and exit.
```
#### Example
```
geedim search -c landsat8_c2_l2 -b 23.9 -33.6 24 -33.5 -s 2019-01-01 -e 2019-02-01
```

### Download / Export
Images can be downloaded / exported by specifying their ID(s), or search result image(s) can be downloaded / exported by chaining the `search` and `download` / `export` sub-commands.  The following auxiliary bands are included in downloaded / exported images:

- FILL_MASK: Filled / captured pixels 
- CLOUD_MASK: Cloudy pixels 
- SHADOW_MASK: Shadow pixels
- VALID_MASK: Filled and cloud / shadow-free pixels
- SCORE: Distance to nearest cloud or shadow (m) 

#### Download
```
geedim download --help
```
```
Usage: geedim download [OPTIONS]

  Download image(s), with cloud and shadow masking

Options:
  -i, --id TEXT                   Earth engine image ID(s).
  -b, --bbox FLOAT...             Region defined by bounding box co-ordinates
                                  in WGS84 (xmin, ymin, xmax, ymax).  [One of
                                  --bbox or --region is required.]

  -r, --region FILE               Region defined by geojson or raster file.
                                  [One of --bbox or --region is required.]

  -dd, --download-dir DIRECTORY   Download image file(s) to this directory.
                                  [default: cwd]

  -c, --crs TEXT                  Reproject image(s) to this CRS (WKT or EPSG
                                  string).  [default: source CRS]

  -s, --scale FLOAT               Resample image bands to this pixel
                                  resolution (m).  [default: minimum of the
                                  source band resolutions]

  -m, --mask / -nm, --no-mask     Do/don't apply (cloud and shadow) nodata
                                  mask(s).  [default: --no-mask]

  -sr, --scale-refl / -nsr, --no-scale-refl
                                  Scale reflectance bands from 0-10000.
                                  [default: --no-scale-refl]

  -o, --overwrite                 Overwrite the destination file if it exists.
                                  [default: prompt the user for confirmation]

  --help                          Show this message and exit.
  ```

#### Export
```
geedim export --help
```
```
Usage: geedim export [OPTIONS]

  Export image(s) to Google Drive, with cloud and shadow masking

Options:
  -i, --id TEXT                   Earth engine image ID(s).
  -b, --bbox FLOAT...             Region defined by bounding box co-ordinates
                                  in WGS84 (xmin, ymin, xmax, ymax).  [One of
                                  --bbox or --region is required.]

  -r, --region FILE               Region defined by geojson or raster file.
                                  [One of --bbox or --region is required.]

  -df, --drive-folder TEXT        Export image(s) to this Google Drive folder.
                                  [default: root]

  -c, --crs TEXT                  Reproject image(s) to this CRS (WKT or EPSG
                                  string).  [default: source CRS]

  -s, --scale FLOAT               Resample image bands to this pixel
                                  resolution (m).  [default: minimum of the
                                  source band resolutions]

  -m, --mask / -nm, --no-mask     Do/don't apply (cloud and shadow) nodata
                                  mask(s).  [default: --no-mask]

  -sr, --scale-refl / -nsr, --no-scale-refl
                                  Scale reflectance bands from 0-10000.
                                  [default: --no-scale-refl]

  -w, --wait / -nw, --no-wait     Wait / don't wait for export to complete.
                                  [default: --wait]

  --help                          Show this message and exit.
```
#### Examples
```
geedim download -i LANDSAT/LC08/C02/T1_L2/LC08_172083_20190128 -b 23.9 -33.6 24 -33.5 --scale-refl --mask
```
```
geedim export -i LANDSAT/LC08/C02/T1_L2/LC08_172083_20190128 -b 23.9 -33.6 24 -33.5 -df geedim_test --scale-refl --mask
```

### Composite
Form a single composite image from a specified stack, using one of the following methods:
- `q_mosaic`: Use the pixel with the highest score (i.e. distance to cloud / shadow).
- `mosaic`: Use the first unmasked pixel.
- `median`: Use the median of the unmasked pixels.
- `medoid`: Use the [medoid](https://www.mdpi.com/2072-4292/5/12/6481) of the unmasked pixels.


The `composite` sub-command must be chained with one of `download` / `export` to get the resulting composite image.  It can also be chained with `search` to form a composite of the search results.  


```
geedim composite --help
```
```
Usage: geedim composite [OPTIONS]

  Create a cloud-free composite image

Options:
  -i, --id TEXT                   Earth engine image ID(s).
  -cm, --method [q_mosaic|mosaic|median|medoid]
                                  Compositing method to use.  [default:
                                  q_mosaic]

  -m, --mask / -nm, --no-mask     Do/don't apply (cloud and shadow) nodata
                                  mask(s) before compositing.  [default:
                                  --mask]

  -sr, --scale-refl / -nsr, --no-scale-refl
                                  Scale reflectance bands from 0-10000.
                                  [default: --no-scale-refl]

  --help                          Show this message and exit.
```
#### Example
Composite the results of a search and download the result.
```
geedim search -c landsat8_c2_l2 -s 2019-02-01 -e 2019-03-01 --bbox 23 -33 23.2 -33.2 --mask composite -cm q_mosaic download --scale 30 --crs EPSG:3857
```
## API
### Example 
```python
import ee
from geedim import collection, image, export

ee.Initialize()     #initialise earth engine

# geojson region to search / download
region = {"type": "Polygon",
          "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]}

# make collection and search
gd_collection = collection.Collection('sentinel2_sr')
res_df = gd_collection.search('2019-01-10', '2019-01-21', region)
print(gd_collection.summary_key)
print(gd_collection.summary)

# create and download an image
im = image.Sentinel2SrClImage.from_id('COPERNICUS/S2_SR/20190115T080251_20190115T082230_T35HKC')
export.download_image(im, 's2_image.tif', region=region)

# composite search results and download
comp_res = gd_collection.composite()
export.download_image(comp_res.image, 's2_comp_image.tif', region=region, crs='EPSG:32735', scale=30)

```

## Known limitations
- GEE limits image downloads to 10MB, images larger than this can exported to Google Drive.
- There is a GEE bug exporting MODIS in its native CRS, this can be worked around by re-projecting to another CRS with the `--crs` `download`/`export` option.  Please star [the issue](https://issuetracker.google.com/issues/194561313) if it affects you.

## License
This project is licensed under the terms of the [Apache-2.0 License](LICENSE).

## Contributing
Contributions are welcome!  Please report any bugs or contact me with questions [here](https://github.com/dugalh/geedim/issues).

## Author
**Dugal Harris** - [dugalh@gmail.com](mailto:dugalh@gmail.com)

## Credits
- Medoid compositing was adapted from [gee_tools](https://github.com/gee-community/gee_tools) under the terms of the [MIT license](https://github.com/gee-community/gee_tools/blob/master/LICENSE).
- The CLI was informed by [landsatxplore](https://github.com/yannforget/landsatxplore).