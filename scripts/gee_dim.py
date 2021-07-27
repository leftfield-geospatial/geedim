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
import argparse
import pathlib
from datetime import datetime

import dateutil
import ee
from geedim import search as dim_util
from geedim import get_logger

# conda install -c conda-forge earthengine-api
# conda install -c conda-forge folium
# rio bounds NGI_3322A_2010_Subsection_Source.vrt > NGI_3322A_2010_Subsection_Source_Bounds.geojson


# src_filename = pathlib.Path(r'V:\Data\HomonimEgs\NGI_3322A_2010_HotSpotSeamLineEg\Source\NGI_3322A_2010_HotSpotSeamLineEg_Source.vrt')    #2010-01-22 - 2010-02-01
# src_date = '2010-01-26'
# src_filename = pathlib.Path(r'V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Source\NGI_3323DA_2015_GefSite_Source.vrt')    #2010-01-22 - 2010-02-01
# src_date = '2015-05-08'
##

logger = get_logger(__name__)
def parse_arguments():
    collection_info = dim_util.load_collection_info()
    parser = argparse.ArgumentParser(description='Search and download surface reflectance imagery from Google Earth Engine (GEE)')
    # parser.add_argument('extent_file', help='path specifying source image/vector file whose spatial extent should be covered', type=str,
    #                     metavar='extent_file', nargs='+')
    parser.add_argument('extent_file', help='image file whose spatial extent should be covered',
                        type=str)
    parser.add_argument('-d', '--date', help='capture date to search around e.g. \'2015-01-28\' '
                                             '(default: use creation time of the <extent_file>)', type=str)
    parser.add_argument('-c', '--collection',
                        help='image collection to search: \'landsat7\'=LANDSAT/LE07/C02/T1_L2, '
                             '\'landsat8\'=LANDSAT/LC08/C02/T1_L2, \'sentinel2_toa\'=COPERNICUS/S2, \'sentinel2_sr\'=COPERNICUS/S2_SR, '
                             '\'modis\'=MODIS/006/MCD43A4, *=valid GEE image collection name',
                        choices=list(collection_info.keys()), type=str)
    parser.add_argument('-o', '--output_filename',
                        help='download zipfile name (default: save zip to current directory)',
                        type=str)
    return parser.parse_args()


def main(args):
    """
    Search and download surface reflectance imagery from Google Earth Engine (GEE)

    Parameters
    ----------
    args :  ArgumentParser.parse_args()
            Run `python get_gee_ref_im.py -h` to see help on arguments
    """

    ## check arguments
    if not pathlib.Path(args.extent_file).exists():
        raise Exception(f'Extent file {args.extent_file} does not exist')

    if args.output_filename is None:
        args.output_filename = pathlib.Path(args.extent_file).cwd().joinpath(f'{args.collection}.zip')
    else:
        args.output_filename = pathlib.Path(args.output_filename)

    if args.date is None:
        extent_ctime = pathlib.Path(args.extent_file).stat().st_ctime
        args.date = datetime.fromtimestamp(extent_ctime)
        logger.warning(f'No date specified, using file creation date: {args.date}')

    args.date = dateutil.parser.parse(args.date)

    ## initialise GEE
    ee.Authenticate()
    ee.Initialize()

    ## get extents and search
    if args.collection == 'landsat7':
        min_images = 2  # composite of >1 image to get rid of scanline error
    else:
        min_images = 1
    if False:
        src_bbox_wgs84, crs = dim_util.get_image_bounds(args.extent_file, expand=5)
        link, image = dim_util.search_image_collection(args.collection, src_bbox_wgs84, args.date, min_images=min_images,
                                                       crs=crs, bands=None, cloud_mask=lambda x: x)
    else:
        if 'landsat' in args.collection:
            ref_image = dim_util.EeLandsatRefImage(args.extent_file, collection=args.collection)
        elif 'sentinel' in args.collection:
            ref_image = dim_util.Sentinel2ImSearch(args.extent_file, collection=args.collection)
        else:
            ref_image = dim_util.ImSearch(args.extent_file, collection=args.collection)

        link, image = ref_image.search(args.date, min_images=min_images)

    dim_util.download_image(link, args.output_filename)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
