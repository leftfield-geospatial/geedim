"""
    Homonim: Radiometric homogenisation of aerial and satellite imagery
    Copyright (C) 2021 Dugal Harris
    Email: dugalh@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import argparse
import pathlib
from datetime import datetime

import dateutil
import ee
from geedim import utils as imutil
from homonim import get_logger

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
    collection_info = imutil.load_collection_info()
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
        src_bbox_wgs84, crs = imutil.get_image_bounds(args.extent_file, expand=5)
        link, image = imutil.search_image_collection(args.collection, src_bbox_wgs84, args.date, min_images=min_images,
                                                     crs=crs, bands=None, cloud_mask=lambda x: x)
    else:
        if 'landsat' in args.collection:
            ref_image = imutil.EeLandsatRefImage(args.extent_file, collection=args.collection)
        elif 'sentinel' in args.collection:
            ref_image = imutil.Sentinel2EeImage(args.extent_file, collection=args.collection)
        else:
            ref_image = imutil.EeRefImage(args.extent_file, collection=args.collection)

        link, image = ref_image.search(args.date, min_images=min_images)

    imutil.download_image(link, args.output_filename)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
