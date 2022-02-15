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
# Functions to download and export Earth Engine images

import os
import pathlib
import re
import time
import zipfile
from xml.dom import minidom
from xml.etree import ElementTree

import click
import ee
import numpy as np
import requests

from geedim import image

_footprint_key = "system:footprint"


def write_pam_xml(obj, filename):
    """
    Write image metadata to PAM xml file (as supported by GDAL/QGIS)

    Parameters
    ----------
    obj : ee.Image, geedim.image.Image, dict
          Image object whose metadata to write
    filename : str, pathlib.Path
               Path of the output file (use *.aux.xml)
    """
    # extract image metadata dict
    if isinstance(obj, dict):
        gd_info = obj
    elif isinstance(obj, ee.Image):
        gd_info = image.get_info(obj)
    elif isinstance(obj, image.Image):
        gd_info = obj.info
    else:
        raise TypeError(f"Unsupported type: {obj.__class__}")

    # remove footprint if it exists
    if _footprint_key in gd_info["properties"]:
        gd_info["properties"].pop(_footprint_key)

    # construct xml tree
    root = ElementTree.Element("PAMDataset")
    prop_meta = ElementTree.SubElement(root, "Metadata")
    for key, val in gd_info["properties"].items():
        item = ElementTree.SubElement(prop_meta, "MDI", attrib=dict(key=key))
        item.text = str(val)

    for band_i, band_dict in enumerate(gd_info["bands"]):
        band_elem = ElementTree.SubElement(root, "PAMRasterBand", attrib=dict(band=str(band_i + 1)))
        if "id" in band_dict:
            desc = ElementTree.SubElement(band_elem, "Description")
            desc.text = band_dict["id"]
        band_meta = ElementTree.SubElement(band_elem, "Metadata")
        for key, val in band_dict.items():
            item = ElementTree.SubElement(band_meta, "MDI", attrib=dict(key=key.upper()))
            item.text = str(val)

    # write indented xml string to file (excluding header)
    xml_str = minidom.parseString(ElementTree.tostring(root)).childNodes[0].toprettyxml(indent="   ")
    with open(filename, "w") as f:
        f.write(xml_str)


class _ExportImage(image.Image):
    """ Helper class for determining export/download crs, scale and region parameters"""

    def __init__(self, image_obj, name="Image", exp_region=None, exp_crs=None, exp_scale=None, resampling='near'):
        if isinstance(image_obj, image.Image):
            image.Image.__init__(self, image_obj.ee_image)
            self._info = image_obj.info
        else:
            image.Image.__init__(self, image_obj)

        self.name = name
        self.exp_region = exp_region
        self.exp_crs = exp_crs
        self.exp_scale = exp_scale
        self.resampling = resampling
        # TODO - what resampling to use and whether to expose CLI/API

    def parse_attributes(self):
        """ Set the exp_region, exp_crs and exp_scale attributes """

        if self.id is None:
            self._info["id"] = pathlib.Path(self.name).stem

        # if the image is in WGS84 and or has no scale (probable composite), then exit
        if (not self.scale and not self.exp_scale) or (not self.crs and not self.exp_crs):
            raise ValueError(f'{self.info["id"]} appears to be a composite in WGS84, specify a scale and CRS')

        # set resampling if image is not a composite
        if self.scale and self.crs and self.resampling != 'near':
            self._ee_image = self._ee_image.resample(self.resampling)

        # If CRS is the native MODIS CRS, then exit due to GEE bug
        if (self.crs == "SR-ORG:6974") and not self.exp_crs:
            raise ValueError(
                "There is an earth engine bug exporting in SR-ORG:6974, specify another CRS: "
                "https://issuetracker.google.com/issues/194561313"
            )

        if self.exp_crs is None:
            self.exp_crs = self.crs     # CRS corresponding to minimum scale band
        if self.exp_scale is None:
            self.exp_scale = self.scale  # minimum scale of the bands
        if self.exp_region is None:
            # use image granule footprint
            if _footprint_key in self.info["properties"]:
                self.exp_region = self.info["properties"][_footprint_key]
                click.secho(f'{self.info["id"]}: region not specified, setting to image footprint')
            else:
                raise AttributeError(f'{self.info["id"]} does not have a footprint, specify a region to download')

        if isinstance(self.exp_region, dict):
            self.exp_region = ee.Geometry(self.exp_region)


def export_image(image_obj, filename, folder="", region=None, crs=None, scale=None, resampling='near', wait=True):
    """
    Export an image to a GeoTiff in Google Drive

    Parameters
    ----------
    image_obj : ee.Image, geedim.image.Image
               The image to export
    filename : str
               The name of the task and destination file
    folder : str, optional
             Google Drive folder to export to (default: root).
    region : dict, geojson, ee.Geometry, optional
             Region of interest (WGS84) to export (default: export the entire image granule if it has one).
    crs : str, optional
          WKT, EPSG etc specification of CRS to export to.  Where image bands have different CRSs, all are re-projected
          to this CRS.
          (default: use the CRS of the minimum scale band if available).
    scale : float, optional
            Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this scale.
            (default: use the minimum scale of image bands if available).
    resampling : str, optional
           Resampling method: ("near"|"bilinear"|"bicubic") (default: "near")
    wait : bool
           Wait for the export to complete before returning (default: True)

    Returns
    -------
    ee.batch.Task
        Earth Engine export task object
    """
    exp_image = _ExportImage(image_obj, name=filename, exp_region=region, exp_crs=crs, exp_scale=scale,
                             resampling=resampling)
    exp_image.parse_attributes()

    # create export task and start
    task = ee.batch.Export.image.toDrive(
        image=exp_image.ee_image,
        region=exp_image.exp_region,
        description=filename[:100],
        folder=folder,
        fileNamePrefix=filename,
        scale=exp_image.exp_scale,
        crs=exp_image.exp_crs,
        maxPixels=1e9,
    )

    task.start()

    if wait:  # wait for completion
        monitor_export_task(task)

    return task


def monitor_export_task(task, label=None):
    """
    Monitor and display the progress of an export task

    Parameters
    ----------
    task : ee.batch.Task
           WW task to monitor
    label: str, optional
           Optional label for progress display (default: use task description)
    """
    toggles = r"-\|/"
    toggle_count = 0
    pause = 0.5
    bar_len = 100
    status = ee.data.getOperation(task.name)

    if label is None:
        label = f'{status["metadata"]["description"][:80]}:'

    # wait for export preparation to complete, displaying a spin toggle
    while "progress" not in status["metadata"]:
        time.sleep(pause)
        status = ee.data.getOperation(task.name)  # get task status
        click.echo(f"\rPreparing {label} {toggles[toggle_count % 4]}", nl=False)
        toggle_count += 1
    click.echo(f"\rPreparing {label}  done")

    # wait for export to complete, displaying a progress bar
    with click.progressbar(length=bar_len, label=f"Exporting {label}") as bar:
        while ("done" not in status) or (not status["done"]):
            time.sleep(pause)
            status = ee.data.getOperation(task.name)  # get task status
            progress = status["metadata"]["progress"] * bar_len
            bar.update(progress - bar.pos)  # update with progress increment
        bar.update(bar_len - bar.pos)

    if status["metadata"]["state"] != "SUCCEEDED":
        raise IOError(f"Export failed \n{status}")


def download_image(image_obj, filename, region=None, crs=None, scale=None, resampling='near', overwrite=False):
    """
    Download an image as a GeoTiff

    Parameters
    ----------
    image_obj : ee.Image, geedim.image.Image
               The image to export
    filename : str, pathlib.Path
               Name of the destination file
    region : dict, geojson, ee.Geometry, optional
             Region of interest (WGS84) to export (default: export the entire image granule if it has one).
    crs : str, optional
          WKT, EPSG etc specification of CRS to export to.  Where image bands have different CRSs, all are re-projected
          to this CRS.
          (default: use the CRS of the minimum scale band if available).
    scale : float, optional
            Pixel scale (m) to export to.  Where image bands have different scales, all are re-projected to this scale.
            (default: use the minimum scale of image bands if available).
    resampling : str, optional
           Resampling method: ("near"|"bilinear"|"bicubic") (default: "near")
    overwrite : bool, optional
                Overwrite the destination file if it exists, otherwise prompt the user (default: True)
    """
    exp_image = _ExportImage(image_obj, name=str(filename), exp_region=region, exp_crs=crs, exp_scale=scale,
                             resampling=resampling)
    filename = pathlib.Path(filename)
    exp_image.parse_attributes()

    # get download link
    try:
        link = exp_image.ee_image.getDownloadURL(dict(
            scale=exp_image.exp_scale,
            crs=exp_image.exp_crs,
            fileFormat="GeoTIFF",
            filePerBand=False,
            region=exp_image.exp_region,
        ))
    except ee.ee_exception.EEException as ex:
        # Add to exception message when the image is too large to download
        if re.match(r"Total request size \(.*\) must be less than or equal to .*", str(ex)):
            raise IOError(f"The requested image is too large to download, reduce its size, or export\n({str(ex)})")
        else:
            raise ex

    # download zip file
    tif_filename = filename.parent.joinpath(filename.stem + ".tif")  # force to tif file
    zip_filename = tif_filename.parent.joinpath("geedim_download.zip")
    with requests.get(link, stream=True) as r:
        r.raise_for_status()
        csize = 8192
        with open(zip_filename, "wb") as f:
            with click.progressbar(
                    r.iter_content(chunk_size=csize),
                    label=f"{filename.stem[:80]}:",
                    length=np.ceil(int(r.headers["Content-length"]) / csize).astype('int'),
                    show_pos=True,
            ) as bar:
                # override the bar's formatting function to show progress in MB
                bar.format_pos = (
                    lambda: f"{bar.pos * csize / (2 ** 20):.1f}/{bar.length * csize / (2 ** 20):.1f} MB\r")
                for chunk in bar:
                    f.write(chunk)

    # extract image tif from zip file
    zip_tif_filename = zip_filename.parent.joinpath(zipfile.ZipFile(zip_filename, "r").namelist()[0])
    with zipfile.ZipFile(zip_filename, "r") as zip_file:
        zip_file.extractall(zip_filename.parent)
    os.remove(zip_filename)

    # rename to extracted tif file to specified filename
    if zip_tif_filename != tif_filename:
        while tif_filename.exists():
            if overwrite or click.confirm(f"{tif_filename.name} exists, do you want to overwrite?", default="n"):
                os.remove(tif_filename)
            else:
                tif_filename = click.prompt("Please enter another filename", type=str, default=None)
                tif_filename = pathlib.Path(tif_filename)
        os.rename(zip_tif_filename, tif_filename)

    # write image metadata to pam xml file
    write_pam_xml(exp_image.info, str(tif_filename) + ".aux.xml")

    return link
