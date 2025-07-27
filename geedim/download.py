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
import os
import warnings

import ee
import rasterio as rio
from fsspec.core import OpenFile

from geedim.enums import Driver, ExportType
from geedim.image import ImageAccessor
from geedim.tile import Tiler

logger = logging.getLogger(__name__)


class BaseImage(ImageAccessor):
    def __init__(self, ee_image: ee.Image):
        """
        A class for encapsulating an Earth Engine image.

        .. deprecated:: 2.0.0
            Please use the :class:`ee.Image.gd <geedim.image.ImageAccessor>` accessor
            instead.

        :param ee_image:
            Image to encapsulate.
        """
        warnings.warn(
            f"'{self.__class__.__name__}' is deprecated and will be removed in a "
            f"future release.  Please use the 'ee.Image.gd' accessor instead.",
            category=FutureWarning,
            stacklevel=2,
        )
        super().__init__(ee_image)

    @classmethod
    def from_id(cls, image_id: str) -> BaseImage:
        """
        Create a BaseImage instance from an Earth Engine image ID.

        :param image_id:
           Image ID.

        :return:
            Image.
        """
        ee_image = ee.Image(image_id)
        gd_image = cls(ee_image)
        return gd_image

    @property
    def ee_image(self) -> ee.Image:
        """Encapsulated Earth Engine image."""
        return self._ee_image

    @ee_image.setter
    def ee_image(self, value: ee.Image):
        for attr in ['_mi', '_min_projection', 'info', 'dtype']:
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        self._ee_image = value

    @property
    def name(self) -> str | None:
        """Image name (:attr:`~geedim.image.ImageAccessor.id` with slashes replaced
        by dashes).
        """
        return self.id.replace('/', '-') if self.id else None

    @property
    def transform(self) -> rio.Affine | None:
        transform = super().transform
        return rio.Affine(*transform) if transform else None

    @property
    def footprint(self) -> dict | None:
        """GeoJSON geometry of the image extent.  ``None`` if the image has no fixed
        projection.
        """
        return self.geometry

    @property
    def has_fixed_projection(self) -> bool:
        """Whether the image has a fixed projection."""
        return self.shape is not None

    @property
    def refl_bands(self) -> list[str]:
        """List of spectral band names."""
        return self.specBands

    @property
    def band_properties(self) -> list[dict[str, Any]]:
        """List of STAC band properties."""
        return super().bandProps

    @staticmethod
    def monitor_export(task: ee.batch.Task, label: str | None = None) -> None:
        """
        Monitor and display the progress of an export task.

        :param task:
            Earth Engine task to monitor (as returned by :meth:`export`).
        :param label:
            Optional label for progress display.  Defaults to the task description.
        """
        ImageAccessor.monitorTask(task, label)

    def export(
        self,
        filename: str,
        type: ExportType = ImageAccessor._default_export_type,
        folder: str | None = None,
        wait: bool = True,
        **export_kwargs,
    ) -> ee.batch.Task:
        """
        Export the image to a raster file on Google Drive, Earth Engine asset,
        or raster file on Google Cloud Storage, using a batch task.

        :param filename:
            Destination file or asset name (excluding extension).
        :param type:
            Export type.
        :param folder:
            Google Drive folder (when ``type`` is
            :attr:`~geedim.enums.ExportType.drive`), Earth Engine asset project (when
            ``type`` is :attr:`~geedim.enums.ExportType.asset`), or Google Cloud
            Storage bucket (when ``type`` is :attr:`~geedim.enums.ExportType.cloud`).
            Can include sub-folders.  If ``type`` is
            :attr:`~geedim.enums.ExportType.asset` and ``folder`` is not supplied,
            ``filename`` should be a valid Earth Engine asset ID. If ``type`` is
            :attr:`~geedim.enums.ExportType.cloud` then ``folder`` is required.
        :param wait:
            Whether to wait for the export to complete before returning.
        :param export_kwargs:
            Arguments to :meth:`~geedim.image.ImageAccessor.prepareForExport`.

        :return:
            Export task, started if ``wait`` is ``False``, or completed if ``wait``
            is ``True``.
        """
        export_image = ImageAccessor(self.prepareForExport(**export_kwargs))
        return export_image.toGoogleCloud(filename, type=type, folder=folder, wait=wait)

    def download(
        self,
        filename: os.PathLike | str | OpenFile,
        overwrite: bool = False,
        num_threads: int | None = None,
        nodata: bool | int | float = True,
        driver: str | Driver = Driver.gtiff,
        max_tile_size: float = Tiler._default_max_tile_size,
        max_tile_dim: int = Tiler._ee_max_tile_dim,
        max_tile_bands: int = Tiler._ee_max_tile_bands,
        max_requests: int = Tiler._max_requests,
        max_cpus: int | None = None,
        **export_kwargs,
    ) -> None:
        """
        Export the image to a GeoTIFF file.

        The image is retrieved as separate tiles which are downloaded and
        decompressed concurrently.  Tile size can be controlled with
        ``max_tile_size``, ``max_tile_dim`` and ``max_tile_bands``, and download /
        decompress concurrency with ``max_requests`` and ``max_cpus``.

        GeoTIFF default namespace tags are written with
        :attr:`~geedim.image.ImageAccessor.properties`, and band tags with
        :attr:`band_properties`.

        :param file:
            Destination file.  Can be a path or URI string, or an
            :class:`~fsspec.core.OpenFile` object in binary mode (``'wb'``).
        :param overwrite:
            Whether to overwrite the destination file if it exists.
        :param num_threads:
            Deprecated and has no effect. ``max_requests`` and ``max_cpus`` can be
            used to limit concurrency.
        :param nodata:
            How to set the GeoTIFF nodata tag.  If ``True`` (the default), the nodata
            tag is set to :attr:`~geedim.image.ImageAccessor.nodata`. Otherwise,
            if ``False``, the nodata tag is not set.  A custom value can also be
            provided, in which case the nodata tag is set to this value. Usually,
            a custom value would be supplied when the image has been unmasked with
            ``ee.Image.unmask(nodata)``.
        :param driver:
            File format driver.
        :param max_tile_size:
            Maximum tile size (MB).  Should be less than the `Earth Engine size limit
            <https://developers.google.com/earth-engine/apidocs/ee-image
            -getdownloadurl>`__ (32 MB).
        :param max_tile_dim:
            Maximum tile width / height (pixels).  Should be less than the `Earth
            Engine limit <https://developers.google.com/earth-engine/apidocs/ee-image
            -getdownloadurl>`__ (10000).
        :param max_tile_bands:
            Maximum number of tile bands.  Should be less than the `Earth Engine
            limit <https://developers.google.com/earth-engine/reference/rest/v1
            /projects.image/computePixels>`__ (1024).
        :param max_requests:
            Maximum number of concurrent tile downloads.  Should be less than the
            `max concurrent requests quota
            <https://developers.google.com/earth-engine/guides/usage
            #adjustable_quota_limits>`__.
        :param max_cpus:
            Maximum number of tiles to decompress concurrently.  Defaults to one less
            than the number of CPUs, or one, whichever is greater.  Values larger
            than the default can stall the asynchronous event loop and are not
            recommended.
        :param export_kwargs:
            Arguments to :meth:`~geedim.image.ImageAccessor.prepareForExport`.
        """
        if num_threads is not None:
            warnings.warn(
                "'num_threads' is deprecated and has no effect.  'max_requests' and "
                "'max_cpus' can be used to limit concurrency.",
                category=FutureWarning,
                stacklevel=2,
            )

        export_image = ImageAccessor(self.prepareForExport(**export_kwargs))
        export_image.toGeoTIFF(
            filename,
            overwrite=overwrite,
            nodata=nodata,
            driver=driver,
            max_tile_size=max_tile_size,
            max_tile_dim=max_tile_dim,
            max_tile_bands=max_tile_bands,
            max_requests=max_requests,
            max_cpus=max_cpus,
        )
