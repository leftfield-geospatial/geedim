Changelog
=========

v2.0.0 - 2025-08-22
-------------------

This release provides a new API via ``gd`` accessors on the ``ee.Image`` and ``ee.ImageCollection`` Earth Engine classes.  Existing export functionality has been expanded to allow image and image collection export to various formats.  Cloud masking has been extended to support more collections and masking options.  With a substantial part of the package having been rewritten, some breaking changes were necessary.

Breaking changes
~~~~~~~~~~~~~~~~

- The default Landsat cloud mask is more aggressive.  Following `<https://gis.stackexchange.com/a/473652>`__, it now includes dilated cloud, and all medium confidence cloud, shadow & cirrus pixels.
- The ``MaskedCollection.stac`` property now returns a STAC dictionary and not a ``StacItem`` instance.
- ``MaskedImage`` doesn't add mask bands to composite images.  Previously mask bands were added to composite images that had no existing mask bands.
- When cloud masking is not supported, ``MaskedImage.mask_clouds()`` leaves the image unaltered instead of applying a fill mask.
- ``MaskedImage.scale`` is in units of its ``crs``, not meters.
- The ``MaskedImage.search()`` ``end_date`` parameter defaults to a millisecond after ``start_date``, not one day.
- The ``geedim download`` and ``geedim export`` CLI commands name image files with their Earth Engine index rather than ID.
- The piped value of the CLI ``--region`` option is no longer implicitly used by subsequent commands in the pipeline.

Deprecations
~~~~~~~~~~~~

- The ``MaskedImage`` and ``MaskedCollection`` classes are deprecated.  The ``ee.Image.gd`` and ``ee.ImageCollection.gd`` accessors should be used instead.

Features
~~~~~~~~

- Provide the API via ``ee.Image.gd`` and ``ee.ImageCollection.gd`` accessors.
- Provide client-side access to image and collection properties.
- Allow images and image collections to be exported to GeoTIFF file, NumPy array, Xarray DataArray / Dataset and Google Cloud platforms.
- Support exporting to Cloud Optimised GeoTIFF.
- Allow setting a custom nodata value when exporting to GeoTIFF (#21).
- Support specifying file / directory paths as remote URIs with ``fsspec``.
- Extend cloud masking support to Landsat C2 collections.
- Allow saturated, non-physical reflectance or aerosol pixels to be included in cloud masks.
- Add a CLI ``--buffer`` option for buffering the ``--region`` / ``--bbox``.

Packaging
~~~~~~~~~

- Increase the minimum Python version to 3.11.
- Add ``fsspec`` and AIOHTTP dependencies.

Documentation
~~~~~~~~~~~~~

- Update the site theme & layout.
- Add new getting started sections.

Internal changes
~~~~~~~~~~~~~~~~

- Rewrite tiled downloading with AIOHTTP and custom retries (#22, #26, #30).
- Rewrite STAC retrieval with AIOHTTP.

v1.9.1 - 2025-05-13
-------------------

Updates for compatibility with click 8.2.0.

Fixes:
~~~~~~

- Fix CLI ``--overwrite`` option to work around pallets/click#2894.
- Fix CLI choice options to work with the new ``click.Choice`` class.

v1.9.0 - 2024-10-27
-------------------

This release adds support for cloud masking Sentinel-2 images with the
Cloud Score+ dataset, restores multi-threaded GeoTIFF overview building,
and makes a number of other bug fixes and minor updates.

The Cloud Score+ method (``cloud-score``) is superior to the existing
Sentinel-2 mask methods. It has been made the default, and the existing
(``qa`` and ``cloud-prob`` ) methods deprecated.

Breaking changes:
~~~~~~~~~~~~~~~~~

- Change the default cloud mask method for Sentinel-2 images to
  ``cloud-score``.

Deprecations
~~~~~~~~~~~~

- Deprecate the Sentinel-2 ``cloud-prob`` and ``qa`` cloud mask methods.

Features:
~~~~~~~~~

- Add a Cloud Score+ mask method for Sentinel-2 images
  (``cloud-score``).
- Build GeoTIFF overviews with multiple threads.
- Change the ``BaseImage.date`` property to be time zone aware.

.. _fixes-1:

Fixes:
~~~~~~

- Fix the shadow cast direction for the Sentinel-2 ``cloud-prob`` and
  ``qa`` cloud mask methods.
- Provide the ``BaseImage.scale`` property in meters for all CRSs.
- Allow export of images with positive y-axis geo-transforms on the
  source pixel grid.
- Fix ``average`` resampling to use the minimum scale projection (for
  e.g. Sentinel-2 images that have bands without fixed projections).

Packaging:
~~~~~~~~~~

- Increase the minimum required python version to 3.8.
- Pin the Rasterio version for multi-threaded overviews.
- Pin the earthengine-api version to remove the
  ``ee.Image.getDownloadUrl()`` thread lock.

Documentation:
~~~~~~~~~~~~~~

- Update examples to use harmonised Sentinel-2 images and the
  ``cloud-score`` mask method.

Internal changes:
~~~~~~~~~~~~~~~~~

- Remove thread lock on calls to ``ee.Image.getDownloadUrl()`` for tile
  download.
- Download tiles as GeoTIFFs rather than zipped GeoTIFFs.
- Simplify pixel grid maintenance when exporting.
- Update deprecated calls to ``datetime.utcfromtimestamp()``.

v1.8.1 - 2024-07-12
-------------------

This is a bugfix release that deals with Sentinel-2 images that are
missing required cloud mask data, and an Earth Engine quirk relating to
image data type and ``nodata`` value.

.. _fixes-2:

Fixes:
~~~~~~

- Create fully masked cloud masks when Sentinel-2 images are missing
  QA60 or cloud probability data (#24).
- Work around
  `350528377 <https://issuetracker.google.com/issues/350528377>`__ by
  changing the GeoTIFF ``nodata`` value for floating point data type
  images to ``float('-inf')``.

.. _packaging-1:

Packaging:
~~~~~~~~~~

- Pin Rasterio for compatibility with ``float('-inf')`` ``nodata`` .

v1.8.0 - 2024-06-21
-------------------

This release adds logic for retrying tile downloads and includes fixes
for compatibility with NumPy 2.

.. _features-1:

Features:
~~~~~~~~~

- Improve download reliability by retrying corrupt or incomplete tiles
  (#22).
- Update STAC URLs.

.. _fixes-3:

Fixes:
~~~~~~

- Update deprecated Numpy calls for compatibility with Numpy 2.

v1.7.2 - 2023-06-10
-------------------

- Build overviews in a single thread to work around
  https://github.com/OSGeo/gdal/issues/7921.
- Update STAC urls.

v1.7.1 - 2023-05-10
-------------------

- Allow download / export of a subset of image bands with a ``bands``
  API parameter and ``--band-name`` CLI option.

v1.7.0 - 2022-12-11
-------------------

- Simplify the ``medoid`` module to reduce memory usage and computation.
- Change ``cloudless_portion`` search parameter to be the portion of
  filled pixels, rather than portion of ``region``.
- Only find ``region`` portions when searching with
  ``cloudless_portion`` or ``fill_portion`` filters (improves speed).
- Where possible, apply ``custom_filter`` before cloud detection in
  search (improves speed).
- Add a Sentinel-2 ``medoid`` composite tutorial.
- Fix Sentinel-2 shadow projection, and ``qa`` cloud mask naming bugs.
- Update the documentation and STAC catalog.

v1.6.1 - 2022-11-14
-------------------

- Fix unexpected argument exporting to Google Cloud Storage.

v1.6.0 - 2022-11-12
-------------------

- Add API and CLI support for exporting to Earth Engine asset and Google
  Cloud Storage.
- Allow command line chaining of Earth Engine asset export with
  download.
- Update documentation, and add section on user memory limits.

v1.5.3 - 2022-09-25
-------------------

- Update STAC URLs.

v1.5.2 - 2022-09-23
-------------------

- Support downloading MODIS images in their native CRS.
- Fix boundedness test on download to include the MODIS case.

v1.5.1 - 2022-09-18
-------------------

- Always allow download with ``ee.Geometry`` type ``region`` (fixes #6).

v1.5.0 - 2022-08-30
-------------------

- Add ``crs_transform`` and ``shape`` parameters to the download /
  export API and CLI.
- Add a download / export CLI ``--like`` option, that uses a template
  image to specify ``crs``, ``crs_transform`` and ``shape``.
- Download / export on the Earth Engine image pixel grid when possible.

v1.4.0 - 2022-08-03
-------------------

- Add ``max_tile_size`` and ``max_tile_dim`` parameters to download CLI
  and API for working around EE “*user memory limit exceeded*” errors.

v1.3.2 - 2022-07-22
-------------------

- Ensure download progress reaches 100% on success.

v1.3.1 - 2022-07-21
-------------------

- Bugfix for large downloads causing a segmentation fault (all python
  versions).

v1.3.0 - 2022-07-18
-------------------

- Resolve #2 by adding ``BIGTIFF`` support to downloaded images whose
  uncompressed size is larger than 4GB.
- Allow for the inclusion of user-specified properties in ``search``
  results and collection properties.
- Allow for custom ``search`` filters.
- Fix an issue with 4 band images being misinterpreted as *RGBA*.
- Work around a Python 3.10 issue with concurrent tile downloads.
- Update the STAC URL data.

v1.2.0 - 2022-06-20
-------------------

- Add cloud/shadow mask support for harmonised Sentinel-2 collections.
- Add scale/offset download/export option that uses STAC information to
  convert bands to floating point values representing physical
  quantities.
- Abbreviate ``geedim`` collection names, apply ``yapf`` code autoformat
  & update docs.

v1.1.2 - 2022-06-16
-------------------

- Fix PyPI readme format.

v1.1.1 - 2022-06-16
-------------------

- CLI and API documentation improvements.
- Sphinx config and RST content added for building docs.
- Add notebook tutorial.
- Clip Landsat cloud distance at a maximum.
- Allow repeat cloud/shadow masking on filtered collections with
  different config.
- Add yapf style file.
- Other minor bug fixes.

v1.0.1 - 2022-05-27
-------------------

- Remove the dependency on pip with a new spinner class
- Display spinner in CLI search while waiting

v1.0.0 - 2022-05-26
-------------------

- Tiled image downloading for files larger than the EE size limit
- Extend search/composite/download to apply to all EE imagery
- Improve piping of images and configuration between chained commands
- Add ``config`` command to configure cloud/shadow masking
- Add support for Landsat-9
- Rewrite unit tests with pytest
- Remove pandas dependency, replacing with tabulate
- Add logging
- Restructure & simplify API

v0.4.0 - 2022-02-16
-------------------

- Add support for Landsat 4 & 5 collections
- Cloud/shadow masking and compositing fix for non-native scales
- Masking performance improvement

v0.3.1 - 2021-10-29
-------------------

- Fix Landsat7 SLE masking
- Remove noise from Sentinel2 shadow mask

v0.3.0 - 2021-10-28
-------------------

- CLI and API options added for selecting the resampling method
- Default EE masks (where surface reflectance == 0) incorporated into
  shadow mask
- Fixed search stats to reflect validity of region rather than image
- Reflectance scaling (–scale-refl) removed
- Unit tests for checking image content

v0.2.3 - 2021-09-21
-------------------

- Unnecessary mask and scale-refl options removed from search API and
  CLI
- Unit tests clean previous downloads and overwrite by default
- Github workflows now run on python 3.6 and 3.x (latest) only

v0.1.5 - 2021-09-15
-------------------

- First release

