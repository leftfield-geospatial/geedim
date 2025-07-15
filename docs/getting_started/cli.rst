Command line
============

.. Geedim command line functionality is accessed with the |geedim|_ command, and its sub-commands.

Much of the API functionality can also be accessed on the command line with the |geedim|_ command, and its sub-commands.

Get help on ``geedim`` with:

.. code-block:: 

   geedim --help

and help on an ``geedim`` sub-command with:

.. code-block:: 

   geedim <sub-command> --help

Paths and URIs
--------------

Command line and API file / directory parameters can be specified as local paths or remote URIs.  Geedim uses `fsspec <https://github.com/fsspec/filesystem_spec>`__ for file IO, which provides built-in support for `a number of remote file systems <https://filesystem-spec.readthedocs.io/en/stable/api.html#implementations>`__.  Support for other remote systems is available by installing `the relevant extension package <https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations>`__.  See the `fsspec documentation <https://filesystem-spec.readthedocs.io/en/stable/features.html#configuration>`__ if your file system requires credentials or other configuration.

Command chaining
----------------

Multiple |geedim|_ commands can be chained together in a pipeline where image results from previous command(s) form inputs to the current command.  For example, to download a composite of the images produced by a search, the ``search``, ``composite`` and ``download`` commands would be chained.  Cloud / shadow configuration and ``--region`` / ``--bbox`` options are also piped between commands to save repeating these for multiple commands.  More detail on what each command reads from and outputs to the pipeline are given in the sections below.

Cloud / shadow configuration
----------------------------

|config|_ configures cloud / shadow masking for subsequent commands in the pipeline.  Commands use a default configuration when they're not chained after |config|_.  E.g. this configures Sentinel-2 masking to use a threshold of 0.7 on the 'cs_cdf' Cloud Score+ band:

.. code-block:: 

    geedim config --score 0.7 --cs-band cs_cdf

Filtering image collections
----------------------------

|search|_ searches (filters) an image collection with user criteria and displays a table of the resulting images and their properties.  The resulting images are added to any images already in the pipeline, and piped out for use by subsequent commands.  E.g. this filters the Sentinel-2 surface reflectance collection on date range, region bounds, and a lower limit of 60% on the cloud / shadow - free portion of region:

.. code-block:: 

    geedim search --collection COPERNICUS/S2_SR_HARMONIZED --start-date 2024-10-01 --end-date 2025-04-01 --bbox 24.35 -33.75 24.45 -33.65 --cloudless-portion 60

When |search|_ is chained after |config|_, it uses the piped configuration to find the the cloud / shadow - free portions.  E.g.:

.. code-block:: 

    geedim config --score 0.7 --cs-band cs_cdf search --collection COPERNICUS/S2_SR_HARMONIZED --start-date 2024-10-01 --end-date 2025-04-01 --bbox 24.35 -33.75 24.45 -33.65 --cloudless-portion 60

.. _geotiff:

Exporting images to GeoTIFF
---------------------------

|download|_ exports images to GeoTIFF files.  Input images can be piped from previous commands or provided with :option:`--id <geedim-download --id>`.  All input images are piped out for use by subsequent commands.

The export pixel grid, bounds and data type are defined automatically based on the first input image by default.  User pixel grid and bounds can be supplied with :option:`--crs <geedim-download --crs>`, :option:`--bbox <geedim-download --bbox>` / :option:`--region <geedim-download --region>` and :option:`--scale <geedim-download --scale>` / :option:`--shape <geedim-download --shape>`; or :option:`--crs <geedim-download --crs>`, :option:`--crs-transform  <geedim-download --crs-transform >` and :option:`--shape <geedim-download --shape>`.   The data type can be modified with :option:`--dtype  <geedim-download --dtype>`. E.g.:

.. code-block::

    geedim download --id COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC --crs EPSG:3857 --bbox 24.35 -33.75 24.45 -33.65 --scale 30 --dtype uint16

Fill (validity) masks are added to exported images, as are cloud / shadow mask and related bands when supported.  Masks can be applied with :option:`--mask <geedim-download --mask>`.  Any cloud / shadow configuration piped with |config|_ is used to form the cloud / shadow masks.  E.g.:

.. code-block::

    geedim config --score 0.7 --cs-band cs_cdf download --id COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC --crs EPSG:3857 --bbox 24.35 -33.75 24.45 -33.65 --scale 30 --dtype uint16 --mask

The :option:`--split <geedim-download --split>` option controls whether a file is exported for each input image (the default), or each band of the input image(s).  E.g. this pipes images from a search and exports an image for each of the ``B2``, ``B3`` and ``B4`` bands:

.. code-block::

    geedim search --collection COPERNICUS/S2_SR_HARMONIZED --start-date 2024-11-10 --end-date 2024-11-20 --bbox 24.35 -33.75 24.45 -33.65 download --region - --band-name B2 --band-name B3 --band-name B4 --split bands


Exporting images to Google cloud
--------------------------------

|export|_ export image(s) to Google Drive, Earth Engine asset or Google Cloud Storage.

The :option:`--type <geedim-export --type>` and :option:`--folder <geedim-export --folder>` options specify the export destination.  E.g. to export an image to an Earth Engine asset in the ``'geedim'`` project:

.. code-block::

    geedim export --id COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC --type asset --folder geedim --crs EPSG:3857 --bbox 24.35 -33.75 24.45 -33.65 --scale 30 --dtype uint16

Export pixel grid and bounds, cloud / shadow masking, image / band splitting, and piping behaviours are the same as with |download|_, and share the same options.  See that :ref:`section <geotiff>` for details.


.. |geedim| replace:: ``geedim``
.. _geedim: ../reference/cli.html#geedim

.. |config| replace:: ``geedim config``
.. _config: ../reference/cli.html#geedim-config

.. |search| replace:: ``geedim search``
.. _search: ../reference/cli.html#geedim-search

.. |download| replace:: ``geedim download``
.. _download: ../reference/cli.html#geedim-download

.. |export| replace:: ``geedim export``
.. _export: ../reference/cli.html#geedim-export
