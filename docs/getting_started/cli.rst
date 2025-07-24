Command line
============

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

Multiple |geedim|_ commands can be chained together in a pipeline where image results from previous commands form inputs to the current command.  For example, to download a composite of the images produced by a search, the ``search``, ``composite`` and ``download`` commands would be chained.  Cloud mask configuration and ``--region`` / ``--bbox`` options are also piped between commands to save repeating these for multiple commands.  More detail on what each command reads from and outputs to the pipeline are given in the sections below.

Cloud configuration
-------------------

|config|_ configures cloud masking for subsequent commands in the pipeline.  Commands use a default configuration when they're not chained after |config|_.  E.g. this configures Sentinel-2 masking to use a threshold of 0.7 on the ``'cs_cdf'`` Cloud Score+ band:

.. code-block:: 

    geedim config --score 0.7 --cs-band cs_cdf

Filtering image collections
---------------------------

|search|_ searches (filters) an image collection and displays a table of filtered image properties.  Filter criteria can include a lower limit on the cloud-free portion of :option:`--bbox <geedim-search --bbox>` / :option:`--region <geedim-search --region>`.  E.g. this filters the Sentinel-2 surface reflectance collection on date range, region bounds, and cloud-free portion:

.. code-block:: 

    geedim search --collection COPERNICUS/S2_SR_HARMONIZED --start-date 2024-10-01 --end-date 2025-04-01 --bbox 24.35 -33.75 24.45 -33.65 --cloudless-portion 60

Filtered images are added to any images already in the pipeline, and piped out for use by subsequent commands.  When |search|_ is chained after |config|_, it uses the piped configuration to find the the cloud-free portions.  E.g.:

.. code-block:: 

    geedim config --score 0.7 --cs-band cs_cdf search --collection COPERNICUS/S2_SR_HARMONIZED --start-date 2024-10-01 --end-date 2025-04-01 --bbox 24.35 -33.75 24.45 -33.65 --cloudless-portion 60

.. _geotiff:

Exporting images to GeoTIFF
---------------------------

|download|_ exports images to GeoTIFF files.  Input images can be piped from previous commands or provided with :option:`--id <geedim-download --id>`.  All input images are piped out for use by subsequent commands.

The export projection, bounds and data type are defined automatically based on the first input image by default.  User projection and bounds can be supplied with :option:`--crs <geedim-download --crs>`, :option:`--bbox <geedim-download --bbox>` / :option:`--region <geedim-download --region>` and :option:`--scale <geedim-download --scale>` / :option:`--shape <geedim-download --shape>`; or :option:`--crs <geedim-download --crs>`, :option:`--crs-transform  <geedim-download --crs-transform >` and :option:`--shape <geedim-download --shape>`.   The data type can be modified with :option:`--dtype  <geedim-download --dtype>`. E.g.:

.. code-block::

    geedim download --id COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC --crs EPSG:3857 --bbox 24.35 -33.75 24.45 -33.65 --scale 30 --dtype uint16

Masks and related bands are added to exported images.  Cloud masks can be applied with :option:`--mask <geedim-download --mask>`, when supported.  Any configuration piped with |config|_ is used to form the cloud masks.  E.g.:

.. code-block::

    geedim config --score 0.7 --cs-band cs_cdf download --id COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC --crs EPSG:3857 --bbox 24.35 -33.75 24.45 -33.65 --scale 30 --dtype uint16 --mask

The :option:`--split <geedim-download --split>` option controls whether a file is exported for each input image (the default), or each band of the input images.  Exported files are named with the Earth Engine image index when they correspond to images, or band name when they correspond to bands.  E.g. this pipes images from a search and exports an image for each of the ``'B2'``, ``'B3'`` and ``'B4'`` bands:

.. code-block::

    geedim search --collection COPERNICUS/S2_SR_HARMONIZED --start-date 2024-11-10 --end-date 2024-11-20 --bbox 24.35 -33.75 24.45 -33.65 download --region - --band-name B2 --band-name B3 --band-name B4 --split bands

Exporting images to Google cloud
--------------------------------

|export|_ export image(s) to Google Drive, Earth Engine asset or Google Cloud Storage.

The :option:`--type <geedim-export --type>` and :option:`--folder <geedim-export --folder>` options specify the export destination.  E.g. to export an image to an Earth Engine asset in the ``'geedim'`` project:

.. code-block::

    geedim export --id COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC --type asset --folder geedim --crs EPSG:3857 --bbox 24.35 -33.75 24.45 -33.65 --scale 30 --dtype uint16

Export projection and bounds, cloud masking, image / band splitting, and piping behaviours are the same as with |download|_, and share the same options.  See that :ref:`section <geotiff>` for details.

Compositing images
------------------

|composite|_ creates a composite of input images.  Input images can be piped from previous commands, or specified with :option:`--id <geedim-composite --id>`.  The composite image is piped out for use by subsequent commands.  |download|_ or |export|_ should be chained after |composite|_ to export the composite image, which will be named ``'{--method NAME}-COMP'``.  E.g. this creates a cloud-free ``'median'`` composite from search result images, and exports to a GeoTIFF:

.. code-block::

    geedim search --collection COPERNICUS/S2_SR_HARMONIZED --start-date 2024-10-01 --end-date 2025-04-01 --bbox 24.35 -33.75 24.45 -33.65 --cloudless-portion 60 composite --method median download --crs EPSG:3857 --region - --scale 30 --dtype uint16

Cloud is masked from input images by default.  This can be disabled with :option:`--no-mask <geedim-composite --no-mask>`.  A compositing method can be specified with :option:`--method <geedim-composite --method>`.  The :class:`~geedim.enums.CompositeMethod` reference documents supported values.  The :attr:`~geedim.enums.CompositeMethod.mosaic`, :attr:`~geedim.enums.CompositeMethod.q_mosaic`, and :attr:`~geedim.enums.CompositeMethod.medoid` methods prioritise images in their sort order i.e. when more than one image pixel qualifies for selection, they select the first one.  Images can be sorted by closeness to :option:`--date <geedim-composite --date>`, or by the cloud-free portion of :option:`--bbox <geedim-composite --bbox>` /  :option:`--region <geedim-composite --region>`.  If none of the sorting options are provided, images are sorted by capture date.

Memory limit error
------------------

Exporting a composite with |download|_ could raise a ``'User memory limit exceeded'`` in some unusual cases.  |export|_ is not subject to the `limit on user memory <https://developers.google.com/earth-engine/guides/usage#per-request_memory_footprint>`__ which causes this error, and using it for export is recommended in this situation.  The composite can first be exported to Earth Engine asset with |export|_, and then the asset image exported to GeoTIFF with |download|_.  E.g.:

.. code-block::

    geedim search --collection COPERNICUS/S2_SR_HARMONIZED --start-date 2021-01-01 --end-date 2023-01-01 --bbox 24.35 -33.75 24.45 -33.65 composite --method median export --type asset --folder geedim --crs EPSG:3857 --region - --scale 10 --dtype uint16
    geedim download --id projects/geedim/assets/MEDIAN-COMP


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

.. |composite| replace:: ``geedim composite``
.. _composite: ../reference/cli.html#geedim-composite
