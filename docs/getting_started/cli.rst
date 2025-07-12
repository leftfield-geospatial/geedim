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

Multiple |geedim|_ commands can be chained together in a pipeline where image results from previous command(s) form inputs to the current command.  For example, to download a composite of the images produced by a search, the ``search``, ``composite`` and ``download`` commands would be chained.  Cloud / shadow configuration and ``--region`` / ``--bbox`` options are also piped between commands to save repeating these for multiple commands.  The details of what each command reads from and outputs to the pipeline are given in the sections below.

Cloud / shadow configuration
----------------------------

|config|_ configures cloud / shadow masking for subsequent commands in the pipeline.  Commands use a default configuration when they're not chained after |config|_.  E.g. this configures Sentinel-2 masking to use a threshold of 0.7 on the 'cs_cdf' Cloud Score+ band:

.. code-block:: 

    geedim config --score 0.7 --cs-band cs_cdf

Filtering image collections
----------------------------

|search|_ searches (filters) an image collection with user criteria and displays a table of the resulting images and their properties.  The resulting images are added to any images already in the pipeline, and piped out for use by subsequent commands.  E.g. this filters the Sentinel-2 surface reflectance collection on date range, region bounds, and a lower limit of 60% on the cloud/shadow-free portion of region:

.. code-block:: 

    geedim search -c COPERNICUS/S2_SR_HARMONIZED -s 2024-10-01 -e 2025-04-01 -b 24.35 -33.75 24.45 -33.65 -cp 60

When |search|_ is chained after |config|_, it uses the piped cloud / shadow configuration.  E.g.:

.. code-block:: 

    geedim config --score 0.7 --cs-band cs_cdf search -c COPERNICUS/S2_SR_HARMONIZED -s 2024-10-01 -e 2025-04-01 -b 24.35 -33.75 24.45 -33.65 -cp 60


.. |geedim| replace:: ``geedim``
.. _geedim: ../reference/cli/cli.html#geedim

.. |config| replace:: ``geedim config``
.. _config: ../reference/cli/cli.html#geedim-config

.. |search| replace:: ``geedim search``
.. _search: ../reference/cli/cli.html#geedim-search

.. |download| replace:: ``geedim download``
.. _download: ../reference/cli/cli.html#geedim-download

.. |export| replace:: ``geedim download``
.. _export: ../reference/cli/cli.html#geedim-export
