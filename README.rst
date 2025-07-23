|Tests| |codecov| |PyPI version| |conda-forge version| |docs| |License|

Geedim
======

.. description_start

Geedim provides a Python API and command line toolkit for exporting and cloud masking Google Earth Engine (GEE) imagery.  Images and Image collections can be exported to:

- GeoTIFF file
- NumPy array
- Xarray Dataset / DataArray
- Google Cloud platforms

And cloud masking is supported on:

- Landsat 4-9 `collection 2 <https://developers.google.com/earth-engine/datasets/catalog/landsat>`__ images
- Sentinel-2 `TOA <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED>`__ and `surface reflectance <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>`__ images

.. description_end

.. install_start

Installation
------------

To install from PyPI:

.. code:: shell

   pip install geedim

To install from conda-forge:

.. code:: shell

   conda install -c conda-forge geedim

To support exporting to Xarray, use ``pip install geedim[xarray]`` or ``conda install -c conda-forge geedim xarray`` instead.

A registered Google Cloud project is required for `access to Earth Engine <https://developers.google.com/earth-engine/guides/access#create-a-project>`__.  Once installation and registration is done, Earth Engine should be authenticated:

.. code:: shell

   earthengine authenticate

.. install_end

Examples
--------

API
~~~

Geedim provides access to its functionality through the ``gd`` accessor on the ``ee.Image`` and ``ee.ImageCollection`` `GEE <https://github.com/google/earthengine-api>`__ classes.  This example exports a 6 month cloud-free composite of Sentinel-2 surface reflectance imagery to a GeoTIFF file:

.. code:: python

    import ee

    # import geedim to enable accessors
    import geedim  # noqa: F401

    ee.Initialize()

    # filter collection based on cloudless portion etc.
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    filt_coll = coll.gd.filter(
        '2021-10-01', '2022-04-01', region=region, cloudless_portion=60
    )

    # print image properties
    print(filt_coll.gd.schemaTable)
    print(filt_coll.gd.propertiesTable)

    # create a cloud-free composite & download
    comp_im = filt_coll.gd.composite('median')
    prep_im = comp_im.gd.prepareForExport(
        crs='EPSG:3857', region=region, scale=10, dtype='uint16'
    )
    prep_im.gd.toGeoTIFF('s2_comp.tif')


Command line interface
~~~~~~~~~~~~~~~~~~~~~~

Much of the API functionality can also be accessed on the command line with ``geedim`` and its sub-commands.  As in the API example, this exports a 6-month cloud-free composite of Sentinel-2 surface reflectance imagery to a GeoTIFF file:

.. code:: shell

    geedim search -c COPERNICUS/S2_SR_HARMONIZED -s 2024-10-01 -e 2025-04-01 -b 24.35 -33.75 24.45 -33.65 -cp 60 composite -cm median download -c EPSG:3857 -r - -s 10 -dt uint16


Documentation
-------------

See `geedim.readthedocs.io <https://geedim.readthedocs.io/>`__ for usage, contribution and reference documentation.

License
-------

This project is licensed under the terms of the `Apache-2.0 License <https://github.com/leftfield-geospatial/geedim/blob/main/LICENSE>`__.

Credits
-------

-  Tiled downloading was inspired by the `MIT licensed <https://github.com/cordmaur/GEES2Downloader/blob/main/LICENSE>`__ `GEES2Downloader <https://github.com/cordmaur/GEES2Downloader>`__ project.
-  Medoid compositing, and the accessor approach to extending the `GEE API <https://github.com/google/earthengine-api>`__, were adapted from `geetools <https://github.com/gee-community/geetools>`__ under terms of the
   `MIT license <https://github.com/gee-community/geetools/blob/master/LICENSE>`__.
-  Sentinel-2 cloud masking was adapted from `ee_extra <https://github.com/r-earthengine/ee_extra>`__ under
   terms of the `Apache-2.0 license <https://github.com/r-earthengine/ee_extra/blob/master/LICENSE>`__

.. TODO: include a section on why geedim and not Xee?


.. |Tests| image:: https://github.com/leftfield-geospatial/geedim/actions/workflows/run-unit-tests.yml/badge.svg
   :target: https://github.com/leftfield-geospatial/geedim/actions/workflows/run-unit-tests.yml
.. |codecov| image:: https://codecov.io/gh/leftfield-geospatial/geedim/branch/main/graph/badge.svg?token=69GZNQ3TI3
   :target: https://codecov.io/gh/leftfield-geospatial/geedim
.. |PyPI version| image:: https://img.shields.io/pypi/v/geedim.svg
   :target: https://pypi.org/project/geedim/
.. |conda-forge version| image:: https://img.shields.io/conda/vn/conda-forge/geedim.svg
   :alt: conda-forge
   :target: https://anaconda.org/conda-forge/geedim
.. |docs| image:: https://readthedocs.org/projects/geedim/badge/?version=latest
   :target: https://geedim.readthedocs.io/en/latest/?badge=latest
.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
