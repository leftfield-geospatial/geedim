|Tests| |codecov| |PyPI version| |Anaconda-Server Badge| |License|

``geedim``
==========

.. short_descr_start

Search, composite, and download `Google Earth Engine <https://earthengine.google.com/>`__ imagery, without size limits.

.. short_descr_end

.. description_start

Description
-----------

``geedim`` provides a command line interface and API for searching, compositing and downloading satellite imagery from
Google Earth Engine (EE). It optionally performs cloud / shadow masking, and cloud / shadow-free compositing on
supported collections. Images and composites can be downloaded, or exported to Google Drive. Images larger than the EE
size limit are split and downloaded as separate tiles, then re-assembled into a single GeoTIFF.

.. description_end

See the documentation site for more detail: https://geedim.readthedocs.io/.

.. supp_im_start

Supported imagery
~~~~~~~~~~~~~~~~~

Any EE imagery can be searched, composited and downloaded by ``geedim``. Cloud / shadow masking is supported on the
following collections:

+--------------------------------+----------------------------------------+--------------------------------------------+
| ``geedim`` name                | EE name                                | Description                                |
+================================+========================================+============================================+
| landsat4-c2-l2                 | `LANDSAT/LT04/C02/T1_L2 <https://de    | Landsat 4, collection 2, tier 1, level 2   |
|                                | velopers.google.com/earth-engine/datas | surface reflectance                        |
|                                | ets/catalog/LANDSAT_LT04_C02_T1_L2>`__ |                                            |
+--------------------------------+----------------------------------------+--------------------------------------------+
| landsat5-c2-l2                 | `LANDSAT/LT05/C02/T1_L2 <https://de    | Landsat 5, collection 2, tier 1, level 2   |
|                                | velopers.google.com/earth-engine/datas | surface reflectance                        |
|                                | ets/catalog/LANDSAT_LT05_C02_T1_L2>`__ |                                            |
+--------------------------------+----------------------------------------+--------------------------------------------+
| landsat7-c2-l2                 | `LANDSAT/LE07/C02/T1_L2 <https://de    | Landsat 7, collection 2, tier 1, level 2   |
|                                | velopers.google.com/earth-engine/datas | surface reflectance                        |
|                                | ets/catalog/LANDSAT_LE07_C02_T1_L2>`__ |                                            |
+--------------------------------+----------------------------------------+--------------------------------------------+
| landsat8-c2-l2                 | `LANDSAT/LC08/C02/T1_L2 <https://de    | Landsat 8, collection 2, tier 1, level 2   |
|                                | velopers.google.com/earth-engine/datas | surface reflectance                        |
|                                | ets/catalog/LANDSAT_LC08_C02_T1_L2>`__ |                                            |
+--------------------------------+----------------------------------------+--------------------------------------------+
| landsat9-c2-l2                 | `LANDSAT/LC09/C02/T1_L2 <https://de    | Landsat 9, collection 2, tier 1, level 2   |
|                                | velopers.google.com/earth-engine/datas | surface reflectance                        |
|                                | ets/catalog/LANDSAT_LC09_C02_T1_L2>`__ |                                            |
+--------------------------------+----------------------------------------+--------------------------------------------+
| sentinel2-toa                  | `COPERNICUS/S2 <h                      | Sentinel-2, level 1C, top of atmosphere    |
|                                | ttps://developers.google.com/earth-eng | reflectance                                |
|                                | ine/datasets/catalog/COPERNICUS_S2>`__ |                                            |
+--------------------------------+----------------------------------------+--------------------------------------------+
| sentinel2-sr                   | `COPERNICUS/S2_SR <http                | Sentinel-2, level 2A, surface reflectance  |
|                                | s://developers.google.com/earth-engine |                                            |
|                                | /datasets/catalog/COPERNICUS_S2_SR>`__ |                                            |
+--------------------------------+----------------------------------------+--------------------------------------------+

.. supp_im_end

.. install_start

Installation
------------

Requirements
~~~~~~~~~~~~

``geedim`` is a python 3 package, and requires users to be registered with `Google Earth
Engine <https://signup.earthengine.google.com>`__.

conda
~~~~~

Under Windows, using ``conda`` is the easiest way to resolve binary dependencies. The
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ installation provides a minimal ``conda``.

.. code:: shell

   conda install -c conda-forge geedim

pip
~~~

.. code:: shell

   pip install geedim

Authentication
~~~~~~~~~~~~~~

Following installation, Earth Engine should be authenticated:

.. code:: shell

   earthengine authenticate

.. install_end

Getting started
---------------

Command line interface
~~~~~~~~~~~~~~~~~~~~~~

.. cli_start

``geedim`` command line functionality is accessed through the commands:

-  ``search``: Search for images.
-  ``composite``: Create a composite image.
-  ``download``: Download image(s).
-  ``export``: Export image(s) to Google Drive.
-  ``config``: Configure cloud / shadow masking.

Get help on ``geedim`` with:

.. code:: shell

   geedim --help

and help on a ``geedim`` command with:

.. code:: shell

   geedim <command> --help

Examples
^^^^^^^^

Search for Landsat-8 images.

.. code:: shell

   geedim search -c landsat8-c2-l2 -s 2021-06-01 -e 2021-07-01 --bbox 24 -33 24.1 -33.1

Download a Landsat-8 image with cloud / shadow mask applied.

.. code:: shell

   geedim download -i LANDSAT/LC08/C02/T1_L2/LC08_172083_20210610 --bbox 24 -33 24.1 -33.1 --mask

Command pipelines
~~~~~~~~~~~~~~~~~

Multiple ``geedim`` commands can be chained together in a pipeline where image results from the previous command form
inputs to the current command. For example, if the ``composite`` command is chained with ``download`` command, the
created composite image will be downloaded, or if the ``search`` command is chained with the ``composite`` command, the
search result images will be composited.

Common command options are also piped between chained commands. For example, if the ``config`` command is chained with
other commands, the configuration specified with ``config`` will be applied to subsequent commands in the pipeline. Many
command combinations are possible.

.. cli_end

.. _examples-1:

Examples
^^^^^^^^

Composite two Landsat-7 images and download the result:

.. code:: shell

   geedim composite -i LANDSAT/LE07/C02/T1_L2/LE07_173083_20100203 -i LANDSAT/LE07/C02/T1_L2/LE07_173083_20100219 download --bbox 22 -33.1 22.1 -33 --crs EPSG:3857 --scale 30

Composite the results of a Landsat-8 search and download the result.

.. code:: shell

   geedim search -c landsat8-c2-l2 -s 2019-02-01 -e 2019-03-01 --bbox 23 -33 23.2 -33.2 composite -cm q-mosaic download --scale 30 --crs EPSG:3857

Search for Sentinel-2 SR images with a cloudless portion of at least 60%, using the ``qa`` mask-method to identify
clouds:

.. code:: shell

   geedim config --mask-method qa search -c sentinel2-sr --cloudless-portion 60 -s 2022-01-01 -e 2022-01-14 --bbox 24 -34 24.5 -33.5

API
~~~

Example
^^^^^^^

.. code:: python

   import geedim as gd

   gd.Initialize()  # initialise earth engine

   # geojson region to search / download
   region = {
       "type": "Polygon",
       "coordinates": [[[24, -33.6], [24, -33.53], [23.93, -33.53], [23.93, -33.6], [24, -33.6]]]
   }

   # make collection and search
   coll = gd.MaskedCollection.from_name('COPERNICUS/S2_SR')
   coll = coll.search('2019-01-10', '2019-01-21', region)
   print(coll.schema_table)
   print(coll.properties_table)

   # create and download an image
   im = gd.MaskedImage.from_id('COPERNICUS/S2_SR/20190115T080251_20190115T082230_T35HKC')
   im.download('s2_image.tif', region=region)

   # composite search results and download
   comp_im = coll.composite()
   comp_im.download('s2_comp_image.tif', region=region, crs='EPSG:32735', scale=30)

License
-------

This project is licensed under the terms of the `Apache-2.0 License <LICENSE>`__.

Contributing
------------

See the `documentation <https://geedim.readthedocs.io/en/latest/contributing.html>`__ for details.

Credits
-------

-  Tiled downloading was inspired by the work in `GEES2Downloader <https://github.com/cordmaur/GEES2Downloader>`__ under
   terms of the `MIT license <https://github.com/cordmaur/GEES2Downloader/blob/main/LICENSE>`__.
-  Medoid compositing was adapted from `gee_tools <https://github.com/gee-community/gee_tools>`__ under the terms of the
   `MIT license <https://github.com/gee-community/gee_tools/blob/master/LICENSE>`__.
-  Sentinel-2 cloud / shadow masking was adapted from `ee_extra <https://github.com/r-earthengine/ee_extra>`__ under
   terms of the `Apache-2.0 license <https://github.com/r-earthengine/ee_extra/blob/master/LICENSE>`__

Author
------

**Dugal Harris** - dugalh@gmail.com

.. |Tests| image:: https://github.com/dugalh/geedim/actions/workflows/run-unit-tests.yml/badge.svg
   :target: https://github.com/dugalh/geedim/actions/workflows/run-unit-tests.yml
.. |codecov| image:: https://codecov.io/gh/dugalh/geedim/branch/main/graph/badge.svg?token=69GZNQ3TI3
   :target: https://codecov.io/gh/dugalh/geedim
.. |PyPI version| image:: https://badge.fury.io/py/geedim.svg
   :target: https://badge.fury.io/py/geedim
.. |Anaconda-Server Badge| image:: https://anaconda.org/conda-forge/geedim/badges/version.svg
   :target: https://anaconda.org/conda-forge/geedim
.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
