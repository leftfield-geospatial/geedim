API
===

Getting started
---------------

This section gives a quick overview of the API.  You can also take a look at the :ref:`tutorial <tutorial>`.

Initialisation
^^^^^^^^^^^^^^

:meth:`~geedim.utils.Initialize` provides a shortcut for initialising the Earth Engine API.

.. literalinclude:: examples/api_getting_started.py
    :language: python
    :start-after: [initialise-start]
    :end-before: [initialise-end]


Searching image collections
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Any Earth Engine image collection <https://developers.google.com/earth-engine/datasets/catalog>`_ can be searched with
:class:`~geedim.collection.MaskedCollection`.  Here, we search for
`Landsat-8 surface reflectance <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2>`_
images over Stellenbosch, South Africa.

.. literalinclude:: examples/api_getting_started.py
    :language: python
    :start-after: [search-start]
    :end-before: [search-end]

The output:

.. _search results:

.. code:: shell

    ABBREV    DESCRIPTION
    --------- ---------------------------------------
    ID        Earth Engine image id
    DATE      Image capture date/time (UTC)
    FILL      Portion of region pixels that are valid (%)
    CLOUDLESS Portion of filled pixels that are cloud/shadow free (%)
    GRMSE     Orthorectification RMSE (m)
    SAA       Solar azimuth angle (deg)
    SEA       Solar elevation angle (deg)

    ID                                          DATE              FILL CLOUDLESS GRMSE   SAA   SEA
    ------------------------------------------- ---------------- ----- --------- ----- ----- -----
    LANDSAT/LC08/C02/T1_L2/LC08_175083_20190101 2019-01-01 08:35 99.83     55.62  8.48 79.36 59.20
    LANDSAT/LC08/C02/T1_L2/LC08_175084_20190101 2019-01-01 08:35 99.79     60.35  9.71 77.28 58.67
    LANDSAT/LC08/C02/T1_L2/LC08_175083_20190117 2019-01-17 08:35 99.98     94.90  8.84 76.98 56.79
    LANDSAT/LC08/C02/T1_L2/LC08_175084_20190117 2019-01-17 08:35 99.97     95.07  9.75 75.13 56.21
    LANDSAT/LC08/C02/T1_L2/LC08_175083_20190202 2019-02-02 08:34 99.91     95.82  8.46 71.91 54.00
    LANDSAT/LC08/C02/T1_L2/LC08_175084_20190202 2019-02-02 08:35 99.87     95.21  9.21 70.34 53.30

Image creation and download
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Images can be created, masked and downloaded with the :class:`~geedim.mask.MaskedImage` class.  Typically, one would
pass the Earth Engine image ID to :meth:`.MaskedImage.from_id` to create the image.

.. literalinclude:: examples/api_getting_started.py
    :language: python
    :start-after: [image-download-start]
    :end-before: [image-download-end]

Compositing
^^^^^^^^^^^

Let's form a cloud/shadow-free composite of the search result images, using the *q-mosaic* method, then download
the result.  By specifying the ``region`` parameter to :meth:`.MaskedCollection.composite`, we prioritise selection
of pixels from the least cloudy images when forming the composite.

.. note::
    When downloading composite images, the ``region``, ``crs`` and ``scale`` parameters must be specified, as the image
    has no fixed (known) projection.

.. literalinclude:: examples/api_getting_started.py
    :language: python
    :start-after: [composite-start]
    :end-before: [composite-end]


Cloud/shadow masking
^^^^^^^^^^^^^^^^^^^^

All :class:`~geedim.mask.MaskedImage` and :class:`~geedim.collection.MaskedCollection` methods that involve cloud/shadow
masking (:meth:`.MaskedImage.from_id`, :meth:`.MaskedCollection.search`, and :meth:`.MaskedCollection.composite`)
take optional cloud/shadow masking ``**kwargs``.  See :meth:`.MaskedImage.__init__` for a description of these
parameters.

Here, we create and download a cloud/shadow masked
`Sentinel-2 image <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR>`_, specifying a cloud probability threshold of 30%.

.. literalinclude:: examples/api_getting_started.py
    :language: python
    :start-after: [mask-start]
    :end-before: [mask-end]


Image metadata
^^^^^^^^^^^^^^

``geedim`` populates downloaded files with metadata from the source Earth Engine image, and the associated STAC entry.
The next code snippet uses `rasterio <https://github.com/rasterio/rasterio>`_ to read the metadata of the downloaded
Sentinel-2 image.

.. literalinclude:: examples/api_getting_started.py
    :language: python
    :start-after: [metadata-start]
    :end-before: [metadata-end]

Output:

.. code:: shell

    Image properties:
     {'AOT_RETRIEVAL_ACCURACY': '0', 'AREA_OR_POINT': 'Area', 'CLOUDY_PIXEL_PERCENTAGE': '4.228252', 'CLOUD_COVERAGE_ASSESSMENT': '4.228252', 'CLOUD_SHADOW_PERCENTAGE': '0.353758', 'DARK_FEATURES_PERCENTAGE': '1.390344', 'DATASTRIP_ID': 'S2A_OPER_MSI_L2A_DS_MTI__20190101T112242_S20190101T084846_N02.11', 'DATATAKE_IDENTIFIER': 'GS2A_20190101T082331_018422_N02.11', 'DATATAKE_TYPE': 'INS-NOBS', 'DEGRADED_MSI_DATA_PERCENTAGE': '0', 'FORMAT_CORRECTNESS': 'PASSED', 'GENERAL_QUALITY': 'PASSED', 'GENERATION_TIME': '1546341762000', 'GEOMETRIC_QUALITY': 'PASSED', 'GRANULE_ID': 'L2A_T34HCH_A018422_20190101T084846', 'HIGH_PROBA_CLOUDS_PERCENTAGE': '1.860266', 'LICENSE': 'https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#terms-of-use', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B1': '197.927117994', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B10': '200.473257333', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B11': '199.510962726', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B12': '198.925728126', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B2': '204.233801024', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B3': '201.623624653', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B4': '200.124411228', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B5': '199.531415295', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B6': '199.06932777', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B7': '198.686746475', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B8': '202.762429499', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B8A': '198.389627535', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B9': '197.722038042', 'MEAN_INCIDENCE_ZENITH_ANGLE_B1': '3.18832352559', 'MEAN_INCIDENCE_ZENITH_ANGLE_B10': '2.74480253282', 'MEAN_INCIDENCE_ZENITH_ANGLE_B11': '2.9240632909', 'MEAN_INCIDENCE_ZENITH_ANGLE_B12': '3.12457836272', 'MEAN_INCIDENCE_ZENITH_ANGLE_B2': '2.56258512587', 'MEAN_INCIDENCE_ZENITH_ANGLE_B3': '2.66821142821', 'MEAN_INCIDENCE_ZENITH_ANGLE_B4': '2.78791939543', 'MEAN_INCIDENCE_ZENITH_ANGLE_B5': '2.86049380258', 'MEAN_INCIDENCE_ZENITH_ANGLE_B6': '2.93757718579', 'MEAN_INCIDENCE_ZENITH_ANGLE_B7': '3.01912758709', 'MEAN_INCIDENCE_ZENITH_ANGLE_B8': '2.61179829178', 'MEAN_INCIDENCE_ZENITH_ANGLE_B8A': '3.10418395274', 'MEAN_INCIDENCE_ZENITH_ANGLE_B9': '3.28253454154', 'MEAN_SOLAR_AZIMUTH_ANGLE': '74.331216318', 'MEAN_SOLAR_ZENITH_ANGLE': '27.589988524', 'MEDIUM_PROBA_CLOUDS_PERCENTAGE': '0.774948', 'MGRS_TILE': '34HCH', 'NODATA_PIXEL_PERCENTAGE': '2.7e-05', 'NOT_VEGETATED_PERCENTAGE': '72.305781', 'PROCESSING_BASELINE': '02.11', 'PRODUCT_ID': 'S2A_MSIL2A_20190101T082331_N0211_R121_T34HCH_20190101T112242', 'RADIATIVE_TRANSFER_ACCURACY': '0', 'RADIOMETRIC_QUALITY': 'PASSED', 'REFLECTANCE_CONVERSION_CORRECTION': '1.03413456106', 'SATURATED_DEFECTIVE_PIXEL_PERCENTAGE': '0', 'SENSING_ORBIT_DIRECTION': 'DESCENDING', 'SENSING_ORBIT_NUMBER': '121', 'SENSOR_QUALITY': 'PASSED', 'SNOW_ICE_PERCENTAGE': '0.0156', 'SOLAR_IRRADIANCE_B1': '1884.69', 'SOLAR_IRRADIANCE_B10': '367.15', 'SOLAR_IRRADIANCE_B11': '245.59', 'SOLAR_IRRADIANCE_B12': '85.25', 'SOLAR_IRRADIANCE_B2': '1959.72', 'SOLAR_IRRADIANCE_B3': '1823.24', 'SOLAR_IRRADIANCE_B4': '1512.06', 'SOLAR_IRRADIANCE_B5': '1424.64', 'SOLAR_IRRADIANCE_B6': '1287.61', 'SOLAR_IRRADIANCE_B7': '1162.08', 'SOLAR_IRRADIANCE_B8': '1041.63', 'SOLAR_IRRADIANCE_B8A': '955.32', 'SOLAR_IRRADIANCE_B9': '812.92', 'SPACECRAFT_NAME': 'Sentinel-2A', 'system-asset_size': '1820790758', 'system-index': '20190101T082331_20190101T084846_T34HCH', 'system-time_end': '1546332584000', 'system-time_start': '1546332584000', 'THIN_CIRRUS_PERCENTAGE': '1.593038', 'UNCLASSIFIED_PERCENTAGE': '1.202621', 'VEGETATION_PERCENTAGE': '18.241563', 'WATER_PERCENTAGE': '2.262083', 'WATER_VAPOUR_RETRIEVAL_ACCURACY': '0'}
    Band names:
     ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'AOT', 'WVP', 'SCL', 'TCI_R', 'TCI_G', 'TCI_B', 'MSK_CLDPRB', 'MSK_SNWPRB', 'QA10', 'QA20', 'QA60', 'FILL_MASK', 'CLOUD_MASK', 'CLOUDLESS_MASK', 'SHADOW_MASK', 'CLOUD_PROB', 'CLOUD_DIST')
    Band 1 properties:
     {'center_wavelength': '0.4439', 'description': 'Aerosols', 'gsd': '60', 'name': 'B1', 'scale': '0.0001', 'wavelength': '443.9nm (S2A) / 442.3nm (S2B)'}


Computed images and user memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Earth engine has a size limit of 32 MB on `download requests <https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl>`_.  ``geedim`` avoids exceeding this by tiling downloads.  However, Earth engine also has a `limit on user memory <https://developers.google.com/earth-engine/guides/usage#per-request_memory_footprint>`_ for image computations.  In some situations, this limit can be exceeded when downloading large computed images.  This generates a *user memory limit exceeded* error.  (such as custom user images or ``geedim`` generated composites).  Unfortunately, there is no way for ``geedim`` to adjust tiles to avoid exceeding this limit, as the memory requirements of a computation are not known in advance.  The user has two options for working around this error:

1) max_tile_size
~~~~~~~~~~~~~~~~

Decreasing the ``max_tile_size`` argument to :meth:`geedim.mask.MaskedImage.download` reduces the user memory required by computations.  The user would need to experiment to find a reduced value that solves any memory limit problem.  :option:`--max-tile-size <geedim-download --max-tile-size>` is the equivalent option on the command line.

.. literalinclude:: examples/api_getting_started.py
    :language: python
    :start-after: [max_tile_size-start]
    :end-before: [max_tile_size-end]

2) Exporting
~~~~~~~~~~~~~

Exporting the image to an Earth Engine asset, and then downloading.  Exporting images is not subject to the user memory limit, and once exported, computation on the asset image is complete.  The exported asset image can then be downloaded in the standard way.

.. literalinclude:: examples/api_getting_started.py
    :language: python
    :start-after: [export-asset-download-start]
    :end-before: [export-asset-download-end]


Reference
---------

MaskedImage
^^^^^^^^^^^

.. currentmodule:: geedim.mask

.. autoclass:: MaskedImage
    :special-members: __init__

.. rubric:: Methods

.. autosummary::
    :toctree: _generated

    ~MaskedImage.from_id
    ~MaskedImage.mask_clouds
    ~MaskedImage.download
    ~MaskedImage.export
    ~MaskedImage.monitor_export


.. rubric:: Attributes

.. autosummary::
    :toctree: _generated

    ~MaskedImage.ee_image
    ~MaskedImage.id
    ~MaskedImage.date
    ~MaskedImage.crs
    ~MaskedImage.scale
    ~MaskedImage.footprint
    ~MaskedImage.transform
    ~MaskedImage.shape
    ~MaskedImage.count
    ~MaskedImage.dtype
    ~MaskedImage.size
    ~MaskedImage.has_fixed_projection
    ~MaskedImage.name
    ~MaskedImage.properties
    ~MaskedImage.band_properties


MaskedCollection
^^^^^^^^^^^^^^^^

.. currentmodule:: geedim.collection

.. autoclass:: MaskedCollection
    :special-members: __init__


.. rubric:: Methods

.. autosummary::
    :toctree: _generated/

    ~MaskedCollection.from_name
    ~MaskedCollection.from_list
    ~MaskedCollection.search
    ~MaskedCollection.composite


.. rubric:: Attributes

.. autosummary::
    :toctree: _generated/

    ~MaskedCollection.ee_collection
    ~MaskedCollection.name
    ~MaskedCollection.image_type
    ~MaskedCollection.properties
    ~MaskedCollection.properties_table
    ~MaskedCollection.schema
    ~MaskedCollection.schema_table
    ~MaskedCollection.refl_bands


enums
^^^^^

CompositeMethod
~~~~~~~~~~~~~~~

.. currentmodule:: geedim.enums

.. autoclass:: CompositeMethod
    :members:

CloudMaskMethod
~~~~~~~~~~~~~~~

.. currentmodule:: geedim.enums

.. autoclass:: CloudMaskMethod
    :members:

ResamplingMethod
~~~~~~~~~~~~~~~~~

.. currentmodule:: geedim.enums

.. autoclass:: ResamplingMethod
    :members:

ExportType
~~~~~~~~~~

.. currentmodule:: geedim.enums

.. autoclass:: ExportType
    :members:


Initialize
^^^^^^^^^^

.. currentmodule:: geedim.utils

.. autofunction:: Initialize


.. _tutorial:

Tutorial
--------

.. toctree::
    :maxdepth: 1

    examples/l7_composite.ipynb
