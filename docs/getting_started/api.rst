API
===

.. TODO: make all code cross-refs use :meth:`ee.Image.gd.method() <geedim.image.ImageAccessor.method>` style?

Initialisation
--------------

Geedim extends the `GEE API <https://github.com/google/earthengine-api>`__ with the
|ee.Image.gd|_ and |ee.ImageCollection.gd|_ accessors.  To enable the accessors, Geedim must be imported:

.. literalinclude:: api.py
    :language: python
    :start-after: [initialise]
    :end-before: [end initialise]

If you use a linter, you may need to include the ``# noqa: F401`` suppression comment after the ``geedim`` import to prevent it being removed.

Cloud / shadow
--------------

Masking
~~~~~~~

Cloud / shadow masking is supported on Landsat 4-9 `collection 2 <https://developers.google.com/earth-engine/datasets/catalog/landsat>`__ images and Sentinel-2 `TOA <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED>`__ and `surface reflectance <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>`__ images.  The ``cloudShadowSupport`` property is ``True`` on images / collections with cloud / shadow support:

.. literalinclude:: api.py
    :language: python
    :start-after: [cloud support]
    :end-before: [end cloud support]

Image and collection accessors share the same interface for cloud / shadow masking.  Cloud / shadow masks and related bands can be added with ``addMaskBands()``, and image mask(s) updated with ``maskClouds()``:

.. literalinclude:: api.py
    :language: python
    :start-after: [mask]
    :end-before: [end mask]

The :meth:`~geedim.image.ImageAccessor.addMaskBands` reference documents the masking parameters for images / collections.

Filtering
~~~~~~~~~

Collections can be filtered on the filled or cloud / shadow - free portions of a given region with :meth:`~geedim.collection.ImageCollectionAccessor.filter`.  For convenience, this method allows date, region and custom filters to be included too:

.. literalinclude:: api.py
    :language: python
    :start-after: [filter]
    :end-before: [end filter]

The :attr:`~geedim.collection.ImageCollectionAccessor.schemaTable` and :attr:`~geedim.collection.ImageCollectionAccessor.propertiesTable` properties allow the collection contents to be displayed.  :attr:`~geedim.collection.ImageCollectionAccessor.schemaPropertyNames` defines a set of image :attr:`~geedim.collection.ImageCollectionAccessor.properties` to include in the tables:

.. literalinclude:: api.py
    :language: python
    :start-after: [tables]
    :end-before: [end tables]

Compositing
~~~~~~~~~~~~

Collections can be composited using :meth:`~geedim.collection.ImageCollectionAccessor.composite`.  By default, cloud / shadow is masked in the component images before forming the composite by default.  E.g. to form a cloud / shadow - free :attr:`~geedim.enums.CompositeMethod.median` composite:

.. literalinclude:: api.py
    :language: python
    :start-after: [composite]
    :end-before: [end composite]

:class:`~geedim.enums.CompositeMethod` documents supported values for the ``method`` parameter.  Optionally, images can be prioritised by closeness to a provided ``date``, or by cloud / shadow - free portion of a ``region``.  See the :meth:`~geedim.collection.ImageCollectionAccessor.composite` reference for details.

Exporting
---------

Preparation
~~~~~~~~~~~

Images are exported on the pixel grid and bounds given by their :attr:`~geedim.image.ImageAccessor.crs`, :attr:`~geedim.image.ImageAccessor.transform` and :attr:`~geedim.image.ImageAccessor.shape` properties; and with data type given by their :attr:`~geedim.image.ImageAccessor.dtype` property:

.. literalinclude:: api.py
    :language: python
    :start-after: [image grid]
    :end-before: [end image grid]

Collections are exported on the pixel grid, bounds and data type given by the first collection image.

Both the image and collection accessor have a ``prepareForExport()`` method with the same parameters.  This can be called before exporting to change the pixel grid, bounds and data type:

.. note::

    This is required for:

    - images without a fixed projection (e.g. composites)
    - collections whose images don't have a fixed projection, or don't share the same pixel grid, bounds and data type

.. literalinclude:: api.py
    :language: python
    :start-after: [image prepare for export]
    :end-before: [end image prepare for export]

Pixel grid and bounds can be defined with the ``crs``, ``region`` and ``scale`` / ``shape``; or ``crs``, ``crs_transform`` and ``shape`` parameters.  Other parameters alter resampling, selected bands and scale / offset - see the :meth:`ee.Image.gd.prepareForExport() <geedim.image.ImageAccessor.prepareForExport>` or :meth:`ee.ImageCollection.gd.prepareForExport() <geedim.collection.ImageCollectionAccessor.prepareForExport>` docs for details.

GeoTIFF
~~~~~~~

.. _geotiff-image:

Image
^^^^^

:meth:`ee.Image.toGeoTIFF() <geedim.image.ImageAccessor.toGeoTIFF>` exports an image to a GeoTIFF file:

.. literalinclude:: api.py
    :language: python
    :start-after: [image geotiff]
    :end-before: [end image geotiff]

Image :attr:`~geedim.image.ImageAccessor.properties` are written to the GeoTIFF file default namespace tags, and :attr:`~geedim.image.ImageAccessor.bandProps` are written to the band tags:

.. literalinclude:: api.py
    :language: python
    :start-after: [image geotiff tags]
    :end-before: [end image geotiff tags]

Collection
^^^^^^^^^^

:meth:`ee.ImageCollection.toGeoTIFF() <geedim.collection.ImageCollectionAccessor.toGeoTIFF>` exports a collection to GeoTIFF file(s).  The ``split`` parameter controls whether exported files correspond to collection :attr:`~geedim.enums.SplitType.bands` or :attr:`~geedim.enums.SplitType.images`:

.. literalinclude:: api.py
    :language: python
    :start-after: [coll geotiff]
    :end-before: [end coll geotiff]

When ``split`` is images, image :attr:`~geedim.image.ImageAccessor.properties` are written to the GeoTIFF file default namespace tags, and :attr:`~geedim.image.ImageAccessor.bandProps` are written to the band tags (see the :ref:`image <geotiff-image>` example).

Nodata
^^^^^^

By default, GeoTIFF file nodata tag(s) are set to the :attr:`~geedim.image.ImageAccessor.nodata` value of their corresponding image(s).  Both :meth:`ee.Image.toGeoTIFF() <geedim.image.ImageAccessor.toGeoTIFF>` and :meth:`ee.ImageCollection.toGeoTIFF() <geedim.collection.ImageCollectionAccessor.toGeoTIFF>` have a ``nodata`` parameter that allows this to be changed.  E.g.:

.. literalinclude:: api.py
    :language: python
    :start-after: [geotiff nodata]
    :end-before: [end geotiff nodata]


NumPy
~~~~~

Image
^^^^^

:meth:`ee.Image.toNumPy() <geedim.image.ImageAccessor.toNumPy>` exports an image to a NumPy :class:`~numpy.ndarray`:

.. literalinclude:: api.py
    :language: python
    :start-after: [image numpy]
    :end-before: [end image numpy]

Collection
^^^^^^^^^^

:meth:`ee.ImageCollection.toNumPy() <geedim.collection.ImageCollectionAccessor.toNumPy>` exports a collection to a NumPy :class:`~numpy.ndarray`.  The ``split`` parameter controls if the the layout of collection bands and images in the exported array:

.. literalinclude:: api.py
    :language: python
    :start-after: [coll numpy]
    :end-before: [end coll numpy]

Masked and structured arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both :meth:`ee.Image.toNumPy() <geedim.image.ImageAccessor.toNumPy>` and :meth:`ee.ImageCollection.toNumPy() <geedim.collection.ImageCollectionAccessor.toNumPy>` have ``masked`` and ``structured`` parameters.  The ``masked`` parameter controls whether the exported array has masked pixels set to :attr:`~geedim.image.ImageAccessor.nodata`, or is a :class:`~numpy.ma.MaskedArray`.  The ``structured`` parameter controls whether the exported array has a `numerical <https://numpy.org/devdocs//user/basics.types.html#numerical-data-types>`__ or `structured <https://numpy.org/doc/stable/user/basics.rec.html#structured-arrays>`__ data type.  E.g.:

.. literalinclude:: api.py
    :language: python
    :start-after: [numpy masked structured]
    :end-before: [end numpy masked structured]

Xarray
~~~~~~

Image
^^^^^

:meth:`ee.Image.toXarray() <geedim.image.ImageAccessor.toXarray>` exports an image to a Xarray :class:`~xarray.core.dataarray.DataArray`:

.. literalinclude:: api.py
    :language: python
    :start-after: [image xarray]
    :end-before: [end image xarray]

Collection
^^^^^^^^^^

:meth:`ee.ImageCollection.toXarray() <geedim.collection.ImageCollectionAccessor.toXarray>` exports a collection to a Xarray :class:`~xarray.core.dataset.Dataset`.  The ``split`` parameter controls whether dataset variables correspond to collection :attr:`~geedim.enums.SplitType.bands` or :attr:`~geedim.enums.SplitType.images`:

.. literalinclude:: api.py
    :language: python
    :start-after: [coll xarray]
    :end-before: [end coll xarray]

Masking
^^^^^^^

Both :meth:`ee.Image.toXarray() <geedim.image.ImageAccessor.toXarray>` and :meth:`ee.ImageCollection.toXarray() <geedim.collection.ImageCollectionAccessor.toXarray>` have a ``masked`` parameter that controls whether exported masked pixels are set to :attr:`~geedim.image.ImageAccessor.nodata`, or to NaN.  If they are set to NaN, the export data type will be converted to a floating point type able to represent the data:

.. literalinclude:: api.py
    :language: python
    :start-after: [xarray masked]
    :end-before: [end xarray masked]

See the Xarray documentation on `missing values <https://docs.xarray.dev/en/stable/user-guide/computation.html#missing-values>`__ for background.

Attributes
^^^^^^^^^^

DataArray / Dataset attributes include :attr:`~geedim.image.ImageAccessor.crs`, :attr:`~geedim.image.ImageAccessor.transform` and ``nodata`` values for compatibility with `rioxarray <https://github.com/corteva/rioxarray>`__, as well as ``ee`` and ``stac`` JSON strings of the Earth Engine property and STAC dictionaries.

.. |ee.Image.gd| replace:: ``ee.Image.gd``
.. |ee.ImageCollection.gd| replace:: ``ee.ImageCollection.gd``
.. _ee.Image.gd: ../../reference/api/api.html#geedim.image.ImageAccessor
.. _ee.ImageCollection.gd: ../../reference/api/api.html#geedim.collection.ImageCollectionAccessor