API
===

Initialisation
--------------

Geedim extends the `GEE API <https://github.com/google/earthengine-api>`__ with the
:class:`ee.Image.gd <geedim.image.ImageAccessor>` and :class:`ee.ImageCollection.gd <geedim.collection.ImageCollectionAccessor>` accessors.  To enable the accessors, Geedim must be imported:

.. jupyter-execute::
    :hide-code:

    from functools import partial

    import ee
    import geedim as gd
    from geedim import utils

    # prevent widget display
    utils.get_tqdm_kwargs = partial(utils.get_tqdm_kwargs, display=False)

    # initialise using EE_SERVICE_ACC_PRIVATE_KEY env variable if present
    gd.Initialize()

.. jupyter-input::

    import ee
    import geedim  # noqa: F401

    ee.Initialize()

If you use a linter, you may need to include the ``# noqa: F401`` suppression comment after the ``geedim`` import to prevent it being removed.

Cloud masking
-------------

Cloud masking is supported on Landsat 4-9 `collection 2 <https://developers.google.com/earth-engine/datasets/catalog/landsat>`__ images and Sentinel-2 `TOA <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED>`__ and `surface reflectance <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>`__ images.  The ``cloudSupport`` property is ``True`` on images / collections with cloud mask support:

.. jupyter-execute::

    im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')
    im.gd.cloudSupport


Image and collection accessors share the same interface for cloud masking.  Masks and related bands can be added with ``addMaskBands()``, and cloud masks applied with ``maskClouds()``:

.. jupyter-execute::

    # add mask bands created with a threshold of 0.7 on the 'cs_cdf' Cloud Score+ band
    im = im.gd.addMaskBands(score=0.7, cs_band='cs_cdf')
    print(im.gd.bandNames)

    im = im.gd.maskClouds()

The :meth:`~geedim.image.ImageAccessor.addMaskBands` reference documents the masking parameters for images / collections.

Filtering
---------

Collections can be filtered on the filled or cloud-free portions of a given ``region``, and other criteria, with :meth:`~geedim.collection.ImageCollectionAccessor.filter`:

.. jupyter-execute::

    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    # filter by date range, region bounds, and a lower limit of 60% on the cloud-free
    # portion of region
    filt_coll = coll.gd.filter(
        '2021-10-01', '2022-04-01', region=region, cloudless_portion=60
    )

The :attr:`~geedim.collection.ImageCollectionAccessor.schemaTable` and :attr:`~geedim.collection.ImageCollectionAccessor.propertiesTable` properties allow the collection contents to be displayed.  :attr:`~geedim.collection.ImageCollectionAccessor.schemaPropertyNames` defines a set of image properties to include in the tables:

.. jupyter-execute::

    # include the VEGETATION_PERCENTAGE property in schemaTable & propertiesTable
    filt_coll.gd.schemaPropertyNames += ('VEGETATION_PERCENTAGE',)

    print(filt_coll.gd.schemaTable)
    print(filt_coll.gd.propertiesTable)

Compositing
-----------

Collections can be composited using :meth:`~geedim.collection.ImageCollectionAccessor.composite`.  By default, cloud is masked in the component images before compositing.  E.g. to form a cloud-free :attr:`~geedim.enums.CompositeMethod.median` composite:

.. jupyter-execute::

    # create and filter a collection
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    filt_coll = coll.gd.filter('2021-10-01', '2022-04-01', region=region)

    # composite
    comp_im = filt_coll.gd.composite(method='median')

:class:`~geedim.enums.CompositeMethod` documents supported values for the ``method`` parameter.  The :attr:`~geedim.enums.CompositeMethod.mosaic`, :attr:`~geedim.enums.CompositeMethod.q_mosaic`, and :attr:`~geedim.enums.CompositeMethod.medoid` methods prioritise images in their sort order i.e. when more than one image pixel qualifies for selection, they select the first one.  Images can be sorted by closeness to the ``date`` parameter, or by cloud-free portion of the ``region`` parameter.  If neither ``date`` or ``region`` are supplied, images are sorted by capture date.

Exporting
---------

Preparation
~~~~~~~~~~~

Images are exported with the projection and bounds given by their :attr:`~geedim.image.ImageAccessor.crs`, :attr:`~geedim.image.ImageAccessor.transform` and :attr:`~geedim.image.ImageAccessor.shape` properties; and with data type given by their :attr:`~geedim.image.ImageAccessor.dtype` property:

.. jupyter-execute::

    im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')

    print(im.gd.crs)
    print(im.gd.transform)
    print(im.gd.shape)
    print(im.gd.dtype)

Collections are exported with the projection, bounds and data type given by the first collection image.

Both the image and collection accessors have a ``prepareForExport()`` method with the same parameters.  This can be called before exporting to change the projection, bounds and data type:

.. note::

    This is required for:

    - images without a fixed projection (e.g. composites)
    - collections whose images don't have a fixed projection, or don't share the same projection, bounds and data type

.. jupyter-execute::

    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    prep_im = im.gd.prepareForExport(
        crs='EPSG:3857', region=region, scale=30, dtype='uint16'
    )

Projection and bounds can be defined with the ``crs``, ``region`` and ``scale`` / ``shape``; or ``crs``, ``crs_transform`` and ``shape`` parameters.  Other parameters alter resampling, selected bands and scale / offset - see the :meth:`ee.Image.gd.prepareForExport() <geedim.image.ImageAccessor.prepareForExport>` or :meth:`ee.ImageCollection.gd.prepareForExport() <geedim.collection.ImageCollectionAccessor.prepareForExport>` docs for details.

GeoTIFF
~~~~~~~

Image
^^^^^

:meth:`ee.Image.gd.toGeoTIFF() <geedim.image.ImageAccessor.toGeoTIFF>` exports an image to a GeoTIFF file:

.. jupyter-execute::

    # create and prepare an image
    im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    prep_im = im.gd.prepareForExport(region=region, scale=30, dtype='uint16')

    # export
    prep_im.gd.toGeoTIFF('s2.tif')

.. _geotiff-tags:

Image :attr:`~geedim.image.ImageAccessor.properties` are written to the GeoTIFF default namespace tags, and :attr:`~geedim.image.ImageAccessor.bandProps` are written to the band tags:

.. jupyter-execute::

    import rasterio as rio

    with rio.open('s2.tif') as ds:
        # default namespace tags
        print(ds.tags())
        # band 1 tags
        print(ds.tags(bidx=1))

.. jupyter-execute::
    :hide-code:

    # clean up image export
    from pathlib import Path
    Path('s2.tif').unlink(missing_ok=True)

Collection
^^^^^^^^^^

:meth:`ee.ImageCollection.gd.toGeoTIFF() <geedim.collection.ImageCollectionAccessor.toGeoTIFF>` exports a collection to GeoTIFF files.  The ``split`` parameter controls whether exported files correspond to collection :attr:`~geedim.enums.SplitType.bands` or :attr:`~geedim.enums.SplitType.images`:

.. jupyter-execute::

    from pathlib import Path

    # create and prepare a collection (with two images and three bands)
    coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    coll = coll.filterBounds(region).limit(2)
    prep_coll = coll.gd.prepareForExport(
        region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
    )

    # create export directory
    dirname = Path('s2')
    dirname.mkdir()

    # export (one file for each collection band)
    prep_coll.gd.toGeoTIFF(dirname, split='bands')

    # display exported files
    [fp.name for fp in dirname.glob('*.tif')]

.. jupyter-execute::
    :hide-code:

    # clean up collection export
    for f in dirname.glob('*.tif'):
        f.unlink()
    if dirname.exists():
        dirname.rmdir()


When ``split`` is :attr:`~geedim.enums.SplitType.images`, image :attr:`~geedim.image.ImageAccessor.properties` are written to the GeoTIFF default namespace tags, and :attr:`~geedim.image.ImageAccessor.bandProps` are written to the band tags (see the :ref:`image <geotiff-tags>` example).

Nodata
^^^^^^

By default, GeoTIFF file nodata tags are set to the :attr:`~geedim.image.ImageAccessor.nodata` value of their corresponding images.  Both :meth:`ee.Image.gd.toGeoTIFF() <geedim.image.ImageAccessor.toGeoTIFF>` and :meth:`ee.ImageCollection.gd.toGeoTIFF() <geedim.collection.ImageCollectionAccessor.toGeoTIFF>` have a ``nodata`` parameter that allows this to be changed.  E.g.:

.. jupyter-execute::

    # set masked pixels to a new nodata value
    nodata = 65535
    prep_im = prep_im.unmask(nodata)

    # export, setting nodata to the new value
    prep_im.gd.toGeoTIFF('s2_nodata.tif', nodata=nodata)

    # display GeoTIFF nodata
    with rio.open('s2_nodata.tif') as ds:
        print(ds.nodata)

.. jupyter-execute::
    :hide-code:

    # clean up nodata export
    Path('s2_nodata.tif').unlink(missing_ok=True)

Paths and URIs
^^^^^^^^^^^^^^

The ``file`` argument in :meth:`ee.Image.gd.toGeoTIFF() <geedim.image.ImageAccessor.toGeoTIFF>` and ``dirname`` argument in :meth:`ee.ImageCollection.gd.toGeoTIFF() <geedim.collection.ImageCollectionAccessor.toGeoTIFF>` can be local paths or remote URIs.  See the :ref:`related note <getting_started/cli:paths and uris>` in the command line section for more information.

NumPy
~~~~~

Image
^^^^^

:meth:`ee.Image.gd.toNumPy() <geedim.image.ImageAccessor.toNumPy>` exports an image to a NumPy :class:`~numpy.ndarray`:

.. jupyter-execute::

    # create and prepare an image (with 3 bands)
    im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    prep_im = im.gd.prepareForExport(
        region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
    )

    # export (3D array with bands along the third dimension)
    array = prep_im.gd.toNumPy()

    # display array format
    print(type(array))
    print(array.shape)
    print(array.dtype)


Collection
^^^^^^^^^^

:meth:`ee.ImageCollection.gd.toNumPy() <geedim.collection.ImageCollectionAccessor.toNumPy>` exports a collection to a NumPy :class:`~numpy.ndarray`.  The ``split`` parameter controls the layout of collection bands and images in the exported array:

.. jupyter-execute::

    # create and prepare a collection (with two images and three bands)
    coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    coll = coll.filterBounds(region).limit(2)
    prep_coll = coll.gd.prepareForExport(
        region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
    )

    # export (4D array with bands along the third, and images along the fourth dimension)
    array = prep_coll.gd.toNumPy(split='bands')

    # display array format
    print(type(array))
    print(array.shape)
    print(array.dtype)

Masking and data type
^^^^^^^^^^^^^^^^^^^^^

Both :meth:`ee.Image.gd.toNumPy() <geedim.image.ImageAccessor.toNumPy>` and :meth:`ee.ImageCollection.gd.toNumPy() <geedim.collection.ImageCollectionAccessor.toNumPy>` have ``masked`` and ``structured`` parameters.  The ``masked`` parameter controls whether the exported array has masked pixels set to :attr:`~geedim.image.ImageAccessor.nodata`, or is a :class:`~numpy.ma.MaskedArray`.  The ``structured`` parameter controls whether the exported array has a `numerical <https://numpy.org/devdocs//user/basics.types.html#numerical-data-types>`__ or `structured <https://numpy.org/doc/stable/user/basics.rec.html#structured-arrays>`__ data type.  E.g.:

.. jupyter-execute::

    # export (2D masked array with a structured dtype representing the bands)
    array = prep_im.gd.toNumPy(masked=True, structured=True)

    # display array format
    print(type(array))
    print(array.shape)
    print(array.dtype)

Xarray
~~~~~~

.. jupyter-execute::
    :hide-code:

    import xarray as xr

    # don't expand data and attributes in the Xarray html repr
    _ = xr.set_options(display_expand_attrs=False, display_expand_data=False)

Image
^^^^^

:meth:`ee.Image.gd.toXarray() <geedim.image.ImageAccessor.toXarray>` exports an image to a Xarray :class:`~xarray.core.dataarray.DataArray`:

.. jupyter-execute::

    # create and prepare image
    im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    prep_im = im.gd.prepareForExport(
        region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
    )

    # export (3D DataArray)
    da = prep_im.gd.toXarray()

    da

Collection
^^^^^^^^^^

:meth:`ee.ImageCollection.gd.toXarray() <geedim.collection.ImageCollectionAccessor.toXarray>` exports a collection to a Xarray :class:`~xarray.core.dataset.Dataset`.  The ``split`` parameter controls whether dataset variables correspond to collection :attr:`~geedim.enums.SplitType.bands` or :attr:`~geedim.enums.SplitType.images`:

.. jupyter-execute::

    # create and prepare a collection (with two images and three bands)
    coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    coll = coll.filterBounds(region).limit(2)
    prep_coll = coll.gd.prepareForExport(
        region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
    )

    # export (Dataset with bands as variables)
    ds = prep_coll.gd.toXarray(split='bands')

    ds

Masking
^^^^^^^

Both :meth:`ee.Image.gd.toXarray() <geedim.image.ImageAccessor.toXarray>` and :meth:`ee.ImageCollection.gd.toXarray() <geedim.collection.ImageCollectionAccessor.toXarray>` have a ``masked`` parameter that controls whether exported masked pixels are set to :attr:`~geedim.image.ImageAccessor.nodata`, or to NaN.  If they are set to NaN, the export data type will be converted to a floating point type able to represent the data:

.. jupyter-execute::

    # create and prepare a cloud masked image
    im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')
    im = im.gd.addMaskBands().gd.maskClouds()
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    prep_im = im.gd.prepareForExport(
        region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
    )

    # export, setting masked pixels to NaN
    da = prep_im.gd.toXarray(masked=True)

    # check for NaN pixels and floating point data type
    print(da.isnull().any())
    print(da.dtype)

See the Xarray documentation on `missing values <https://docs.xarray.dev/en/stable/user-guide/computation.html#missing-values>`__ for background.

Attributes
^^^^^^^^^^

DataArray / Dataset attributes include ``crs``, ``transform`` and ``nodata`` values for compatibility with `rioxarray <https://github.com/corteva/rioxarray>`__, as well as ``ee`` and ``stac`` JSON strings of the Earth Engine property and STAC dictionaries.


Google cloud
~~~~~~~~~~~~

.. Use jupyter-input/jupyter-output in this section instead of jupyter-execute to speed up doc builds

Image
^^^^^

:meth:`ee.Image.gd.toGoogleCloud() <geedim.image.ImageAccessor.toGoogleCloud>` exports an image to Google Drive, Earth Engine asset or Google Cloud Storage:

.. jupyter-input::

    # create and prepare image
    im = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20211220T080341_20211220T082827_T35HKC')
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    prep_im = im.gd.prepareForExport(region=region, scale=30, dtype='uint16')

    # export to Earth Engine asset 's2' in the 'geedim' project, waiting for completion
    _ = prep_im.gd.toGoogleCloud('s2', type='asset', folder='geedim', wait=True)

    # display asset image info
    ee.Image('projects/geedim/assets/s2').getInfo()

.. jupyter-output::

    {'type': 'Image', 'bands': [{'id': 'B1', 'data_type': {'type': 'PixelType', ...


Collection
^^^^^^^^^^

:meth:`ee.ImageCollection.gd.toGoogleCloud() <geedim.collection.ImageCollectionAccessor.toGoogleCloud>` exports a collection to Google Drive, Earth Engine asset or Google Cloud Storage.  The ``split`` parameter controls whether exported files / assets correspond to collection :attr:`~geedim.enums.SplitType.bands` or :attr:`~geedim.enums.SplitType.images`:

.. jupyter-input::

    # create and prepare a collection (with two images and three bands)
    coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    coll = coll.filterBounds(region).limit(2)
    prep_coll = coll.gd.prepareForExport(
        region=region, scale=30, dtype='uint16', bands=['B4', 'B3', 'B2']
    )

    # export to Earth Engine assets in the 'geedim' project, waiting for completion
    # (one asset for each collection band)
    _ = prep_coll.gd.toGoogleCloud(type='asset', folder='geedim', wait=True, split='bands')

    # display the info of the first asset image
    ee.Image('projects/geedim/assets/B4').getInfo()

.. jupyter-output::

    {'type': 'Image', 'bands': [{'id': 'B_20180510T075611_20180510T082300_T35HKC', ...

Additional arguments
^^^^^^^^^^^^^^^^^^^^

Depending on the ``type`` parameter, ``toGoogleCloud()`` calls one of the ``Export.image.toDrive()``, ``Export.image.toAsset()`` or ``Export.image.toCloudStorage()`` Earth Engine functions to perform the export.  :meth:`ee.Image.gd.toGoogleCloud() <geedim.image.ImageAccessor.toGoogleCloud>` and :meth:`ee.ImageCollection.gd.toGoogleCloud() <geedim.collection.ImageCollectionAccessor.toGoogleCloud>` allow additional keyword arguments to be passed to the ``type`` relevant Earth Engine function.  See the |toDrive|_, |toAsset|_ or |toCloudStorage|_ docs for supported parameters.  E.g.

.. jupyter-input::

    # export to Google Drive using the TFRecord format
    _ = prep_im.gd.toGoogleCloud(
        's2',
        type='drive',
        folder='geedim',
        fileFormat='TFRecord',
        formatOptions={'patchDimensions': [256, 256], 'compressed': True},
    )

Tiling
~~~~~~

The ``toGeoTIFF()``, ``toNumPy()`` and ``toXarray()`` methods divide images into tiles for export.  Tiles are downloaded and decompressed concurrently, then reassembled into the target export format.  Tile size can be controlled with the ``max_tile_size``, ``max_tile_dim`` and ``max_tile_bands`` parameters.  Download concurrency can be controlled with the ``max_requests``, and decompress concurrency with the ``max_cpus`` parameter.  Each parameter has an upper limit - see the ``toGeoTIFF()``, ``toNumPy()`` or ``toXarray()`` :doc:`reference docs <../reference/api>` for details.  For most uses, the tiling parameters can be left on their default values.

User memory limit error
~~~~~~~~~~~~~~~~~~~~~~~

Exporting computed images with ``toGeoTIFF()``, ``toNumPy()`` or ``toXarray()`` could raise a ``'User memory limit exceeded'`` error in some unusual cases.  Earth Engine raises this error if a computation exceeds the `limit on user memory <https://developers.google.com/earth-engine/guides/usage#per-request_memory_footprint>`__.  E.g.:

.. jupyter-execute::
    :raises:

    # create a 2 year cloud-free median composite
    coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    region = ee.Geometry.Rectangle(24.35, -33.75, 24.45, -33.65)
    coll = coll.gd.filter('2021-01-01', '2023-01-01', region=region)
    comp_im = coll.gd.composite(method='median')

    # prepare the composite for export
    prep_im = comp_im.gd.prepareForExport(
        crs='EPSG:3857', region=region, scale=10, dtype='uint16'
    )

    # attempt export to NumPy array
    array = prep_im.gd.toNumPy()

``toGoogleCloud()`` is not subject to the limit and using it for export is recommended in this situation.  Images can first be exported to Earth Engine asset with ``toGoogleCloud()``, and then the computed assets exported to a target format with one of ``toGeoTIFF()``, ``toNumPy()`` or ``toXarray()``.  E.g.:

.. jupyter-input::

    # export composite to Earth Engine asset 's2-comp' in the 'geedim' project
    _ = prep_im.gd.toGoogleCloud('s2-comp', type='asset', folder='geedim', wait=True)

    # export the asset to a NumPy array
    array = ee.Image('projects/geedim/assets/s2-comp').gd.toNumPy()

.. |toDrive| replace:: ``Export.image.toDrive()``
.. |toAsset| replace:: ``Export.image.toAsset()``
.. |toCloudStorage| replace:: ``Export.image.toCloudStorage()``
.. _toDrive: https://developers.google.com/earth-engine/apidocs/export-image-todrive
.. _toAsset: https://developers.google.com/earth-engine/apidocs/export-image-toasset
.. _toCloudStorage: https://developers.google.com/earth-engine/apidocs/export-image-tocloudstorage
