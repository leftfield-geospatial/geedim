API
===

The main ``geedim`` classes are :class:`geedim.mask.MaskedImage` and :class:`geedim.collection.MaskedCollection`.

Getting started
---------------

Example
^^^^^^^

.. include:: ../README.rst
    :start-after: api_example_start
    :end-before: api_example_end


Usage
-----

geedim.MaskedImage
^^^^^^^^^^^^^^^^^^

.. currentmodule:: geedim.mask

.. autosummary::
    :toctree: _generated

    MaskedImage

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
    ~MaskedImage.footprint
    ~MaskedImage.crs
    ~MaskedImage.transform
    ~MaskedImage.scale
    ~MaskedImage.shape
    ~MaskedImage.count
    ~MaskedImage.dtype
    ~MaskedImage.size
    ~MaskedImage.has_fixed_projection
    ~MaskedImage.properties
    ~MaskedImage.name
    ~MaskedImage.stac
    ~MaskedImage.band_props


geedim.MaskedCollection
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: geedim.collection

.. autosummary::
    :toctree: _generated/

    MaskedCollection

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
    ~MaskedCollection.stac
    ~MaskedCollection.stats_scale
    ~MaskedCollection.refl_bands


geedim.enums
^^^^^^^^^^^^

.. currentmodule:: geedim.enums

.. autosummary::
    :toctree: _generated/

    CompositeMethod
    CloudMaskMethod
    ResamplingMethod
