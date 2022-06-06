API
===

The main ``geedim`` classes are :class:`~geedim.mask.MaskedImage` and :class:`~geedim.collection.MaskedCollection`.

Getting started
---------------

Example
^^^^^^^

.. include:: ../README.rst
    :start-after: api_example_start
    :end-before: api_example_end


Usage
-----

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


geedim.enums
^^^^^^^^^^^^

.. currentmodule:: geedim.enums

.. autosummary::
    :toctree: _generated/

    CompositeMethod
    CloudMaskMethod
    ResamplingMethod
