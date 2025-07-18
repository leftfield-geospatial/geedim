Contributing
============

Contributions are welcome.  Please report bugs and make feature requests with the github `issue tracker
<https://github.com/leftfield-geospatial/geedim/issues>`_.

Development environment
-----------------------

``geedim`` uses the `rasterio <https://github.com/rasterio/rasterio>`_ package, which has binary dependencies.  Under
Windows, it is easiest to resolve these dependencies by working in a ``conda`` environment.  You can set this up with:

.. code:: shell

    conda create -n <environment name> python=3.9 -c conda-forge
    conda activate <environment name>
    conda install -c conda-forge earthengine-api rasterio click requests tqdm tabulate pytest

If you are using Linux, or macOS, you may want to create a clean virtual python environment.  Once the environment is
set up, create a fork of the ``geedim`` github repository, and clone it:

.. code:: shell

    git clone https://github.com/<username>/geedim.git

Finally, install the local ``geedim`` package into your python environment:

.. code:: shell

    pip install -e geedim


Development guide
-----------------

Cloud/shadow masking
^^^^^^^^^^^^^^^^^^^^

If you want to add cloud masking support for a new Earth Engine image collection, you should subclass
``geedim.mask.CloudMaskedImage``, and implement at least the ``_aux_image()`` method.  Then add a new entry to
``geedim.schema.collection_schema``.

Testing
^^^^^^^

Please include `pytest <https://docs.pytest.org>`__ tests with your code.  The existing tests require the user
to be registered with `Google Earth Engine <https://signup.earthengine.google.com>`__.  Installing the `pytest-xdist
<https://github.com/pytest-dev/pytest-xdist>`_ plugin will help speed the testing process.  For ``conda`` users:

.. code:: shell

    conda install -c conda-forge pytest-xdist

Or, using ``pip``:

.. code:: shell

    pip install pytest-xdist

You can then run the tests from the root of the ``geedim`` repository with:

.. code:: shell

    pytest -v -n auto tests

Style
^^^^^

Please include `NumPy docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_ with
your code. Try to conform to the ``geedim`` code style.  You can auto-format with
`yapf <https://github.com/google/yapf>`__ and the included
`.style.yapf <https://github.com/leftfield-geospatial/geedim/blob/main/ .style.yapf>`__ configuration file:

.. code::

    yapf --style .style.yapf -i <file path>

