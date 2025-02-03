# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from geedim.schema import cloud_coll_table
from geedim.version import __version__

# -- Project information -----------------------------------------------------

project = 'geedim'
copyright = '2022-2023, Leftfield Geospatial'
author = 'Leftfield Geospatial'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_click',
    'jupyter_sphinx',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for autodoc -------------------------------------------------
# see https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
# autodoc_mock_imports = ['rasterio', 'click']
autosummary_generate = True
autoclass_content = 'class'
autodoc_class_signature = 'separated'
autodoc_member_order = 'bysource'
autodoc_typehints = 'both'
# autodoc_typehints_format = 'short'

# -- Options for nbsphinx --------------------------------------------------
# env.docname will be e.g. examples/l7_composite.ipynb.  The `../` is to
# reference it from itself. preferable to link to actual version of the file
# at the time of the doc build, than a hyperlink to github.
# see https://github.com/aazuspan/wxee/blob/main/docs/conf.py for other examples
nbsphinx_prolog = """
.. note::

   This page was generated from a Jupyter notebook. To run and interact with it, 
   you can download it :download:`here <../{{ env.docname }}.ipynb>`.
"""
nbsphinx_widgets_path = ''
nbsphinx_requirejs_path = ''

# -- Options for intersphinx ---------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'rasterio': ('https://rasterio.readthedocs.io/en/stable/', None),
    'gdal': ('https://gdal.org/', None),
    'fsspec': ('https://filesystem-spec.readthedocs.io/en/latest/', None),
    'affine': ('https://affine.readthedocs.io/en/latest/', None),
    'aiohttp': ('https://docs.aiohttp.org/en/stable/', None),
}

# -- Generate cloud/shadow supported collection tables for github README and RTD
# docs
with open('cloud_coll_rtd.rst', 'w') as f:
    f.write(cloud_coll_table(descr_join='\n\n'))
