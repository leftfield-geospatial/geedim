# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path information -----------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from geedim.version import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Geedim'
copyright = '2021, Leftfield Geospatial'
author = 'Leftfield Geospatial'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx_click',
    'sphinx_copybutton',
    # 'jupyter_sphinx',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_temp', 'examples']
# nitpicky = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
# html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for autodoc -----------------------------------------------------
# see https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autoclass_content = 'both'

# -- Options for intersphinx ---------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'rasterio': ('https://rasterio.readthedocs.io/en/stable/', None),
    'gdal': ('https://gdal.org/', None),
    'fsspec': ('https://filesystem-spec.readthedocs.io/en/latest/', None),
    'affine': ('https://affine.readthedocs.io/en/latest/', None),
}

# -- Options for pygments -----------------------------------------------------
highlight_language = 'none'

# -- Options for autosectionlabel ----------------------------------------------------
# Make sure the target is unique
autosectionlabel_prefix_document = True
# autosectionlabel_maxdepth = 3  # avoid duplicate section labels for CLI examples

# -- Options for nbsphinx --------------------------------------------------
# env.docname will be e.g. examples/l7_composite.ipynb.  The `../` is to
# reference it from itself. preferable to link to actual version of the file
# at the time of the doc build, than a hyperlink to github.
# see https://github.com/aazuspan/wxee/blob/main/docs/conf.py for other examples
# TODO: update old geedim nbsphinx options
nbsphinx_prolog = """
.. note::

   This page was generated from a Jupyter notebook. To run and interact with it, 
   you can download it :download:`here <../{{ env.docname }}.ipynb>`.
"""
nbsphinx_widgets_path = ''
nbsphinx_requirejs_path = ''
