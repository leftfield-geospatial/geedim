# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
from geedim.version import __version__

project = 'Geedim'
copyright = 'Geedim contributors'
author = 'Leftfield Geospatial'
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',
    'sphinx_click',
    'sphinx_copybutton',
    'jupyter_sphinx',
]

# -- Options for source files ------------------------------------------------
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['jupyter-sphinx.css', 'custom.css']
html_title = f'Geedim {__version__}'

html_theme_options = {
    # copied from https://github.com/pradyunsg/furo/blob/main/docs/conf.py
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/leftfield-geospatial/geedim',
            'html': """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            'class': '',
        }
    ],
}

# -- Options for pygments -----------------------------------------------------
highlight_language = 'none'

# -- Options for intersphinx ---------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'fsspec': ('https://filesystem-spec.readthedocs.io/en/latest/', None),
}

# -- Options for autodoc -----------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autoclass_content = 'both'

# -- Options for autosectionlabel ---------------------------------------------
autosectionlabel_prefix_document = True  # make sure the target is unique
autosectionlabel_maxdepth = 2  # avoid duplicate section labels

# -- Options for linkcheck ----------------------------------------------------
linkcheck_ignore = ['https://www.mdpi.com/*', '../reference/*']

# -- Options for jupyter-sphinx -----------------------------------------------
jupyter_execute_kwargs = {
    'timeout': 300,
    'allow_errors': True,
    # prevents inclusion of jupyter widget style sheet
    'store_widget_state': False,
}
