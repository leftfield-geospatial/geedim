"""
   Copyright 2021 Dugal Harris - dugalh@gmail.com

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import sys
from pathlib import Path

from setuptools import setup, find_packages
"""
 Build and upload to testpypi:
     conda install -c conda-forge build twine
     python -m build
     python -m twine upload --repository testpypi dist/*

 Install from testpypi:
    python -m pip install --extra-index-url https://test.pypi.org/simple/ geedim

 Install local development version:
    pip install -e .
"""

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.rst').read_text()
sys.path[0:0] = ['geedim']
from version import __version__

setup(
    name='geedim',
    version=__version__,
    description='Search, composite and download Google Earth Engine imagery.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Dugal Harris',
    author_email='dugalh@gmail.com',
    url='https://github.com/dugalh/geedim',
    license='Apache-2.0',
    packages=find_packages(include=['geedim']),
    package_data={'geedim': ['data/ee_stac_urls.json']},
    install_requires=[
        'numpy>=1.19',
        'rasterio>=1.1',
        'click>=8',
        'tqdm>=4.6',
        'earthengine-api>=0.1.2',
        'requests>=2.2',
        'tabulate>=0.8',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    keywords=[
        'earth engine', 'satellite imagery', 'search', 'download', 'composite', 'cloud', 'shadow',
    ],
    entry_points={'console_scripts': ['geedim=geedim.cli:cli']},
    project_urls={
        'Documentation': 'https://geedim.readthedocs.io',
        'Source': 'https://github.com/dugalh/geedim',
    },
)  # yapf: disable
