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
long_description = (this_directory / "README.md").read_text()

sys.path[0:0] = ['geedim']
from version import __version__

setup(
    name="geedim",
    version=__version__,
    description="Google Earth Engine image download",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Dugal Harris",
    author_email="dugalh@gmail.com",
    url="https://github.com/dugalh/geedim",
    license="Apache-2.0",
    packages=find_packages(exclude=['tests', 'data'], include=['geedim']),
    install_requires=["pandas>=1.1, <2", "earthengine-api>=0.1.2, <1", "click>=8, <9", "requests>=2.2, <3",
                      "numpy>=1"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["geedim=geedim.cli:cli"]},
)
