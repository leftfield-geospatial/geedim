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
from setuptools import setup, find_packages

# To install local development version use:
#    pip install -e .

setup(
    name='geedim',
    version='0.1.0',
    description='Google Earth Engine image download',
    author='Dugal Harris',
    author_email='dugalh@gmail.com',
    url='https://github.com/dugalh/geedim/blob/main/setup-py',
    license='Apache-2.0',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'rasterio>=1.2',
        'pandas>=1.3',
        'earthengine-api >= 0.1.2'
    ],
)
