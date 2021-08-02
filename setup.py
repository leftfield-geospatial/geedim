"""
    Homonim: Radiometric homogenisation of aerial and satellite imagery
    Copyright (C) 2021 Dugal Harris
    Email: dugalh@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from setuptools import setup, find_packages

# To install local development version use:
#    pip install -e .

setup(
    name='homonim',
    version='0.1.0',
    description='Radiometric homogenisation of aerial and satellite imagery',
    author='Dugal Harris',
    author_email='dugalh@gmail.com',
    url='https://github.com/dugalh/homonim/blob/main/setup-py',
    license='AGPLv3',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.2',
        'rasterio>=1.1',
    ],
)
