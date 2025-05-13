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

from enum import Enum

class _StrChoiceEnum(str, Enum):
    """String value enumeration class that can be used with a ``click.Choice()`` parameter type."""

    def __repr__(self):
        return self._value_

    def __str__(self):
        return self._value_

    @property
    def name(self):
        # override for click>=8.2.0 Choice options which match passed values to Enum names
        return self._value_


class CompositeMethod(_StrChoiceEnum):
    """
    Enumeration for the compositing method, i.e. the method for finding a composite pixel from the stack of
    corresponding input image pixels.
    """

    q_mosaic = 'q-mosaic'
    """ 
    Use the unmasked pixel with the highest cloud distance (distance to nearest cloud). Where more than one pixel has 
    the same cloud distance, the first one in the stack is selected.     
    """

    mosaic = 'mosaic'
    """ Use the first unmasked pixel in the stack. """

    medoid = 'medoid'
    """
    Use the medoid of the unmasked pixels.  This is the pixel from the image having the minimum sum of spectral 
    distances to the rest of the images. 
    Maintains the original relationship between bands. See https://www.mdpi.com/2072-4292/5/12/6481 for detail.
    """

    median = 'median'
    """ Use the median of the unmasked pixels. """

    mode = 'mode'
    """ Use the mode of the unmasked pixels. """

    mean = 'mean'
    """ Use the mean of the unmasked pixels. """

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_


class CloudMaskMethod(_StrChoiceEnum):
    """Enumeration for the Sentinel-2 cloud masking method."""

    cloud_prob = 'cloud-prob'
    """Threshold the Sentinel-2 Cloud Probability."""

    qa = 'qa'
    """Bit mask the `QA60` quality assessment band."""

    cloud_score = 'cloud-score'
    """Threshold the Sentinel-2 Cloud Score+."""

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_


class CloudScoreBand(_StrChoiceEnum):
    """Enumeration for the Sentinel-2 Cloud Score+ band used with the :attr:`~CloudMaskMethod.cloud_score` cloud
    masking method.
    """

    cs = 'cs'
    """Pixel quality score based on spectral distance from a (theoretical) clear reference."""

    cs_cdf = 'cs_cdf'
    """Value of the cumulative distribution function of possible cs values for the estimated cs value."""

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_


class ResamplingMethod(_StrChoiceEnum):
    """Enumeration for the resampling method."""

    near = 'near'
    """ Nearest neighbour. """

    bilinear = 'bilinear'
    """ Bilinear. """

    bicubic = 'bicubic'
    """ Bicubic. """

    average = 'average'
    """ Average (recommended for downsampling). """

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_


class ExportType(_StrChoiceEnum):
    """Enumeration for the export type."""

    drive = 'drive'
    """ Export to Google Drive. """

    asset = 'asset'
    """ Export to an Earth Engine asset. """

    cloud = 'cloud'
    """ Export to Google Cloud Storage. """

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_


class SpectralDistanceMetric(_StrChoiceEnum):
    """Enumeration for the spectral distance metric."""

    sam = 'sam'
    """ Spectral angle mapper. """

    sid = 'sid'
    """ Spectral information divergence. """

    sed = 'sed'
    """ Squared euclidean distance. """

    emd = 'emd'
    """ Earth movers distance. """

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_
