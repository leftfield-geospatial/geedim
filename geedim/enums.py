# Copyright The Geedim Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from enum import Enum


class _StrChoiceEnum(str, Enum):
    """String value enumeration that can be used with a ``click.Choice()`` parameter
    type.
    """

    def __repr__(self):
        return self._value_

    def __str__(self):
        return self._value_

    @property
    def name(self):
        # override for click>=8.2.0 Choice options which match passed values to Enum
        # names
        return self._value_


class CompositeMethod(_StrChoiceEnum):
    """Enumeration for the compositing method i.e. the method for finding a
    composite pixel from the corresponding component image pixels.
    """

    q_mosaic = 'q-mosaic'
    """Use the unmasked pixel with the highest cloud distance (distance to nearest
    cloud). When more than one pixel has the same cloud distance, the first one is
    used.
    """

    mosaic = 'mosaic'
    """Use the first unmasked pixel."""

    medoid = 'medoid'
    """Medoid of the unmasked pixels i.e. the pixel from the image with the minimum
    sum of spectral distances to the rest of the images.  Where more than one pixel
    has the same summed distance, the first one is used.  See
    https://www.mdpi.com/2072-4292/5/12/6481 for detail.
    """

    median = 'median'
    """Median of the unmasked pixels."""

    mode = 'mode'
    """Mode of the unmasked pixels."""

    mean = 'mean'
    """Mean of the unmasked pixels."""


class CloudMaskMethod(_StrChoiceEnum):
    """Enumeration for the Sentinel-2 cloud masking method."""

    cloud_prob = 'cloud-prob'
    """Threshold the Sentinel-2 Cloud Probability.

    .. deprecated:: 1.9.0
        Please use the :attr:`cloud_score` method.
    """

    qa = 'qa'
    """Bit mask the ``QA60`` quality assessment band.

    .. deprecated:: 1.9.0
        Please use the :attr:`cloud_score` method.
    """

    cloud_score = 'cloud-score'
    """Threshold the `Sentinel-2 Cloud Score+
    <https://developers.google.com/earth-engine/datasets/catalog
    /GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED#description>`__.
    """


class CloudScoreBand(_StrChoiceEnum):
    """Enumeration for the `Sentinel-2 Cloud Score+
    <https://developers.google.com/earth-engine/datasets/catalog
    /GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED#description>`__ band to use with the
    :attr:`~CloudMaskMethod.cloud_score` cloud masking method.
    """

    cs = 'cs'
    """Pixel quality score based on spectral distance from a (theoretical) clear
    reference.
    """

    cs_cdf = 'cs_cdf'
    """Value of the cumulative distribution function of possible ``cs`` values for the
    estimated ``cs`` value.
    """


class ResamplingMethod(_StrChoiceEnum):
    """Enumeration for the resampling method."""

    near = 'near'
    """Nearest neighbour."""

    bilinear = 'bilinear'
    """Bilinear."""

    bicubic = 'bicubic'
    """Bicubic."""

    average = 'average'
    """Average."""


class ExportType(_StrChoiceEnum):
    """Enumeration for the export type."""

    drive = 'drive'
    """Export to Google Drive."""

    asset = 'asset'
    """Export to Earth Engine asset."""

    cloud = 'cloud'
    """Export to Google Cloud Storage."""


class SpectralDistanceMetric(_StrChoiceEnum):
    """Enumeration for the spectral distance metric."""

    sam = 'sam'
    """Spectral angle mapper."""

    sid = 'sid'
    """Spectral information divergence."""

    sed = 'sed'
    """Squared euclidean distance."""

    emd = 'emd'
    """Earth movers distance."""


class SplitType(_StrChoiceEnum):
    """Enumeration for how an image collection is split when exporting."""

    bands = 'bands'
    """Split collection by band."""

    images = 'images'
    """Split collection by image."""


class Driver(_StrChoiceEnum):
    """Enumeration for the image file format."""

    gtiff = 'gtiff'
    """GeoTIFF."""

    cog = 'cog'
    """Cloud Optimised GeoTIFF."""
