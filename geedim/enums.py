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


class CompositeMethod(str, Enum):
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
    Use the medoid of the unmasked pixels.  This is the pixel from the image having the minimum summed difference (
    across bands) from the median of the image stack. Maintains the original relationship between bands. See 
    https://www.mdpi.com/2072-4292/5/12/6481 for detail.    
    """

    median = 'median'
    """ Use the median of the unmasked pixels. """

    mode = 'mode'
    """ Use the mode of the unmasked pixels. """

    mean = 'mean'
    """ Use the mean of the unmasked pixels. """


class CloudMaskMethod(str, Enum):
    """ Enumeration for the Sentinel-2 cloud masking method. """
    cloud_prob = 'cloud-prob'
    """ Threshold the corresponding image from the Sentinel-2 cloud probability collection. """

    qa = 'qa'
    """ Use the `QA60` quality assessment band. """


class ResamplingMethod(str, Enum):
    """ Enumeration for the resampling method. """
    near = 'near'
    """ Nearest neighbour. """

    bilinear = 'bilinear'
    """ Bilinear. """

    bicubic = 'bicubic'
    """ Bicubic. """

    average = 'average'
    """ Average (recommended for downsampling). """
