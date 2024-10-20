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


class GeedimError(Exception):
    """Base exception class."""


class UnfilteredError(GeedimError):
    """Raised when attempting to retrieve the properties of an unfiltered image collection."""


class InputImageError(GeedimError):
    """Raised when there is a problem with the images making up a collection."""


class TileError(GeedimError):
    """Raised when there is a problem downloading an image tile."""


class GeedimWarning(UserWarning):
    """Base warning class."""
