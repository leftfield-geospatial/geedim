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

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import rasterio as rio
from rasterio.windows import Window

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """Tile representation."""

    col_off: int
    row_off: int
    width: int
    height: int
    image_transform: Sequence[float] = field(repr=False)

    shape: tuple[int, int] = field(init=False, repr=False)
    window: Window = field(init=False, repr=False)
    tile_transform: Sequence[float] = field(init=False, repr=False)

    def __post_init__(self):
        self.shape = (self.height, self.width)
        self.window = Window(self.col_off, self.row_off, self.width, self.height)
        transform = rio.Affine(*self.image_transform) * rio.Affine.translation(
            self.col_off, self.row_off
        )
        self.tile_transform = transform[:6]
