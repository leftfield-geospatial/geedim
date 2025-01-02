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
from functools import cached_property
from typing import Sequence

import rasterio as rio
from rasterio.windows import Window

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tile:
    """Description of a 3D image tile.

    Defines the tile band, row & column extents, and provides properties to assist accessing it
    in an Earth Engine image, Rasterio dataset or array.
    """

    # band, row & col extents of the tile in the source image (included in repr)
    band_start: int
    row_start: int
    col_start: int
    band_stop: int
    row_stop: int
    col_stop: int
    # source image geo-referencing transform (excluded from repr)
    image_transform: Sequence[float] = field(repr=False)

    @dataclass(frozen=True)
    class Slices:
        """3D slices that make dimensions explicit."""

        band: slice
        row: slice
        col: slice

    @cached_property
    def dimensions(self) -> tuple[int, int]:
        """Tile (width, height) dimensions in pixels."""
        return (
            self.col_stop - self.col_start,
            self.row_stop - self.row_start,
        )

    @cached_property
    def count(self) -> int:
        """Number of tile bands."""
        return self.band_stop - self.band_start

    @cached_property
    def indexes(self) -> range:
        """Tile bands as a range of source image band indexes in one-based / Rasterio convention."""
        return range(self.band_start + 1, self.band_stop + 1)

    @cached_property
    def window(self) -> Window:
        """Tile row & column region in the source image as a Rasterio window."""
        return Window(self.col_start, self.row_start, *self.dimensions)

    @cached_property
    def tile_transform(self) -> list[float]:
        """Tile geo-referencing transform."""
        transform = rio.Affine(*self.image_transform) * rio.Affine.translation(
            self.col_start, self.row_start
        )
        return transform[:6]

    @cached_property
    def slices(self) -> Slices:
        """Slices defining the 3D tile extent in a source image array."""
        return self.Slices(
            slice(self.band_start, self.band_stop),
            slice(self.row_start, self.row_stop),
            slice(self.col_start, self.col_stop),
        )
