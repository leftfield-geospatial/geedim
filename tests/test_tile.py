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

from __future__ import annotations

import itertools
import os
from collections.abc import Sequence
from dataclasses import dataclass

import ee
import numpy as np
import pytest
import rasterio as rio
import rasterio.io
from rasterio import Affine, RasterioIOError
from rasterio.windows import Window

from geedim.image import ImageAccessor
from geedim.tile import Tile, Tiler


@dataclass
class MockImageAccessor:
    """Mock of ImageAccessor that provides image properties used by Tiler.__init__() and
    Tiler._tiles().
    """

    crs: str = 'EPSG:3857'
    transform: tuple[float, ...] = (1, 0, 0, 0, -1, 0)
    shape: tuple[int, int] = (1, 1)
    count: int = 1
    dtype: str = 'uint8'
    size: int = 1


def test_tile():
    """Test Tile."""
    band_start, row_start, col_start = (10, 200, 300)
    band_stop, row_stop, col_stop = (20, 400, 600)
    im_tform = (10, 0, 10000, 0, -20, 20000)
    tile = Tile(
        band_start, row_start, col_start, band_stop, row_stop, col_stop, im_tform
    )
    assert tile.shape == (row_stop - row_start, col_stop - col_start)
    assert tile.count == (band_stop - band_start)
    assert tile.window == Window(col_start, row_start, *tile.shape[::-1])
    tile_tform = (Affine(*im_tform) * Affine.translation(col_start, row_start))[:6]
    assert tile.tile_transform == tile_tform
    assert tile.slices.col == slice(col_start, col_stop)
    assert tile.slices.row == slice(row_start, row_stop)
    assert tile.slices.band == slice(band_start, band_stop)


def test_tiler_init():
    """Test Tiler.__init__() parameters have the expected effects."""
    image = MockImageAccessor(shape=(1000, 1000), count=10, dtype='uint16')

    # max_tile_size
    max_tile_size = 1
    tiler = Tiler(image, max_tile_size=max_tile_size)
    tile_size = np.prod(tiler._tile_shape) * np.dtype(image.dtype).itemsize
    assert tile_size / 2**20 < max_tile_size

    # max_tile_dim
    max_tile_dim = 1
    tiler = Tiler(image, max_tile_dim=max_tile_dim)
    assert max(tiler._tile_shape[1:]) == max_tile_dim

    # max_tile_bands
    max_tile_bands = 1
    tiler = Tiler(image, max_tile_bands=max_tile_bands)
    assert tiler._tile_shape[0] == max_tile_bands

    # max_requests and max_cpus
    max_requests = max_cpus = 1
    tiler = Tiler(image, max_requests=max_requests)
    assert tiler._limit_requests._value == max_requests
    assert tiler._limit_cpus._value == max(os.cpu_count() - 1, 1)
    tiler = Tiler(image, max_cpus=max_cpus)
    assert tiler._limit_cpus._value == max_cpus


def test_tiler_init_error():
    """Test Tiler.__init__() raises an error when the image has a non-fixed
    projection.
    """
    with pytest.raises(ValueError, match='fixed'):
        _ = Tiler(MockImageAccessor(shape=None))


def test_tiler_init_warning():
    """Test Tiler.__init__() issues a warning when the image is large."""
    with pytest.warns(RuntimeWarning, match='export size'):
        _ = Tiler(MockImageAccessor(size=11e9))


def test_tiler_context():
    """Test Tiler context manager shuts down the executor on exit."""
    with Tiler(MockImageAccessor()) as tiler:
        assert tiler._executor._shutdown is False
    assert tiler._executor._shutdown is True


def test_tiler_get_tile_shape():
    """Test Tiler._get_tile_shape()."""
    # test tile size and shape with different image dimensions (that don't exceed
    # max_tile_dim or max_tile_bands), and with max_tile_size values that span the
    # size of a GeoTIFF tile
    dtype = 'float64'
    dtype_size = np.dtype(dtype).itemsize
    tiler = Tiler(MockImageAccessor())

    for mts, count, height, width in itertools.product(
        [1, 3, 5], range(1, 1002, 500), range(1, 2002, 500), range(1, 2002, 500)
    ):
        max_tile_size = mts * 2**20
        min_tile_shape = np.array([1, 512, 512])  # size of GeoTIFF tile
        if max_tile_size < min_tile_shape.prod() * dtype_size:
            min_tile_shape = np.array([1, 1, 1])

        # patch the tiler _im attribute rather than creating a new Tiler on each
        # iteration
        image = MockImageAccessor(shape=(height, width), count=count, dtype=dtype)
        tiler._im = image
        tile_shape = np.array(tiler._get_tile_shape(max_tile_size=mts))
        tile_size = tile_shape.prod() * dtype_size
        im_shape = np.array((image.count, *image.shape))

        # sanity tests on tile shape and size
        assert all(tile_shape >= 1)
        assert all(tile_shape <= im_shape)
        assert all(tile_shape[1:] <= Tiler._ee_max_tile_dim)
        assert tile_shape[0] <= Tiler._ee_max_tile_bands
        assert tile_size <= max_tile_size

        # test tile dimensions lie either on the image bounds, or are multiples of
        # min_tile_shape dimensions
        assert all(
            (tile_shape == im_shape)
            | (np.round(tile_shape / min_tile_shape) == tile_shape / min_tile_shape)
        )

        # if the image consists of >1 tile, test the tile size against a rough lower
        # bound
        if any(tile_shape < im_shape):
            assert tile_size > max_tile_size / 2

    # test tile size is halved for *int8 dtypes
    max_tile_size = 1
    for dtype in ['int8', 'uint8']:
        image = MockImageAccessor(shape=(1024, 1024), count=1024, dtype=dtype)
        tiler._im = image
        tile_shape = np.array(tiler._get_tile_shape(max_tile_size=max_tile_size))
        tile_size = tile_shape.prod() * np.dtype(image.dtype).itemsize
        assert tile_size <= (max_tile_size * 2**20) / 2


def test_tiler_tiles():
    """Test continuity and coverage of Tiler._tiles()."""

    def tile_union(tiles: Sequence[Tile]) -> Tile:
        """Return the union of the tiles."""
        start_args = [
            min(map(lambda t: getattr(t, attr), tiles))
            for attr in ['band_start', 'row_start', 'col_start']
        ]
        stop_args = [
            max(map(lambda t: getattr(t, attr), tiles))
            for attr in ['band_stop', 'row_stop', 'col_stop']
        ]
        return Tile(*start_args, *stop_args, tiles[0].image_transform)

    im_shape = (300, 400, 500)
    image = MockImageAccessor(shape=im_shape[1:], count=im_shape[0], dtype='uint16')
    tiler = Tiler(image, max_tile_size=1)

    # test tile continuity
    tiles = [*tiler._tiles()]
    prev_tile = tiles[0]
    for tile in tiles[1:]:
        assert (
            tile.band_start == prev_tile.band_stop
            or tile.row_start == prev_tile.row_stop
            or tile.col_start == prev_tile.col_stop
        )
        prev_tile = tile

    # test tile coverage
    acc_tile = tile_union(tiles)
    assert (acc_tile.band_start, acc_tile.row_start, acc_tile.col_start) == (0, 0, 0)
    assert (acc_tile.band_stop, acc_tile.row_stop, acc_tile.col_stop) == im_shape


def test_tile_map_tile_retries(
    prepared_image: ImageAccessor,
    prepared_image_array: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test Tiler._map_tile() retry behaviour."""

    class MockImage(ee.Image):
        """Mocked ee.Image."""

        errors = 0
        max_errors = 1

        def getDownloadURL(self, *args, **kwargs):
            """Raise an error on the first max_errors calls then return a URL as
            usual.
            """
            if MockImage.errors < MockImage.max_errors:
                MockImage.errors += 1
                raise ConnectionError('Mock error')
            else:
                return super().getDownloadURL(*args, **kwargs)

    monkeypatch.setattr(ee, 'Image', MockImage)
    array = np.ones(
        (*prepared_image.shape, prepared_image.count), dtype=prepared_image.dtype
    )

    def write_tile(tile: Tile, tile_array: np.ndarray):
        """Write tile_array into array."""
        tile_array = np.moveaxis(tile_array, 0, -1)
        array[tile.slices.row, tile.slices.col, tile.slices.band] = tile_array

    with Tiler(prepared_image) as tiler:
        # test a tile is downloaded correctly after max_errors retries
        with monkeypatch.context() as m:
            # set the default values for the Tiler._map_tile() max_retries and
            # backoff_factor kwargs
            m.setattr(Tiler._map_tile, '__defaults__', (MockImage.max_errors, 0))
            tiler.map_tiles(write_tile, masked=False)

        assert MockImage.errors == MockImage.max_errors
        mask = array != prepared_image.nodata
        assert (mask == ~prepared_image_array.mask).all()
        assert (array == prepared_image_array).all()

        # test an error is raised when a tile is not downloaded correctly within
        # max_errors retries
        MockImage.errors = 0
        with (
            monkeypatch.context() as m,
            pytest.raises(ConnectionError, match='Mock error'),
        ):
            # set the default values for the Tiler._map_tile() max_retries and
            # backoff_factor kwargs
            m.setattr(Tiler._map_tile, '__defaults__', (0, 0))
            tiler.map_tiles(write_tile, masked=False)


@pytest.mark.parametrize('masked', [False, True])
def test_tiler_map_tiles(
    prepared_image: ImageAccessor, prepared_image_array: np.ndarray, masked: bool
):
    """Test Tiler.map_tiles()."""

    def write_tile(tile: Tile, tile_array: np.ndarray):
        """Write tile_array into array."""
        tile_array = np.moveaxis(tile_array, 0, -1)
        array[tile.slices.row, tile.slices.col, tile.slices.band] = tile_array

    # choose max_tile_dim and max_tile_bands to have 2 tiles along each dimension,
    # and max_requests and max_cpus to reach their limits
    with Tiler(
        prepared_image,
        max_tile_dim=11,
        max_tile_bands=2,
        max_requests=4,
        max_cpus=4,
    ) as tiler:
        assert len([*tiler._tiles()]) == 8
        array_type = np.ma.ones if masked else np.ones
        array = array_type(
            (*prepared_image.shape, prepared_image.count),
            dtype=prepared_image.dtype,
        )
        tiler.map_tiles(write_tile, masked=masked)

        mask = ~array.mask if masked else array != prepared_image.nodata
        assert (mask == ~prepared_image_array.mask).all()
        assert (array == prepared_image_array).all()
