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

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import ee
import numpy as np
import pytest
import rasterio as rio
from click.testing import CliRunner
from rasterio.features import bounds
from rasterio.transform import from_bounds

from geedim import cli, enums
from geedim.collection import ImageCollectionAccessor
from tests.conftest import accessors_from_collections, transform_bounds


@pytest.fixture(scope='session')
def region_100ha_file(
    region_100ha: dict, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """GeoJSON polygon file."""
    out_file = tmp_path_factory.mktemp('data').joinpath('region_100ha.json')
    with open(out_file, 'w') as f:
        json.dump(region_100ha, f)
    return out_file


@pytest.fixture(scope='session')
def raster_file(region_100ha: dict, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Georeferenced raster file."""
    out_file = tmp_path_factory.mktemp('data').joinpath('raster.tif')
    shape = (300, 400)
    crs = 'EPSG:4326'
    transform = from_bounds(*bounds(region_100ha), *shape[::-1])
    array = np.ones((1, *shape), dtype='uint8')
    profile = dict(
        driver='GTiff',
        crs=crs,
        transform=transform,
        width=array.shape[2],
        height=array.shape[1],
        count=array.shape[0],
        dtype=array.dtype,
    )

    with rio.open(out_file, 'w', **profile) as ds:
        ds.write(array)

    return out_file


@pytest.fixture
def patch_filter(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Patched ImageCollectionAccessor.filter() for testing passed kwargs."""
    passed_kwargs = []

    def filter(self, **kwargs):
        """Mock filter() that stores passed kwargs."""
        passed_kwargs.append(kwargs)
        coll = ee.ImageCollection([])
        # cache mock EE info to avoid getInfo() via the search command
        coll.gd._info = {
            'id': 'MOCK-ID',
            'features': [
                {'properties': {'system:index': 'MOCK-INDEX-1', 'system:time_start': 0}}
            ],
        }
        return coll

    monkeypatch.setattr(ImageCollectionAccessor, 'filter', filter)
    return passed_kwargs


@pytest.fixture
def patch_prepare_export_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple, dict]]:
    """Patched _prepare_export_collection() for testing passed args."""
    passed_args = []

    def _prepare_export_collection(*args, **kwargs):
        """Mock _prepare_export_collection() that stores passed args."""
        passed_args.append((args, kwargs))
        return ee.ImageCollection([])

    monkeypatch.setattr(cli, '_prepare_export_collection', _prepare_export_collection)
    return passed_args


@pytest.fixture
def patch_to_geotiff(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple, dict]]:
    """Patched ImageCollectionAccessor.toGeoTIFF() for testing passed args."""
    passed_args = []

    def toGeoTIFF(self, *args, **kwargs):
        """Mock toGeoTIFF() that stores passed args."""
        passed_args.append((args, kwargs))

    monkeypatch.setattr(ImageCollectionAccessor, 'toGeoTIFF', toGeoTIFF)
    return passed_args


@pytest.fixture
def patch_to_google_cloud(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Patched ImageCollectionAccessor.toGoogleCloud() for testing passed kwargs."""
    passed_kwargs = []

    def toGoogleCloud(self, **kwargs):
        """Mock toGoogleCloud() that stores passed kwargs."""
        passed_kwargs.append(kwargs)

    monkeypatch.setattr(ImageCollectionAccessor, 'toGoogleCloud', toGoogleCloud)
    return passed_kwargs


@pytest.fixture
def patch_from_images(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Patched ImageCollectionAccessor.fromImages() for testing passed args."""
    passed_args = []

    def fromImages(*args) -> ee.Image:
        """Mock fromImages() that stores passed args."""
        passed_args.append(args)
        return ee.ImageCollection([])

    monkeypatch.setattr(ImageCollectionAccessor, 'fromImages', fromImages)
    return passed_args


@pytest.fixture
def patch_composite(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Patched ImageCollectionAccessor.composite() for testing passed kwargs."""
    passed_kwargs = []

    def composite(self, **kwargs) -> ee.Image:
        """Mock composite() that stores passed kwargs."""
        passed_kwargs.append(kwargs)
        return ee.Image('MOCK-ID/MOCK-COMP')

    monkeypatch.setattr(ImageCollectionAccessor, 'composite', composite)
    return passed_kwargs


@pytest.fixture
def prepare_export_collection_kwargs(
    region_100ha: dict[str, Any], raster_file: Path
) -> dict[str, Any]:
    """Dictionary of kwargs for _prepare_export_collection()."""
    # in normal use, some of _prepare_export_collection() kwargs are mutually
    # exclusive or redundant, but here all kwargs are included for testing conversion
    # & passing of the CLI options
    with rio.open(raster_file, 'r') as ds:
        like = dict(crs=ds.crs.to_wkt(), crs_transform=ds.transform[:6], shape=ds.shape)
    return dict(
        geometry=region_100ha,
        crs='EPSG:3857',
        buffer=500,
        like=like,
        mask=True,
        scale=100,
        crs_transform=(10, 0, 1000, 0, -10, 2000),
        shape=(300, 400),
        dtype='uint16',
        bands=('B1', 'B2'),
        resampling=enums.ResamplingMethod.bilinear,
        scale_offset=True,
    )


@pytest.fixture
def prepare_export_collection_cli_str(
    prepare_export_collection_kwargs: dict[str, Any], raster_file: Path
):
    """CLI options string for prepare_export_collection_kwargs."""
    kwargs = prepare_export_collection_kwargs
    bounds_str = ' '.join(map(str, bounds(kwargs['geometry'])))
    transform_str = ' '.join(map(str, kwargs['crs_transform']))
    bands_str = ' '.join([f'-bn {b}' for b in kwargs['bands']])
    cli_str = (
        f'-c {kwargs["crs"]} -b {bounds_str} -buf {kwargs["buffer"]} -s '
        f'{kwargs["scale"]} -ct {transform_str} -sh {kwargs["shape"][0]} '
        f'{kwargs["shape"][1]} -l {raster_file} -dt {kwargs["dtype"]} {bands_str} -rs '
        f'{kwargs["resampling"]}'
    )
    cli_str += ' ' + ('-m' if kwargs['mask'] else '-nm')
    cli_str += ' ' + ('-so' if kwargs['scale_offset'] else '-nso')
    return cli_str


def test_crs_cb(raster_file: Path, tmp_path: Path):
    """Test the --crs callback."""
    # create CRS text file
    crs = 'EPSG:4326'
    crs_file = tmp_path.joinpath('crs.txt')
    with open(crs_file, 'w') as f:
        f.write(crs)

    ctx = click.Context(cli.download)
    value = cli._crs_cb(ctx, None, crs)
    assert value == crs
    value = cli._crs_cb(ctx, None, raster_file)
    assert value == rio.CRS.from_string(crs).to_wkt()
    value = cli._crs_cb(ctx, None, crs_file)
    assert value == crs


def test_bbox_cb(region_100ha: dict[str, Any]):
    """Test the --bbox callback."""
    ctx = click.Context(cli.search)
    ctx.obj = {}
    bounds_ = bounds(region_100ha)
    cli._bbox_cb(ctx, None, bounds_)
    assert ctx.params['geometry'] == region_100ha

    # None
    ctx = click.Context(cli.search)
    ctx.obj = {}
    cli._bbox_cb(ctx, None, None)
    assert ctx.params['geometry'] is None


def test_region_cb(
    region_100ha_file: Path, raster_file: Path, region_100ha: dict[str, Any]
):
    """Test the --region callback."""
    # raster
    ctx = click.Context(cli.search)
    ctx.obj = {}
    cli._region_cb(ctx, None, str(raster_file))
    assert bounds(ctx.params['geometry']) == pytest.approx(
        bounds(region_100ha), abs=1e-6
    )

    # geojson
    ctx = click.Context(cli.search)
    ctx.obj = {}
    cli._region_cb(ctx, None, str(region_100ha_file))
    assert ctx.params['geometry'] == region_100ha

    # piping
    cli._region_cb(ctx, None, '-')
    assert ctx.params['geometry'] == region_100ha
    ctx.obj = {}
    with pytest.raises(click.BadParameter, match='No piped bounds'):
        _ = cli._region_cb(ctx, None, '-')

    # None
    ctx = click.Context(cli.search)
    ctx.obj = {}
    cli._region_cb(ctx, None, None)
    assert ctx.params['geometry'] is None


def test_like_cb(raster_file: Path):
    """Test the --like callback."""
    ctx = click.Context(cli.download)
    ctx.obj = {}
    value = cli._like_cb(ctx, None, raster_file)

    with rio.open(raster_file, 'r') as ds:
        exp_value = dict(
            crs=ds.crs.to_wkt(), crs_transform=ds.transform[:6], shape=ds.shape
        )
    assert value == exp_value


def test_search(
    patch_filter: list[dict],
    region_100ha: dict[str, Any],
    runner: CliRunner,
    tmp_path: Path,
):
    """Test the search command."""
    start_date = '2023-01-01'
    end_date = '2022-01-02'
    bounds_str = ' '.join(map(str, bounds(region_100ha)))
    buffer = 500
    fill_portion = 90
    cloudless_portion = 50
    custom_filter = 'CLOUD_COVER<50'
    add_props = 'CLOUD_COVER'
    res_file = tmp_path.joinpath('results.json')

    cli_str = (
        f'search -c MOCK-ID -s {start_date} -e {end_date} -b {bounds_str} -buf '
        f'{buffer} -fp {fill_portion} -cp {cloudless_portion} -cf {custom_filter} -ap '
        f'{add_props} -op {res_file}'
    )
    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output

    # test propertiesTable is printed
    assert 'MOCK-INDEX-1' in res.output

    # test filter options are passed
    filter_kwargs = patch_filter.pop(-1)
    assert filter_kwargs['start_date'] == datetime.fromisoformat(start_date)
    assert filter_kwargs['end_date'] == datetime.fromisoformat(end_date)
    assert filter_kwargs['region'] == ee.Geometry(region_100ha).buffer(buffer)
    assert filter_kwargs['fill_portion'] == fill_portion
    assert filter_kwargs['cloudless_portion'] == cloudless_portion
    assert filter_kwargs['custom_filter'] == custom_filter

    # test --add-props is added to table
    assert add_props in res.output

    # test results file was written
    assert res_file.exists()


def test_config_search_pipe(patch_filter: list[dict], runner: CliRunner):
    """Test search uses cloud mask configuration piped from config."""
    cli_str = 'config -nms search -c MOCK-ID'
    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output
    passed_kwargs = patch_filter.pop(-1)
    assert passed_kwargs['mask_shadows'] is False


def test_prepare_export_collection_buffer(
    l9_sr_image_id: str, region_100ha: dict[str, Any]
):
    """Test _prepare_export_collection() buffer parameter."""
    images = [ee.Image(l9_sr_image_id)]
    crs = 'EPSG:3857'
    buffer = 500
    buffer_bounds = (
        ee.Geometry(region_100ha).buffer(buffer).bounds(maxError=1, proj=crs).getInfo()
    )
    buffer_bounds = bounds(buffer_bounds)

    coll = cli._prepare_export_collection(
        images, geometry=region_100ha, buffer=buffer, crs=crs
    )
    assert coll.gd._first.crs == crs
    assert bounds(coll.gd._first.geometry) == pytest.approx(buffer_bounds, abs=100)


def test_prepare_export_collection_like(
    l9_sr_image_id: str, region_100ha: dict[str, Any]
):
    """Test _prepare_export_collection() like parameter."""
    images = [ee.Image(l9_sr_image_id)]
    crs = 'EPSG:4326'
    shape = (300, 400)
    transform = from_bounds(*bounds(region_100ha), *shape[::-1])[:6]

    coll = cli._prepare_export_collection(
        images, like=dict(crs=crs, crs_transform=transform, shape=shape)
    )

    assert coll.gd._first.crs == crs
    assert coll.gd._first.transform == transform
    assert coll.gd._first.shape == shape


def test_prepare_export_collection_cloud_kwargs(l9_sr_image_id: str):
    """Test _prepare_export_collection() cloud_kwargs parameter."""
    # adapted from test_collection.test_add_mask_bands()
    images = [ee.Image(l9_sr_image_id)]
    coll = cli._prepare_export_collection(images, cloud_kwargs=dict(mask_shadows=False))

    assert set(coll.gd._first.bandNames).issuperset(
        ['CLOUDLESS_MASK', 'CLOUD_DIST', 'FILL_MASK']
    )
    assert 'SHADOW_MASK' not in coll.gd._first.bandNames


def test_prepare_export_collection_mask(
    l9_sr_image_id: str, region_100ha: dict[str, Any]
):
    """Test _prepare_export_collection() mask parameter."""
    # adapted from test_collection.test_mask_clouds()
    images = [ee.Image(l9_sr_image_id)]
    colls = [
        cli._prepare_export_collection(images, mask=mask) for mask in [False, True]
    ]
    colls = accessors_from_collections(colls)

    # test mask bands always added
    for coll in colls:
        assert set(coll._first.bandNames).issuperset(
            ['CLOUDLESS_MASK', 'CLOUD_DIST', 'FILL_MASK']
        )

    def aggregate_mask_sum(image: ee.Image, sums: ee.List) -> ee.List:
        """Add the sum of the image masks to the sums list."""
        sum_ = (
            image.mask()
            .reduceRegion('sum', geometry=region_100ha)
            .values()
            .reduce('mean')
        )
        return ee.List(sums).add(sum_)

    mask_sums = [
        coll._ee_coll.iterate(aggregate_mask_sum, ee.List([])) for coll in colls
    ]
    mask_sums = ee.List(mask_sums).getInfo()

    assert len(mask_sums[0]) == len(mask_sums[1]) == len(images)
    for unmasked_sum, masked_sum in zip(*mask_sums, strict=True):
        assert unmasked_sum > masked_sum


def test_prepare_export_collection_other(
    l9_sr_image_id: str, region_100ha: dict[str, Any]
):
    """Test other _prepare_export_collection() kwargs not tested above."""
    images = [ee.Image(l9_sr_image_id)]
    crs = 'EPSG:3857'
    scale = 60
    dtype = 'float32'
    bands = ['SR_B4', 'SR_B3', 'SR_B2']

    coll = cli._prepare_export_collection(
        images,
        geometry=region_100ha,
        crs=crs,
        scale=scale,
        dtype=dtype,
        bands=bands,
        scale_offset=True,
    )
    first = coll.gd._first
    max_band = 'SR_B3'
    max = (
        coll.first().select(max_band).reduceRegion(reducer='max', geometry=region_100ha)
    )
    max = max.getInfo()[max_band]

    assert first.crs == crs
    region_bounds = transform_bounds(region_100ha, crs)
    image_bounds = transform_bounds(first.geometry, crs)
    assert image_bounds == pytest.approx(region_bounds, abs=scale)
    assert first.scale == scale
    assert first.dtype == dtype
    assert first.bandNames == bands
    assert 0 < max < 1.5  # test scale_offset


def test_download(
    patch_prepare_export_collection: list[tuple[tuple, dict]],
    patch_to_geotiff: list[tuple[tuple, dict]],
    prepare_export_collection_kwargs: dict[str, Any],
    prepare_export_collection_cli_str: str,
    runner: CliRunner,
    tmp_path: Path,
):
    """Test the download command."""
    # toGeoTIFF() kwargs to test against
    tg_kwargs = dict(
        split=enums.SplitType.bands,
        nodata=False,
        driver=enums.Driver.cog,
        max_tile_size=32,
        max_tile_dim=5000,
        max_requests=16,
        max_cpus=1,
        overwrite=True,
    )

    # form a CLI string containing all options
    ee_id = 'MOCK-ID/MOCK-INDEX'
    cli_str = f'download -i {ee_id} ' + prepare_export_collection_cli_str
    cli_str += (
        f' -sp {tg_kwargs["split"]} -nn -dv {tg_kwargs["driver"]} '
        f'-mts {tg_kwargs["max_tile_size"]} -mtd {tg_kwargs["max_tile_dim"]} '
        f'-mr {tg_kwargs["max_requests"]} -mc {tg_kwargs["max_cpus"]} -o -dd {tmp_path}'
    )

    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output

    # test _prepare_export_collection() args
    passed_args, passed_kwargs = patch_prepare_export_collection.pop(-1)
    assert passed_args[0] == [ee.Image(ee_id)]
    assert all(
        [passed_kwargs[k] == v for k, v in prepare_export_collection_kwargs.items()]
    )

    # test toGeoTIFF() args
    passed_args, passed_kwargs = patch_to_geotiff.pop(-1)
    assert Path(passed_args[0].path) == tmp_path
    assert passed_kwargs == tg_kwargs


def test_config_download_pipe(
    patch_prepare_export_collection: list[tuple[tuple, dict]],
    patch_to_geotiff: list[tuple[tuple, dict]],
    runner: CliRunner,
):
    """Test the download command uses cloud mask configuration piped from config."""
    cli_str = 'config -nms download -i MOCK-ID/MOCK-INDEX'
    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output
    _, passed_kwargs = patch_prepare_export_collection.pop(-1)
    assert passed_kwargs['cloud_kwargs'] == dict(mask_shadows=False)


def test_search_download_pipe(
    patch_filter: list[dict],
    patch_prepare_export_collection: list[tuple[tuple, dict]],
    patch_to_geotiff: list[tuple[tuple, dict]],
    runner: CliRunner,
):
    """Test the download command adds --id images to images piped from search."""
    cli_str = 'search -c MOCK-ID download -i MOCK-ID/MOCK-INDEX-2'
    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output
    exp_images = [ee.Image('MOCK-ID/MOCK-INDEX-1'), ee.Image('MOCK-ID/MOCK-INDEX-2')]
    passed_args, _ = patch_prepare_export_collection.pop(-1)
    assert set(passed_args[0]) == set(exp_images)


def test_download_download_pipe(
    patch_prepare_export_collection: list[tuple[tuple, dict]],
    patch_to_geotiff: list[tuple[tuple, dict]],
    runner: CliRunner,
):
    """Test the download command pipes input images to downstream commands."""
    cli_str = 'download -i MOCK-ID/MOCK-INDEX download'
    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output
    # there should be two sets of args - test the last
    passed_args, _ = patch_prepare_export_collection.pop(1)
    assert passed_args[0] == [ee.Image('MOCK-ID/MOCK-INDEX')]


def test_export(
    patch_prepare_export_collection: list[tuple[tuple, dict]],
    patch_to_google_cloud: list[tuple[tuple, dict]],
    prepare_export_collection_kwargs: dict[str, Any],
    prepare_export_collection_cli_str: str,
    runner: CliRunner,
):
    """Test the export command."""
    # toGoogleCloud() kwargs to test against
    tg_kwargs = dict(
        type=enums.ExportType.cloud,
        folder='test',
        split=enums.SplitType.bands,
        wait=False,
    )

    # form a CLI string containing all options
    ee_id = 'MOCK-ID/MOCK-INDEX'
    cli_str = f'export -i {ee_id} ' + prepare_export_collection_cli_str
    cli_str += (
        f' -t {tg_kwargs["type"]} -f {tg_kwargs["folder"]} -sp {tg_kwargs["split"]} -nw'
    )

    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output

    # test _prepare_export_collection() args
    passed_args, passed_kwargs = patch_prepare_export_collection.pop(-1)
    assert passed_args[0] == [ee.Image(ee_id)]
    assert all(
        [passed_kwargs[k] == v for k, v in prepare_export_collection_kwargs.items()]
    )

    # test toGoogleCloud() kwargs
    passed_kwargs = patch_to_google_cloud.pop(-1)
    assert passed_kwargs == tg_kwargs


def test_config_export_pipe(
    patch_prepare_export_collection: list[tuple[tuple, dict]],
    patch_to_google_cloud: list[tuple[tuple, dict]],
    runner: CliRunner,
):
    """Test the export command uses cloud mask configuration piped from config."""
    cli_str = 'config -nms export -i MOCK-ID/MOCK-INDEX'
    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output
    _, passed_kwargs = patch_prepare_export_collection.pop(-1)
    assert passed_kwargs['cloud_kwargs'] == dict(mask_shadows=False)


def test_search_export_pipe(
    patch_filter: list[dict],
    patch_prepare_export_collection: list[tuple[tuple, dict]],
    patch_to_geotiff: list[tuple[tuple, dict]],
    runner: CliRunner,
):
    """Test the export command adds --id images to images piped from search."""
    cli_str = 'search -c MOCK-ID export -i MOCK-ID/MOCK-INDEX-2'
    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output
    exp_images = [ee.Image('MOCK-ID/MOCK-INDEX-1'), ee.Image('MOCK-ID/MOCK-INDEX-2')]
    passed_args, _ = patch_prepare_export_collection.pop(-1)
    assert set(passed_args[0]) == set(exp_images)


def test_export_export_pipe(
    patch_prepare_export_collection: list[tuple[tuple, dict]],
    patch_to_google_cloud: list[tuple[tuple, dict]],
    runner: CliRunner,
):
    """Test the export command pipes input images to downstream commands."""
    cli_str = 'export -i MOCK-ID/MOCK-INDEX export'
    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output
    # there should be two sets of args - test the last
    passed_args, _ = patch_prepare_export_collection.pop(1)
    assert passed_args[0] == [ee.Image('MOCK-ID/MOCK-INDEX')]


def test_composite(
    patch_from_images: list[tuple],
    patch_composite: list[dict],
    region_100ha: dict[str, Any],
    runner: CliRunner,
):
    """Test the composite command."""
    ee_id = 'MOCK-ID/MOCK-INDEX'
    method = enums.CompositeMethod.median
    resampling = enums.ResamplingMethod.bilinear
    bounds_str = ' '.join(map(str, bounds(region_100ha)))
    date = '2023-01-01'
    cli_str = (
        f'composite -i {ee_id} -cm {method} -nm -rs {resampling} -b '
        f'{bounds_str} -d {date}'
    )

    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output

    # test fromImages() args
    passed_args = patch_from_images.pop(-1)
    assert passed_args[0] == [ee.Image(ee_id)]

    # test composite() kwargs
    passed_kwargs = patch_composite.pop(-1)
    assert passed_kwargs['method'] == method
    assert passed_kwargs['mask'] is False
    assert passed_kwargs['resampling'] == resampling
    assert passed_kwargs['region'] == region_100ha
    assert passed_kwargs['date'] == datetime.fromisoformat(date)


def test_config_composite_pipe(
    patch_from_images: list[tuple],
    patch_composite: list[dict],
    runner: CliRunner,
):
    """Test the composite command uses cloud mask configuration piped from config."""
    cli_str = 'config -nms composite -i MOCK-ID/MOCK-INDEX'
    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output
    passed_kwargs = patch_composite.pop(-1)
    assert passed_kwargs['mask_shadows'] is False


def test_search_composite_pipe(
    patch_filter: list[dict],
    patch_from_images: list[tuple],
    patch_composite: list[dict],
    runner: CliRunner,
):
    """Test the composite command adds --id images to images piped from search."""
    cli_str = 'search -c MOCK-ID composite -i MOCK-ID/MOCK-INDEX-2'
    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output
    comp_images = [ee.Image('MOCK-ID/MOCK-INDEX-1'), ee.Image('MOCK-ID/MOCK-INDEX-2')]
    passed_args = patch_from_images.pop(-1)
    assert set(passed_args[0]) == set(comp_images)


def test_composite_download_pipe(
    patch_from_images: list[tuple],
    patch_composite: list[dict],
    patch_prepare_export_collection: list[tuple[tuple, dict]],
    patch_to_geotiff: list[dict],
    runner: CliRunner,
):
    """Test the composite command pipes the composite image to downstream commands."""
    cli_str = 'composite -i MOCK-ID/MOCK-INDEX download'
    res = runner.invoke(cli.cli, cli_str.split())
    assert res.exit_code == 0, res.output
    passed_args, _ = patch_prepare_export_collection.pop(-1)
    assert passed_args[0] == [ee.Image('MOCK-ID/MOCK-COMP')]
