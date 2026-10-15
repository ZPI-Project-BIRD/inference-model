#!/usr/bin/env python3
"""
Build an aerial-building segmentation dataset from Geoportal Poland.

INPUT:
    Two WGS84 corners:
        lat1, lon1
        lat2, lon2

OUTPUT:
    <output>/
        imagery/
            tile_000_000.tif
            ...
        buildings/
            OT_BUBD_A_....gpkg
        masks/
            tile_000_000_mask.png
            ...
            mask_summary.csv
        previews/
            ...
        _temp/
            downloaded BDOT packages
            extracted OT_BUBD_A files

PIPELINE:
    1. Convert WGS84 -> EPSG:2180.
    2. Find every Polish powiat intersecting the AOI using the official
       Geoportal PRG Powiaty FeatureServer.
    3. Download the corresponding official BDOT10k county GML package(s).
    4. Extract OT_BUBD_A building polygons.
    5. Download 10 cm orthophotos from the official Geoportal High Resolution WCS.
    6. Rasterize buildings into masks aligned exactly to each TIFF.

The script is resumable:
    - existing valid imagery TIFFs are skipped
    - existing masks are skipped
    - downloaded BDOT ZIPs are reused

Dependencies:
    pip install requests rasterio geopandas shapely pyproj pillow pandas
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import box


from common import (
    wgs84_to_2180_bbox, bdot_package_url, extract_building_gml,
    load_buildings, save_buildings, create_masks,
)

WCS_URL = (
    "https://mapy.geoportal.gov.pl/"
    "wss/service/PZGIK/ORTO/WCS/HighResolution"
)

WCS_COVERAGE = "Orthoimagery_High_Resolution"

POWIAT_URL = (
    "https://mapy.geoportal.gov.pl/"
    "wss/ims/maps/PRG_gugik_wyszukiwarka/"
    "FeatureServer/1/query"
)

BDOT_BASE_URL = (
    "https://opendata.geoportal.gov.pl/"
    "bdot10k/schemat2021"
)

BDOT_WFS_URL = (
    "https://mapy.geoportal.gov.pl/"
    "wss/service/PZGIK/BDOT/WFS/PobieranieBDOT10k"
)

CRS = "EPSG:2180"
WGS84 = "EPSG:4326"

RESOLUTION = 0.10
TILE_SIZE = 2048

BUILDING_CLASS = "OT_BUBD_A"
ALL_TOUCHED = True
NUM_PREVIEWS = 20

TIMEOUT_CONNECT = 30
TIMEOUT_DOWNLOAD = 300
MAX_RETRIES = 8

def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 "
            "(Python Geoportal building-dataset downloader)"
        ),
        "Connection": "close",
    })
    return session


def request_with_retries(
    method: str,
    url: str,
    *,
    params=None,
    stream: bool = False,
    timeout=None,
):
    """
    Generic retry helper for flaky Geoportal connections.
    """
    if timeout is None:
        timeout = (TIMEOUT_CONNECT, TIMEOUT_DOWNLOAD)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session = make_session()

            response = session.request(
                method,
                url,
                params=params,
                stream=stream,
                timeout=timeout,
            )

            if not stream:
                response.content

            session.close()

            response.raise_for_status()
            return response

        except Exception as exc:
            last_error = exc

            print(
                f"  Request failed "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt == MAX_RETRIES:
                break

            delay = min(60.0, 2 ** (attempt - 1))
            delay += random.uniform(0, 2)

            print(f"  Retrying in {delay:.1f}s...")
            time.sleep(delay)

    raise RuntimeError(
        f"Request failed after {MAX_RETRIES} attempts: {url}"
    ) from last_error

def find_intersecting_powiaty(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
):
    """
    Find all powiats whose official PRG geometry intersects
    the AOI.

    This intentionally uses the official PRG FeatureServer,
    rather than the BDOT10k WFS index, because the latter's
    WFS 1.1.0 BBOX/axis-order behavior can be misleading.
    """

    print()
    print("=" * 70)
    print("IDENTIFYING POWIATS")
    print("=" * 70)

    params = {
        "f": "json",
        "where": "1=1",
        "geometry": json.dumps({
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "spatialReference": {
                "wkid": 2180
            },
        }),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 2180,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "teryt,nazwa",
        "returnGeometry": "false",
    }

    response = request_with_retries(
        "GET",
        POWIAT_URL,
        params=params,
        timeout=(TIMEOUT_CONNECT, 120),
    )

    data = response.json()

    if "error" in data:
        raise RuntimeError(
            f"Powiat service returned an error:\n"
            f"{data['error']}"
        )

    features = data.get("features", [])

    if not features:
        raise RuntimeError(
            "No powiat intersects the requested AOI."
        )

    powiaty = []

    for feature in features:
        attrs = feature.get("attributes", {})

        teryt = str(attrs.get("teryt", "")).strip()
        nazwa = str(attrs.get("nazwa", "")).strip()

        if not teryt:
            continue

        powiaty.append({
            "teryt": teryt,
            "nazwa": nazwa,
        })

    unique = {}
    for p in powiaty:
        unique[p["teryt"]] = p

    powiaty = list(unique.values())

    print()
    for p in powiaty:
        print(
            f"  {p['teryt']}  {p['nazwa']}"
        )

    print()
    print(
        f"Powiaty intersecting AOI: {len(powiaty)}"
    )

    return powiaty

def download_file(
    url: str,
    destination: Path,
):
    if destination.exists() and destination.stat().st_size > 0:
        print(
            f"  Already exists: {destination.name}"
        )
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"  Downloading: {url}"
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            session = make_session()

            with session.get(
                url,
                stream=True,
                timeout=(
                    TIMEOUT_CONNECT,
                    600,
                ),
            ) as response:

                response.raise_for_status()

                content_type = (
                    response.headers
                    .get("Content-Type", "")
                    .lower()
                )

                first_chunk = None

                temp_path = destination.with_suffix(
                    destination.suffix + ".part"
                )

                with open(
                    temp_path,
                    "wb",
                ) as f:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if not chunk:
                            continue

                        if first_chunk is None:
                            first_chunk = chunk[:100]

                        f.write(chunk)

                session.close()

            with open(
                temp_path,
                "rb",
            ) as f:
                signature = f.read(4)

            if signature != b"PK\x03\x04":
                text = temp_path.read_bytes()[:1000]

                try:
                    text = text.decode(
                        "utf-8",
                        errors="replace",
                    )
                except Exception:
                    text = repr(text)

                temp_path.unlink(
                    missing_ok=True
                )

                raise RuntimeError(
                    "Downloaded file does not look like a ZIP.\n"
                    f"Beginning of response:\n{text}"
                )

            temp_path.replace(destination)

            print(
                f"  Saved: {destination}"
            )

            return

        except Exception as exc:
            last_error = exc

            print(
                f"  Download failed "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{type(exc).__name__}: {exc}"
            )

            part = destination.with_suffix(
                destination.suffix + ".part"
            )

            part.unlink(
                missing_ok=True
            )

            if attempt == MAX_RETRIES:
                break

            delay = min(60.0, 2 ** (attempt - 1))
            delay += random.uniform(0, 2)

            print(
                f"  Retrying in {delay:.1f}s..."
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Could not download {url}"
    ) from last_error

def download_wcs_tile(
    xmin,
    ymin,
    xmax,
    ymax,
    tile_path,
    width,
    height,
):
    """
    Download and validate one WCS GeoTIFF.
    """

    if tile_path.exists():
        try:
            with rasterio.open(tile_path) as old:
                b = old.bounds
                tol = max((xmax - xmin) / max(width, 1), (ymax - ymin) / max(height, 1), 0.02)
                matches = (
                    old.width == width and old.height == height
                    and old.crs is not None and old.crs.to_epsg() == 2180
                    and abs(b.left - xmin) <= tol and abs(b.bottom - ymin) <= tol
                    and abs(b.right - xmax) <= tol and abs(b.top - ymax) <= tol
                )
                if matches:
                    print(f"  Existing, footprint verified: {tile_path.name}")
                    return True
                print(f"  Cached tile footprint differs from request; redownloading: {tile_path.name}")
        except Exception:
            print(f"  Existing tile is invalid; redownloading: {tile_path.name}")
        tile_path.unlink(missing_ok=True)

    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "COVERAGE": WCS_COVERAGE,
        "CRS": CRS,
        "BBOX": (
            f"{xmin:.3f},"
            f"{ymin:.3f},"
            f"{xmax:.3f},"
            f"{ymax:.3f}"
        ),
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "GEOTIFF",
    }

    for attempt in range(1, MAX_RETRIES + 1):

        print(
            f"  Downloading {tile_path.name} "
            f"(attempt {attempt}/{MAX_RETRIES})"
        )

        try:
            session = make_session()

            response = session.get(
                WCS_URL,
                params=params,
                timeout=(
                    TIMEOUT_CONNECT,
                    TIMEOUT_DOWNLOAD,
                ),
            )

            session.close()

            response.raise_for_status()

            content_type = (
                response.headers
                .get("Content-Type", "")
                .lower()
            )

            if (
                "xml" in content_type
                or response.content[:100]
                .lstrip()
                .startswith(b"<?xml")
            ):
                raise RuntimeError(
                    "WCS returned XML instead of GeoTIFF:\n"
                    + response.text[:1500]
                )

            with rasterio.MemoryFile(
                response.content
            ) as memfile:

                with memfile.open() as src:

                    if src.width <= 0 or src.height <= 0:
                        raise RuntimeError(
                            "WCS returned an empty raster."
                        )

                    if src.crs is None:
                        raise RuntimeError(
                            "Downloaded raster has no CRS."
                        )

                    data = src.read()
                    profile = src.profile.copy()
                    profile.update(
                        crs=CRS,
                        transform=rasterio.transform.from_origin(
                            xmin, ymax,
                            (xmax - xmin) / src.width,
                            (ymax - ymin) / src.height,
                        ),
                        width=src.width,
                        height=src.height,
                    )
                    with rasterio.open(tile_path, "w", **profile) as dst:
                        dst.write(data)

                    print(
                        f"    {src.width}x{src.height}, {src.count} bands, "
                        f"source CRS={src.crs}; saved bounds={rasterio.transform.array_bounds(src.height, src.width, profile['transform'])}"
                    )

            return True

        except Exception as exc:

            print(
                f"    Failed: "
                f"{type(exc).__name__}: {exc}"
            )

            tile_path.unlink(
                missing_ok=True
            )

            if attempt == MAX_RETRIES:
                return False

            delay = min(
                60.0,
                2 ** (attempt - 1),
            )
            delay += random.uniform(0, 2)

            time.sleep(delay)

    return False

def download_imagery(
    xmin,
    ymin,
    xmax,
    ymax,
    imagery_dir,
):
    print()
    print("=" * 70)
    print("DOWNLOADING ORTHOPHOTO")
    print("=" * 70)

    imagery_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    width_m = xmax - xmin
    height_m = ymax - ymin

    total_width_px = math.ceil(
        width_m / RESOLUTION
    )

    total_height_px = math.ceil(
        height_m / RESOLUTION
    )

    tiles_x = math.ceil(
        total_width_px / TILE_SIZE
    )

    tiles_y = math.ceil(
        total_height_px / TILE_SIZE
    )

    print(
        f"AOI size: "
        f"{width_m:.1f}m x {height_m:.1f}m"
    )

    print(
        f"Raster size: "
        f"{total_width_px:,} x "
        f"{total_height_px:,} pixels"
    )

    print(
        f"Tiles: {tiles_x} x {tiles_y} "
        f"= {tiles_x * tiles_y}"
    )

    failures = []

    number = 0

    for iy in range(tiles_y):

        for ix in range(tiles_x):

            number += 1

            tile_xmin = (
                xmin
                + ix * TILE_SIZE * RESOLUTION
            )

            tile_ymin = (
                ymin
                + iy * TILE_SIZE * RESOLUTION
            )

            tile_xmax = min(
                tile_xmin
                + TILE_SIZE * RESOLUTION,
                xmax,
            )

            tile_ymax = min(
                tile_ymin
                + TILE_SIZE * RESOLUTION,
                ymax,
            )

            tile_width = math.ceil(
                (
                    tile_xmax - tile_xmin
                )
                / RESOLUTION
            )

            tile_height = math.ceil(
                (
                    tile_ymax - tile_ymin
                )
                / RESOLUTION
            )

            tile_name = (
                f"tile_{iy:03d}_{ix:03d}.tif"
            )

            tile_path = (
                imagery_dir / tile_name
            )

            print()
            print(
                f"[{number}/{tiles_x * tiles_y}]"
            )

            success = download_wcs_tile(
                tile_xmin,
                tile_ymin,
                tile_xmax,
                tile_ymax,
                tile_path,
                tile_width,
                tile_height,
            )

            if not success:
                failures.append(
                    tile_name
                )

    print()

    if failures:
        print(
            f"FAILED TILES: {len(failures)}"
        )

        for name in failures:
            print(
                f"  {name}"
            )

        raise RuntimeError(
            "Some orthophoto tiles failed. "
            "Run the script again to resume."
        )

    print(
        "All orthophoto tiles downloaded."
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download Geoportal orthophotos and "
            "BDOT10k building labels for a WGS84 AOI."
        )
    )
    parser.add_argument("--lat1", type=float, required=True, help="First corner latitude.")
    parser.add_argument("--lon1", type=float, required=True, help="First corner longitude.")
    parser.add_argument("--lat2", type=float, required=True, help="Second corner latitude.")
    parser.add_argument("--lon2", type=float, required=True, help="Second corner longitude.")
    parser.add_argument("--output", type=Path, default=Path("dataset"), help="Output directory.")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE, help="Orthophoto tile size in pixels.")
    parser.add_argument("--resolution", type=float, default=RESOLUTION, help="Ground resolution in meters/pixel.")
    parser.add_argument("--no-masks", action="store_true", help="Download/process data but don't create masks.")
    parser.add_argument("--no-imagery", action="store_true", help="Skip orthophoto download.")
    return parser.parse_args()


def main():
    global TILE_SIZE
    global RESOLUTION

    args = parse_args()

    TILE_SIZE = args.tile_size
    RESOLUTION = args.resolution

    output_dir = args.output

    imagery_dir = (
        output_dir / "imagery"
    )

    buildings_dir = (
        output_dir / "buildings"
    )

    masks_dir = (
        output_dir / "masks"
    )

    previews_dir = (
        output_dir / "previews"
    )

    temp_dir = (
        output_dir / "_temp"
    )

    for directory in (
        imagery_dir,
        buildings_dir,
        masks_dir,
        previews_dir,
        temp_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    xmin, ymin, xmax, ymax = (
        wgs84_to_2180_bbox(
            args.lat1,
            args.lon1,
            args.lat2,
            args.lon2,
        )
    )

    print()
    print("=" * 70)
    print("GEOPORTAL BUILDING DATASET")
    print("=" * 70)

    print()
    print("WGS84 corners:")
    print(
        f"  ({args.lat1}, {args.lon1})"
    )
    print(
        f"  ({args.lat2}, {args.lon2})"
    )

    print()
    print("EPSG:2180 AOI:")
    print(
        f"  X: {xmin:.3f} -> {xmax:.3f}"
    )
    print(
        f"  Y: {ymin:.3f} -> {ymax:.3f}"
    )

    print()
    print(
        f"Ground size: "
        f"{xmax - xmin:.1f}m x "
        f"{ymax - ymin:.1f}m"
    )

    powiaty = find_intersecting_powiaty(
        xmin,
        ymin,
        xmax,
        ymax,
    )

    print()
    print("=" * 70)
    print("DOWNLOADING BDOT10k")
    print("=" * 70)

    all_gml_files = []

    for p in powiaty:

        teryt = p["teryt"]
        name = p["nazwa"]

        print()
        print(
            f"Powiat {teryt}: {name}"
        )

        url = bdot_package_url(
            teryt
        )

        zip_path = (
            temp_dir
            / f"{teryt}_GML.zip"
        )

        download_file(
            url,
            zip_path,
        )

        extracted_dir = (
            temp_dir
            / teryt
            / "gml"
        )

        gml_files = extract_building_gml(
            zip_path,
            extracted_dir,
        )

        all_gml_files.extend(
            gml_files
        )

    buildings = load_buildings(
        all_gml_files,
        (xmin, ymin, xmax, ymax),
    )

    building_output = (
        buildings_dir
        / f"{BUILDING_CLASS}_AOI.gpkg"
    )

    save_buildings(
        buildings,
        building_output,
    )

    if not args.no_imagery:

        download_imagery(
            xmin,
            ymin,
            xmax,
            ymax,
            imagery_dir,
        )

    else:

        print()
        print(
            "Skipping imagery download."
        )

    if not args.no_masks:

        create_masks(
            buildings,
            imagery_dir,
            masks_dir,
            previews_dir,
        )

    else:

        print()
        print(
            "Skipping mask generation."
        )

    print()
    print("=" * 70)
    print("DATASET COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Output: {output_dir.resolve()}"
    )

    print(
        f"Imagery: {imagery_dir.resolve()}"
    )

    print(
        f"Buildings: {building_output.resolve()}"
    )

    if not args.no_masks:
        print(
            f"Masks: {masks_dir.resolve()}"
        )


if __name__ == "__main__":
    main()
