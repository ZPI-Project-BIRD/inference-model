#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import box


from common import (
    wgs84_to_2180_bbox, bdot_package_url, extract_building_gml,
    load_buildings, save_buildings, create_masks,
)

ORTHO_WFS_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WFS/Skorowidze"
ORTHO_FEATURE_TYPE = "gugik:SkorowidzOrtofomapy2025"
POWIAT_URL = (
    "https://mapy.geoportal.gov.pl/wss/ims/maps/PRG_gugik_wyszukiwarka/"
    "FeatureServer/1/query"
)
BDOT_BASE_URL = "https://opendata.geoportal.gov.pl/bdot10k/schemat2021"

CRS = "EPSG:2180"
WGS84 = "EPSG:4326"
BUILDING_CLASS = "OT_BUBD_A"
DEFAULT_RESOLUTION = 0.25
DEFAULT_TILE_SIZE = 2048
ALL_TOUCHED = True
NUM_PREVIEWS = 20
MAX_RETRIES = 7
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 600


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Geoportal-CIR-BuildingDataset/1.0",
        "Connection": "close",
    })
    return session


def request_with_retries(method, url, *, params=None, stream=False, timeout=None):
    if timeout is None:
        timeout = (CONNECT_TIMEOUT, READ_TIMEOUT)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with make_session() as session:
                response = session.request(
                    method, url, params=params, stream=stream, timeout=timeout
                )
                if not stream:
                    response.content
                response.raise_for_status()
                return response
        except Exception as exc:
            last_error = exc
            print(f"  Request failed ({attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                delay = min(60.0, 2 ** (attempt - 1)) + random.uniform(0, 1.5)
                print(f"  Retrying in {delay:.1f}s")
                time.sleep(delay)
    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url}") from last_error


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def child_text(node, wanted_name: str) -> str:
    for child in list(node):
        if local_name(child.tag).lower() == wanted_name.lower():
            return " ".join(t.strip() for t in child.itertext() if t.strip())
    return ""


def feature_bbox(feature):
    for node in feature.iter():
        if local_name(node.tag).lower() == "envelope":
            corners = []
            for child in list(node):
                lname = local_name(child.tag).lower()
                if lname in ("lowercorner", "uppercorner"):
                    vals = (child.text or "").split()
                    if len(vals) >= 2:
                        corners.append((float(vals[0]), float(vals[1])))
            if len(corners) == 2:
                (axis1_a, axis2_a), (axis1_b, axis2_b) = corners
                x1, y1 = axis2_a, axis1_a
                x2, y2 = axis2_b, axis1_b
                return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return None


def query_cir_sheets(bbox, output_gml: Path, feature_type: str):
    xmin, ymin, xmax, ymax = bbox
    aoi = box(xmin, ymin, xmax, ymax)
    base_params = {
        "SERVICE": "WFS",
        "VERSION": "1.1.0",
        "REQUEST": "GetFeature",
        "TYPENAME": feature_type,
        "SRSNAME": CRS,
    }

    def parse_response(raw):
        root = ET.fromstring(raw)
        parsed = []
        for node in root.iter():
            if local_name(node.tag).lower() not in ("featuremember", "member"):
                continue
            children = [c for c in list(node) if isinstance(c.tag, str)]
            if not children:
                continue
            feature = children[0]
            attrs = {}
            for child in list(feature):
                key = local_name(child.tag)
                if key.lower() in ("msgeometry", "boundedby"):
                    continue
                value = " ".join(t.strip() for t in child.itertext() if t.strip())
                if value:
                    attrs[key] = value
            attrs["_bbox"] = feature_bbox(feature)
            parsed.append(attrs)
        return parsed

    attempts = [
        ("axis-order Y,X", f"{ymin},{xmin},{ymax},{xmax},{CRS}"),
        ("GIS-order X,Y", f"{xmin},{ymin},{xmax},{ymax},{CRS}"),
    ]
    chosen_features = []
    chosen_raw = None
    chosen_label = None
    best_score = -1

    print("Querying official orthophoto index (testing WFS axis order)...")
    for label, bbox_value in attempts:
        params = dict(base_params, BBOX=bbox_value)
        response = request_with_retries("GET", ORTHO_WFS_URL, params=params)
        raw = response.content
        if b"ExceptionReport" in raw[:3000] or b"ServiceException" in raw[:3000]:
            print(f"  {label}: service returned an exception; trying other order.")
            continue
        try:
            features = parse_response(raw)
        except ET.ParseError:
            print(f"  {label}: response was not valid XML; trying other order.")
            continue

        cir_with_urls = [
            feat for feat in features
            if feat.get("kolor", "").strip().upper() == "CIR"
            and feat.get("url_do_pobrania", "").strip().startswith(("https://", "http://"))
        ]
        intersecting = [
            feat for feat in cir_with_urls
            if feat.get("_bbox") and box(*feat["_bbox"]).intersects(aoi)
        ]
        score = len(intersecting) * 1000 + (len(cir_with_urls) if cir_with_urls else 0)
        print(
            f"  {label}: {len(features)} features, "
            f"{len(cir_with_urls)} CIR URLs, {len(intersecting)} envelope intersections"
        )
        if score > best_score:
            best_score = score
            chosen_features = intersecting if intersecting else cir_with_urls
            chosen_raw = raw
            chosen_label = label
        if intersecting:
            break

    if chosen_raw is None:
        raise RuntimeError(
            "Both WFS BBOX axis-order requests failed. Check the service and "
            "the saved AOI coordinates."
        )

    output_gml.parent.mkdir(parents=True, exist_ok=True)
    output_gml.write_bytes(chosen_raw)
    print(f"Using WFS query order: {chosen_label}")

    cir = []
    seen_urls = set()
    for feat in chosen_features:
        url = feat.get("url_do_pobrania", "").strip()
        if url in seen_urls:
            continue
        seen_urls.add(url)
        cir.append(feat)

    print(f"Index features in selected response: {len(parse_response(chosen_raw))}")
    print(f"CIR sheets selected for AOI: {len(cir)}")
    for feat in cir:
        bounds = feat.get("_bbox")
        if bounds:
            print(f"  Index envelope: {bounds}")
        print(
            f"  {feat.get('godlo', 'unknown')} | "
            f"{feat.get('piksel', '?')} m | "
            f"{feat.get('url_do_pobrania')}"
        )
    if not cir:
        raise RuntimeError(
            "No CIR sheets were found for this AOI in either WFS BBOX axis order. "
            "Check coordinates, feature type/year, and the saved _temp/cir_index.gml."
        )
    return cir


def safe_filename(url: str) -> str:
    name = unquote(Path(urlparse(url).path).name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "cir_sheet.tif"


def download_file(url: str, destination: Path, *, expected_zip=False):
    if destination.exists() and destination.stat().st_size > 1024:
        if expected_zip:
            try:
                with zipfile.ZipFile(destination):
                    print(f"  Reusing package: {destination.name}")
                    return
            except zipfile.BadZipFile:
                destination.unlink(missing_ok=True)
        else:
            try:
                with rasterio.open(destination) as src:
                    if src.count >= 3 and src.crs:
                        print(f"  Reusing CIR image: {destination.name}")
                        return
            except Exception:
                destination.unlink(missing_ok=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Downloading {url}")
            with make_session() as session:
                with session.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as response:
                    response.raise_for_status()
                    with temp.open("wb") as f:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                f.write(chunk)
            if expected_zip:
                with zipfile.ZipFile(temp):
                    pass
            else:
                with rasterio.open(temp) as src:
                    if src.count < 3:
                        raise RuntimeError(
                            f"Expected at least 3 bands, found {src.count}"
                        )
                    if src.crs is None:
                        raise RuntimeError("Downloaded TIFF has no CRS")
                    print(
                        f"    Valid TIFF: {src.width}x{src.height}, "
                        f"{src.count} bands, CRS={src.crs}, dtype={src.dtypes[0]}"
                    )
            temp.replace(destination)
            print(f"  Saved: {destination}")
            return
        except Exception as exc:
            last_error = exc
            temp.unlink(missing_ok=True)
            print(f"  Download/validation failed ({attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(min(60, 2 ** (attempt - 1)) + random.uniform(0, 1.5))
    raise RuntimeError(f"Could not download {url}") from last_error


def find_intersecting_powiaty(bbox):
    xmin, ymin, xmax, ymax = bbox
    params = {
        "f": "json",
        "where": "1=1",
        "geometry": json.dumps({
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "spatialReference": {"wkid": 2180},
        }),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 2180,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "teryt,nazwa",
        "returnGeometry": "false",
    }
    response = request_with_retries("GET", POWIAT_URL, params=params, timeout=(30, 120))
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Powiat service error: {data['error']}")
    result = {}
    for feat in data.get("features", []):
        attrs = feat.get("attributes", {})
        code = str(attrs.get("teryt", "")).strip()
        if code:
            result[code] = str(attrs.get("nazwa", "")).strip()
    if not result:
        raise RuntimeError("No powiat found for the AOI.")
    print("\nIntersecting powiaty:")
    for code, name in result.items():
        print(f"  {code} {name}")
    return [{"teryt": code, "nazwa": name} for code, name in result.items()]


def create_cir_tiles(cir_paths, bbox, imagery_dir, resolution, tile_size):
    xmin, ymin, xmax, ymax = bbox
    imagery_dir.mkdir(parents=True, exist_ok=True)

    total_width = math.ceil((xmax - xmin) / resolution)
    total_height = math.ceil((ymax - ymin) / resolution)
    tiles_x = math.ceil(total_width / tile_size)
    tiles_y = math.ceil(total_height / tile_size)

    print("\nCreating training tiles from CIR sheets...")
    print(f"AOI: {(xmax-xmin):.1f} m x {(ymax-ymin):.1f} m")
    print(f"Resolution: {resolution} m/pixel")
    print(f"Pixel grid: {total_width:,} x {total_height:,}")
    print(f"Tiles: {tiles_y} rows x {tiles_x} columns = {tiles_x*tiles_y}")

    source_handles = []
    warped_handles = []
    try:
        for path in cir_paths:
            src = rasterio.open(path)
            source_handles.append(src)
            if src.crs is None:
                raise RuntimeError(f"Missing CRS in source CIR TIFF: {path}")

            print(
                f"  Source {path.name}: CRS={src.crs}, "
                f"size={src.width}x{src.height}, bounds={src.bounds}"
            )

            vrt = WarpedVRT(
                src,
                crs=CRS,
                resolution=resolution,
                resampling=rasterio.enums.Resampling.bilinear,
                nodata=src.nodata if src.nodata is not None else 0,
            )
            warped_handles.append(vrt)
            print(f"    Reprojected view bounds: {vrt.bounds}")

        aoi = box(xmin, ymin, xmax, ymax)
        written = 0

        for iy in range(tiles_y):
            for ix in range(tiles_x):
                txmin = xmin + ix * tile_size * resolution
                tymax = ymax - iy * tile_size * resolution
                width = min(tile_size, max(0, total_width - ix * tile_size))
                height = min(tile_size, max(0, total_height - iy * tile_size))
                if width <= 0 or height <= 0:
                    continue

                txmax = txmin + width * resolution
                tymin = tymax - height * resolution
                tile_bounds = (txmin, tymin, txmax, tymax)
                tile_geom = box(*tile_bounds)

                overlapping = [
                    vrt for vrt in warped_handles
                    if box(*vrt.bounds).intersects(tile_geom)
                ]
                if not overlapping:
                    print(f"  No source CIR coverage for tile row={iy}, col={ix}; skipped.")
                    continue

                name = f"tile_{iy:03d}_{ix:03d}.tif"
                out_path = imagery_dir / name
                if out_path.exists():
                    try:
                        with rasterio.open(out_path) as old:
                            b = old.bounds
                            tol = max(resolution * 0.51, 0.02)
                            matches = (
                                old.width == tile_size and old.height == tile_size
                                and old.crs is not None and old.crs.to_epsg() == 2180
                                and old.count >= 3
                                and abs(b.left - txmin) <= tol
                                and abs(b.top - tymax) <= tol
                                and abs(old.transform.a - resolution) <= 1e-8
                                and abs(old.transform.e + resolution) <= 1e-8
                            )
                            if matches:
                                print(f"  Existing, grid verified: {name}")
                                written += 1
                                continue
                            print(f"  Cached tile grid differs; rebuilding: {name}")
                    except Exception:
                        pass
                    out_path.unlink(missing_ok=True)

                merged, transform = merge(
                    overlapping,
                    bounds=tile_bounds,
                    res=(resolution, resolution),
                    indexes=[1, 2, 3],
                    nodata=0,
                    method="first",
                )
                if merged.shape[1] == 0 or merged.shape[2] == 0:
                    print(f"  Empty tile: {name}; skipped.")
                    continue

                count, mh, mw = merged.shape
                padded = np.zeros((3, tile_size, tile_size), dtype=merged.dtype)
                padded[:, :min(mh, tile_size), :min(mw, tile_size)] = merged[:3, :tile_size, :tile_size]

                transform = from_origin(txmin, tymax, resolution, resolution)
                profile = {
                    "driver": "GTiff",
                    "height": tile_size,
                    "width": tile_size,
                    "count": 3,
                    "dtype": padded.dtype,
                    "crs": CRS,
                    "transform": transform,
                    "compress": "deflate",
                    "predictor": 2 if np.issubdtype(padded.dtype, np.integer) else 3,
                }
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(padded)
                    dst.set_band_description(1, "CIR_band_1")
                    dst.set_band_description(2, "CIR_band_2")
                    dst.set_band_description(3, "CIR_band_3")
                written += 1
                print(f"  Wrote {name}: {tile_size}x{tile_size}, 3 bands")

        if written == 0:
            details = "\n".join(
                f"  {path.name}: CRS={src.crs}, bounds={src.bounds}"
                for path, src in zip(cir_paths, source_handles)
            )
            raise RuntimeError(
                "No CIR training tiles were created. None of the downloaded "
                "CIR TIFF footprints overlap the AOI after reprojecting to "
                f"{CRS}. AOI bounds: {bbox}\nSource metadata:\n{details}\n"
                "Check that the selected WGS84 corners are in the same area "
                "as the CIR sheets and inspect the source TIFF metadata."
            )
    finally:
        for vrt in warped_handles:
            vrt.close()
        for src in source_handles:
            src.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Build a building-segmentation dataset from CIR orthophotos and BDOT10k.")
    parser.add_argument("--lat1", type=float, required=True, help="First corner latitude")
    parser.add_argument("--lon1", type=float, required=True, help="First corner longitude")
    parser.add_argument("--lat2", type=float, required=True, help="Second corner latitude")
    parser.add_argument("--lon2", type=float, required=True, help="Second corner longitude")
    parser.add_argument("--output", type=Path, default=Path("cir_building_dataset"), help="Output directory")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="Tile edge in pixels")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, help="Target ground resolution in metres/pixel (default 0.25 for sample CIR sheets)")
    parser.add_argument("--ortho-type", default=ORTHO_FEATURE_TYPE, help="WFS index layer")
    parser.add_argument("--no-masks", action="store_true", help="Skip mask generation")
    parser.add_argument("--no-imagery", action="store_true", help="Skip CIR download and tiling")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tile_size <= 0 or args.resolution <= 0:
        raise SystemExit("--tile-size and --resolution must be positive.")

    output = args.output
    imagery_dir = output / "imagery"
    buildings_dir = output / "buildings"
    masks_dir = output / "masks"
    previews_dir = output / "previews"
    temp_dir = output / "_temp"
    cir_dir = temp_dir / "cir_sheets"
    gml_dir = temp_dir / "bdot_gml"

    for directory in (imagery_dir, buildings_dir, masks_dir, previews_dir, cir_dir, gml_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bbox = wgs84_to_2180_bbox(args.lat1, args.lon1, args.lat2, args.lon2)
    xmin, ymin, xmax, ymax = bbox
    if xmin == xmax or ymin == ymax:
        raise SystemExit("The two corners must define a non-zero area.")

    print("=" * 72)
    print("GEOPORTAL CIR BUILDING-SEGMENTATION DATASET")
    print("=" * 72)
    print(f"WGS84 corners: ({args.lat1}, {args.lon1}) and ({args.lat2}, {args.lon2})")
    print(f"EPSG:2180 AOI: X={xmin:.3f}..{xmax:.3f}, Y={ymin:.3f}..{ymax:.3f}")
    print(f"Ground area: {(xmax-xmin):.1f} x {(ymax-ymin):.1f} m")
    print(f"CIR output resolution: {args.resolution} m/pixel; tile size: {args.tile_size}")

    if not args.no_imagery:
        cir_features = query_cir_sheets(
            bbox, temp_dir / "cir_index.gml", args.ortho_type
        )
        cir_paths = []
        for feat in cir_features:
            url = feat["url_do_pobrania"].strip()
            dest = cir_dir / safe_filename(url)
            download_file(url, dest, expected_zip=False)
            cir_paths.append(dest)

        aoi_geom = box(*bbox)
        valid_paths = []
        for path in cir_paths:
            try:
                with rasterio.open(path) as src:
                    bounds = (src.bounds.left, src.bounds.bottom,
                              src.bounds.right, src.bounds.top)
                    print(f"Downloaded sheet bounds: {path.name} | CRS={src.crs} | {bounds}")
                    if src.crs and src.crs.to_epsg() == 2180 and box(*bounds).intersects(aoi_geom):
                        valid_paths.append(path)
            except Exception as exc:
                print(f"  Could not inspect {path.name}: {exc}")

        if not valid_paths:
            raise RuntimeError(
                "The downloaded CIR sheets do not overlap the AOI in EPSG:2180. "
                "This usually indicates a WFS BBOX axis-order issue or an incorrect "
                "index response. No tiles were generated; check the WFS query and "
                "the printed AOI/sheet bounds."
            )
        create_cir_tiles(valid_paths, bbox, imagery_dir, args.resolution, args.tile_size)
    else:
        print("\nSkipping CIR download and tile generation.")

    powiaty = find_intersecting_powiaty(bbox)
    all_gml = []
    print("\nDownloading/extracting BDOT10k building data...")
    for p in powiaty:
        code = p["teryt"]
        url = bdot_package_url(code)
        zip_path = temp_dir / f"{code}_GML.zip"
        download_file(url, zip_path, expected_zip=True)
        extracted = extract_building_gml(zip_path, gml_dir / code)
        all_gml.extend(extracted)

    buildings = load_buildings(all_gml, bbox)
    building_path = buildings_dir / f"{BUILDING_CLASS}_AOI.gpkg"
    save_buildings(buildings, building_path)

    if not args.no_masks:
        create_masks(buildings, imagery_dir, masks_dir, previews_dir)
    else:
        print("\nSkipping mask generation.")

    print("\n" + "=" * 72)
    print("DATASET COMPLETE")
    print(f"Output: {output.resolve()}")
    print(f"CIR imagery: {imagery_dir.resolve()}")
    print(f"Building polygons: {building_path.resolve()}")
    if not args.no_masks:
        print(f"Masks: {masks_dir.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
