"""Raster inference pipeline with selectable paper or rectangle geometry.

Detections are streamed: after each tile row is inferred, every connected
building segment that can no longer grow (it does not touch the boundary with
rows still being inferred) is converted to final geometry immediately and
handed to an optional ``on_detections`` callback, instead of waiting for the
whole raster to finish.

An optional ``clip_area`` (a polygon in EPSG:4326 lon/lat, e.g. the project
rectangle) restricts inference to the part of the sheet inside that area, and
pixels outside it are never turned into detections.
"""
from math import ceil, floor
from pathlib import Path
import time
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import shapes, rasterize
from shapely.affinity import affine_transform
from rasterio.warp import transform_geom
from rasterio.windows import Window
from shapely.geometry import box, mapping, shape
from shapely.validation import make_valid
from shapely.ops import unary_union

from .model import iter_predict_raster, to_rgb
from .geometry_rectangle import (
    polygon_parts, fill_small_holes, orthogonalize, decompose_building,
)
from .geometry_paper import regularize_polygon
from .outputs import save_quicklook, save_results_json


class _BuildingExtractor:
    """Turns raw segment polygons into building / rectangle features."""

    def __init__(self, args, geometry_mode, source_name):
        self.args = args
        self.geometry_mode = geometry_mode
        self.source_name = source_name
        self.building_count = 0

    def extract(self, poly):
        """Return (building_feats, rectangle_feats) for one segment polygon."""
        args = self.args
        building_feats, rectangle_feats = [], []
        if poly.area < args.min_area_m2:
            return building_feats, rectangle_feats

        if self.geometry_mode == "paper":
            # The paper algorithm owns its simplification/coarse/fine stages.
            # Do not pre-simplify or orthogonalize this input a second time.
            result = regularize_polygon(poly)
            if result is None or result.is_empty:
                return building_feats, rectangle_feats
            for final in polygon_parts(result):
                if final.area < args.min_area_m2:
                    continue
                self.building_count += 1
                building_feats.append({
                    "geometry": final,
                    "building_id": self.building_count,
                    "source": self.source_name,
                    "area_m2": float(final.area),
                    "perimeter_m": float(final.length),
                    "segment_area_m2": float(poly.area),
                    "geometry_mode": "paper",
                    "threshold": args.threshold,
                })
            return building_feats, rectangle_feats

        prepared = fill_small_holes(poly, args.hole_area_m2)
        if args.simplify_m > 0:
            prepared = prepared.simplify(args.simplify_m, preserve_topology=True)
        if getattr(args, "orthogonalize", False):
            prepared = orthogonalize(prepared, args.orthogonal_angle_deg)
        if not prepared.is_valid:
            prepared = make_valid(prepared)

        for building in polygon_parts(prepared):
            if building.area < args.min_area_m2:
                continue
            decomposition = decompose_building(
                building,
                waste_limit=args.max_rectangle_waste,
                max_parts=args.max_rectangle_parts,
                min_part_area=args.min_rectangle_area_m2,
                snap_distance=getattr(args, "rectangle_snap_distance", 0.5),
                snap_angle_deg=getattr(args, "rectangle_snap_angle_deg", 8.0),
                gap_close_distance=getattr(args, "rectangle_gap_close_distance", 0.5),
            )
            outlines = [g for g, _ in decomposition if g is not None and not g.is_empty]
            if not outlines:
                continue
            self.building_count += 1
            building_id = self.building_count
            outline_union = unary_union(outlines).buffer(0)
            for final in polygon_parts(outline_union):
                building_feats.append({
                    "geometry": final,
                    "building_id": building_id,
                    "source": self.source_name,
                    "area_m2": float(final.area),
                    "perimeter_m": float(final.length),
                    "segment_area_m2": float(building.area),
                    "rectangle_count": len(outlines),
                    "geometry_mode": "rectangles",
                    "threshold": args.threshold,
                })
            for idx, (outline, waste) in enumerate(decomposition, start=1):
                rectangle_feats.append({
                    "geometry": outline,
                    "building_id": building_id,
                    "part_id": idx,
                    "source": self.source_name,
                    "area_m2": float(outline.area),
                    "waste_ratio": float(waste),
                    "threshold": args.threshold,
                })
        return building_feats, rectangle_feats


def _complete_segments(window_mask, at_end):
    """Split a mask window into finished segments and the top of unfinished ones.

    Polygons are returned in window pixel coordinates. A segment touching the
    bottom row of the window may still continue into rows that are not final
    yet, so it is deferred unless ``at_end`` is set. Returns
    ``(complete_pixel_polygons, deferred_top_row_or_None)``.
    """
    complete, deferred_top = [], None
    bottom = window_mask.shape[0]
    for geom_mapping, value in shapes(window_mask.astype(np.uint8), mask=window_mask,
                                      connectivity=8):
        if value != 1:
            continue
        geom = shape(geom_mapping)
        _, miny, _, maxy = geom.bounds
        if not at_end and maxy >= bottom:
            top = int(miny)
            deferred_top = top if deferred_top is None else min(deferred_top, top)
        else:
            complete.append(geom)
    return complete, deferred_top


CLIP_AREA_CRS = "EPSG:4326"


def lonlat_box(min_lon, min_lat, max_lon, max_lat):
    """Lon/lat rectangle as a polygon suitable for ``clip_area``."""
    return box(min(min_lon, max_lon), min(min_lat, max_lat),
               max(min_lon, max_lon), max(min_lat, max_lat))


class _ClippedSource:
    """Read-only view of a rasterio dataset restricted to a pixel window.

    Exposes the subset of the dataset API the pipeline uses (height, width,
    crs, transform, profile, read) in coordinates local to the window.
    """

    def __init__(self, src, window):
        self.src, self.window = src, window
        self.height, self.width = int(window.height), int(window.width)
        self.crs = src.crs
        self.transform = src.window_transform(window)
        self.profile = src.profile.copy()
        self.profile.update(height=self.height, width=self.width, transform=self.transform)
        # Source block sizes may not fit the smaller extent when writing outputs.
        for key in ("blockxsize", "blockysize", "tiled"):
            self.profile.pop(key, None)

    def read(self, window=None):
        if window is None:
            return self.src.read(window=self.window)
        return self.src.read(window=Window(
            self.window.col_off + window.col_off, self.window.row_off + window.row_off,
            window.width, window.height,
        ))


def _clip_to_area(src, clip_area):
    """Return (source_view, inside_mask_or_None, area_in_raster_crs_or_None).

    The view is None when the raster does not overlap the area.
    """
    if clip_area is None:
        return src, None, None
    # Densify edges so the lon/lat rectangle stays accurate after reprojection.
    area = shape(transform_geom(CLIP_AREA_CRS, src.crs, mapping(clip_area.segmentize(0.0005))))
    inverse = ~src.transform
    minx, miny, maxx, maxy = area.bounds
    cols, rows = zip(*(inverse * xy for xy in
                       [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]))
    c0, c1 = max(0, floor(min(cols))), min(src.width, ceil(max(cols)))
    r0, r1 = max(0, floor(min(rows))), min(src.height, ceil(max(rows)))
    if c1 <= c0 or r1 <= r0:
        return None, None, None
    view = _ClippedSource(src, Window(c0, r0, c1 - c0, r1 - r0))
    inside = rasterize([(area, 1)], out_shape=(view.height, view.width),
                       transform=view.transform, fill=0, dtype="uint8").astype(bool)
    if not inside.any():
        return None, None, None
    print(f"[DEBUG] Clipped {src.width}x{src.height} sheet to project area: "
          f"window cols {c0}:{c1}, rows {r0}:{r1} ({view.width}x{view.height}); "
          f"{int(inside.sum())} pixels inside", flush=True)
    if inside.all():
        return view, None, None
    return view, inside, area


def _clip_features(features, area, min_area):
    """Cut feature geometries to ``area``; drop pieces that vanish or get too small."""
    kept = []
    for feat in features:
        geom = feat["geometry"]
        if area.contains(geom):
            kept.append(feat)
            continue
        clipped = geom.intersection(area)
        parts = [g for g in polygon_parts(clipped) if g.area >= min_area]
        if not parts:
            continue
        geom = unary_union(parts) if len(parts) > 1 else parts[0]
        feat = dict(feat, geometry=geom, area_m2=float(geom.area))
        if "perimeter_m" in feat:
            feat["perimeter_m"] = float(geom.length)
        kept.append(feat)
    return kept


def process_file(path, model, processor, args, device, on_detections=None, clip_area=None):
    """Process a raster using args.geometry_mode ('paper' or 'rectangles').

    ``on_detections(building_feats)`` is called with each batch of newly
    finished buildings as soon as they are known, while inference continues.

    ``clip_area`` is an optional shapely polygon in EPSG:4326 (lon/lat). When
    given, only the part of the raster inside it is processed and every
    detection is confined to it. A raster that does not overlap the area
    yields no detections.
    """
    total_started = time.perf_counter()
    geometry_mode = getattr(args, "geometry_mode", "rectangles").lower()
    if geometry_mode not in {"paper", "rectangles"}:
        raise ValueError("geometry_mode must be 'paper' or 'rectangles'")

    print(f"\n[DEBUG] Processing {path}; geometry_mode={geometry_mode}", flush=True)
    extractor = _BuildingExtractor(args, geometry_mode, path.name)
    building_feats, rectangle_feats = [], []
    shape_count = 0
    geometry_seconds = 0.0

    with rasterio.open(path) as dataset:
        src = dataset
        if src.crs is None:
            raise ValueError(f"{path} has no CRS")
        if not src.crs.is_projected:
            raise ValueError(
                f"{path} CRS {src.crs} is not projected; reproject to a metric CRS "
                "before using area/distance thresholds."
            )
        src, inside, area = _clip_to_area(dataset, clip_area)
        if src is None:
            print(f"[DEBUG] {path} does not overlap the clip area; no detections", flush=True)
            return [], dataset.crs
        transform, crs, profile = src.transform, src.crs, src.profile.copy()
        h, w = src.height, src.width
        prob = np.zeros((h, w), dtype=np.float32)
        # Pixels that already belong to an emitted segment.
        consumed = np.zeros((h, w), dtype=bool)
        pending_start = 0  # first row that may still hold an unfinished segment

        stage_started = time.perf_counter()
        print("[DEBUG] Stage: streaming model inference starting", flush=True)
        for y0, y1, rows in iter_predict_raster(src, model, processor, args, device):
            if inside is not None:
                rows = np.where(inside[y0:y1], rows, 0.0)
            prob[y0:y1] = rows
            at_end = y1 >= h
            geom_started = time.perf_counter()
            window_mask = (prob[pending_start:y1] >= args.threshold) & ~consumed[pending_start:y1]
            complete, deferred_top = _complete_segments(window_mask, at_end)

            if complete:
                consumed[pending_start:y1] |= rasterize(
                    ((g, 1) for g in complete), out_shape=window_mask.shape,
                    fill=0, dtype="uint8",
                ).astype(bool)
                wt = transform * Affine.translation(0, pending_start)
                params = [wt.a, wt.b, wt.d, wt.e, wt.c, wt.f]
                new_buildings = []
                for pixel_geom in complete:
                    shape_count += 1
                    geom = affine_transform(pixel_geom, params)
                    if not geom.is_valid:
                        geom = make_valid(geom)
                    for poly in polygon_parts(geom):
                        b, r = extractor.extract(poly)
                        if area is not None:
                            # Regularized outlines can overshoot the cut edge slightly.
                            b = _clip_features(b, area, args.min_area_m2)
                            r = _clip_features(r, area, 0.0)
                        new_buildings.extend(b)
                        rectangle_feats.extend(r)
                building_feats.extend(new_buildings)
                geometry_seconds += time.perf_counter() - geom_started
                if new_buildings and on_detections is not None:
                    on_detections(new_buildings)
            else:
                geometry_seconds += time.perf_counter() - geom_started

            pending_start = y1 if deferred_top is None else pending_start + deferred_top

        print(f"[DEBUG] Stage: inference + streaming geometry finished in "
              f"{time.perf_counter() - stage_started:.1f}s (geometry {geometry_seconds:.1f}s); "
              f"shapes={shape_count}, buildings={len(building_feats)}", flush=True)
        stage_started = time.perf_counter()
        print("[DEBUG] Stage: reading raster and creating RGB preview", flush=True)
        rgb_full = to_rgb(src.read())
        print(f"[DEBUG] Stage: RGB preview ready in {time.perf_counter() - stage_started:.1f}s; "
              f"shape={rgb_full.shape}", flush=True)

    mask = prob >= args.threshold
    print(f"[DEBUG] Mask: {int(np.count_nonzero(mask))}/{mask.size} positive pixels", flush=True)
    if getattr(args, "save_probability", False):
        prof = profile.copy()
        prof.update(count=1, dtype="float32", nodata=None, compress="deflate")
        with rasterio.open(args.output / f"{path.stem}_probability.tif", "w", **prof) as dst:
            dst.write(prob.astype(np.float32), 1)
    if getattr(args, "save_mask", False):
        prof = profile.copy()
        prof.update(count=1, dtype="uint8", nodata=0, compress="deflate")
        with rasterio.open(args.output / f"{path.stem}_mask.tif", "w", **prof) as dst:
            dst.write(mask.astype(np.uint8), 1)

    args.output.mkdir(parents=True, exist_ok=True)
    png_path = args.output / f"{path.stem}_{geometry_mode}_outline_preview.png"
    save_quicklook(rgb_full, prob, mask, building_feats, transform, png_path, args.threshold)

    import geopandas as gpd
    out = args.output / f"{path.stem}_buildings.gpkg"
    gdf = gpd.GeoDataFrame(building_feats, geometry="geometry", crs=crs)
    gdf.to_file(out, layer="buildings", driver="GPKG")
    if rectangle_feats:
        rect_gdf = gpd.GeoDataFrame(rectangle_feats, geometry="geometry", crs=crs)
        rect_gdf.to_file(out, layer="rectangle_footprints", driver="GPKG", mode="a")
    print(f"  Geometry mode: {geometry_mode}; building outlines: {len(gdf)}; "
          f"rectangle-derived parts: {len(rectangle_feats)} -> {out}", flush=True)

    json_out = args.output / f"{path.stem}_buildings.geojson"
    save_results_json(building_feats, json_out, path.name, crs)
    print(f"[DEBUG] Pipeline completed in {time.perf_counter() - total_started:.1f}s", flush=True)
    return building_feats, crs
