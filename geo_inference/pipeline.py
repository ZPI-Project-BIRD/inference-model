"""Raster inference pipeline with selectable paper or rectangle geometry."""
from pathlib import Path
import time
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.validation import make_valid
from shapely.ops import unary_union

from .model import predict_raster, to_rgb
from .geometry_rectangle import (
    polygon_parts, fill_small_holes, orthogonalize, decompose_building,
)
from .geometry_paper import regularize_polygon
from .outputs import save_quicklook, save_results_json


def process_file(path, model, processor, args, device):
    """Process a raster using args.geometry_mode ('paper' or 'rectangles')."""
    total_started = time.perf_counter()
    geometry_mode = getattr(args, "geometry_mode", "rectangles").lower()
    if geometry_mode not in {"paper", "rectangles"}:
        raise ValueError("geometry_mode must be 'paper' or 'rectangles'")

    print(f"\n[DEBUG] Processing {path}; geometry_mode={geometry_mode}", flush=True)
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no CRS")
        if not src.crs.is_projected:
            raise ValueError(
                f"{path} CRS {src.crs} is not projected; reproject to a metric CRS "
                "before using area/distance thresholds."
            )
        stage_started = time.perf_counter()
        print("[DEBUG] Stage: model inference starting", flush=True)
        prob = predict_raster(src, model, processor, args, device)
        print(f"[DEBUG] Stage: model inference finished in {time.perf_counter() - stage_started:.1f}s; "
              f"prob shape={prob.shape}, dtype={prob.dtype}", flush=True)
        transform, crs, profile = src.transform, src.crs, src.profile.copy()
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

    building_feats, rectangle_feats = [], []
    stage_started = time.perf_counter()
    shape_count = 0
    print(f"[DEBUG] Extracting polygons with {geometry_mode} geometry", flush=True)
    for geom_mapping, value in shapes(mask.astype(np.uint8), mask=mask,
                                      transform=transform, connectivity=8):
        shape_count += 1
        if value != 1:
            continue
        geom = shape(geom_mapping)
        if not geom.is_valid:
            geom = make_valid(geom)

        for poly in polygon_parts(geom):
            if poly.area < args.min_area_m2:
                continue

            if geometry_mode == "paper":
                # The paper algorithm owns its simplification/coarse/fine stages.
                # Do not pre-simplify or orthogonalize this input a second time.
                result = regularize_polygon(poly)
                if result is None or result.is_empty:
                    continue
                for final in polygon_parts(result):
                    if final.area < args.min_area_m2:
                        continue
                    building_feats.append({
                        "geometry": final,
                        "building_id": len(building_feats) + 1,
                        "source": path.name,
                        "area_m2": float(final.area),
                        "perimeter_m": float(final.length),
                        "segment_area_m2": float(poly.area),
                        "geometry_mode": "paper",
                        "threshold": args.threshold,
                    })
                continue

            # Rectangle mode: shared pre-cleanup, followed by rectangle
            # decomposition and dissolved perimeter generation.
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
                building_id = len(building_feats) + 1
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
                outline_union = unary_union(outlines).buffer(0)
                for final in polygon_parts(outline_union):
                    building_feats.append({
                        "geometry": final,
                        "building_id": building_id,
                        "source": path.name,
                        "area_m2": float(final.area),
                        "perimeter_m": float(final.length),
                        "segment_area_m2": float(building.area),
                        "rectangle_count": len(outlines),
                        "geometry_mode": "rectangles",
                        "threshold": args.threshold,
                    })
                # The current rectangle module returns dissolved footprint
                # polygons (not the individual internal rectangles).
                for idx, (outline, waste) in enumerate(decomposition, start=1):
                    rectangle_feats.append({
                        "geometry": outline,
                        "building_id": building_id,
                        "part_id": idx,
                        "source": path.name,
                        "area_m2": float(outline.area),
                        "waste_ratio": float(waste),
                        "threshold": args.threshold,
                    })

    print(f"[DEBUG] Geometry finished in {time.perf_counter() - stage_started:.1f}s; "
          f"shapes={shape_count}, buildings={len(building_feats)}", flush=True)
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
