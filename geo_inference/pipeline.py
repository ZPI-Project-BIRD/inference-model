"""Process one georeferenced raster from prediction through export."""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.validation import make_valid
from shapely.ops import unary_union
from .model import predict_raster, to_rgb
from .geometry import polygon_parts, fill_small_holes, orthogonalize, decompose_building
from .outputs import save_quicklook, save_results_json

def process_file(path, model, processor, args, device):
    print(f"\nProcessing {path}")
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no CRS")
        if not src.crs.is_projected:
            raise ValueError(f"{path} CRS {src.crs} is not projected; reproject to EPSG:2180 before reporting m².")
        prob = predict_raster(src, model, processor, args, device)
        transform, crs, profile = src.transform, src.crs, src.profile.copy()
        # Build a display RGB image from the full raster for the PNG quicklook.
        rgb_full = to_rgb(src.read())
        pixel_area = abs(transform.a * transform.e)
        if pixel_area <= 0:
            raise ValueError("Invalid raster pixel size")

    mask = prob >= args.threshold
    if args.save_probability:
        prof = profile.copy()
        prof.update(count=1, dtype="float32", nodata=None, compress="deflate")
        with rasterio.open(args.output / f"{path.stem}_probability.tif", "w", **prof) as dst:
            dst.write(prob.astype(np.float32), 1)
    if args.save_mask:
        prof = profile.copy()
        prof.update(count=1, dtype="uint8", nodata=0, compress="deflate")
        with rasterio.open(args.output / f"{path.stem}_mask.tif", "w", **prof) as dst:
            dst.write(mask.astype(np.uint8), 1)

    building_feats = []
    rectangle_feats = []
    from shapely.ops import unary_union
    for mapping, value in shapes(mask.astype(np.uint8), mask=mask,
                                 transform=transform, connectivity=8):
        if value != 1:
            continue
        geom = shape(mapping)
        if not geom.is_valid:
            geom = make_valid(geom)
        for poly in polygon_parts(geom):
            if poly.area < args.min_area_m2:
                continue
            poly = fill_small_holes(poly, args.hole_area_m2)
            if args.simplify_m > 0:
                poly = poly.simplify(args.simplify_m, preserve_topology=True)
            if args.orthogonalize:
                poly = orthogonalize(poly, args.orthogonal_angle_deg)
            if not poly.is_valid:
                poly = make_valid(poly)
            for building in polygon_parts(poly):
                if building.area < args.min_area_m2:
                    continue
                building_id = len(building_feats) + 1
                decomposition = decompose_building(
                    building, args.max_rectangle_waste, args.max_rectangle_parts,
                    args.min_rectangle_area_m2)
                rects = [r for r, _ in decomposition]
                # Build the exported outline from the rectangle decomposition.
                # Intersecting with the original mask here would erase the rectangle
                # approximation and make the result look identical to segmentation.
                rectangle_coverage = unary_union(rects).buffer(0)
                outline = rectangle_coverage
                for idx, (rect, waste) in enumerate(decomposition, start=1):
                    rectangle_feats.append({
                        "geometry": rect, "building_id": building_id,
                        "part_id": idx, "source": path.name,
                        "area_m2": float(rect.area), "waste_ratio": float(waste),
                        "threshold": args.threshold
                    })
                for final in polygon_parts(outline):
                    building_feats.append({
                        "geometry": final, "building_id": building_id,
                        "source": path.name, "area_m2": float(final.area),
                        "perimeter_m": float(final.length),
                        "segment_area_m2": float(building.area),
                        "rectangle_count": len(rects), "threshold": args.threshold
                    })
    feats = building_feats
    png_path = args.output / f"{path.stem}_outline_preview.png"
    save_quicklook(rgb_full, prob, mask, feats, transform, png_path, args.threshold)
    print(f"  PNG preview: {png_path}")

    import geopandas as gpd
    gdf = gpd.GeoDataFrame(feats, geometry="geometry", crs=crs)
    out = args.output / f"{path.stem}_buildings.gpkg"
    gdf.to_file(out, layer="buildings", driver="GPKG")
    rect_out = gpd.GeoDataFrame(rectangle_feats, geometry="geometry", crs=crs)
    rect_out.to_file(out, layer="rectangles", driver="GPKG", mode="a")
    print(f"  Building outlines: {len(gdf)}; rectangles: {len(rect_out)} -> {out}")
    json_out = args.output / f"{path.stem}_buildings.geojson"
    save_results_json(feats, json_out, path.name, crs)
    print(f"  GeoJSON: {json_out}")
    return feats, crs
