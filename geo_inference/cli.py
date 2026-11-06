"""CLI for running inference on local GeoTIFFs."""
import argparse, sys
from pathlib import Path
from .model import DEVICE_CHOICES, describe_device, load_model, resolve_device
from .pipeline import lonlat_box, process_file

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Saved best_model directory")
    p.add_argument("--input", type=Path, required=True, help="Input GeoTIFF or directory of GeoTIFFs")
    p.add_argument("--output", type=Path, required=True, help="Output directory")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--min-area-m2", type=float, default=4.0)
    p.add_argument("--hole-area-m2", type=float, default=1.0)
    p.add_argument("--simplify-m", type=float, default=0.05)
    p.add_argument("--max-rectangle-waste", type=float, default=0.18,
                   help="Split a building when this fraction of its enclosing rotated rectangle is outside the building (0-1).")
    p.add_argument("--max-rectangle-parts", type=int, default=8,
                   help="Maximum rectangles used to approximate one building.")
    p.add_argument("--min-rectangle-area-m2", type=float, default=2.0,
                   help="Discard decomposition pieces smaller than this area.")
    p.add_argument("--orthogonalize", action="store_true",
                   help="Snap near-horizontal/vertical edges; can distort rotated buildings")
    p.add_argument("--orthogonal-angle-deg", type=float, default=10.0)
    p.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                   help="auto = CUDA if present, else Apple GPU (MPS), else CPU")
    p.add_argument("--save-probability", action="store_true")
    p.add_argument("--save-mask", action="store_true")
    p.add_argument("--clip-bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                   help="Only detect inside this EPSG:4326 lon/lat rectangle (e.g. a project area).")
    return p.parse_args()

def iter_tiffs(path):
    if path.is_file():
        if path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Expected GeoTIFF: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted([x for x in path.iterdir() if x.suffix.lower() in (".tif", ".tiff")])
    if not files:
        raise FileNotFoundError(f"No TIFF files in {path}")
    return files

def main():
    args = parse_args()
    if not 0 < args.threshold < 1: raise ValueError("--threshold must be between 0 and 1")
    if args.tile_size < 32 or args.overlap < 0 or args.overlap >= args.tile_size: raise ValueError("Require tile-size >= 32 and 0 <= overlap < tile-size")
    if args.batch_size < 1 or args.min_area_m2 < 0 or args.hole_area_m2 < 0 or args.simplify_m < 0: raise ValueError("Invalid batch size / area / simplification value")
    if not 0 <= args.max_rectangle_waste < 1: raise ValueError("--max-rectangle-waste must be in [0, 1)")
    if args.max_rectangle_parts < 1 or args.min_rectangle_area_m2 < 0: raise ValueError("Invalid rectangle decomposition settings")
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(describe_device(device))
    model, processor = load_model(args.model, device)
    clip_area = lonlat_box(*args.clip_bbox) if args.clip_bbox else None
    all_features, merged_crs = [], None
    for path in iter_tiffs(args.input):
        features, crs = process_file(path, model, processor, args, device, clip_area=clip_area)
        if merged_crs is None: merged_crs = crs
        if crs == merged_crs: all_features.extend(features)
        else: print(f"Not adding {path.name} to merged output: CRS differs", file=sys.stderr)
    if all_features:
        import geopandas as gpd
        merged = gpd.GeoDataFrame(all_features, geometry="geometry", crs=merged_crs)
        output = args.output / "all_buildings.gpkg"
        merged.to_file(output, layer="buildings", driver="GPKG")
        from .outputs import save_results_json
        save_results_json(all_features, args.output / "all_buildings.geojson", "all_buildings", merged_crs)
        print(f"Merged polygons: {output} ({len(merged)} features)")
    print(f"Done. Output directory: {args.output}")

if __name__ == "__main__": main()
