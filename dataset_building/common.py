from pathlib import Path
import zipfile
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import box

CRS = "EPSG:2180"
WGS84 = "EPSG:4326"
BDOT_BASE_URL = "https://opendata.geoportal.gov.pl/bdot10k/schemat2021"
BUILDING_CLASS = "OT_BUBD_A"
ALL_TOUCHED = True
NUM_PREVIEWS = 20

def wgs84_to_2180_bbox(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
):
    transformer = Transformer.from_crs(
        WGS84,
        CRS,
        always_xy=True,
    )

    x1, y1 = transformer.transform(lon1, lat1)
    x2, y2 = transformer.transform(lon2, lat2)

    return (
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    )

def bdot_package_url(teryt: str) -> str:
    if len(teryt) < 4:
        raise ValueError(
            f"Unexpected TERYT county code: {teryt}"
        )

    woj = teryt[:2]

    return (
        f"{BDOT_BASE_URL}/"
        f"{woj}/{teryt}_GML.zip"
    )

def extract_building_gml(
    zip_path: Path,
    destination_dir: Path,
):
    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"  Inspecting {zip_path.name}"
    )

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()

        candidates = [
            name
            for name in names
            if (
                BUILDING_CLASS in Path(name).name.upper()
                and (
                    name.lower().endswith(".gml")
                    or name.lower().endswith(".xml")
                )
            )
        ]

        if not candidates:
            raise RuntimeError(
                f"Could not find {BUILDING_CLASS} "
                f"in {zip_path}"
            )

        print(
            f"  Found {len(candidates)} building file(s):"
        )

        extracted = []

        for name in candidates:
            print(
                f"    {name}"
            )

            archive.extract(
                name,
                destination_dir,
            )

            extracted.append(
                destination_dir / name
            )

        return extracted

def load_buildings(gml_files, bbox):
    print()
    print("=" * 70)
    print("LOADING BUILDING POLYGONS")
    print("=" * 70)

    aoi_geom = box(*bbox)

    frames = []

    for gml in gml_files:
        print()
        print(
            f"Reading: {gml}"
        )

        buildings = gpd.read_file(
            gml
        )

        print(
            f"  Objects: {len(buildings):,}"
        )

        print(
            f"  CRS: {buildings.crs}"
        )

        if buildings.crs is None:
            buildings = buildings.set_crs(
                CRS
            )
        elif buildings.crs.to_epsg() != 2180:
            buildings = buildings.to_crs(
                CRS
            )

        buildings = buildings[
            buildings.geometry.notna()
        ].copy()

        buildings = buildings[
            ~buildings.geometry.is_empty
        ].copy()

        buildings = buildings[
            buildings.geometry.intersects(
                aoi_geom
            )
        ].copy()

        print(
            f"  Intersecting AOI: "
            f"{len(buildings):,}"
        )

        if not buildings.empty:
            frames.append(
                buildings[
                    ["geometry"]
                ].copy()
            )

    if not frames:
        raise RuntimeError(
            "No OT_BUBD_A buildings intersect the AOI."
        )

    buildings = gpd.GeoDataFrame(
        pd.concat(
            frames,
            ignore_index=True,
        ),
        geometry="geometry",
        crs=CRS,
    )

    buildings = buildings[
        buildings.geometry.notna()
    ].copy()

    buildings = buildings[
        ~buildings.geometry.is_empty
    ].copy()

    invalid = ~buildings.geometry.is_valid

    if invalid.any():
        print()
        print(
            f"Repairing {invalid.sum():,} "
            "invalid geometries..."
        )

        buildings.loc[
            invalid,
            "geometry"
        ] = (
            buildings.loc[
                invalid,
                "geometry"
            ].buffer(0)
        )

    buildings = buildings[
        buildings.geometry.notna()
    ].copy()

    buildings = buildings[
        ~buildings.geometry.is_empty
    ].copy()

    print()
    print(
        f"Final building geometries: {len(buildings):,}"
    )
    print(f"Building CRS used for masks: {buildings.crs}")
    print(f"Building bounds (EPSG:2180): {tuple(round(v, 3) for v in buildings.total_bounds)}")

    return buildings

def save_buildings(
    buildings,
    output_path: Path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Saving building polygons:"
    )
    print(
        f"  {output_path}"
    )

    buildings.to_file(
        output_path,
        layer=BUILDING_CLASS,
        driver="GPKG",
    )

def create_masks(
    buildings,
    imagery_dir,
    mask_dir,
    preview_dir,
):
    print()
    print("=" * 70)
    print("CREATING BUILDING MASKS")
    print("=" * 70)

    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = sorted(
        imagery_dir.glob("*.tif")
    )

    if not image_files:
        raise RuntimeError(
            "No orthophoto TIFFs found."
        )

    for old_preview in Path(preview_dir).glob("*_mask.png"):
        old_preview.unlink(missing_ok=True)

    summary = []

    preview_count = 0

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path.name}"
        )

        mask_path = (
            mask_dir
            / f"{image_path.stem}_mask.png"
        )

        mask_path.unlink(missing_ok=True)

        with rasterio.open(
            image_path
        ) as src:
            width = src.width
            height = src.height
            transform = src.transform
            crs = src.crs

            if crs.to_epsg() != 2180:
                raise RuntimeError(
                    f"Unexpected imagery CRS "
                    f"for {image_path}: {crs}"
                )

            tile_polygon = box(
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            )

            tile_buildings = buildings[
                buildings.geometry.intersects(
                    tile_polygon
                )
            ]

            print(
                f"  Buildings: "
                f"{len(tile_buildings)}"
            )

            if tile_buildings.empty:
                mask = np.zeros(
                    (height, width),
                    dtype=np.uint8,
                )

            else:
                shapes = [
                    (geom, 1)
                    for geom
                    in tile_buildings.geometry
                    if (
                        geom is not None
                        and not geom.is_empty
                    )
                ]

                mask = rasterize(
                    shapes=shapes,
                    out_shape=(
                        height,
                        width,
                    ),
                    transform=transform,
                    fill=0,
                    default_value=1,
                    dtype="uint8",
                    all_touched=ALL_TOUCHED,
                )

        building_pixels = int(
            np.count_nonzero(mask)
        )

        percentage = (
            100
            * building_pixels
            / mask.size
        )

        print(
            f"  Building pixels: "
            f"{building_pixels:,} "
            f"({percentage:.2f}%)"
        )

        Image.fromarray(
            mask * 255,
            mode="L",
        ).save(
            mask_path
        )

        if preview_count < NUM_PREVIEWS:
            preview_path = (
                preview_dir
                / mask_path.name
            )

            Image.fromarray(
                mask * 255,
                mode="L",
            ).save(
                preview_path
            )

            preview_count += 1

        summary.append({
            "image": str(image_path),
            "mask": str(mask_path),
            "width": width,
            "height": height,
            "buildings": len(tile_buildings),
            "building_pixels": building_pixels,
            "building_percentage": percentage,
        })

    summary_df = pd.DataFrame(
        summary
    )

    summary_path = (
        mask_dir / "mask_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    nonempty = summary_df[
        summary_df["building_pixels"] > 0
    ]

    print()
    print("=" * 70)
    print("MASKS COMPLETE")
    print("=" * 70)

    print(
        f"Tiles: "
        f"{len(summary_df):,}"
    )

    print(
        f"Tiles with buildings: "
        f"{len(nonempty):,}"
    )

    print(
        f"Empty tiles: "
        f"{len(summary_df) - len(nonempty):,}"
    )

    print(
        f"Total building pixels: "
        f"{summary_df['building_pixels'].sum():,}"
    )

    print(
        f"Summary: {summary_path}"
    )
