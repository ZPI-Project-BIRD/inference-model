import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import mapping

def save_quicklook(rgb, prob, mask, features, transform, out_path, threshold):
    from rasterio.features import rasterize

    geometries = [(f["geometry"], 1) for f in features]
    if geometries:
        outline_raster = rasterize(
            geometries,
            out_shape=mask.shape,
            transform=transform,
            fill=0,
            default_value=1,
            all_touched=False,
            dtype="uint8",
        )
    else:
        outline_raster = np.zeros(mask.shape, dtype=np.uint8)

    max_dim = 1800
    scale = max(1, int(math.ceil(max(mask.shape) / max_dim)))
    rgb_v = rgb[::scale, ::scale]
    prob_v = prob[::scale, ::scale]
    mask_v = mask[::scale, ::scale]
    outline_v = outline_raster[::scale, ::scale]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb_v)
    axes[0].set_title("Orthophoto")
    axes[1].imshow(prob_v, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Building probability")
    axes[2].imshow(mask_v, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Binary mask (threshold {threshold:.2f})")
    axes[3].imshow(rgb_v)
    axes[3].contour(outline_v, levels=[0.5], colors=["red"], linewidths=0.6)
    axes[3].set_title(f"Rectangle-derived outlines ({len(features)} polygons)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_results_json(features, output_path, source_name, crs):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "type": "FeatureCollection",
        "name": source_name,
        "crs": {"type": "name", "properties": {"name": str(crs)}},
        "features": [
            {"type": "Feature", "geometry": mapping(item["geometry"]),
             "properties": {k: v for k, v in item.items() if k != "geometry"}}
            for item in features
        ],
    }
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
