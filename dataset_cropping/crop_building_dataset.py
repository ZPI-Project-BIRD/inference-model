#!/usr/bin/env python3
"""Prepare image/mask crop manifests for building-segmentation training."""
from __future__ import annotations

import argparse
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageEnhance, ImageFilter
from rasterio.windows import Window

CROP_SIZE = 512
BLOCK_SIZE_M = 1_000
SEED = 67
DEFAULT_CROPS = {"train": 25_000, "val": 3_000, "test": 3_000}
CATEGORY_FRACTIONS = {"positive": 0.70, "mixed": 0.10, "background": 0.20}
POSITIVE_MIN, MIXED_MIN = 0.05, 0.002
MIN_VALID_FRACTION, MIN_BRIGHTNESS, MAX_BRIGHTNESS = 0.98, 10, 245


def find_images(area: Path) -> list[Path]:
    return sorted(p for p in area.rglob("*.tif") if "mask" not in p.stem.lower())


def find_mask(image: Path) -> Path | None:
    name = f"{image.stem}_mask.png"
    candidates = [image.parent.parent / "masks" / name, image.parent / "masks" / name, image.with_name(name)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    root = image
    for _ in range(3):
        root = root.parent
    return next(iter(root.rglob(name)), None)


@lru_cache(maxsize=32)
def _read_mask(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def read_mask(path: str | Path) -> np.ndarray:
    return _read_mask(str(Path(path).resolve()))


def image_stats(path: Path) -> dict:
    with rasterio.open(path) as src:
        data = src.read(indexes=list(range(1, min(src.count, 3) + 1))).astype(np.float32)
        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        gray = data.mean(axis=0)
        return {
            "width": src.width, "height": src.height, "crs": str(src.crs),
            "xmin": src.bounds.left, "ymin": src.bounds.bottom,
            "xmax": src.bounds.right, "ymax": src.bounds.top,
            "valid_fraction": float(np.isfinite(data).all(axis=0).mean()),
            "black_fraction": float(np.all(data == 0, axis=0).mean()),
            "brightness": float(data.mean()),
            "clipped_low": float((data <= 2).mean()),
            "clipped_high": float((data >= 253).mean()),
            "gradient_energy": float((np.diff(gray, axis=1) ** 2).mean() + (np.diff(gray, axis=0) ** 2).mean()),
        }


def tile_is_good(stats: dict) -> bool:
    return stats["valid_fraction"] >= MIN_VALID_FRACTION and MIN_BRIGHTNESS <= stats["brightness"] <= MAX_BRIGHTNESS


def geographic_block(stats: dict) -> tuple[int, int]:
    x = (stats["xmin"] + stats["xmax"]) / 2
    y = (stats["ymin"] + stats["ymax"]) / 2
    return math.floor(x / BLOCK_SIZE_M), math.floor(y / BLOCK_SIZE_M)


def assign_splits(records: list[dict], seed: int) -> dict[tuple, str]:
    blocks = sorted({(r["area"], r["block_x"], r["block_y"]) for r in records})
    random.Random(seed).shuffle(blocks)
    if len(blocks) < 6:
        print("WARNING: fewer than 6 geographic blocks; split reliability is limited.")
    n_test = max(1, round(len(blocks) * 0.15))
    n_val = max(1, round(len(blocks) * 0.15))
    if n_test + n_val >= len(blocks):
        n_test, n_val = 1, 1
    return {block: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, block in enumerate(blocks)}


def crop_fraction(mask: np.ndarray, x: int, y: int, size: int) -> float:
    return float(mask[y:y + size, x:x + size].mean())


def matches_category(fraction: float, category: str) -> bool:
    if category == "positive":
        return fraction >= POSITIVE_MIN
    if category == "mixed":
        return MIXED_MIN <= fraction < POSITIVE_MIN
    if category == "background":
        return fraction < MIXED_MIN
    raise ValueError(f"Unknown category: {category}")


def find_crop(mask: np.ndarray, size: int, category: str, rng: random.Random, attempts: int = 250):
    """Find a crop in the requested class, returning (x, y, building_fraction) or None."""
    height, width = mask.shape
    if height < size or width < size:
        return None
    max_x, max_y = width - size, height - size
    for _ in range(attempts):
        x, y = rng.randint(0, max_x), rng.randint(0, max_y)
        fraction = crop_fraction(mask, x, y, size)
        if matches_category(fraction, category):
            return x, y, fraction
    stride = 128
    xs, ys = list(range(0, max_x + 1, stride)), list(range(0, max_y + 1, stride))
    if xs[-1] != max_x: xs.append(max_x)
    if ys[-1] != max_y: ys.append(max_y)
    positions = [(x, y) for y in ys for x in xs]
    rng.shuffle(positions)
    for x, y in positions:
        fraction = crop_fraction(mask, x, y, size)
        if matches_category(fraction, category):
            return x, y, fraction
    return None


def usable_tiles(records: list[dict], split: str) -> list[dict]:
    candidates = [r for r in records if r["split"] == split]
    usable = []
    for record in candidates:
        try:
            with rasterio.open(record["image"]) as src:
                if src.width >= CROP_SIZE and src.height >= CROP_SIZE:
                    usable.append(record)
        except Exception as exc:
            print(f"WARNING: cannot read {record['image']}: {exc}")
    if len(usable) < len(candidates):
        print(f"  Skipped {len(candidates) - len(usable)} undersized/unreadable {split} tiles.")
    if not usable:
        raise RuntimeError(f"No usable source tiles for split '{split}'.")
    return usable


def make_manifest(records: list[dict], split: str, count: int, output_csv: Path, seed: int, strict_train: bool = False) -> pd.DataFrame:
    candidates = usable_tiles(records, split)
    rng = random.Random(seed)
    if strict_train:
        targets = {"positive": round(count * CATEGORY_FRACTIONS["positive"]), "mixed": round(count * CATEGORY_FRACTIONS["mixed"])}
        targets["background"] = count - targets["positive"] - targets["mixed"]
        requested = [category for category, n in targets.items() for _ in range(n)]
        rng.shuffle(requested)
        print("Training targets: " + ", ".join(f"{k}={v:,}" for k, v in targets.items()))
    else:
        requested = [rng.choices(list(CATEGORY_FRACTIONS), weights=list(CATEGORY_FRACTIONS.values()))[0] for _ in range(count)]

    rows = []
    failed: dict[str, set[str]] = {category: set() for category in CATEGORY_FRACTIONS}
    print(f"Creating {split} manifest: {count:,} crops from {len(candidates):,} tiles")
    for i, category in enumerate(requested, 1):
        pool = [r for r in candidates if r["image"] not in failed[category]]
        found = None
        for _ in range(min(100, max(len(pool), 1))):
            if not pool:
                break
            record = rng.choice(pool)
            result = find_crop(read_mask(record["mask"]), CROP_SIZE, category, rng)
            if result is not None:
                x, y, fraction = result
                found = record, x, y, fraction
                break
            failed[category].add(record["image"])
            pool = [r for r in pool if r["image"] != record["image"]]
        if found is None:
            raise RuntimeError(f"Could not find a '{category}' crop for {split}. Check that your masks contain enough examples of each class.")
        record, x, y, fraction = found
        rows.append({"area": record["area"], "image": record["image"], "mask": record["mask"], "split": split,
                     "x": x, "y": y, "crop_size": CROP_SIZE, "category": category, "building_fraction": fraction,
                     "block_x": record["block_x"], "block_y": record["block_y"]})
        if i % 1000 == 0:
            print(f"  {i:,}/{count:,}")

    df = pd.DataFrame(rows)
    if strict_train:
        actual = df["category"].value_counts().reindex(targets.keys(), fill_value=0).to_dict()
        if actual != targets:
            raise RuntimeError(f"Training class counts differ from targets: expected {targets}, got {actual}")
        for row in df.itertuples():
            if not matches_category(row.building_fraction, row.category):
                raise RuntimeError(f"Invalid crop category in generated manifest: {row.category}")
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df):,} rows to {output_csv}")
    return df


class BuildingCropDataset:
    """PyTorch Dataset returning image tensors (3,H,W) and binary mask tensors (H,W)."""
    def __init__(self, manifest_csv: str | Path, augment: bool = False):
        self.df = pd.read_csv(manifest_csv)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load_crop(self, row):
        x, y, size = int(row.x), int(row.y), int(row.crop_size)
        with rasterio.open(row.image) as src:
            image = src.read(indexes=[1, 2, 3], window=Window(x, y, size, size))
        mask = np.asarray(Image.open(row.mask).convert("L"))[y:y + size, x:x + size]
        image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image, (mask > 0).astype(np.uint8) * 255

    def _augment(self, image, mask):
        rng = random.Random()
        if rng.random() < 0.5:
            image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
        if rng.random() < 0.5:
            image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
        turns = rng.randint(0, 3)
        if turns:
            image, mask = np.rot90(image, turns).copy(), np.rot90(mask, turns).copy()
        if rng.random() < 0.35:
            angle = rng.uniform(-15, 15)
            image = np.asarray(Image.fromarray(image).rotate(angle, resample=Image.Resampling.BICUBIC))
            mask = np.asarray(Image.fromarray(mask).rotate(angle, resample=Image.Resampling.NEAREST))
        picture = Image.fromarray(image)
        if rng.random() < 0.50: picture = ImageEnhance.Brightness(picture).enhance(rng.uniform(0.85, 1.15))
        if rng.random() < 0.50: picture = ImageEnhance.Contrast(picture).enhance(rng.uniform(0.85, 1.20))
        if rng.random() < 0.35: picture = ImageEnhance.Color(picture).enhance(rng.uniform(0.85, 1.15))
        if rng.random() < 0.20: picture = picture.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7)))
        image = np.asarray(picture).astype(np.float32)
        if rng.random() < 0.20:
            image = 255 * np.power(np.clip(image / 255, 0, 1), rng.uniform(0.85, 1.15))
        if rng.random() < 0.20:
            image += np.random.normal(0, rng.uniform(1, 4), image.shape)
        return np.clip(image, 0, 255).astype(np.uint8), (mask > 127).astype(np.uint8)

    def __getitem__(self, index):
        import torch
        row = self.df.iloc[index]
        image, mask = self._load_crop(row)
        if self.augment:
            image, mask = self._augment(image, mask)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        mask = torch.from_numpy(mask.copy()).float()
        return image, mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/validation/test crop manifests from orthophoto TIFFs and building-mask PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python prepare_building_dataset.py --areas milicz włóki --output building_training",
    )
    parser.add_argument("--areas", nargs="+", default=["dataset"], help="Directories containing orthophoto TIFFs and matching masks.")
    parser.add_argument("--output", type=Path, default=Path("building_training"), help="Directory for CSV manifests and tile reports.")
    parser.add_argument("--train-crops", type=int, default=DEFAULT_CROPS["train"], help="Number of training crops to sample.")
    parser.add_argument("--val-crops", type=int, default=DEFAULT_CROPS["val"], help="Number of validation crops to sample.")
    parser.add_argument("--test-crops", type=int, default=DEFAULT_CROPS["test"], help="Number of test crops to sample.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducible sampling and splits.")
    parser.add_argument("--train-only", action="store_true", help="Regenerate train.csv only; leave existing val.csv and test.csv untouched.")
    args = parser.parse_args()
    if min(args.train_crops, args.val_crops, args.test_crops) < 1:
        parser.error("crop counts must be positive integers")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    for area_arg in args.areas:
        area = Path(area_arg)
        if not area.exists():
            print(f"WARNING: area directory does not exist: {area}")
            continue
        images = find_images(area)
        print(f"{area}: found {len(images):,} TIFF images")
        for image in images:
            mask = find_mask(image)
            if mask is None:
                print(f"  MISSING MASK: {image}")
                continue
            stats = image_stats(image)
            block_x, block_y = geographic_block(stats)
            records.append({"area": area.name, "image": str(image.resolve()), "mask": str(mask.resolve()),
                            "block_x": block_x, "block_y": block_y, "good_tile": tile_is_good(stats), **stats})

    if not records:
        raise RuntimeError("No image/mask pairs found. Check --areas and the expected mask naming/layout.")
    pd.DataFrame(records).to_csv(args.output / "source_tiles.csv", index=False)
    good = [r for r in records if r["good_tile"]]
    if not good:
        raise RuntimeError("No tiles passed quality checks (valid pixels and brightness).")
    split_map = assign_splits(good, args.seed)
    for record in good:
        record["split"] = split_map[(record["area"], record["block_x"], record["block_y"])]
    split_df = pd.DataFrame(good)
    split_df.to_csv(args.output / "source_tiles_with_splits.csv", index=False)
    print(f"Matched pairs: {len(records):,}; quality-passing tiles: {len(good):,}")
    print("Tiles by split:\n" + split_df["split"].value_counts().to_string())

    train = make_manifest(good, "train", args.train_crops, args.output / "train.csv", args.seed + 1, strict_train=True)
    if args.train_only:
        print("Train manifest rebuilt; existing validation and test manifests were left untouched.")
        return
    val = make_manifest(good, "val", args.val_crops, args.output / "val.csv", args.seed + 2)
    test = make_manifest(good, "test", args.test_crops, args.output / "test.csv", args.seed + 3)
    for name, df in (("TRAIN", train), ("VAL", val), ("TEST", test)):
        print(f"\n{name}: {len(df):,} crops")
        print(df["category"].value_counts().to_string())
        print(f"Mean building fraction: {df['building_fraction'].mean():.4f}")
    print(f"\nOutput directory: {args.output.resolve()}")


if __name__ == "__main__":
    main()
