#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from train_model.train_buildings import BuildingDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SegFormer at different thresholds."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("building_training"),
        help="Dataset directory containing val.csv.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/segformer_b2/best_model"),
        help="Path to the saved best_model directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("threshold_results"),
        help="Directory for results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[
            0.20, 0.25, 0.30, 0.35, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65,
            0.70, 0.75, 0.80,
        ],
    )
    parser.add_argument(
		"--max-images",
		type=int,
		default=200,
		help="Maximum number of validation images to evaluate (0 = all)"
	)
    return parser.parse_args()


def main():
    args = parse_args()

    if not (args.data / "val.csv").exists():
        raise FileNotFoundError(args.data / "val.csv")

    if not (args.model / "config.json").exists():
        raise FileNotFoundError(
            f"Could not find model config in {args.model}"
        )

    if any(not 0 < t < 1 for t in args.thresholds):
        raise ValueError("All thresholds must be between 0 and 1.")

    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Validation dataset: {args.data / 'val.csv'}")

    processor = SegformerImageProcessor.from_pretrained(
        str(args.model)
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(args.model)
    ).to(device)
    model.eval()

    dataset = BuildingDataset(
        args.data / "val.csv",
        processor,
        augment=False,
    )

    if args.max_images > 0:
        dataset = torch.utils.data.Subset(
            dataset,
            range(min(args.max_images, len(dataset)))
        )

    print(f"Evaluating {len(dataset)} validation images")

    loader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=device.type == "cuda",
	)

    if len(dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")

    counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in args.thresholds
    }

    per_image_rows = []

    with torch.inference_mode():
        sample_index = 0

        for batch_index, batch in enumerate(loader, start=1):
            pixels = batch["pixel_values"].to(
                device, non_blocking=True
            )
            targets = batch["labels"].to(
                device, non_blocking=True
            ).bool()

            logits = model(pixel_values=pixels).logits
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.sigmoid(logits)

            for i in range(pixels.shape[0]):
                target = targets[i, 0]
                probability = probabilities[i, 0]

                for threshold in args.thresholds:
                    prediction = probability >= threshold

                    tp = (prediction & target).sum().item()
                    fp = (prediction & ~target).sum().item()
                    fn = (~prediction & target).sum().item()

                    counts[threshold]["tp"] += tp
                    counts[threshold]["fp"] += fp
                    counts[threshold]["fn"] += fn

                    precision = tp / (tp + fp) if tp + fp else 0.0
                    recall = tp / (tp + fn) if tp + fn else 0.0
                    dice_denominator = 2 * tp + fp + fn
                    iou_denominator = tp + fp + fn

                    dice = (
                        2 * tp / dice_denominator
                        if dice_denominator else 1.0
                    )
                    iou = (
                        tp / iou_denominator
                        if iou_denominator else 1.0
                    )

                    per_image_rows.append({
                        "image_index": sample_index,
                        "threshold": threshold,
                        "precision": precision,
                        "recall": recall,
                        "dice": dice,
                        "iou": iou,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    })

                sample_index += 1

            print(
                f"Processed batch {batch_index}/{len(loader)}"
            )

    rows = []

    for threshold in args.thresholds:
        tp = counts[threshold]["tp"]
        fp = counts[threshold]["fp"]
        fn = counts[threshold]["fn"]

        rows.append({
            "threshold": threshold,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "dice": (
                2 * tp / (2 * tp + fp + fn)
                if 2 * tp + fp + fn else 1.0
            ),
            "iou": (
                tp / (tp + fp + fn)
                if tp + fp + fn else 1.0
            ),
            "TP": tp,
            "FP": fp,
            "FN": fn,
        })

    results = pd.DataFrame(rows).sort_values("threshold")
    results.to_csv(
        args.output / "threshold_metrics.csv",
        index=False,
    )

    pd.DataFrame(per_image_rows).to_csv(
        args.output / "per_image_metrics.csv",
        index=False,
    )

    print("\n=== Global validation metrics ===")
    print(
        results[
            ["threshold", "precision", "recall", "dice", "iou"]
        ].to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "dice": "{:.4f}".format,
                "iou": "{:.4f}".format,
            },
        )
    )

    best_iou = results.loc[results["iou"].idxmax()]
    best_dice = results.loc[results["dice"].idxmax()]
    best_precision = results.loc[
        results["precision"].idxmax()
    ]

    print(
        f"\nBest IoU: {best_iou['iou']:.4f} "
        f"at threshold {best_iou['threshold']:.2f}"
    )
    print(
        f"Best Dice: {best_dice['dice']:.4f} "
        f"at threshold {best_dice['threshold']:.2f}"
    )
    print(
        f"Best precision: {best_precision['precision']:.4f} "
        f"at threshold {best_precision['threshold']:.2f}"
    )

    print(f"\nResults saved in: {args.output.resolve()}")


if __name__ == "__main__":
    main()