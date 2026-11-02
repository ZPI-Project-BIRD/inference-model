#!/usr/bin/env python3
"""Train a SegFormer model to segment building footprints in aerial imagery."""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    SegformerModel,
    get_cosine_schedule_with_warmup,
)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SegFormer-B2 for building-footprint segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=Path("building_training"), help="Folder containing train.csv, val.csv and optional test.csv.")
    parser.add_argument("--output", type=Path, default=Path("runs/segformer_b2"))
    parser.add_argument("--model", default="nvidia/mit-b2", help="Pretrained SegFormer encoder checkpoint.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--accumulation", type=int, default=4, help="Number of batches per optimizer update.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-every", type=int, default=1, help="Save validation previews every N epochs.")
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pt in the output folder.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation) < 1:
        parser.error("epochs, batch-size and accumulation must be positive.")
    if args.workers < 0 or args.preview_every < 1 or args.preview_count < 1:
        parser.error("workers must be non-negative; preview settings must be positive.")
    if not 0 < args.warmup_ratio < 1:
        parser.error("warmup-ratio must be between 0 and 1.")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BuildingDataset(Dataset):
    """Read crop coordinates from a CSV and load image/mask pairs on demand."""

    def __init__(self, csv_path, processor, augment=False):
        self.rows = pd.read_csv(csv_path)
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = 90 * random.randint(1, 3)
            image = image.rotate(angle, expand=True)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)

        if random.random() < 0.7:
            pixels = np.asarray(image, dtype=np.float32)
            pixels = pixels * random.uniform(0.85, 1.15) + random.uniform(-12, 12)
            gray = (0.299 * pixels[..., 0] + 0.587 * pixels[..., 1] +
                    0.114 * pixels[..., 2])[..., None]
            pixels = gray + random.uniform(0.90, 1.10) * (pixels - gray)
            image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

        if random.random() < 0.15:
            pixels = np.asarray(image, dtype=np.float32)
            pixels += np.random.normal(0, 2, pixels.shape)
            image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
        return image, mask

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        x, y, size = (int(row[key]) for key in ("x", "y", "crop_size"))

        # PIL opens the TIFF and crops only the requested window.
        with Image.open(row["image"]) as source:
            image = source.convert("RGB").crop((x, y, x + size, y + size))
        with Image.open(row["mask"]) as source:
            mask = source.convert("L").crop((x, y, x + size, y + size))

        if self.augment:
            image, mask = self._augment(image, mask)

        labels = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.int64)
        encoded = self.processor(
            images=np.asarray(image),
            return_tensors="pt",
            do_resize=False,
        )
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": torch.from_numpy(labels).unsqueeze(0),
        }


class DiceBCELoss(nn.Module):
    """Combine pixel-wise binary cross-entropy with Dice loss."""

    def forward(self, logits, target):
        logits = F.interpolate(logits, size=target.shape[-2:],
                               mode="bilinear", align_corners=False)
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * target).sum(dim=(1, 2, 3))
        total = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = ((2 * intersection + 1) / (total + 1)).mean()
        return 0.5 * (1 - dice) + 0.5 * bce


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    intersection = predicted = actual = 0

    for batch in tqdm(loader, desc="validate", leave=False):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["labels"].to(device, non_blocking=True).bool()
        with autocast(device):
            logits = model(pixel_values=pixels).logits
            logits = F.interpolate(logits, size=target.shape[-2:],
                                   mode="bilinear", align_corners=False)
        prediction = torch.sigmoid(logits) >= 0.5
        intersection += (prediction & target).sum().item()
        predicted += prediction.sum().item()
        actual += target.sum().item()

    eps = 1e-7
    union = predicted + actual - intersection
    iou = intersection / max(union, eps)
    return {
        "iou": iou,
        "dice": 2 * intersection / max(predicted + actual, eps),
        "precision": intersection / max(predicted, eps),
        "recall": intersection / max(actual, eps),
    }


def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn,
                    device, accumulation):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    progress = tqdm(loader, desc="train", leave=False)

    for step, batch in enumerate(progress, start=1):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["labels"].to(device, non_blocking=True)
        with autocast(device):
            logits = model(pixel_values=pixels).logits
            loss = loss_fn(logits, target) / accumulation

        scaler.scale(loss).backward()
        if step % accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch_loss = loss.item() * accumulation
        total_loss += batch_loss
        progress.set_postfix(loss=f"{batch_loss:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / max(len(loader), 1)


def denormalize(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(image * STD + MEAN, 0, 1)


@torch.no_grad()
def save_previews(model, dataset, device, output_path, count=6):
    """Save image, label, prediction and error-overlay examples."""
    if not len(dataset):
        return
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
    fig, axes = plt.subplots(len(indices), 4, figsize=(14, 3.2 * len(indices)),
                             squeeze=False)

    for row_idx, index in enumerate(indices):
        item = dataset[int(index)]
        pixels = item["pixel_values"].unsqueeze(0).to(device)
        with autocast(device):
            logits = model(pixel_values=pixels).logits
        logits = F.interpolate(logits, size=item["labels"].shape[-2:],
                               mode="bilinear", align_corners=False)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        prediction = probability >= 0.5
        target = item["labels"][0].numpy().astype(bool)
        image = denormalize(item["pixel_values"])

        overlay = image.copy()
        overlay[target] = 0.65 * overlay[target] + 0.35 * np.array([0, 1, 0])
        overlay[prediction & ~target] = (
            0.65 * overlay[prediction & ~target] + 0.35 * np.array([0, 1, 0]))
        overlay[target & ~prediction] = (
            0.65 * overlay[target & ~prediction] + 0.35 * np.array([0, 1, 0]))

        panels = [
            (image, "Aerial image", None),
            (target, "Ground truth", "gray"),
            (prediction, f"Prediction (max {probability.max():.2f})", "gray"),
            (overlay, "Green=TP, red=FP, blue=FN", None),
        ]
        for axis, (picture, title, cmap) in zip(axes[row_idx], panels):
            axis.imshow(picture, cmap=cmap, vmin=0, vmax=1 if cmap else None)
            axis.set_title(title)
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_dataset_examples(dataset, output_path, count=8):
    """Save a small visual check of training images and their masks."""
    if not len(dataset):
        return
    indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 3 * len(indices)),
                             squeeze=False)
    for row_idx, index in enumerate(indices):
        item = dataset[int(index)]
        axes[row_idx, 0].imshow(denormalize(item["pixel_values"]))
        axes[row_idx, 0].set_title("Training crop")
        axes[row_idx, 1].imshow(item["labels"][0], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("Building mask")
        for axis in axes[row_idx]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_history(history, output_dir):
    if not history:
        return
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train loss")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(frame["epoch"], frame["iou"], label="IoU")
    axes[1].plot(frame["epoch"], frame["dice"], label="Dice")
    axes[1].plot(frame["epoch"], frame["precision"], label="Precision")
    axes[1].plot(frame["epoch"], frame["recall"], label="Recall")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_dashboard.png", dpi=150)
    plt.close(fig)


def save_json(data, path):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_model(model_name, device):
    """Load pretrained encoder weights and add a binary segmentation decoder."""
    processor = SegformerImageProcessor.from_pretrained(model_name, do_resize=False)
    encoder = SegformerModel.from_pretrained(model_name)
    config = SegformerConfig.from_pretrained(
        model_name, num_labels=1, id2label={0: "building"},
        label2id={"building": 0},
    )
    model = SegformerForSemanticSegmentation(config)
    model.segformer.load_state_dict(encoder.state_dict())
    del encoder
    model.to(device)
    model.gradient_checkpointing_enable()
    return model, processor


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_csv, val_csv = args.data / "train.csv", args.data / "val.csv"
    test_csv = args.data / "test.csv"
    for path in (train_csv, val_csv):
        if not path.exists():
            raise FileNotFoundError(path)

    model, processor = load_model(args.model, device)
    train_ds = BuildingDataset(train_csv, processor, augment=True)
    val_ds = BuildingDataset(val_csv, processor)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    if not len(train_loader):
        raise RuntimeError("Training loader is empty. Reduce batch size or add more crops.")

    print(f"Training crops: {len(train_ds):,} | validation crops: {len(val_ds):,}")
    save_dataset_examples(train_ds, args.output / "training_examples.png")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation)
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = DiceBCELoss()
    history, best_iou, start_epoch = [], -1.0, 1
    latest_path = args.output / "checkpoint_latest.pt"
    history_path = args.output / "history.json"

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_iou = checkpoint.get("best_iou", -1.0)
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"Resuming at epoch {start_epoch}; best IoU so far: {best_iou:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn,
            device, args.accumulation,
        )
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics,
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} | loss {train_loss:.4f} | "
            f"IoU {metrics['iou']:.4f} | Dice {metrics['dice']:.4f} | "
            f"precision {metrics['precision']:.4f} | recall {metrics['recall']:.4f}"
        )
        save_json(history, history_path)
        save_history(history, args.output)

        improved = metrics["iou"] > best_iou
        if epoch == 1 or epoch % args.preview_every == 0 or improved:
            preview_dir = args.output / "previews"
            preview_dir.mkdir(exist_ok=True)
            save_previews(model, val_ds, device,
                          preview_dir / f"epoch_{epoch:03d}.png",
                          args.preview_count)

        if improved:
            best_iou = metrics["iou"]
            model.save_pretrained(args.output / "best_model")
            processor.save_pretrained(args.output / "best_model")
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_iou": best_iou, "args": vars(args),
            }, args.output / "best.pt")
            print(f"  Saved new best model (IoU {best_iou:.4f})")

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_iou": best_iou, "args": vars(args),
        }, latest_path)

    print(f"Training finished. Best validation IoU: {best_iou:.4f}")
    print(f"Results saved to {args.output.resolve()}")

    if test_csv.exists() and (args.output / "best_model").exists():
        print("Evaluating best model on test set...")
        best_model = SegformerForSemanticSegmentation.from_pretrained(
            args.output / "best_model"
        ).to(device)
        test_ds = BuildingDataset(test_csv, processor)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        test_metrics = evaluate(best_model, test_loader, device)
        print("Test: " + " | ".join(f"{key} {value:.4f}"
                                   for key, value in test_metrics.items()))
        save_json(test_metrics, args.output / "test_metrics.json")


if __name__ == "__main__":
    main()
    