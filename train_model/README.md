# Building segmentation training

This script trains a SegFormer model to detect building footprints in aerial images. It reads crop manifests created by the dataset-cropping script.

## Requirements

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Usage

Run from the repository root:

```bash
python train_model/train_buildings.py --data building_training --output runs/segformer_b2
```

The data directory must contain `train.csv` and `val.csv`. If it also contains `test.csv`, the script evaluates the best model on the test set after training.

Use `--help` to see all options. To continue training from the latest checkpoint, add `--resume`.

## Output

The output directory contains model checkpoints, training history, validation previews, and metrics. Training uses a CUDA GPU if available; otherwise, it runs on CPU.