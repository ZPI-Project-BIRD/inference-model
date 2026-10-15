# Building-dataset cropping script

This script creates train, validation, and test crop manifests from orthophoto TIFFs and their matching building-mask PNGs.

## Input files

Place TIFF images and masks in the area directory. Each mask should be named after its image with `_mask.png` appended to the image filename.

## Usage

```bash
python dataset_cropping/crop_building_dataset.py --areas path/to/area --output building_training
```

Replace `path/to/area` with the directory containing your images and masks. Use `--help` to see all options, including crop counts and `--train-only`.

The output directory contains the crop manifests (`train.csv`, `val.csv`, and `test.csv`) and source-tile reports.

## Crop distribution

You can change the crop-category distribution by editing `CATEGORY_FRACTIONS` in `crop_building_dataset.py`:

```python
CATEGORY_FRACTIONS = {"positive": 0.20, "mixed": 0.30, "background": 0.50}
```

These values set the exact category proportions for training crops and the sampling weights for validation and test crops.