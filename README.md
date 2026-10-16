# inference-model

## Dependecies

All in `requirements.txt`

## About

Training pipliene for the inference model for detecting buildings.

1. Download and build the dataset using scripts in dataset building.
2. Create train, validation, and test crop manifests using the script in dataset cropping.
3. Run the training script using the crop manifests created in the previous step.

## Trained models

The following models were trained using this pipeline:

- [Building Segmentation Poland — B2](https://huggingface.co/McAntek/building-segmentation-poland-b2)
- [Building Segmentation Poland — B3](https://huggingface.co/McAntek/building-segmentation-poland-b3)