# inference-model

## Dependecies

All in `requirements.txt`

## About

Training pipliene for the inference model for detecting buildings.

1. Download and build the dataset using scripts in dataset building. The script dowloads data from geoportal. First it dowloads the building data.
2. Create train, validation, and test crop manifests using the script in dataset cropping.
3. Run the training script using the crop manifests created in the previous step.

## Trained models

The following models were trained using this pipeline:

- [Building Segmentation Poland — B2](https://huggingface.co/McAntek/building-segmentation-poland-b2)
- [Building Segmentation Poland — B3](https://huggingface.co/McAntek/building-segmentation-poland-b3)


First I tested different avaible models I could find on Hugging Face. They did perform well, when it came to areas with a lot of buildings, but struggled when it came to country side areas with low amounts of buildings. 

Thus I decided to train our own model. While testing the models, I used data from geoportal. After a bit of reaserch I discovered that you can use the data from geoportal for model training.

Next up was finding a way to download the data, and geoportal actually has a feature for that - wcs. It dowloads high resulution orthophotomaps. I used this, along with the building map to crete a dataset of masks, that I can later on train the detection ai on. That is what the script in `dataset_building` is for.

The model I decided to use is the **Nvidia Segformer b3**. B3, bc its the biggest one that will train on a rtx 4060 in a resonable time. I'm still experimaenting with b5, but will see how that works out. The model takes in 512x512 images, so I needed to prepare the dataset further. I decided that instead of cutting each image and saving the cut parts, the better approach would be saving the cuts in a file. Cutting an image to a desired size is not computationally heavy, but saving each cut does take up a lot of storage space. So that is what `dataset cropping does`.

Once the crop manifests are ready, the training script reads each row and loads the corresponding image and mask crop directly from the original files. This means the dataset stores crop locations instead of creating a separate image file for every sample. The script starts with a SegFormer encoder pretrained on general image data, then adds a segmentation head that predicts whether each pixel belongs to a building. During training, it applies random flips and rotations to both the image and mask, as well as small changes to image brightness and color, to help the model handle variation in aerial imagery.

The model is trained using a combination of binary cross-entropy and Dice loss, which compares its pixel predictions with the building masks. After each training epoch, the script evaluates the model on the validation crops and records IoU, Dice, precision, and recall. It saves the checkpoint with the best validation IoU as the final model, while also keeping the latest checkpoint so training can be resumed. If a test manifest is available, the best model is evaluated on it after training; this gives a final performance check on data that was not used to update the model weights.

Right now the poligonazartion has two modes, will see what to do with them, one rectangular, one just aligning the sides of the polygon. 

Also more models in training, will udate readme when the training finishes.