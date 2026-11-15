# inference-model

## Dependencies

All in `requirements.txt`

## Run the inference database worker

The worker requires:

- PostgreSQL on `localhost:5432` with the Geoportal migrations applied;
- Floci on `http://localhost.floci.io:4566`;
- downloaded feature TIFFs and pending recognition jobs in the database.

From this repository's root, create the environment and download the trained B3
checkpoint:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/hf download McAntek/building-segmentation-poland-b3 \
  --revision c950b74ced2867457054decc4264b4588329cb31 \
  --local-dir .cache/models/building-segmentation-poland-b3
```

Create the local configuration and start the worker:

```bash
cp .env.example .env
.venv/bin/python -u -m geo_inference.database_worker
```

The checked-in example is ready for the local stack. In particular, these
values connect the worker to Floci's feature storage:

```dotenv
MINIO_ENDPOINT=localhost.floci.io:4566
MINIO_ACCESS_KEY=test
MINIO_SECRET_KEY=test
MINIO_SOURCE_BUCKET=features
MINIO_SECURE=false
```

`MINIO_ENDPOINT` is a `host:port` value without an `http://` prefix. The
downloader writes each `<feature-id>.tif` object to its `S3_SOURCE_BUCKET`,
which is `features` in the local Floci configuration, and stores that object
key in `geoportal.features.object_name`. The inference worker therefore must
use `MINIO_SOURCE_BUCKET=features` as well; `tiles` is a different bucket.

The worker validates the bucket and database schema at startup, loads the model
from `MODEL_DIR`, and then immediately begins claiming pending jobs. Results are
stored in PostgreSQL and under `RESULTS_DIR`. Stop it with `Ctrl+C`.

See [`geo_inference/README.md`](geo_inference/README.md) for the configuration
and processing details.

## About

Training pipliene for the inference model for detecting buildings.

1. Download and build the dataset using scripts in dataset building. The script dowloads data from geoportal. First it dowloads the building data.
2. Create train, validation, and test crop manifests using the script in dataset cropping.
3. Run the training script using the crop manifests created in the previous step.

## Trained models

The following models were trained using this pipeline:

- [Building Segmentation Poland — B2](https://huggingface.co/McAntek/building-segmentation-poland-b2)
- [Building Segmentation Poland — B3](https://huggingface.co/McAntek/building-segmentation-poland-b3)
- **[Building Segmentation Poland — B5](https://huggingface.co/McAntek/building-segmentation-poland-b5)**



First I tested different avaible models I could find on Hugging Face. They did perform well, when it came to areas with a lot of buildings, but struggled when it came to country side areas with low amounts of buildings. 

Thus I decided to train our own model. While testing the models, I used data from geoportal. After a bit of reaserch I discovered that you can use the data from geoportal for model training.

Next up was finding a way to download the data, and geoportal actually has a feature for that - wcs. It dowloads high resulution orthophotomaps. I used this, along with the building map to crete a dataset of masks, that I can later on train the detection ai on. That is what the script in `dataset_building` is for.

The model I decided to use is the **Nvidia Segformer b3**. B3, bc its the biggest one that will train on a rtx 4060 in a resonable time. I'm still experimaenting with b5, but will see how that works out. The model takes in 512x512 images, so I needed to prepare the dataset further. I decided that instead of cutting each image and saving the cut parts, the better approach would be saving the cuts in a file. Cutting an image to a desired size is not computationally heavy, but saving each cut does take up a lot of storage space. So that is what `dataset cropping does`.

Once the crop manifests are ready, the training script reads each row and loads the corresponding image and mask crop directly from the original files. This means the dataset stores crop locations instead of creating a separate image file for every sample. The script starts with a SegFormer encoder pretrained on general image data, then adds a segmentation head that predicts whether each pixel belongs to a building. During training, it applies random flips and rotations to both the image and mask, as well as small changes to image brightness and color, to help the model handle variation in aerial imagery.

The model is trained using a combination of binary cross-entropy and Dice loss, which compares its pixel predictions with the building masks. After each training epoch, the script evaluates the model on the validation crops and records IoU, Dice, precision, and recall. It saves the checkpoint with the best validation IoU as the final model, while also keeping the latest checkpoint so training can be resumed. If a test manifest is available, the best model is evaluated on it after training; this gives a final performance check on data that was not used to update the model weights.

Right now the poligonazartion has two modes, will see what to do with them, one rectangular, one just aligning the sides of the polygon. 

Also more models in training, will udate readme when the training finishes.
