# Geo Inference

Python worker for running building detection jobs from the Geoportal database.

## Run

From the inference repository root:

```bash
# Create an isolated environment and install the worker dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Download the documented trained B3 checkpoint
.venv/bin/hf download McAntek/building-segmentation-poland-b3 \
  --revision c950b74ced2867457054decc4264b4588329cb31 \
  --local-dir .cache/models/building-segmentation-poland-b3

# Create your local configuration (the example targets local Floci/PostgreSQL)
cp .env.example .env

# Check the Python files
.venv/bin/python -m py_compile \
  geo_inference/database_worker.py \
  geo_inference/pipeline.py

# Start the database worker
PYTHONUNBUFFERED=1 .venv/bin/python -u -m geo_inference.database_worker
```

The worker polls PostgreSQL for pending recognition jobs, downloads the source
GeoTIFF from the configured object-storage bucket, runs building detection,
and saves the polygon coordinates to the database and to a `detections.json`
file under `RESULTS_DIR`.

Stop the worker with `Ctrl+C`.

## Configuration

Copy `.env.example` to `.env` and adjust the values for your environment.
Keep `.env` local; do not commit credentials or local paths.

For the local stack, Floci exposes its S3-compatible API at
`localhost.floci.io:4566` over HTTP with `test`/`test` credentials. The
downloader uses `S3_SOURCE_BUCKET=features`, so the worker must use the same
bucket through `MINIO_SOURCE_BUCKET=features`. The endpoint value must not
include the URL scheme. If the worker itself runs in Docker, replace the host
connection values with Docker-reachable hostnames while keeping the bucket
name unchanged.
