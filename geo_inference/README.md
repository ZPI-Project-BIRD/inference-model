# Geo Inference

Python worker for running building detection jobs from the Geoportal database.

## Run

From the `inference-model` directory:

```bash
# Create your local configuration
cp .env.example .env

# Edit .env if your paths or connection settings differ
nano .env

# Install dependencies (if not already installed)
.venv/bin/pip install -r requirements.txt

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
