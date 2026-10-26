# Modular building-footprint inference

## Modules
- `geo_inference/model.py`: SegFormer loading and tiled inference.
- `geo_inference/geometry.py`: polygon cleanup and rectangle decomposition.
- `geo_inference/outputs.py`: preview images and GeoJSON writing.
- `geo_inference/pipeline.py`: end-to-end processing of one GeoTIFF; exports preview PNG, GeoJSON and GeoPackage layers.
- `geo_inference/cli.py`: local-file CLI.
- `geo_inference/geodown_consumer.py`: RabbitMQ + MinIO adapter, no PostgreSQL dependency.

## Local use
`python -m geo_inference.cli --model /path/to/best_model --input /path/to/file.tif --output ./output`