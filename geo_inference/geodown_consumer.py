"""geodown RabbitMQ/MinIO adapter. Writes files; does not require PostgreSQL."""
import json, logging, os, re, tempfile, time
from pathlib import Path
from types import SimpleNamespace
import pika, torch
from dotenv import load_dotenv
from minio import Minio
from .model import load_model
from .pipeline import process_file

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("geodown-inference")
def env(name, default=None):
    value = os.getenv(name, default)
    if not value: raise RuntimeError(f"Missing required environment variable: {name}")
    return value
RABBITMQ_URL = env("RABBITMQ_URL", "amqp://app:app@localhost:5672/")
QUEUE = os.getenv("DETECTION_QUEUE", "detect.feature")
MINIO_ENDPOINT = env("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = env("MINIO_ACCESS_KEY", "app")
MINIO_SECRET_KEY = env("MINIO_SECRET_KEY", "app12345")
MINIO_BUCKET = os.getenv("MINIO_SOURCE_BUCKET", "features")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in {"1", "true", "yes"}
MODEL_DIR = Path(env("MODEL_DIR"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results/geodown")).expanduser()
DEVICE_SETTING = os.getenv("MODEL_DEVICE", "auto")
def safe_name(value): return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "feature"
def make_options(output):
    """Model settings, overridable through environment variables."""
    return SimpleNamespace(output=output, threshold=float(os.getenv("MODEL_THRESHOLD", ".5")),
        tile_size=int(os.getenv("MODEL_TILE_SIZE", "512")), overlap=int(os.getenv("MODEL_OVERLAP", "64")),
        batch_size=int(os.getenv("MODEL_BATCH_SIZE", "4")), min_area_m2=float(os.getenv("MODEL_MIN_AREA_M2", "4")),
        hole_area_m2=float(os.getenv("MODEL_HOLE_AREA_M2", "1")), simplify_m=float(os.getenv("MODEL_SIMPLIFY_M", ".05")),
        max_rectangle_waste=float(os.getenv("MODEL_MAX_RECTANGLE_WASTE", ".18")),
        max_rectangle_parts=int(os.getenv("MODEL_MAX_RECTANGLE_PARTS", "8")),
        min_rectangle_area_m2=float(os.getenv("MODEL_MIN_RECTANGLE_AREA_M2", "2")),
        orthogonalize=os.getenv("MODEL_ORTHOGONALIZE", "false").lower() in {"1", "true", "yes"},
        orthogonal_angle_deg=float(os.getenv("MODEL_ORTHOGONAL_ANGLE_DEG", "10")),
        save_probability=os.getenv("MODEL_SAVE_PROBABILITY", "false").lower() in {"1", "true", "yes"},
        save_mask=os.getenv("MODEL_SAVE_MASK", "false").lower() in {"1", "true", "yes"})
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if DEVICE_SETTING == "auto" and torch.cuda.is_available() else "cpu" if DEVICE_SETTING == "auto" else DEVICE_SETTING)
    log.info("Loading model %s on %s", MODEL_DIR, device)
    model, processor = load_model(MODEL_DIR, device)
    minio = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)
    params = pika.URLParameters(RABBITMQ_URL); params.heartbeat = 60; params.blocked_connection_timeout = 30
    def on_message(channel, method, properties, body):
        try:
            message = json.loads(body.decode("utf-8")); feature_id = str(message["featureId"]); object_name = str(message["objectName"])
            output = RESULTS_DIR / safe_name(feature_id); output.mkdir(parents=True, exist_ok=True)
            log.info("Processing %s (%s)", feature_id, object_name)
            with tempfile.TemporaryDirectory(prefix="geodown-inference-") as temp_dir:
                local_tiff = Path(temp_dir) / Path(object_name).name
                minio.fget_object(MINIO_BUCKET, object_name, str(local_tiff))
                features, crs = process_file(local_tiff, model, processor, make_options(output), device)
            manifest = {"featureId": feature_id, "objectName": object_name, "crs": str(crs), "buildingCount": len(features),
                        "geojson": f"{local_tiff.stem}_buildings.geojson", "preview": f"{local_tiff.stem}_outline_preview.png",
                        "geopackage": f"{local_tiff.stem}_buildings.gpkg"}
            (output / "results.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            channel.basic_ack(delivery_tag=method.delivery_tag)
            log.info("Done: %s buildings=%d output=%s", feature_id, len(features), output)
        except Exception:
            log.exception("Could not process geodown message; rejecting without requeue to avoid hot-loop")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    while True:
        try:
            connection = pika.BlockingConnection(params); channel = connection.channel()
            channel.queue_declare(queue=QUEUE, durable=True); channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=QUEUE, on_message_callback=on_message, auto_ack=False)
            log.info("Listening on %s; output directory %s", QUEUE, RESULTS_DIR.resolve()); channel.start_consuming()
        except KeyboardInterrupt:
            log.info("Stopping consumer"); break
        except Exception:
            log.exception("RabbitMQ connection failed; retrying in 5 seconds"); time.sleep(5)
if __name__ == "__main__": main()
