"""Poll recognition jobs; save one coordinate-only JSON array and one DB row per job."""
from __future__ import annotations

import json
import logging
import os
import re
import time
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psycopg
import torch
from dotenv import load_dotenv
from minio import Minio
from psycopg.types.json import Jsonb

from .model import load_model
from .pipeline import process_file

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("geo-inference-db-worker")


def required_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


DATABASE_URL = required_env("DATABASE_URL", "postgresql://app:app@localhost:5434/app")
MINIO_ENDPOINT = required_env("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = required_env("MINIO_ACCESS_KEY", "app")
MINIO_SECRET_KEY = required_env("MINIO_SECRET_KEY", "app12345")
MINIO_BUCKET = os.getenv("MINIO_SOURCE_BUCKET", "features")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in {"1", "true", "yes"}
MODEL_DIR = Path(required_env("MODEL_DIR"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results/database-worker")).expanduser()
DEVICE_SETTING = os.getenv("MODEL_DEVICE", "auto")
POLL_SECONDS = max(1.0, float(os.getenv("JOB_POLL_SECONDS", "3")))


def safe_name(value: str) -> str:
    """Make IDs safe to use as directory names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "item"


def make_model_options(output_dir: Path) -> SimpleNamespace:
    """Read inference options from environment variables."""
    return SimpleNamespace(
        output=output_dir,
        threshold=float(os.getenv("MODEL_THRESHOLD", "0.5")),
        tile_size=int(os.getenv("MODEL_TILE_SIZE", "512")),
        overlap=int(os.getenv("MODEL_OVERLAP", "64")),
        batch_size=int(os.getenv("MODEL_BATCH_SIZE", "4")),
        min_area_m2=float(os.getenv("MODEL_MIN_AREA_M2", "4.0")),
        hole_area_m2=float(os.getenv("MODEL_HOLE_AREA_M2", "1.0")),
        simplify_m=float(os.getenv("MODEL_SIMPLIFY_M", "0.05")),
        max_rectangle_waste=float(os.getenv("MODEL_MAX_RECTANGLE_WASTE", "0.18")),
        max_rectangle_parts=int(os.getenv("MODEL_MAX_RECTANGLE_PARTS", "8")),
        min_rectangle_area_m2=float(os.getenv("MODEL_MIN_RECTANGLE_AREA_M2", "2.0")),
        orthogonalize=os.getenv("MODEL_ORTHOGONALIZE", "false").lower() in {"1", "true", "yes"},
        orthogonal_angle_deg=float(os.getenv("MODEL_ORTHOGONAL_ANGLE_DEG", "10.0")),
        save_probability=os.getenv("MODEL_SAVE_PROBABILITY", "false").lower() in {"1", "true", "yes"},
        save_mask=os.getenv("MODEL_SAVE_MASK", "false").lower() in {"1", "true", "yes"},
    )


def make_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def claim_next_job() -> dict[str, Any] | None:
    """Atomically claim a ready, failed, or stale-started job.

    This follows the backend's eligibility rules and uses SKIP LOCKED so multiple
    workers can safely poll the same queue without claiming the same job.
    """
    sql = """
        WITH next_job AS (
            SELECT j.id
            FROM geoportal.recognition_jobs AS j
            JOIN geoportal.features AS f ON f.id = j.feature_id
            WHERE f.downloaded_at IS NOT NULL
              AND (
                j.status IN ('0_PENDING', '3_FAILED')
                OR (
                    j.status = '1_STARTED'
                    AND j.started_at < NOW() - INTERVAL '30 minutes'
                )
              )
            ORDER BY j.created_at ASC, j.id ASC
            FOR UPDATE OF j SKIP LOCKED
            LIMIT 1
        )
        UPDATE geoportal.recognition_jobs AS j
        SET status = '1_STARTED',
            started_at = NOW(),
            finished_at = NULL,
            error = NULL
        FROM next_job, geoportal.features AS f
        WHERE j.id = next_job.id
          AND f.id = j.feature_id
        RETURNING j.id, j.feature_id, f.object_name
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            if row is None:
                return None
            job_id, feature_id, object_name = row
            return {
                "id": str(job_id),
                "feature_id": feature_id,
                "object_name": object_name,
            }


def mark_job_failed(job_id: str, message: str) -> None:
    """Store a concise failure message and mark the job as failed."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE geoportal.recognition_jobs
                   SET status = '3_FAILED', finished_at = NOW(), error = %s
                   WHERE id = %s""",
                (message[:4000], job_id),
            )


def persist_detections(job_id: str, coordinates: list[Any], json_path: Path) -> None:
    """Persist one coordinate-only JSONB row per job and finish atomically in PostgreSQL."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(coordinates, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_json.replace(json_path)

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Replace older per-building rows when reprocessing an existing job.
            cur.execute(
                "DELETE FROM geoportal.detections WHERE recognition_job_id = %s",
                (job_id,),
            )
            cur.execute(
                """INSERT INTO geoportal.detections (recognition_job_id, bounds)
                   VALUES (%s, %s)""",
                (job_id, Jsonb(coordinates)),
            )
            cur.execute(
                """UPDATE geoportal.recognition_jobs
                   SET status = '2_FINISHED', finished_at = NOW(), error = NULL
                   WHERE id = %s""",
                (job_id,),
            )


def process_job(job: dict[str, Any], model: Any, processor: Any, device: torch.device,
                minio: Minio) -> None:
    job_id = job["id"]
    feature_id = job["feature_id"]
    object_name = job["object_name"]

    if not feature_id:
        raise ValueError("Recognition job has no feature_id; create one job per feature")
    if not object_name:
        raise ValueError(f"Feature {feature_id!r} has no object_name in geoportal.features")

    output_dir = RESULTS_DIR / safe_name(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Starting job=%s feature=%s object=%s", job_id, feature_id, object_name)

    # Keep the source object's filename so output filenames are recognisable.
    job_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="geo-inference-") as temp_dir:
        local_tiff = Path(temp_dir) / Path(object_name).name
        stage_started = time.perf_counter()
        log.info("[DEBUG] Job %s: downloading s3://%s/%s", job_id, MINIO_BUCKET, object_name)
        minio.fget_object(MINIO_BUCKET, object_name, str(local_tiff))
        log.info("[DEBUG] Job %s: download finished in %.1fs; local size=%d bytes",
                 job_id, time.perf_counter() - stage_started, local_tiff.stat().st_size)
        stage_started = time.perf_counter()
        log.info("[DEBUG] Job %s: calling process_file", job_id)
        features, crs = process_file(
            local_tiff, model, processor, make_model_options(output_dir), device
        )
        log.info("[DEBUG] Job %s: process_file returned in %.1fs; detections=%d; crs=%s",
                 job_id, time.perf_counter() - stage_started, len(features), crs)

    # Store only geometry coordinates; omit properties, CRS metadata, scores and IDs.
    all_coordinates = []
    for feature in features:
        geometry = feature.get("geometry")
        if geometry is None or geometry.is_empty:
            continue
        all_coordinates.append(geometry.__geo_interface__["coordinates"])

    json_path = output_dir / "detections.json"
    stage_started = time.perf_counter()
    log.info("[DEBUG] Job %s: saving %d geometries to %s and PostgreSQL",
             job_id, len(all_coordinates), json_path)
    persist_detections(job_id, all_coordinates, json_path)
    log.info("[DEBUG] Job %s: JSON + database save finished in %.1fs",
             job_id, time.perf_counter() - stage_started)
    log.info("Finished job=%s detections=%d output=%s total_elapsed=%.1fs",
             job_id, len(features), output_dir, time.perf_counter() - job_started)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if DEVICE_SETTING == "auto" and torch.cuda.is_available()
        else "cpu" if DEVICE_SETTING == "auto"
        else DEVICE_SETTING
    )
    log.info("Loading model from %s on %s", MODEL_DIR, device)
    model, processor = load_model(MODEL_DIR, device)
    minio = make_minio_client()

    # Fail early if the database is unreachable or the expected schema is absent.
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('geoportal.recognition_jobs'), to_regclass('geoportal.detections')")
            jobs_table, detections_table = cur.fetchone()
            if not jobs_table or not detections_table:
                raise RuntimeError(
                    "Missing geoportal.recognition_jobs or geoportal.detections; apply database migrations first"
                )

    log.info("Polling for pending recognition jobs every %.1f seconds", POLL_SECONDS)
    while True:
        try:
            job = claim_next_job()
            if job is None:
                time.sleep(POLL_SECONDS)
                continue
            try:
                process_job(job, model, processor, device, minio)
            except Exception as exc:
                log.exception("Recognition job %s failed", job["id"])
                try:
                    mark_job_failed(job["id"], f"{type(exc).__name__}: {exc}")
                except Exception:
                    log.exception("Could not mark job %s as failed", job["id"])
        except KeyboardInterrupt:
            log.info("Stopping database worker")
            break
        except Exception:
            log.exception("Database polling failed; retrying in %.1f seconds", POLL_SECONDS)
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
