"""
api/main.py
-----------
FastAPI application for the meter anomaly detection service.

Endpoints
---------
POST /detect
    Main inference endpoint. Accepts one or more raw HES API records,
    runs the full detection pipeline on each, returns structured results.

GET  /health
    Liveness check. Returns service status and model load state.

GET  /model/info
    Returns model artifact metadata (feature schema, thresholds).

POST /model/reload
    Hot-reloads model artifacts from disk without restarting the service.
    Use after retraining.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from api.schemas import (
    DetectRequest,
    DetectBatchResponse,
    DetectResponse,
    DetectionLayers,
    RuleLayerResult,
    ZScoreLayerResult,
    IFLayerResult,
)
from config.settings import MODEL_PATHS, ROLLING_WINDOW_SIZE
from pipeline import run as run_pipeline
from pipeline.if_detector import reload_artifacts

# --------------- logging ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api.main")


# =========================================================
# DB — imported conditionally so the API works without
# a live DB (uses in-memory history fallback).
# =========================================================

try:
    from db.client import (
        init_schema,
        get_last_n_readings,
        insert_raw_reading,
        insert_telemetry,
        insert_anomaly,
        close_pool,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    logger.warning("psycopg2 not available — running without DB persistence.")


# =========================================================
# LIFESPAN — startup / shutdown
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────
    logger.info("Starting meter anomaly detection service ...")

    # Pre-load model artifacts (fail fast if missing)
    try:
        reload_artifacts()
        logger.info("Model artifacts loaded.")
    except FileNotFoundError as e:
        logger.error(f"Model artifacts missing: {e}")
        # Service starts but /detect will return errors until models exist

    # Initialise DB schema if DB is available
    if _DB_AVAILABLE:
        try:
            init_schema()
            logger.info("Database schema verified.")
        except Exception as e:
            logger.warning(f"DB unavailable at startup: {e}. Continuing without DB.")

    yield

    # ── Shutdown ──────────────────────────────────────────
    if _DB_AVAILABLE:
        try:
            close_pool()
        except Exception:
            pass
    logger.info("Service shut down.")


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Meter Anomaly Detection API",
    description=(
        "Detects anomalies in smart meter telemetry using a three-layer pipeline: "
        "rule-based checks, z-score statistical detection, and Isolation Forest ML."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# HELPERS
# =========================================================

def _fetch_history(meter_serial: str, before_timestamp: str) -> list[dict]:
    """
    Fetches the last N readings for a meter from DB.
    Falls back to an empty list if DB is unavailable —
    the pipeline handles missing history gracefully (uses
    current reading for rolling stats).
    """
    if not _DB_AVAILABLE:
        return []

    try:
        return get_last_n_readings(
            meter_serial=meter_serial,
            n=ROLLING_WINDOW_SIZE,
            before_timestamp=before_timestamp,
        )
    except Exception as e:
        logger.warning(f"Could not fetch history for {meter_serial}: {e}")
        return []


def _persist(record, parsed_interval_ts: str, result) -> None:
    """
    Writes raw record, canonical telemetry, and anomaly log
    to DB. Errors are logged but never bubble up to the caller —
    persistence failures must not affect detection responses.
    """
    if not _DB_AVAILABLE:
        return

    try:
        # 1. Raw record
        insert_raw_reading(
            id=record.id,
            meter_serial=record.meterSerial,
            received_at=record.timestamp,
            profile_obis_code=record.obisCode,
            entry_id=record.entryId,
            raw_value=record.rawValue,
        )

        # 2. Parsed telemetry — store canonical dict in raw_data
        if result.features and not result.error:
            # Extract only the canonical electrical values
            # (not derived features — those are recomputed at inference)
            canonical_fields = [
                "energy_consumption", "voltage", "current", "power_factor",
                "apparent_import_energy", "active_export_energy",
                "reactive_import_energy", "reactive_export_energy",
                "active_import_power", "active_export_power", "frequency",
            ]
            raw_data = {
                k: result.features[k]
                for k in canonical_fields
                if k in result.features and result.features[k] is not None
            }
            insert_telemetry(
                meter_serial=record.meterSerial,
                interval_timestamp=parsed_interval_ts,
                raw_data=raw_data,
                received_at=record.timestamp,
                source_raw_id=record.id,
            )

        # 3. Anomaly log (only if flagged)
        if result.is_anomaly and not result.error:
            rb = result.rule_based
            zs = result.zscore
            if_r = result.isolation_forest

            insert_anomaly(
                meter_serial=record.meterSerial,
                interval_timestamp=parsed_interval_ts,
                rule_based_flag=rb.get("is_anomaly", False),
                zscore_flag=zs.get("is_anomaly", False),
                if_flag=if_r.get("is_anomaly", False),
                if_score=if_r.get("anomaly_score"),
                zscore_value=zs.get("z_score"),
                rule_violations=rb.get("violations"),
                feature_snapshot=result.features,
            )

    except Exception as e:
        logger.error(f"DB persistence failed for record {record.id}: {e}")


def _build_response(result) -> DetectResponse:
    """Converts a PipelineResult into a DetectResponse."""

    if result.error:
        return DetectResponse(
            meter_serial=result.meter_serial,
            interval_timestamp=result.interval_timestamp,
            is_anomaly=False,
            error=result.error,
        )

    rb   = result.rule_based
    zs   = result.zscore
    if_r = result.isolation_forest

    layers = DetectionLayers(
        rule_based=RuleLayerResult(
            layer=rb["layer"],
            is_anomaly=rb["is_anomaly"],
            violations=rb.get("violations", []),
            details=rb.get("details", {}),
        ),
        zscore=ZScoreLayerResult(
            layer=zs["layer"],
            is_anomaly=zs["is_anomaly"],
            z_score=zs.get("z_score"),
            spike_ratio=zs.get("spike_ratio"),
            triggers=zs.get("triggers", []),
            details=zs.get("details", {}),
        ),
        isolation_forest=IFLayerResult(
            layer=if_r.get("layer", "isolation_forest"),
            is_anomaly=if_r.get("is_anomaly", False),
            anomaly_score=if_r.get("anomaly_score"),
            prediction=if_r.get("prediction"),
        ),
    )

    return DetectResponse(
        meter_serial=result.meter_serial,
        interval_timestamp=result.interval_timestamp,
        is_anomaly=result.is_anomaly,
        layers=layers,
        features=result.features,
        error=None,
    )


# =========================================================
# ENDPOINTS
# =========================================================

@app.post(
    "/detect",
    response_model=DetectBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Run anomaly detection on one or more meter records",
    tags=["Detection"],
)
async def detect(request: DetectRequest) -> DetectBatchResponse:
    """
    Accepts a batch of raw HES API records and runs the full
    three-layer detection pipeline on each:

    1. **Rule-based** — deterministic checks (negative energy,
       voltage out of range, invalid power factor, etc.)
    2. **Z-score** — statistical deviation from rolling baseline
    3. **Isolation Forest** — multivariate ML anomaly detection

    Each record is processed independently. A record is flagged
    as anomalous if **any** layer fires.

    History for rolling features is fetched from the database
    per meter. If the DB is unavailable, the pipeline falls back
    to using the current reading only (reduced feature quality).
    """
    responses = []

    for record in request.records:
        # ── Fetch rolling history from DB ─────────────────
        history = _fetch_history(
            meter_serial=record.meterSerial,
            before_timestamp=record.timestamp,
        )

        # ── Run pipeline ──────────────────────────────────
        result = run_pipeline(
            api_record=record.model_dump(),
            history=history,
        )

        # ── Persist to DB ─────────────────────────────────
        _persist(record, result.interval_timestamp, result)

        # ── Build response ────────────────────────────────
        responses.append(_build_response(result))

    n_anomalies = sum(1 for r in responses if r.is_anomaly)

    return DetectBatchResponse(
        total=len(responses),
        anomalies=n_anomalies,
        results=responses,
    )


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Service liveness check",
    tags=["Ops"],
)
async def health() -> dict:
    """
    Returns service status and component availability.
    """
    # Check model artifacts
    models_ok = all(
        os.path.exists(os.path.abspath(p))
        for p in MODEL_PATHS.values()
    )

    # Check DB
    db_ok = False
    if _DB_AVAILABLE:
        try:
            from db.client import get_connection
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            db_ok = True
        except Exception:
            db_ok = False

    return {
        "status":    "ok" if models_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "model_artifacts": "ok" if models_ok else "missing",
            "database":        "ok" if db_ok else ("unavailable" if _DB_AVAILABLE else "not_configured"),
        },
    }


@app.get(
    "/model/info",
    status_code=status.HTTP_200_OK,
    summary="Model and feature schema info",
    tags=["Ops"],
)
async def model_info() -> dict:
    """
    Returns the feature schema the model was trained on,
    detection thresholds, and artifact file paths.
    """
    schema_path = os.path.abspath(MODEL_PATHS["feature_schema"])
    if not os.path.exists(schema_path):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not found. Run training/train.py first.",
        )

    schema = joblib.load(schema_path)

    from config.settings import DETECTION_CONFIG, ROLLING_WINDOW_SIZE
    return {
        "feature_schema":    schema,
        "detection_config":  DETECTION_CONFIG,
        "rolling_window":    ROLLING_WINDOW_SIZE,
        "artifact_paths":    {k: os.path.abspath(v) for k, v in MODEL_PATHS.items()},
    }


@app.post(
    "/model/reload",
    status_code=status.HTTP_200_OK,
    summary="Hot-reload model artifacts from disk",
    tags=["Ops"],
)
async def model_reload() -> dict:
    """
    Reloads all model artifacts (Isolation Forest, scaler,
    impute values, feature schema) from disk without restarting
    the service. Call this after retraining.
    """
    try:
        reload_artifacts()
        return {
            "status":  "reloaded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "artifacts": {k: os.path.abspath(v) for k, v in MODEL_PATHS.items()},
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )