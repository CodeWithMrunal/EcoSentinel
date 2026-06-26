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
from time import perf_counter
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from api.schemas import (
    DetectRequest,
    DetectBatchResponse,
    DetectResponse,
    DetectionLayers,
    RuleLayerResult,
    ZScoreLayerResult,
    IFLayerResult,
    AnomalyExplanationResponse,
)
from config.settings import MODEL_PATHS, ROLLING_WINDOW_SIZE, DECISION_ENGINE_CONFIG
from pipeline import run as run_pipeline
from pipeline.feature_engineer import summarize_rolling_state
from pipeline.if_detector import reload_artifacts
from decision_engine.service import run_explanation_task
from fastapi.middleware.cors import CORSMiddleware

# --------------- logging ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api.main")


def _summarize_layer_flags(result) -> str:
    flags = {
        "rule_based": result.rule_based.get("is_anomaly") if result.rule_based else None,
        "zscore": result.zscore.get("is_anomaly") if result.zscore else None,
        "isolation_forest": result.isolation_forest.get("is_anomaly") if result.isolation_forest else None,
    }
    return ", ".join(f"{name}={value}" for name, value in flags.items())


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
        get_anomaly_by_id,
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


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
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


def _persist(record, parsed_interval_ts: str, result) -> Optional[int]:
    """
    Writes raw record, canonical telemetry, and anomaly log
    to DB. Errors are logged but never bubble up to the caller —
    persistence failures must not affect detection responses.

    Returns
    -------
    anomaly_log.id if an anomaly was logged, else None.
    Used by the caller to schedule a decision-engine background task.
    """
    if not _DB_AVAILABLE:
        return None

    anomaly_id = None

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
                flagged_anomalous=bool(result.is_anomaly),
                source_raw_id=record.id,
            )

        # 3. Anomaly log (only if flagged)
        if result.is_anomaly and not result.error:
            rb = result.rule_based
            zs = result.zscore
            if_r = result.isolation_forest

            explanation_status = (
                "pending" if DECISION_ENGINE_CONFIG["enabled"] else None
            )

            anomaly_id = insert_anomaly(
                meter_serial=record.meterSerial,
                interval_timestamp=parsed_interval_ts,
                rule_based_flag=rb.get("is_anomaly", False),
                zscore_flag=zs.get("is_anomaly", False),
                if_flag=if_r.get("is_anomaly", False),
                if_score=if_r.get("anomaly_score"),
                zscore_value=zs.get("z_score"),
                rule_violations=rb.get("violations"),
                feature_snapshot=result.features,
                explanation_status=explanation_status,
            )

    except Exception as e:
        logger.error(f"DB persistence failed for record {record.id}: {e}")

    return anomaly_id


def _log_baseline_snapshot(prefix: str, canonical: dict, history: list[dict], interval_ts: str) -> None:
    summary = summarize_rolling_state(canonical, history, interval_ts, include_current=False)
    logger.info(
        f"{prefix} Historical Sample Count={summary['history_sample_count']}; Rolling Mean={summary['rolling_mean']}; Rolling Standard Deviation={summary['rolling_std']}; Same-Hour Historical Average={summary['historical_avg_same_hour']}."
    )
    logger.info(
        f"{prefix} Updated Sample Count={summary['sample_count']}; Updated Mean={summary['rolling_mean']}; Updated Standard Deviation={summary['rolling_std']}."
    )

def _build_response(
    result,
    anomaly_id: Optional[int] = None,
) -> DetectResponse:
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
            model_used=if_r.get("model_used"),
            features_used=if_r.get("features_used"),
        ),
    )

    explanation_status = None
    if result.is_anomaly and anomaly_id is not None:
        explanation_status = (
            "pending" if DECISION_ENGINE_CONFIG["enabled"] else None
        )

    return DetectResponse(
        meter_serial=result.meter_serial,
        interval_timestamp=result.interval_timestamp,
        is_anomaly=result.is_anomaly,
        layers=layers,
        features=result.features,
        error=None,
        anomaly_id=anomaly_id if result.is_anomaly else None,
        explanation_status=explanation_status,
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
async def detect(
    request: DetectRequest,
    background_tasks: BackgroundTasks,
) -> DetectBatchResponse:
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

    **Decision Engine**: if a record is flagged anomalous and the
    decision engine is enabled, an LLM explanation is generated
    asynchronously as a background task after this response is
    returned. The response includes `anomaly_id` and
    `explanation_status: "pending"` — poll
    `GET /anomalies/{anomaly_id}/explanation` for the result.
    """
    batch_started = perf_counter()
    logger.info(f"/detect received {len(request.records)} record(s).")

    responses = []

    for index, record in enumerate(request.records, start=1):
        record_started = perf_counter()
        logger.info(
            f"[{index}/{len(request.records)}] Processing record id={record.id} meter={record.meterSerial} entry={record.entryId}."
        )

        history_started = perf_counter()
        history = _fetch_history(
            meter_serial=record.meterSerial,
            before_timestamp=record.timestamp,
        )
        logger.info(
            f"[{record.meterSerial}] Retrieved {len(history)} historical reading(s) in {(perf_counter() - history_started) * 1000:.1f} ms."
        )

        pipeline_started = perf_counter()
        result = run_pipeline(
            api_record=record.model_dump(),
            history=history,
        )

        if result.features and not result.error:
            _log_baseline_snapshot(
                prefix=f"[{record.meterSerial}] Before persistence:",
                canonical=result.features,
                history=history,
                interval_ts=result.interval_timestamp,
            )

        logger.info(
            f"[{record.meterSerial}] Pipeline completed in {(perf_counter() - pipeline_started) * 1000:.1f} ms; anomaly={result.is_anomaly}, error={result.error!r}."
        )

        persist_started = perf_counter()
        anomaly_id = _persist(record, result.interval_timestamp, result)
        logger.info(
            f"[{record.meterSerial}] Persistence finished in {(perf_counter() - persist_started) * 1000:.1f} ms; anomaly_id={anomaly_id}."
        )

        if result.features and not result.error:
            post_history = _fetch_history(
                meter_serial=record.meterSerial,
                before_timestamp=None,
            )
            _log_baseline_snapshot(
                prefix=f"[{record.meterSerial}] After persistence:",
                canonical=result.features,
                history=post_history,
                interval_ts=result.interval_timestamp,
            )

        if (
            result.is_anomaly
            and anomaly_id is not None
            and DECISION_ENGINE_CONFIG["enabled"]
            and not result.error
        ):
            rb   = result.rule_based
            zs   = result.zscore
            if_r = result.isolation_forest

            background_tasks.add_task(
                run_explanation_task,
                anomaly_id=anomaly_id,
                meter_serial=result.meter_serial,
                interval_timestamp=result.interval_timestamp,
                features=result.features,
                rule_violations=rb.get("violations", []),
                zscore_value=zs.get("z_score"),
                zscore_triggers=zs.get("triggers", []),
                if_score=if_r.get("anomaly_score"),
                if_model_used=if_r.get("model_used"),
            )
            logger.info(
                f"[{record.meterSerial}] Scheduled Decision Engine explanation task for anomaly_id={anomaly_id}."
            )

        logger.info(
            f"[{record.meterSerial}] Record finished in {(perf_counter() - record_started) * 1000:.1f} ms; layers={_summarize_layer_flags(result)}."
        )

        responses.append(_build_response(result, anomaly_id=anomaly_id))

    n_anomalies = sum(1 for r in responses if r.is_anomaly)
    logger.info(
        f"/detect completed in {(perf_counter() - batch_started) * 1000:.1f} ms with {n_anomalies}/{len(responses)} anomaly result(s)."
    )

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
    logger.info(f"/health checked model artifacts -> {'ok' if models_ok else 'missing'}.")

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
        logger.warning("/model/info requested but feature schema artifact is missing.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not found. Run training/train.py first.",
        )

    raw_schema = joblib.load(schema_path)
    schema = raw_schema["all_features"] if isinstance(raw_schema, dict) else raw_schema
    logger.info(f"/model/info returned feature schema with {len(schema)} feature(s).")

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
        logger.info("/model/reload requested.")
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
    
@app.get(
    "/anomalies/{anomaly_id}/explanation",
    response_model=AnomalyExplanationResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch the Decision Engine explanation for a flagged anomaly",
    tags=["Decision Engine"],
)
async def get_anomaly_explanation(anomaly_id: int) -> AnomalyExplanationResponse:
    """
    Fetches the LLM-generated explanation for a previously flagged
    anomaly. Explanations are generated asynchronously after
    POST /detect returns, so this endpoint may need to be polled.

    `explanation_status` values:
      - "pending"   — LLM call is queued or in progress; poll again shortly
      - "completed" — `explanation` field is populated
      - "failed"    — `explanation_error` describes what went wrong;
                      the anomaly detection itself is still valid,
                      only the explanation generation failed
      - null        — decision engine was disabled when this anomaly
                      was detected; no explanation will be generated

    Typical polling interval: 2-5 seconds, depending on LLM provider
    latency (local Ollama models typically take 3-15 seconds).
    """
    if not _DB_AVAILABLE:
        logger.warning(f"/anomalies/{anomaly_id}/explanation requested but DB is unavailable.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available — explanations are not persisted without a DB.",
        )

    row = get_anomaly_by_id(anomaly_id)
    logger.info(
        f"Fetched explanation row for anomaly_id={anomaly_id}: {'found' if row else 'not found'}."
    )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No anomaly found with id={anomaly_id}",
        )

    return AnomalyExplanationResponse(
        anomaly_id=row["id"],
        meter_serial=row["meter_serial"],
        interval_timestamp=str(row["interval_timestamp"]),
        explanation_status=row.get("explanation_status"),
        explanation=row.get("explanation"),
        explanation_generated_at=(
            str(row["explanation_generated_at"])
            if row.get("explanation_generated_at") else None
        ),
        explanation_error=row.get("explanation_error"),
        rule_violations=row.get("rule_violations"),
        zscore_value=row.get("zscore_value"),
        if_score=row.get("if_score"),
    )