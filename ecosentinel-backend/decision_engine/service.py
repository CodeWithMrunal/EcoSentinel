"""
decision_engine/service.py
-----------------------------
Orchestrates the Decision Engine end to end:

  1. Assemble AnomalyContext from a PipelineResult + DB history
  2. Build the LLM prompt
  3. Call the LLM (provider-agnostic via llm_client)
  4. Parse and validate the JSON response into AnomalyExplanation
  5. Persist the result to anomaly_log (or record failure)

This module is called as a FastAPI background task — it must
never raise in a way that crashes the request/response cycle.
All failures are caught, logged, and persisted as
explanation_status='failed'.
"""

import sys
import os
import json
import logging
from time import perf_counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional
from pydantic import ValidationError

from config.settings import DECISION_ENGINE_CONFIG, ROLLING_WINDOW_SIZE
from decision_engine.schemas import AnomalyContext, AnomalyExplanation, HistoricalReading
from decision_engine.prompt_builder import build_messages
from decision_engine.llm_client import call_llm, get_provider_info, LLMClientError

logger = logging.getLogger(__name__)


# =========================================================
# JSON EXTRACTION
# Local models via Ollama sometimes wrap JSON in markdown
# fences or add stray text despite instructions. This
# extracts the first valid JSON object from the response.
# =========================================================

def _extract_json(raw_text: str) -> dict:
    """
    Extracts a JSON object from raw LLM output.

    Handles:
      - Clean JSON (ideal case)
      - JSON wrapped in ```json ... ``` fences
      - JSON with leading/trailing prose

    Raises
    ------
    json.JSONDecodeError if no valid JSON object can be found.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # If still not starting with '{', find the first '{' and last '}'
    if not text.startswith("{"):
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    return json.loads(text)


# =========================================================
# CONTEXT ASSEMBLY
# =========================================================

def build_context(
    meter_serial:       str,
    interval_timestamp: str,
    features:           dict,
    rule_violations:    list[str],
    zscore_value:       Optional[float],
    zscore_triggers:    list[str],
    if_score:           Optional[float],
    if_model_used:      Optional[str],
    history_raw:        list[dict],
    anomaly_id:         Optional[int] = None,
) -> AnomalyContext:
    """
    Assembles an AnomalyContext from pipeline outputs and raw
    history rows (as returned by db.client.get_last_n_readings).

    Parameters
    ----------
    history_raw : list of {"interval_timestamp": ..., "raw_data": dict}
                  ordered oldest -> newest, as returned by
                  get_last_n_readings(). Will be truncated to
                  DECISION_ENGINE_CONFIG["history_window_size"]
                  most recent entries.
    """
    window = DECISION_ENGINE_CONFIG["history_window_size"]
    trimmed = history_raw[-window:] if len(history_raw) > window else history_raw

    history = [
        HistoricalReading(
            interval_timestamp=str(h["interval_timestamp"]),
            values=h.get("raw_data", {}) or {},
        )
        for h in trimmed
    ]

    # Current values: the canonical raw values for this reading.
    # Pulled from `features` by selecting only the raw (non-derived)
    # canonical keys that are non-null.
    raw_keys = [
        "energy_consumption", "voltage", "current", "power_factor",
        "apparent_import_energy", "active_export_energy",
        "reactive_import_energy", "reactive_export_energy",
        "active_import_power", "active_export_power", "frequency",
    ]
    current_values = {
        k: features[k] for k in raw_keys
        if k in features and features[k] is not None
    }

    logger.info(
        f"Built Decision Engine context for anomaly_id={anomaly_id}: history_rows={len(history)}, current_values={list(current_values.keys())}, rule_violations={rule_violations}, zscore_triggers={zscore_triggers}, if_model_used={if_model_used}."
    )

    return AnomalyContext(
        meter_serial=meter_serial,
        interval_timestamp=str(interval_timestamp),
        current_values=current_values,
        features=features,
        history=history,
        rule_violations=rule_violations,
        zscore_value=zscore_value,
        zscore_triggers=zscore_triggers,
        if_score=if_score,
        if_model_used=if_model_used,
        anomaly_id=anomaly_id,
    )


# =========================================================
# MAIN ENTRY POINT
# =========================================================

def generate_explanation(context: AnomalyContext) -> AnomalyExplanation:
    """
    Generates an LLM explanation for an anomaly context.

    Returns
    -------
    AnomalyExplanation on success.

    Raises
    ------
    LLMClientError       if the LLM call itself fails
                          (connection, timeout, API error).
    json.JSONDecodeError  if the LLM response is not valid JSON
                          even after extraction attempts.
    pydantic.ValidationError if the JSON doesn't match the
                          expected AnomalyExplanation schema.

    Callers (the background task) should catch all of these,
    log them, and persist explanation_status='failed'.
    """
    started = perf_counter()
    messages = build_messages(context)
    logger.info(
        f"Generating explanation prompt for anomaly_id={context.anomaly_id}: messages={len(messages)}, history_rows={len(context.history)}."
    )

    raw_response = call_llm(messages)

    logger.info(
        f"LLM response received for anomaly_id={context.anomaly_id} with {len(raw_response)} character(s)."
    )

    parsed = _extract_json(raw_response)

    provider_info = get_provider_info()
    parsed.setdefault("llm_provider", provider_info["provider"])
    parsed.setdefault("llm_model", provider_info["model"])

    explanation = AnomalyExplanation(**parsed)

    logger.info(
        f"Explanation generated for anomaly_id={context.anomaly_id} in {(perf_counter() - started) * 1000:.1f} ms."
    )

    return explanation


def run_explanation_task(
    anomaly_id:         int,
    meter_serial:       str,
    interval_timestamp: str,
    features:           dict,
    rule_violations:    list[str],
    zscore_value:       Optional[float],
    zscore_triggers:    list[str],
    if_score:           Optional[float],
    if_model_used:      Optional[str],
) -> None:
    """
    Full background task entry point: fetches history from DB,
    builds context, calls the LLM, and persists the result.

    This function is passed to FastAPI's BackgroundTasks and
    runs after the /detect response has already been sent.
    It must never raise — all errors are caught and persisted
    as explanation_status='failed' so the row doesn't stay
    'pending' forever.

    DB access is done here (not in the API layer) to keep this
    module self-contained and independently testable.
    """
    # Import here to avoid circular imports and to allow this
    # module to be used without DB in unit tests.
    from db.client import get_last_n_readings, update_anomaly_explanation

    started = perf_counter()
    logger.info(f"[decision_engine] Generating explanation for anomaly_id={anomaly_id} "
                f"(meter={meter_serial}, ts={interval_timestamp})")

    try:
        history_window = DECISION_ENGINE_CONFIG["history_window_size"]
        history_raw = get_last_n_readings(
            meter_serial=meter_serial,
            n=history_window,
            before_timestamp=interval_timestamp,
        )
    except Exception as e:
        logger.error(f"[decision_engine] Failed to fetch history for "
                     f"anomaly_id={anomaly_id}: {e}")
        history_raw = []
    logger.info(
        f"[decision_engine] Retrieved {len(history_raw)} historical row(s) for anomaly_id={anomaly_id}."
    )

    context = build_context(
        meter_serial=meter_serial,
        interval_timestamp=interval_timestamp,
        features=features,
        rule_violations=rule_violations,
        zscore_value=zscore_value,
        zscore_triggers=zscore_triggers,
        if_score=if_score,
        if_model_used=if_model_used,
        history_raw=history_raw,
        anomaly_id=anomaly_id,
    )

    try:
        explanation = generate_explanation(context)

        update_anomaly_explanation(
            anomaly_id=anomaly_id,
            explanation=explanation.model_dump(),
            status="completed",
            error=None,
        )
        logger.info(f"[decision_engine] Explanation completed for "
                    f"anomaly_id={anomaly_id} (confidence={explanation.confidence}, duration={(perf_counter() - started) * 1000:.1f} ms)")

    except LLMClientError as e:
        logger.error(f"[decision_engine] LLM call failed for anomaly_id={anomaly_id}: {e}")
        _safe_mark_failed(anomaly_id, str(e))

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"[decision_engine] Invalid LLM response for anomaly_id={anomaly_id}: {e}")
        _safe_mark_failed(anomaly_id, f"Invalid LLM response format: {e}")

    except Exception as e:
        logger.error(f"[decision_engine] Unexpected error for anomaly_id={anomaly_id}: {e}")
        _safe_mark_failed(anomaly_id, f"Unexpected error: {e}")


def _safe_mark_failed(anomaly_id: int, error_msg: str) -> None:
    """Marks an anomaly's explanation as failed, swallowing any DB errors."""
    try:
        from db.client import update_anomaly_explanation
        update_anomaly_explanation(
            anomaly_id=anomaly_id,
            explanation=None,
            status="failed",
            error=error_msg[:2000],   # truncate to fit TEXT column comfortably
        )
    except Exception as e:
        logger.error(f"[decision_engine] Could not even mark anomaly_id={anomaly_id} "
                     f"as failed: {e}")