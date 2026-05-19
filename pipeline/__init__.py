"""
pipeline/__init__.py
---------------------
Orchestrates the full detection pipeline end to end:

  Raw API record
      ↓
  obis_parser       — parse rawValue pipe-string
      ↓
  canonical_mapper  — OBIS codes → canonical feature names
      ↓
  feature_engineer  — compute derived features (using DB history)
      ↓
  rule_based        — deterministic rule checks
      ↓
  zscore_detector   — statistical threshold checks
      ↓
  isolation_forest  — ML anomaly detection
      ↓
  PipelineResult    — combined output

Public API:
    from pipeline import run
    result = run(api_record, history)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from pipeline.obis_parser      import parse_api_record, OBISParseError
from pipeline.canonical_mapper import map_to_canonical
from pipeline.feature_engineer import compute_features
from pipeline import rule_based
from pipeline import zscore_detector
from pipeline import if_detector

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    # ── Identifiers ───────────────────────────────────────
    meter_serial:        str
    interval_timestamp:  str

    # ── Overall verdict ───────────────────────────────────
    is_anomaly:          bool

    # ── Per-layer results ─────────────────────────────────
    rule_based:          dict         = field(default_factory=dict)
    zscore:              dict         = field(default_factory=dict)
    isolation_forest:    dict         = field(default_factory=dict)

    # ── Feature snapshot (for anomaly_log) ────────────────
    features:            dict         = field(default_factory=dict)

    # ── Error (set if pipeline could not complete) ────────
    error:               Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "meter_serial":       self.meter_serial,
            "interval_timestamp": self.interval_timestamp,
            "is_anomaly":         self.is_anomaly,
            "layers": {
                "rule_based":       self.rule_based,
                "zscore":           self.zscore,
                "isolation_forest": self.isolation_forest,
            },
            "features": self.features,
            "error":    self.error,
        }


def run(
    api_record: dict,
    history: list[dict],
) -> PipelineResult:
    """
    Runs the complete detection pipeline for one API record.

    Parameters
    ----------
    api_record : dict
        One record from the HES API, e.g.:
        {
            "id": 449618,
            "meterSerial": "E0000002",
            "timestamp": "2025-11-12T04:38:09.523241+00:00",
            "obisCode": "1.0.99.1.0.255",
            "entryId": 5,
            "rawValue": "1,0.0.1.0.0.255,2,2025-11-12 10:00:00,|..."
        }

    history : list[dict]
        Last N readings for this meter from meter_telemetry,
        ordered oldest → newest. Each item:
        {
            "interval_timestamp": str,
            "raw_data": dict          ← canonical feature names (from DB)
        }
        Pass an empty list for meters with no history yet.

    Returns
    -------
    PipelineResult
        Complete result with per-layer outputs and overall verdict.
        On parse/processing error, returns a result with error set
        and is_anomaly=False (do not flag what you cannot assess).
    """

    meter_serial = api_record.get("meterSerial", "UNKNOWN")

    # ── Stage 1: Parse rawValue ────────────────────────────
    try:
        parsed = parse_api_record(api_record)
    except (OBISParseError, KeyError) as e:
        logger.error(f"[{meter_serial}] OBIS parse failed: {e}")
        return PipelineResult(
            meter_serial=meter_serial,
            interval_timestamp="UNKNOWN",
            is_anomaly=False,
            error=f"obis_parse_error: {str(e)}",
        )

    interval_ts = parsed["interval_timestamp"]

    # ── Stage 2: Canonical mapping ────────────────────────
    canonical = map_to_canonical(parsed["readings"])

    if "energy_consumption" not in canonical:
        msg = "energy_consumption OBIS code not found in payload — cannot process."
        logger.error(f"[{meter_serial}] {msg}")
        return PipelineResult(
            meter_serial=meter_serial,
            interval_timestamp=interval_ts,
            is_anomaly=False,
            error=f"missing_energy: {msg}",
        )

    # ── Stage 3: Feature engineering ──────────────────────
    try:
        features = compute_features(
            canonical=canonical,
            interval_ts=interval_ts,
            history=history,
        )
    except Exception as e:
        logger.error(f"[{meter_serial}] Feature engineering failed: {e}")
        return PipelineResult(
            meter_serial=meter_serial,
            interval_timestamp=interval_ts,
            is_anomaly=False,
            error=f"feature_engineering_error: {str(e)}",
        )

    # ── Stage 4: Rule-based detection ─────────────────────
    rule_result = rule_based.check(features)

    # ── Stage 5: Z-score detection ────────────────────────
    zscore_result = zscore_detector.check(features)

    # ── Stage 6: Isolation Forest ─────────────────────────
    try:
        if_result = if_detector.check(features)
    except FileNotFoundError as e:
        logger.error(f"[{meter_serial}] IF model not loaded: {e}")
        if_result = None

    # ── Overall verdict ───────────────────────────────────
    # A reading is anomalous if ANY layer flags it.
    # This is intentionally conservative — the decision engine
    # (Step 5 of the roadmap) will add confidence scoring and
    # root cause analysis on top of this.
    is_anomaly = (
        rule_result.is_anomaly
        or zscore_result.is_anomaly
        or (if_result is not None and if_result.is_anomaly)
    )

    if is_anomaly:
        layers_fired = []
        if rule_result.is_anomaly:   layers_fired.append("rule_based")
        if zscore_result.is_anomaly: layers_fired.append("zscore")
        if if_result and if_result.is_anomaly: layers_fired.append("isolation_forest")
        logger.info(
            f"[{meter_serial}] ANOMALY at {interval_ts} | "
            f"layers: {layers_fired}"
        )

    return PipelineResult(
        meter_serial=meter_serial,
        interval_timestamp=interval_ts,
        is_anomaly=is_anomaly,
        rule_based=rule_result.to_dict(),
        zscore=zscore_result.to_dict(),
        isolation_forest=if_result.to_dict() if if_result else {"error": "model_not_loaded"},
        features=features,
    )