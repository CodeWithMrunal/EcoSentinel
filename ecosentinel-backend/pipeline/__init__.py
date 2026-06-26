"""
pipeline/__init__.py
---------------------
Orchestrates the full detection pipeline end to end.
Energy consumption is NOT required — pipeline continues with
whatever canonical features are available.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from time import perf_counter

from pipeline.obis_parser      import parse_api_record, OBISParseError
from pipeline.canonical_mapper import map_to_canonical
from pipeline.feature_engineer import compute_features
from pipeline import rule_based
from pipeline import zscore_detector
from pipeline import if_detector

logger = logging.getLogger(__name__)


def _summarize_present(values: dict, max_items: int = 8) -> str:
    keys = [k for k, v in values.items() if v is not None]
    if not keys:
        return "[]"
    preview = keys[:max_items]
    if len(keys) > max_items:
        preview.append(f"...(+{len(keys) - max_items} more)")
    return "[" + ", ".join(preview) + "]"


@dataclass
class PipelineResult:
    meter_serial:        str
    interval_timestamp:  str
    is_anomaly:          bool
    rule_based:          dict          = field(default_factory=dict)
    zscore:              dict          = field(default_factory=dict)
    isolation_forest:    dict          = field(default_factory=dict)
    features:            dict          = field(default_factory=dict)
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


def run(api_record: dict, history: list[dict]) -> PipelineResult:
    """
    Runs the complete detection pipeline for one API record.
    Energy consumption is not required; all features degrade
    gracefully when parameters are absent.
    """
    meter_serial = api_record.get("meterSerial", "UNKNOWN")
    pipeline_started = perf_counter()
    logger.info(
        f"[{meter_serial}] Pipeline started with {len(history)} historical reading(s)."
    )

    # ── Stage 1: Parse rawValue ────────────────────────────
    try:
        stage_started = perf_counter()
        parsed = parse_api_record(api_record)
        logger.info(
            f"[{meter_serial}] Parsed rawValue in {(perf_counter() - stage_started) * 1000:.1f} ms; interval_timestamp={parsed['interval_timestamp']}, readings={len(parsed['readings'])}."
        )
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
    stage_started = perf_counter()
    canonical = map_to_canonical(parsed["readings"])
    logger.info(
        f"[{meter_serial}] Canonical mapping completed in {(perf_counter() - stage_started) * 1000:.1f} ms; canonical_features={_summarize_present(canonical)}."
    )

    if not canonical:
        msg = "No recognisable OBIS codes in payload."
        logger.error(f"[{meter_serial}] {msg}")
        return PipelineResult(
            meter_serial=meter_serial,
            interval_timestamp=interval_ts,
            is_anomaly=False,
            error=f"empty_canonical: {msg}",
        )

    if "energy_consumption" not in canonical:
        logger.info(
            f"[{meter_serial}] energy_consumption absent — continuing with: "
            f"{list(canonical.keys())}"
        )

    # ── Stage 3: Feature engineering ──────────────────────
    try:
        stage_started = perf_counter()
        features = compute_features(
            canonical=canonical,
            interval_ts=interval_ts,
            history=history,
        )
        logger.info(
            f"[{meter_serial}] Feature engineering completed in {(perf_counter() - stage_started) * 1000:.1f} ms; non_null_features={_summarize_present(features)}."
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
    stage_started = perf_counter()
    rule_result   = rule_based.check(features)
    logger.info(
        f"[{meter_serial}] Rule-based layer completed in {(perf_counter() - stage_started) * 1000:.1f} ms; anomaly={rule_result.is_anomaly}, violations={rule_result.violations}."
    )

    # ── Stage 5: Z-score detection ────────────────────────
    stage_started = perf_counter()
    zscore_result = zscore_detector.check(features)
    logger.info(
        f"[{meter_serial}] Z-score layer completed in {(perf_counter() - stage_started) * 1000:.1f} ms; anomaly={zscore_result.is_anomaly}, triggers={zscore_result.triggers}, z_score={zscore_result.z_score}, spike_ratio={zscore_result.spike_ratio}."
    )

    # ── Stage 6: Isolation Forest (with group routing) ────
    try:
        stage_started = perf_counter()
        if_result = if_detector.check(features, canonical=canonical)
        logger.info(
            f"[{meter_serial}] Isolation Forest layer completed in {(perf_counter() - stage_started) * 1000:.1f} ms; anomaly={if_result.is_anomaly}, model_used={if_result.model_used}, score={if_result.anomaly_score:.4f}."
        )
    except FileNotFoundError as e:
        logger.error(f"[{meter_serial}] IF model not loaded: {e}")
        if_result = None

    # ── Overall verdict ───────────────────────────────────
    is_anomaly = (
        rule_result.is_anomaly
        or zscore_result.is_anomaly
        or (if_result is not None and if_result.is_anomaly)
    )

    if is_anomaly:
        layers_fired = []
        if rule_result.is_anomaly:                      layers_fired.append("rule_based")
        if zscore_result.is_anomaly:                    layers_fired.append("zscore")
        if if_result and if_result.is_anomaly:          layers_fired.append("isolation_forest")
        logger.info(
            f"[{meter_serial}] ANOMALY at {interval_ts} | layers: {layers_fired}"
        )

    logger.info(
        f"[{meter_serial}] Pipeline finished in {(perf_counter() - pipeline_started) * 1000:.1f} ms; anomaly={is_anomaly}."
    )

    return PipelineResult(
        meter_serial=meter_serial,
        interval_timestamp=interval_ts,
        is_anomaly=is_anomaly,
        rule_based=rule_result.to_dict(),
        zscore=zscore_result.to_dict(),
        isolation_forest=(
            if_result.to_dict() if if_result
            else {"error": "model_not_loaded"}
        ),
        features=features,
    )