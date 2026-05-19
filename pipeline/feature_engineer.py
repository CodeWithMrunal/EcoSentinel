"""
pipeline/feature_engineer.py
-----------------------------
Computes the full fixed feature vector from:
  1. A canonical dict for the current reading
  2. A list of past readings for this meter (from DB)

This module mirrors exactly what training/train.py does during
training, so the feature vector is always consistent.

Input:
    canonical      : dict  — current reading (canonical feature names)
    interval_ts    : str   — current interval timestamp
    history        : list  — last N readings from meter_telemetry
                             each item: {"interval_timestamp": ..., "raw_data": dict}

Output:
    dict — fixed feature vector with all ALL_FEATURES keys.
           Missing optional features are set to None (will be
           imputed by the IF detector using training medians).
"""

import logging
import numpy as np
from datetime import datetime
from typing import Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config.settings import ALL_FEATURES, OPTIONAL_FEATURES

logger = logging.getLogger(__name__)

# Nominal voltage for deviation calculation
NOMINAL_VOLTAGE = 230.0

# Rolling window size for local statistics
ROLLING_WINDOW = 5


def _is_holiday(dt: datetime) -> int:
    """Sunday = holiday proxy (matches training logic)."""
    return 1 if dt.weekday() == 6 else 0


def _extract_energy_series(history: list[dict], current_energy: float) -> list[float]:
    """
    Builds an energy series from history + current reading.
    history items must have raw_data with 'energy_consumption'.
    Missing values in history are forward-filled with the last
    known value — avoids NaN propagation in rolling stats.
    """
    series = []
    last_known = None

    for h in history:
        raw = h.get("raw_data", {})
        # raw_data from DB is already a canonical dict
        val = raw.get("energy_consumption")
        if val is not None:
            try:
                series.append(float(val))
                last_known = float(val)
            except (TypeError, ValueError):
                if last_known is not None:
                    series.append(last_known)
        else:
            if last_known is not None:
                series.append(last_known)

    series.append(current_energy)
    return series


def compute_features(
    canonical: dict,
    interval_ts: str,
    history: list[dict],
) -> dict:
    """
    Computes the full feature vector for one reading.

    Parameters
    ----------
    canonical    : canonical feature dict for the current reading
                   e.g. {"energy_consumption": 1.6, "voltage": 230.1, ...}
    interval_ts  : ISO timestamp string of the current reading
    history      : list of past readings (oldest → newest),
                   each with {"interval_timestamp": ..., "raw_data": dict}
                   raw_data keys are canonical names (as stored in DB).

    Returns
    -------
    dict with ALL_FEATURES keys. Optional features absent from
    canonical are set to None (imputed downstream).
    """

    # ── Parse timestamp ───────────────────────────────────
    try:
        dt = datetime.fromisoformat(str(interval_ts))
    except Exception as e:
        logger.warning(f"Cannot parse interval_timestamp '{interval_ts}': {e}. Using now().")
        dt = datetime.utcnow()

    # ── Core electrical value ─────────────────────────────
    energy = canonical.get("energy_consumption")
    if energy is None:
        logger.error("energy_consumption missing from canonical dict — cannot compute features.")
        raise ValueError("energy_consumption is required but missing from canonical dict.")
    energy = float(energy)

    # ── Time features ─────────────────────────────────────
    hour_of_day = dt.hour
    day_of_week = dt.weekday()
    is_weekend  = 1 if day_of_week >= 5 else 0
    holiday     = _is_holiday(dt)

    # ── Build energy series (history + current) ───────────
    energy_series = _extract_energy_series(history, energy)
    n = len(energy_series)

    # ── Delta (change from previous reading) ──────────────
    if n >= 2:
        delta = energy - energy_series[-2]
    else:
        delta = 0.0

    # ── Rolling stats over last ROLLING_WINDOW values ─────
    window = energy_series[-ROLLING_WINDOW:]
    rolling_mean = float(np.mean(window))
    rolling_std  = float(np.std(window)) if len(window) > 1 else 0.0

    # ── Z-score ───────────────────────────────────────────
    z_score = (energy - rolling_mean) / (rolling_std + 1e-5)

    # ── Spike ratio ───────────────────────────────────────
    spike_ratio = energy / (rolling_mean + 1e-5)

    # ── Historical averages from past readings ────────────
    # Same-hour average
    same_hour_values = [
        float(h["raw_data"]["energy_consumption"])
        for h in history
        if (
            h.get("raw_data", {}).get("energy_consumption") is not None
            and _parse_hour(h.get("interval_timestamp")) == hour_of_day
        )
    ]
    historical_avg_same_hour = (
        float(np.mean(same_hour_values)) if same_hour_values else energy
    )

    # Same-day-type average (weekday vs weekend)
    same_day_type_values = [
        float(h["raw_data"]["energy_consumption"])
        for h in history
        if (
            h.get("raw_data", {}).get("energy_consumption") is not None
            and _parse_is_weekend(h.get("interval_timestamp")) == is_weekend
        )
    ]
    historical_avg_same_day_type = (
        float(np.mean(same_day_type_values)) if same_day_type_values else energy
    )

    # ── Optional electrical features ─────────────────────

    voltage = _optional_float(canonical, "voltage")
    current = _optional_float(canonical, "current")
    pf      = _optional_float(canonical, "power_factor")
    app_e   = _optional_float(canonical, "apparent_import_energy")

    # Current delta
    current_delta = None
    if current is not None:
        prev_current = _last_canonical_value(history, "current")
        if prev_current is not None:
            current_delta = current - prev_current

    # Voltage deviation from nominal
    voltage_deviation = (voltage - NOMINAL_VOLTAGE) if voltage is not None else None

    # Power factor deviation from ideal (1.0)
    power_factor_deviation = (1.0 - pf) if pf is not None else None

    # ── Assemble feature vector ───────────────────────────
    features = {
        # Core
        "energy_consumption":         round(energy, 4),
        "hour_of_day":                hour_of_day,
        "day_of_week":                day_of_week,
        "is_weekend":                 is_weekend,
        "holiday":                    holiday,
        "delta":                      round(delta, 4),
        "rolling_mean":               round(rolling_mean, 4),
        "rolling_std":                round(rolling_std, 4),
        "z_score":                    round(z_score, 4),
        "spike_ratio":                round(spike_ratio, 4),
        "historical_avg_same_hour":   round(historical_avg_same_hour, 4),
        "historical_avg_same_day_type": round(historical_avg_same_day_type, 4),

        # Optional (None if not available for this meter)
        "voltage":                    _round_opt(voltage),
        "current":                    _round_opt(current),
        "power_factor":               _round_opt(pf),
        "apparent_import_energy":     _round_opt(app_e),
        "current_delta":              _round_opt(current_delta),
        "voltage_deviation":          _round_opt(voltage_deviation),
        "power_factor_deviation":     _round_opt(power_factor_deviation),
    }

    # Sanity check — ensure all expected keys are present
    missing = [f for f in ALL_FEATURES if f not in features]
    if missing:
        logger.warning(f"Feature engineering produced missing keys: {missing}")
        for m in missing:
            features[m] = None

    return features


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _optional_float(d: dict, key: str) -> Optional[float]:
    val = d.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _round_opt(val: Optional[float], ndigits: int = 4) -> Optional[float]:
    return round(val, ndigits) if val is not None else None


def _last_canonical_value(history: list[dict], key: str) -> Optional[float]:
    """Returns the most recent value of a canonical key from history."""
    for h in reversed(history):
        val = h.get("raw_data", {}).get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _parse_hour(ts_str: Optional[str]) -> Optional[int]:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(str(ts_str)).hour
    except Exception:
        return None


def _parse_is_weekend(ts_str: Optional[str]) -> Optional[int]:
    if not ts_str:
        return None
    try:
        return 1 if datetime.fromisoformat(str(ts_str)).weekday() >= 5 else 0
    except Exception:
        return None