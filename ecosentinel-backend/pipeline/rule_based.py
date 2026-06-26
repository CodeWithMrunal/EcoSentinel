"""
pipeline/rule_based.py
-----------------------
Layer 1 of the detection pipeline.

Detects obvious, deterministic anomalies that do not require ML:
  - Negative energy values
  - Zero consumption for extended periods
  - Voltage outside safe operating range
  - Power factor out of valid range [0, 1]
  - Current negative
  - Frequency outside safe range (if available)

Input  : feature dict (output of feature_engineer.compute_features)
Output : RuleBasedResult dataclass
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from time import perf_counter
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config.settings import DETECTION_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class RuleBasedResult:
    is_anomaly:  bool
    violations:  list[str]          = field(default_factory=list)
    details:     dict               = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "layer":       "rule_based",
            "is_anomaly":  self.is_anomaly,
            "violations":  self.violations,
            "details":     self.details,
        }


# =========================================================
# THRESHOLDS (from config — all tunable in settings.py)
# =========================================================

V_MIN  = DETECTION_CONFIG["voltage_min"]           # 180 V
V_MAX  = DETECTION_CONFIG["voltage_max"]           # 270 V
PF_MIN = DETECTION_CONFIG["power_factor_min"]      # 0.0
PF_MAX = DETECTION_CONFIG["power_factor_max"]      # 1.0

# Frequency safe operating range (standard: 49–51 Hz)
FREQ_MIN = 49.0
FREQ_MAX = 51.0


def check(features: dict) -> RuleBasedResult:
    """
    Runs all rule checks against the feature dict.

    Parameters
    ----------
    features : dict
        Output of feature_engineer.compute_features().
        Keys match ALL_FEATURES; optional features may be None.

    Returns
    -------
    RuleBasedResult
        is_anomaly=True if any rule fires.
        violations lists short rule IDs (used by decision engine).
        details holds the actual values that triggered each rule.
    """
    started = perf_counter()
    logger.info(
        f"Rule-based evaluation started with available_features={[k for k, v in features.items() if v is not None]}."
    )

    violations = []
    details    = {}

    energy      = features.get("energy_consumption")
    voltage     = features.get("voltage")
    current     = features.get("current")
    pf          = features.get("power_factor")
    frequency   = features.get("frequency") if "frequency" in features else None
    spike_ratio = features.get("spike_ratio")
    rolling_std = features.get("rolling_std")

    # ── 1. Negative energy ────────────────────────────────
    if energy is not None and energy < 0:
        violations.append("negative_energy")
        details["energy_consumption"] = energy
        logger.debug(f"Rule fired: negative_energy ({energy})")

    # ── 2. Zero / near-zero consumption ──────────────────
    # Flag if spike_ratio is ~0 and rolling_std is low (meter is flat-lining,
    # not just an off-peak reading)
    if (
        energy is not None
        and spike_ratio is not None
        and rolling_std is not None
        and energy == 0.0
        and rolling_std < 0.01       # historically non-zero meter now reading 0
    ):
        violations.append("zero_consumption")
        details["energy_consumption"] = energy
        details["rolling_std"]        = rolling_std
        logger.debug(f"Rule fired: zero_consumption (rolling_std={rolling_std})")

    # ── 3. Voltage out of safe range ──────────────────────
    if voltage is not None:
        if voltage < V_MIN:
            violations.append("voltage_too_low")
            details["voltage"] = voltage
            details["voltage_min_threshold"] = V_MIN
            logger.debug(f"Rule fired: voltage_too_low ({voltage} < {V_MIN})")
        elif voltage > V_MAX:
            violations.append("voltage_too_high")
            details["voltage"] = voltage
            details["voltage_max_threshold"] = V_MAX
            logger.debug(f"Rule fired: voltage_too_high ({voltage} > {V_MAX})")

    # ── 4. Power factor out of valid range ────────────────
    if pf is not None:
        if pf < PF_MIN or pf > PF_MAX:
            violations.append("power_factor_out_of_range")
            details["power_factor"]     = pf
            details["power_factor_min"] = PF_MIN
            details["power_factor_max"] = PF_MAX
            logger.debug(f"Rule fired: power_factor_out_of_range ({pf})")

    # ── 5. Negative current ───────────────────────────────
    if current is not None and current < 0:
        violations.append("negative_current")
        details["current"] = current
        logger.debug(f"Rule fired: negative_current ({current})")

    # ── 6. Frequency out of safe range ───────────────────
    if frequency is not None:
        if frequency < FREQ_MIN or frequency > FREQ_MAX:
            violations.append("frequency_out_of_range")
            details["frequency"]     = frequency
            details["frequency_min"] = FREQ_MIN
            details["frequency_max"] = FREQ_MAX
            logger.debug(f"Rule fired: frequency_out_of_range ({frequency})")

    is_anomaly = len(violations) > 0

    logger.info(
        f"Rule-based evaluation finished in {(perf_counter() - started) * 1000:.1f} ms; anomaly={is_anomaly}, violations={violations}."
    )

    return RuleBasedResult(
        is_anomaly=is_anomaly,
        violations=violations,
        details=details,
    )