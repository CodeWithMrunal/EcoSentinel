"""
pipeline/zscore_detector.py
-----------------------------
Layer 2 of the detection pipeline.

Detects statistical deviations using the z-score already
computed by feature_engineer.py.

  - Flags if |z_score| > threshold (sudden spikes or drops)
  - Also checks spike_ratio for extreme multiplier anomalies
    (catches cases where rolling_std is near-zero, making
     z-score unreliable as a standalone signal)

Input  : feature dict (output of feature_engineer.compute_features)
Output : ZScoreResult dataclass
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

# =========================================================
# THRESHOLDS
# =========================================================

ZSCORE_THRESHOLD      = DETECTION_CONFIG["zscore_threshold"]  # 3.0
SPIKE_RATIO_THRESHOLD = 4.0    # energy > 4× rolling mean = extreme spike
DROP_RATIO_THRESHOLD  = 0.1    # energy < 10% of rolling mean = extreme drop
SAME_HOUR_DEVIATION_THRESHOLD = DETECTION_CONFIG["same_hour_deviation_threshold"]


@dataclass
class ZScoreResult:
    is_anomaly:   bool
    z_score:      Optional[float]   = None
    spike_ratio:  Optional[float]   = None
    triggers:     list[str]         = field(default_factory=list)
    details:      dict              = field(default_factory=dict)
    same_hour_deviation: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "layer":       "zscore",
            "is_anomaly":  self.is_anomaly,
            "z_score":     self.z_score,
            "spike_ratio": self.spike_ratio,
            "same_hour_deviation": self.same_hour_deviation,
            "triggers":    self.triggers,
            "details":     self.details,
        }


def check(features: dict) -> ZScoreResult:
    """
    Runs z-score and spike-ratio checks.

    Parameters
    ----------
    features : dict
        Output of feature_engineer.compute_features().

    Returns
    -------
    ZScoreResult
        is_anomaly=True if any statistical threshold is breached.
        triggers lists which checks fired.
        z_score and spike_ratio hold the actual values.
    """
    started = perf_counter()
    logger.info(
        f"Z-score evaluation started with z_score={features.get('z_score')}, spike_ratio={features.get('spike_ratio')}."
    )

    triggers = []
    details  = {}

    z_score     = features.get("z_score")
    spike_ratio = features.get("spike_ratio")
    energy      = features.get("energy_consumption")
    rolling_mean = features.get("rolling_mean")
    rolling_std  = features.get("rolling_std")
    same_hour_avg = features.get("historical_avg_same_hour")

    # ── 1. Z-score threshold ─────────────────────────────
    if z_score is not None:
        abs_z = abs(z_score)
        if abs_z > ZSCORE_THRESHOLD:
            direction = "spike" if z_score > 0 else "drop"
            triggers.append("Z_SCORE_THRESHOLD_EXCEEDED")
            details["z_score"]          = z_score
            details["zscore_threshold"] = ZSCORE_THRESHOLD
            details["direction"]        = direction
            logger.debug(
                f"Z-score trigger: {direction} "
                f"z={z_score:.3f} (threshold={ZSCORE_THRESHOLD})"
            )

    history_sample_count = features.get("history_sample_count")
    if history_sample_count is None:
        history_sample_count = 0

    logger.info(
        f"Statistical baseline before current reading: sample_count={history_sample_count}, rolling_mean={rolling_mean}, rolling_std={rolling_std}, same_hour_avg={same_hour_avg}."
    )

    same_hour_deviation = None
    if energy is not None and same_hour_avg not in (None, 0):
        same_hour_deviation = abs(energy - same_hour_avg) / abs(same_hour_avg)
        if same_hour_deviation > SAME_HOUR_DEVIATION_THRESHOLD:
            triggers.append("SAME_HOUR_DEVIATION_EXCEEDED")
            details["historical_avg_same_hour"] = same_hour_avg
            details["same_hour_deviation"] = same_hour_deviation
            details["same_hour_deviation_threshold"] = SAME_HOUR_DEVIATION_THRESHOLD
            logger.debug(
                f"Same-hour deviation trigger: deviation={same_hour_deviation:.3f} threshold={SAME_HOUR_DEVIATION_THRESHOLD}"
            )

    # ── 2. Spike ratio — extreme multiplier ──────────────
    # Complements z-score: catches spikes when rolling_std ≈ 0
    # (flat baseline) where z-score would explode but may be
    # technically correct. Spike ratio is more interpretable.
    if spike_ratio is not None and energy is not None:
        if spike_ratio > SPIKE_RATIO_THRESHOLD and energy > 0:
            triggers.append("SPIKE_RATIO_THRESHOLD_EXCEEDED")
            details["spike_ratio"]           = spike_ratio
            details["spike_ratio_threshold"] = SPIKE_RATIO_THRESHOLD
            details["energy_consumption"]    = energy
            details["rolling_mean"]          = rolling_mean
            logger.debug(
                f"Spike ratio trigger: {spike_ratio:.2f}× rolling mean"
            )
        elif spike_ratio < DROP_RATIO_THRESHOLD and energy >= 0:
            # Near-zero consumption relative to baseline
            triggers.append("DROP_RATIO_THRESHOLD_EXCEEDED")
            details["spike_ratio"]          = spike_ratio
            details["drop_ratio_threshold"] = DROP_RATIO_THRESHOLD
            details["energy_consumption"]   = energy
            details["rolling_mean"]         = rolling_mean
            logger.debug(
                f"Drop ratio trigger: {spike_ratio:.3f}× rolling mean"
            )

    is_anomaly = len(triggers) > 0

    if len(triggers) > 1:
        details["combined_trigger"] = "BOTH"

    logger.info(
        f"Current reading statistics: current_energy={energy}, computed_z_score={z_score}, threshold_used={ZSCORE_THRESHOLD}, anomaly_decision={is_anomaly}, trigger_reason(s)={triggers or ['none']}."
    )
    updated_sample_count = history_sample_count + (1 if energy is not None and not is_anomaly else 0)
    logger.info(
        f"Updated sample count={updated_sample_count}, updated_mean={rolling_mean}, updated_std={rolling_std}."
    )

    logger.info(
        f"Z-score evaluation finished in {(perf_counter() - started) * 1000:.1f} ms; anomaly={is_anomaly}, triggers={triggers}."
    )

    return ZScoreResult(
        is_anomaly=is_anomaly,
        z_score=z_score,
        spike_ratio=spike_ratio,
        triggers=triggers,
        details=details,
        same_hour_deviation=same_hour_deviation,
    )