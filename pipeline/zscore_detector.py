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


@dataclass
class ZScoreResult:
    is_anomaly:   bool
    z_score:      Optional[float]   = None
    spike_ratio:  Optional[float]   = None
    triggers:     list[str]         = field(default_factory=list)
    details:      dict              = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "layer":       "zscore",
            "is_anomaly":  self.is_anomaly,
            "z_score":     self.z_score,
            "spike_ratio": self.spike_ratio,
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
    triggers = []
    details  = {}

    z_score     = features.get("z_score")
    spike_ratio = features.get("spike_ratio")
    energy      = features.get("energy_consumption")
    rolling_mean = features.get("rolling_mean")
    rolling_std  = features.get("rolling_std")

    # ── 1. Z-score threshold ─────────────────────────────
    if z_score is not None:
        abs_z = abs(z_score)
        if abs_z > ZSCORE_THRESHOLD:
            direction = "spike" if z_score > 0 else "drop"
            triggers.append(f"zscore_{direction}")
            details["z_score"]          = z_score
            details["zscore_threshold"] = ZSCORE_THRESHOLD
            details["direction"]        = direction
            logger.debug(
                f"Z-score trigger: {direction} "
                f"z={z_score:.3f} (threshold={ZSCORE_THRESHOLD})"
            )

    # ── 2. Spike ratio — extreme multiplier ──────────────
    # Complements z-score: catches spikes when rolling_std ≈ 0
    # (flat baseline) where z-score would explode but may be
    # technically correct. Spike ratio is more interpretable.
    if spike_ratio is not None and energy is not None:
        if spike_ratio > SPIKE_RATIO_THRESHOLD and energy > 0:
            triggers.append("extreme_spike_ratio")
            details["spike_ratio"]           = spike_ratio
            details["spike_ratio_threshold"] = SPIKE_RATIO_THRESHOLD
            details["energy_consumption"]    = energy
            details["rolling_mean"]          = rolling_mean
            logger.debug(
                f"Spike ratio trigger: {spike_ratio:.2f}× rolling mean"
            )
        elif spike_ratio < DROP_RATIO_THRESHOLD and energy >= 0:
            # Near-zero consumption relative to baseline
            triggers.append("extreme_drop_ratio")
            details["spike_ratio"]          = spike_ratio
            details["drop_ratio_threshold"] = DROP_RATIO_THRESHOLD
            details["energy_consumption"]   = energy
            details["rolling_mean"]         = rolling_mean
            logger.debug(
                f"Drop ratio trigger: {spike_ratio:.3f}× rolling mean"
            )

    is_anomaly = len(triggers) > 0

    return ZScoreResult(
        is_anomaly=is_anomaly,
        z_score=z_score,
        spike_ratio=spike_ratio,
        triggers=triggers,
        details=details,
    )