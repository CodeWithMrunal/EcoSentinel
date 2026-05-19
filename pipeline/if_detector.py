"""
pipeline/if_detector.py
------------------------
Layer 3 of the detection pipeline.

Loads the trained Isolation Forest model artifacts and runs
inference on the fixed feature vector.

Handles:
  - NaN imputation using training medians
  - Scaling using training StandardScaler
  - Returning anomaly score + binary prediction

Input  : feature dict (output of feature_engineer.compute_features)
Output : IFResult dataclass
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import MODEL_PATHS, ALL_FEATURES

logger = logging.getLogger(__name__)


@dataclass
class IFResult:
    is_anomaly:   bool
    anomaly_score: float          # decision_function output; lower = more anomalous
    prediction:   int             # -1 = anomaly, 1 = normal

    def to_dict(self) -> dict:
        return {
            "layer":         "isolation_forest",
            "is_anomaly":    self.is_anomaly,
            "anomaly_score": round(self.anomaly_score, 6),
            "prediction":    self.prediction,
        }


# =========================================================
# MODEL LOADING
# Artifacts are loaded once at module import time (lazy,
# thread-safe singleton pattern). The pipeline calls
# check() repeatedly; loading only happens on first call.
# =========================================================

_model         = None
_scaler        = None
_impute_values = None
_feature_schema = None


def _load_artifacts():
    global _model, _scaler, _impute_values, _feature_schema

    if _model is not None:
        return   # already loaded

    for name, path in MODEL_PATHS.items():
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"Model artifact '{name}' not found at: {abs_path}\n"
                f"Run training/train.py first to generate model artifacts."
            )

    _model          = joblib.load(MODEL_PATHS["isolation_forest"])
    _scaler         = joblib.load(MODEL_PATHS["scaler"])
    _impute_values  = joblib.load(MODEL_PATHS["impute_values"])
    _feature_schema = joblib.load(MODEL_PATHS["feature_schema"])

    logger.info("Isolation Forest artifacts loaded successfully.")

    # Validate feature schema matches what we expect
    loaded_features = _feature_schema.get("all_features", [])
    if loaded_features != ALL_FEATURES:
        logger.warning(
            f"Feature schema mismatch!\n"
            f"  Expected : {ALL_FEATURES}\n"
            f"  Loaded   : {loaded_features}\n"
            f"Ensure training/train.py and config/settings.py are in sync."
        )


def check(features: dict) -> IFResult:
    """
    Runs Isolation Forest inference on a feature dict.

    Parameters
    ----------
    features : dict
        Output of feature_engineer.compute_features().
        Optional features absent from this meter should be None.

    Returns
    -------
    IFResult
        is_anomaly    : True if model predicts -1
        anomaly_score : raw decision_function score (lower = more anomalous)
        prediction    : -1 or 1
    """
    _load_artifacts()

    # ── Build ordered feature vector ──────────────────────
    # Must follow ALL_FEATURES order exactly (same as training)
    row = []
    for feat in ALL_FEATURES:
        val = features.get(feat)
        if val is None:
            # Use training median for this feature
            imputed = _impute_values.get(feat, 0.0)
            row.append(float(imputed))
        else:
            row.append(float(val))

    X = pd.DataFrame([row], columns=ALL_FEATURES)

    # ── Scale ─────────────────────────────────────────────
    X_scaled = _scaler.transform(X)

    # ── Predict ───────────────────────────────────────────
    prediction    = int(_model.predict(X_scaled)[0])        # -1 or 1
    anomaly_score = float(_model.decision_function(X_scaled)[0])

    is_anomaly = prediction == -1

    if is_anomaly:
        logger.debug(
            f"IF flagged anomaly: score={anomaly_score:.4f}"
        )

    return IFResult(
        is_anomaly=is_anomaly,
        anomaly_score=anomaly_score,
        prediction=prediction,
    )


def reload_artifacts():
    """
    Forces a reload of model artifacts from disk.
    Call this after retraining without restarting the service.
    """
    global _model, _scaler, _impute_values, _feature_schema
    _model = _scaler = _impute_values = _feature_schema = None
    _load_artifacts()
    logger.info("Model artifacts reloaded.")