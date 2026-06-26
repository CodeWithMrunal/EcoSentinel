"""
pipeline/if_detector.py
------------------------
Layer 3 of the detection pipeline.

Model routing logic:
  1. Determine which canonical features are present in the payload.
  2. Look up CAPABILITY_GROUPS for an exact or best-subset match.
  3. Load and run the group-specific model (trained only on that
     group's features — no NaN imputation needed).
  4. If no group matches, fall back to the global model with
     median imputation for missing optional features.

Each group model is loaded once (lazy singleton) and cached.
reload_artifacts() clears all caches and forces a fresh load.
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from time import perf_counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    MODEL_PATHS,
    ALL_FEATURES,
    CAPABILITY_GROUPS,
    DERIVED_FEATURE_MAP,
    group_model_paths,
)

logger = logging.getLogger(__name__)


@dataclass
class IFResult:
    is_anomaly:    bool
    anomaly_score: float
    prediction:    int
    model_used:    str = "global"      # which group or "global"
    features_used: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layer":         "isolation_forest",
            "is_anomaly":    self.is_anomaly,
            "anomaly_score": round(self.anomaly_score, 6),
            "prediction":    self.prediction,
            "model_used":    self.model_used,
            "features_used": self.features_used,
        }


# =========================================================
# ARTIFACT CACHES
# _group_cache: { group_name → {"model": ..., "scaler": ...,
#                                "features": [...]} }
# _global_*   : global fallback model artifacts
# =========================================================

_group_cache: dict = {}
_global_model         = None
_global_scaler        = None
_global_impute_values = None
_global_feature_schema = None


# =========================================================
# DERIVED FEATURE BUILDER
# Given a group's raw canonical features, returns the full
# list of features (raw + derived) that model was trained on.
# =========================================================

def _group_features_for(raw_features: frozenset) -> list[str]:
    """
    Returns the ordered feature list for a capability group:
      raw canonical features + all derivable features.
    Time features (_timestamp) are always included.
    Order: time → energy-derived → raw electrical → electrical-derived
    """
    feats = []

    # Time features always present
    feats += DERIVED_FEATURE_MAP["_timestamp"]

    # Energy-derived features
    if "energy_consumption" in raw_features:
        feats.append("energy_consumption")
        feats += DERIVED_FEATURE_MAP["energy_consumption"]

    # Voltage
    if "voltage" in raw_features:
        feats.append("voltage")
        feats += DERIVED_FEATURE_MAP["voltage"]

    # Current
    if "current" in raw_features:
        feats.append("current")
        feats += DERIVED_FEATURE_MAP["current"]

    # Power factor
    if "power_factor" in raw_features:
        feats.append("power_factor")
        feats += DERIVED_FEATURE_MAP["power_factor"]

    # Other raw features (apparent energy, frequency, etc.)
    other = raw_features - {
        "energy_consumption", "voltage", "current", "power_factor"
    }
    for f in sorted(other):
        if f not in feats:
            feats.append(f)

    # Deduplicate preserving order
    seen = set()
    ordered = []
    for f in feats:
        if f not in seen:
            seen.add(f)
            ordered.append(f)

    return ordered


# =========================================================
# GROUP ROUTING
# =========================================================

def _resolve_group(present_features: frozenset) -> Optional[str]:
    """
    Finds the best-matching capability group for a set of
    present canonical features.

    Matching priority:
      1. Exact match  — present_features == group
      2. Subset match — present_features ⊆ group
         (pick group with most overlap, i.e. smallest superset)
      3. None         → caller uses global fallback

    Only raw canonical features are considered for matching
    (not derived features like delta, z_score etc.).
    """
    # Filter to raw canonical features only (not derived)
    all_derived = set()
    for derived_list in DERIVED_FEATURE_MAP.values():
        all_derived.update(derived_list)
    raw_present = present_features - all_derived

    # 1. Exact match
    for group_name, group_features in CAPABILITY_GROUPS.items():
        if raw_present == group_features:
            return group_name

    # 2. Best subset match (raw_present ⊆ group_features)
    best_group  = None
    best_overlap = -1
    for group_name, group_features in CAPABILITY_GROUPS.items():
        if raw_present.issubset(group_features):
            overlap = len(raw_present & group_features)
            if overlap > best_overlap:
                best_overlap = overlap
                best_group   = group_name

    return best_group


# =========================================================
# ARTIFACT LOADING
# =========================================================

def _load_group(group_name: str) -> Optional[dict]:
    """
    Loads artifacts for a capability group.
    Returns None if artifacts don't exist on disk yet
    (group model hasn't been trained).
    """
    if group_name in _group_cache:
        logger.info(f"Using cached group model '{group_name}'.")
        return _group_cache[group_name]

    paths = group_model_paths(group_name)
    for p in paths.values():
        if not os.path.exists(os.path.abspath(p)):
            logger.warning(
                f"Group model '{group_name}' not found at {p}. "
                f"Will fall back to global model."
            )
            return None

    artifacts = {
        "model":    joblib.load(paths["isolation_forest"]),
        "scaler":   joblib.load(paths["scaler"]),
        "features": joblib.load(paths["feature_schema"])["features"],
    }
    _group_cache[group_name] = artifacts
    logger.info(f"Loaded group model '{group_name}' "
                f"({len(artifacts['features'])} features).")
    return artifacts


def _load_global():
    global _global_model, _global_scaler, _global_impute_values, _global_feature_schema

    if _global_model is not None:
        return

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(os.path.abspath(path)):
            raise FileNotFoundError(
                f"Global model artifact '{name}' not found at: {path}\n"
                f"Run training/train.py first."
            )

    _global_model          = joblib.load(MODEL_PATHS["isolation_forest"])
    _global_scaler         = joblib.load(MODEL_PATHS["scaler"])
    _global_impute_values  = joblib.load(MODEL_PATHS["impute_values"])
    _global_feature_schema = joblib.load(MODEL_PATHS["feature_schema"])
    logger.info("Global fallback model loaded.")


# =========================================================
# INFERENCE
# =========================================================

def check(features: dict, canonical: dict = None) -> IFResult:
    """
    Runs Isolation Forest inference with automatic group routing.

    Parameters
    ----------
    features  : full feature dict from feature_engineer.compute_features()
    canonical : original canonical dict (used to determine which raw
                features are present for group routing).
                If None, inferred from non-None values in features.

    Returns
    -------
    IFResult with model_used indicating which group or "global" was used.
    """
    started = perf_counter()

    # Determine present raw features for routing
    if canonical is not None:
        present = frozenset(k for k, v in canonical.items() if v is not None)
    else:
        # Infer from features dict — use known raw canonical names
        raw_names = set()
        for meta in __import__(
            "config.settings", fromlist=["OBIS_REGISTRY"]
        ).OBIS_REGISTRY.values():
            if meta.get("canonical_name"):
                raw_names.add(meta["canonical_name"])
        present = frozenset(k for k in raw_names if features.get(k) is not None)

    logger.info(f"Isolation Forest routing started with raw features={sorted(present)}.")

    # Route to group model
    group_name = _resolve_group(present)
    if group_name:
        logger.info(f"Resolved capability group '{group_name}' for Isolation Forest inference.")
        group_artifacts = _load_group(group_name)
    else:
        logger.info("No capability group matched; using global Isolation Forest model.")
        group_artifacts = None

    if group_artifacts is not None:
        result = _run_group_model(features, group_name, group_artifacts)
    else:
        result = _run_global_model(features)

    logger.info(
        f"Isolation Forest inference finished in {(perf_counter() - started) * 1000:.1f} ms; anomaly={result.is_anomaly}, model_used={result.model_used}, score={result.anomaly_score:.4f}."
    )
    return result


def _run_group_model(features: dict, group_name: str, artifacts: dict) -> IFResult:
    """
    Runs inference using a group-specific model.
    No NaN imputation — the group model was trained only on
    features that are always present for this group.
    Missing values within the group features are set to 0
    (should not happen in practice for a matched group).
    """
    feat_list = artifacts["features"]

    row = []
    for f in feat_list:
        val = features.get(f)
        row.append(float(val) if val is not None else 0.0)

    X        = pd.DataFrame([row], columns=feat_list)
    X_scaled = artifacts["scaler"].transform(X)
    pred     = int(artifacts["model"].predict(X_scaled)[0])
    score    = float(artifacts["model"].decision_function(X_scaled)[0])

    logger.debug(
        f"Group '{group_name}' IF used {len(feat_list)} feature(s); prediction={pred}, score={score:.4f}."
    )

    if pred == -1:
        logger.debug(f"Group '{group_name}' IF flagged anomaly: score={score:.4f}")

    return IFResult(
        is_anomaly=pred == -1,
        anomaly_score=score,
        prediction=pred,
        model_used=group_name,
        features_used=feat_list,
    )


def _run_global_model(features: dict) -> IFResult:
    """
    Runs inference using the global fallback model.
    NaN imputation applied for missing optional features.
    """
    _load_global()

    row = []
    for feat in ALL_FEATURES:
        val = features.get(feat)
        if val is None:
            imputed = _global_impute_values.get(feat, 0.0)
            row.append(float(imputed))
        else:
            row.append(float(val))

    X        = pd.DataFrame([row], columns=ALL_FEATURES)
    X_scaled = _global_scaler.transform(X)
    pred     = int(_global_model.predict(X_scaled)[0])
    score    = float(_global_model.decision_function(X_scaled)[0])

    logger.debug(
        f"Global IF used {len(ALL_FEATURES)} feature(s); prediction={pred}, score={score:.4f}."
    )

    if pred == -1:
        logger.debug(f"Global IF flagged anomaly: score={score:.4f}")

    return IFResult(
        is_anomaly=pred == -1,
        anomaly_score=score,
        prediction=pred,
        model_used="global",
        features_used=ALL_FEATURES,
    )


def reload_artifacts():
    """Hot-reload all model artifacts from disk."""
    global _group_cache, _global_model, _global_scaler
    global _global_impute_values, _global_feature_schema
    _group_cache = {}
    _global_model = _global_scaler = None
    _global_impute_values = _global_feature_schema = None
    logger.info("Isolation Forest artifact caches cleared.")
    _load_global()
    logger.info("All model artifacts reloaded.")