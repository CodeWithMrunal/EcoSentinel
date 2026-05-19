"""
training/train.py
------------------
Reads the OBIS-format dataset, applies canonical mapping,
builds the fixed feature matrix, and trains the Isolation Forest.

Run from project root:
    python training/train.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import json
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config.settings import (
    OBIS_REGISTRY,
    ALL_FEATURES,
    CORE_FEATURES,
    OPTIONAL_FEATURES,
    MODEL_PATHS,
    DETECTION_CONFIG,
)

# =========================================================
# PATHS
# =========================================================

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset",
    "dynamic_meter_anomaly_dataset.csv"
)

ISOLATION_FOREST_PARAMS = {
    "n_estimators":  200,
    "max_samples":   "auto",
    "contamination": DETECTION_CONFIG["if_contamination"],
    "random_state":  42,
    "n_jobs":        -1,
}

# =========================================================
# CANONICAL MAPPING
# Built from OBIS_REGISTRY — single source of truth.
# Maps OBIS code → canonical feature name.
# is_timestamp entries are excluded (not features).
# =========================================================

OBIS_TO_CANONICAL = {
    obis: meta["canonical_name"]
    for obis, meta in OBIS_REGISTRY.items()
    if not meta["is_timestamp"] and meta["canonical_name"] is not None
}

# =========================================================
# STEP 1 — LOAD
# =========================================================

print("[ 1/5 ] Loading dataset ...")
df_raw = pd.read_csv(CSV_PATH)
print(f"        {len(df_raw)} rows loaded from: {CSV_PATH}")

# =========================================================
# STEP 2 — PARSE raw_data + CANONICAL MAPPING
# Each raw_data is a JSON dict keyed by OBIS codes.
# Map each OBIS key → canonical name.
# Non-OBIS keys (shouldn't exist in real data but safe to
# handle) are passed through as-is.
# =========================================================

print("[ 2/5 ] Parsing raw_data and applying canonical mapping ...")

def parse_and_canonicalize(raw_json_str: str) -> dict:
    raw = json.loads(raw_json_str)
    canonical = {}
    for key, value in raw.items():
        # Skip the timestamp OBIS entry — not a feature
        if key == "0.0.1.0.0.255":
            continue
        canonical_key = OBIS_TO_CANONICAL.get(key, key)
        canonical[canonical_key] = value
    return canonical

parsed = df_raw["raw_data"].apply(parse_and_canonicalize)
df_features = pd.DataFrame(list(parsed))

# Attach identifiers for traceability
df_features["meter_serial"]       = df_raw["meter_serial"].values
df_features["interval_timestamp"] = df_raw["interval_timestamp"].values

print(f"        Canonical columns found: {sorted(df_features.columns.tolist())}")

# =========================================================
# STEP 3 — FEATURE ENGINEERING
# Derived features are computed per-meter in chronological
# order, matching exactly what the inference pipeline does.
# =========================================================

print("[ 3/5 ] Computing derived features per meter ...")

engineered_frames = []

for meter_serial, grp in df_features.groupby("meter_serial"):

    grp = grp.sort_values("interval_timestamp").copy()

    ts = pd.to_datetime(grp["interval_timestamp"])

    # ── Time features ────────────────────────────────────
    grp["hour_of_day"] = ts.dt.hour
    grp["day_of_week"] = ts.dt.dayofweek
    grp["is_weekend"]  = (grp["day_of_week"] >= 5).astype(int)
    grp["holiday"]     = ts.apply(lambda x: 1 if x.weekday() == 6 else 0)

    ec = grp["energy_consumption"]

    # ── Delta ────────────────────────────────────────────
    grp["delta"] = ec.diff().fillna(0)

    # ── Current delta ────────────────────────────────────
    if "current" in grp.columns:
        grp["current_delta"] = grp["current"].diff().fillna(0)

    # ── Rolling features ─────────────────────────────────
    grp["rolling_mean"] = ec.rolling(window=5, min_periods=1).mean()
    grp["rolling_std"]  = ec.rolling(window=5, min_periods=1).std().fillna(0)

    # ── Z-score ──────────────────────────────────────────
    grp["z_score"] = (ec - grp["rolling_mean"]) / (grp["rolling_std"] + 1e-5)

    # ── Spike ratio ──────────────────────────────────────
    grp["spike_ratio"] = ec / (grp["rolling_mean"] + 1e-5)

    # ── Voltage deviation ────────────────────────────────
    if "voltage" in grp.columns:
        grp["voltage_deviation"] = grp["voltage"] - 230.0

    # ── Power factor deviation ───────────────────────────
    if "power_factor" in grp.columns:
        grp["power_factor_deviation"] = 1.0 - grp["power_factor"]

    # ── Historical averages (within this meter's window) ─
    grp["historical_avg_same_hour"] = (
        grp.groupby("hour_of_day")["energy_consumption"]
        .transform("mean")
    )
    grp["historical_avg_same_day_type"] = (
        grp.groupby("is_weekend")["energy_consumption"]
        .transform("mean")
    )

    engineered_frames.append(grp)

df_eng = pd.concat(engineered_frames, ignore_index=True)
print(f"        Engineered dataframe shape: {df_eng.shape}")

# =========================================================
# STEP 4 — BUILD FIXED FEATURE MATRIX
# Ensure every ALL_FEATURES column exists.
# Missing optional features → NaN → median imputation.
# =========================================================

print("[ 4/5 ] Building fixed feature matrix and training model ...")

for col in ALL_FEATURES:
    if col not in df_eng.columns:
        df_eng[col] = np.nan

X = df_eng[ALL_FEATURES].copy()

# Report NaN distribution before imputation
missing = X.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    print("        NaN optional features (median-imputed):")
    for col, count in missing.items():
        print(f"          {col:<30} {count:>5}  ({100*count/len(X):.1f}%)")

impute_values = X.median()
X_imputed = X.fillna(impute_values)

print(f"        Feature matrix: {X_imputed.shape}")

# ── Scale ─────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ── Train ─────────────────────────────────────────────────
model       = IsolationForest(**ISOLATION_FOREST_PARAMS)
model.fit(X_scaled)

predictions = model.predict(X_scaled)
scores      = model.decision_function(X_scaled)
n_anomalies = (predictions == -1).sum()

print(f"        Training complete.")
print(f"        Anomalies flagged: {n_anomalies} / {len(X_scaled)} "
      f"({100*n_anomalies/len(X_scaled):.2f}%)")

# =========================================================
# STEP 5 — SAVE ARTIFACTS
# =========================================================

print("[ 5/5 ] Saving artifacts ...")

os.makedirs(os.path.dirname(MODEL_PATHS["isolation_forest"]), exist_ok=True)

joblib.dump(model,         MODEL_PATHS["isolation_forest"])
joblib.dump(scaler,        MODEL_PATHS["scaler"])
joblib.dump(impute_values, MODEL_PATHS["impute_values"])

feature_schema = {
    "all_features":      ALL_FEATURES,
    "core_features":     CORE_FEATURES,
    "optional_features": OPTIONAL_FEATURES,
}
joblib.dump(feature_schema, MODEL_PATHS["feature_schema"])

for name, path in MODEL_PATHS.items():
    print(f"        {name:<20} → {path}")

# =========================================================
# SANITY CHECK
# =========================================================

print("\n[ Sanity Check ] Inference on sample rows ...")

normal_idx  = list(np.where(predictions == 1)[0][:2])
anomaly_idx = list(np.where(predictions == -1)[0][:1])

for idx in normal_idx + anomaly_idx:
    row       = X_imputed.iloc[[idx]]
    row_sc    = scaler.transform(row)
    pred      = model.predict(row_sc)[0]
    score     = model.decision_function(row_sc)[0]
    label     = "ANOMALY" if pred == -1 else "NORMAL"
    meter     = df_eng.iloc[idx]["meter_serial"]
    ts        = df_eng.iloc[idx]["interval_timestamp"]
    print(f"   [{label}]  score={score:+.4f}  meter={meter}  ts={ts}")

print(f"\n✓ All artifacts saved to: {os.path.dirname(MODEL_PATHS['isolation_forest'])}")