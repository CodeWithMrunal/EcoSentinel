import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIG
# =========================================================

CSV_PATH = "../dataset/dynamic_meter_anomaly_dataset.csv"
MODEL_OUTPUT_DIR = "../models"

ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": 0.05,   # ~5% anomaly rate matches our injection rate
    "random_state": 42,
    "n_jobs": -1
}

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# =========================================================
# CANONICAL MAPPING
# Maps all vendor-specific parameter name variants
# to a single canonical feature name.
# This mirrors what the real preprocessing pipeline
# will do on incoming live payloads.
# =========================================================

CANONICAL_MAP = {
    # Energy
    "energy_consumption":   "energy_consumption",
    "active_energy":        "energy_consumption",
    "Active Energy":        "energy_consumption",
    "Import Active Energy": "energy_consumption",
    "Export Active Energy": "energy_consumption",
    "kWh":                  "energy_consumption",

    # Reactive Energy
    "reactive_energy":         "reactive_energy",
    "Reactive Energy":         "reactive_energy",
    "Import Reactive Energy":  "reactive_energy",
    "Export Reactive Energy":  "reactive_energy",
    "kVARh":                   "reactive_energy",

    # Voltage
    "voltage":        "voltage",
    "Voltage":        "voltage",
    "Line Voltage":   "voltage",
    "Phase Voltage":  "voltage",

    # Current
    "current":       "current",
    "Current":       "current",
    "Phase Current": "current",
    "Line Current":  "current",

    # Power Factor
    "power_factor":  "power_factor",
    "Power Factor":  "power_factor",
    "PF":            "power_factor",
}

# =========================================================
# FEATURE SCHEMA
# This is the FIXED feature vector the model is trained on.
# Optional features are NaN-imputed when absent.
# =========================================================

# Features always present (every meter sends these)
CORE_FEATURES = [
    "energy_consumption",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "holiday",
    "delta",
    "rolling_mean",
    "rolling_std",
    "z_score",
    "spike_ratio",
    "historical_avg_same_hour",
    "historical_avg_same_day_type",
]

# Features present only for certain meter capability groups
OPTIONAL_FEATURES = [
    "voltage",
    "current",
    "power_factor",
    "reactive_energy",
    "current_delta",
    "voltage_deviation",
    "power_factor_deviation",
]

ALL_FEATURES = CORE_FEATURES + OPTIONAL_FEATURES

# =========================================================
# STEP 1 — LOAD CSV
# =========================================================

print("[ 1/5 ] Loading dataset ...")

df_raw = pd.read_csv(CSV_PATH)
print(f"       Loaded {len(df_raw)} rows from '{CSV_PATH}'")

# =========================================================
# STEP 2 — PARSE + CANONICALIZE raw_data PAYLOADS
# =========================================================

print("[ 2/5 ] Parsing and canonicalizing raw_data payloads ...")

def parse_and_canonicalize(raw_json_str):
    """
    Parses a raw JSON payload string and remaps all vendor-specific
    parameter names to canonical names.
    Unknown keys (like derived features already in canonical form)
    are passed through as-is.
    """
    raw = json.loads(raw_json_str)
    canonical = {}
    for key, value in raw.items():
        canonical_key = CANONICAL_MAP.get(key, key)
        canonical[canonical_key] = value
    return canonical

parsed_records = df_raw["raw_data"].apply(parse_and_canonicalize)
df_features = pd.DataFrame(list(parsed_records))

# Attach meter_id and timestamp for traceability
df_features["meter_id"] = df_raw["meter_id"].values
df_features["timestamp"] = df_raw["timestamp"].values

print(f"       Parsed columns: {sorted(df_features.columns.tolist())}")

# =========================================================
# STEP 3 — BUILD FIXED FEATURE MATRIX
# Ensures every row has all ALL_FEATURES columns.
# Optional features missing for a meter are NaN → imputed.
# =========================================================

print("[ 3/5 ] Building fixed feature matrix ...")

# Add any missing optional feature columns as NaN
for col in ALL_FEATURES:
    if col not in df_features.columns:
        df_features[col] = np.nan

X = df_features[ALL_FEATURES].copy()

# Report missingness before imputation
missing_summary = X.isnull().sum()
missing_cols = missing_summary[missing_summary > 0]
if not missing_cols.empty:
    print("       Optional features with NaN (will be imputed with median):")
    for col, count in missing_cols.items():
        pct = 100 * count / len(X)
        print(f"         {col:<30} {count:>5} rows  ({pct:.1f}%)")

# Impute NaN with column median
# Median is more robust than mean for features that may contain anomalies
impute_values = X.median()
X_imputed = X.fillna(impute_values)

print(f"       Feature matrix shape: {X_imputed.shape}")

# =========================================================
# STEP 4 — SCALE FEATURES
# IsolationForest is tree-based and doesn't strictly require
# scaling, but StandardScaler improves anomaly score
# consistency across features with very different ranges
# (e.g. energy_consumption vs voltage_deviation).
# =========================================================

print("[ 4/5 ] Scaling features and training Isolation Forest ...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# =========================================================
# TRAIN ISOLATION FOREST
# =========================================================

model = IsolationForest(**ISOLATION_FOREST_PARAMS)
model.fit(X_scaled)

# Anomaly scores: more negative = more anomalous
# Predictions: -1 = anomaly, 1 = normal
scores = model.decision_function(X_scaled)
predictions = model.predict(X_scaled)

n_anomalies = (predictions == -1).sum()
print(f"       Training complete.")
print(f"       Anomalies flagged on training data: {n_anomalies} / {len(X_scaled)} "
      f"({100 * n_anomalies / len(X_scaled):.2f}%)")

# =========================================================
# STEP 5 — SAVE ARTIFACTS
# =========================================================

print("[ 5/5 ] Saving model artifacts ...")

# Save the trained Isolation Forest model
model_path = os.path.join(MODEL_OUTPUT_DIR, "isolation_forest.joblib")
joblib.dump(model, model_path)
print(f"       Model saved       → {model_path}")

# Save the scaler (must use same scaler at inference time)
scaler_path = os.path.join(MODEL_OUTPUT_DIR, "scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"       Scaler saved      → {scaler_path}")

# Save imputation values (median per feature)
impute_path = os.path.join(MODEL_OUTPUT_DIR, "impute_values.joblib")
joblib.dump(impute_values, impute_path)
print(f"       Impute values saved → {impute_path}")

# Save feature schema so inference script uses exact same column order
feature_schema = {
    "all_features": ALL_FEATURES,
    "core_features": CORE_FEATURES,
    "optional_features": OPTIONAL_FEATURES,
}
schema_path = os.path.join(MODEL_OUTPUT_DIR, "feature_schema.joblib")
joblib.dump(feature_schema, schema_path)
print(f"       Feature schema saved → {schema_path}")

# =========================================================
# QUICK SANITY CHECK — INFERENCE ON A SAMPLE ROW
# =========================================================

print("\n[ Sanity Check ] Running inference on 3 sample rows ...")

def infer(row_index):
    row = X_imputed.iloc[[row_index]]
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    score = model.decision_function(row_scaled)[0]
    label = "ANOMALY" if pred == -1 else "NORMAL"
    print(f"   Row {row_index:>4} → {label}  (anomaly score: {score:.4f})")

# Sample a few normal + flagged rows
normal_indices = list(np.where(predictions == 1)[0][:2])
anomaly_indices = list(np.where(predictions == -1)[0][:1])

for idx in normal_indices + anomaly_indices:
    infer(idx)

print("\n✓ Training complete. All artifacts saved to:", MODEL_OUTPUT_DIR)