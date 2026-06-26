"""
training/train.py
------------------
Trains one Isolation Forest per capability group + one global
fallback model. Includes 80/20 train/test split and full
evaluation (precision, recall, F1, ROC-AUC, confusion matrix).

Run from project root:
    python training/train.py

Outputs
-------
models/
  isolation_forest.joblib    ← global fallback
  scaler.joblib
  impute_values.joblib
  feature_schema.joblib
  group_A/
    isolation_forest.joblib  ← group-specific model
    scaler.joblib
    feature_schema.joblib
  group_B/ ...
  ...
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from config.settings import (
    OBIS_REGISTRY,
    ALL_FEATURES,
    CORE_FEATURES,
    OPTIONAL_FEATURES,
    CAPABILITY_GROUPS,
    DERIVED_FEATURE_MAP,
    MODEL_PATHS,
    DETECTION_CONFIG,
    ROLLING_WINDOW_SIZE,
    group_model_paths,
)
from pipeline.feature_engineer import compute_features

logging.getLogger("pipeline.feature_engineer").setLevel(logging.WARNING)

# =========================================================
# CONFIG
# =========================================================

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset",
    "dynamic_meter_anomaly_dataset.csv"
)

IF_PARAMS = {
    "n_estimators":  200,
    "max_samples":   "auto",
    "contamination": DETECTION_CONFIG["if_contamination"],
    "random_state":  42,
    "n_jobs":        -1,
}

TRAIN_RATIO = 0.8    # 80% of meters used for training, 20% for test

OBIS_TO_CANONICAL = {
    obis: meta["canonical_name"]
    for obis, meta in OBIS_REGISTRY.items()
    if not meta["is_timestamp"] and meta["canonical_name"] is not None
}

# =========================================================
# HELPERS
# =========================================================

def _group_feature_list(raw_features: frozenset) -> list[str]:
    """
    Returns the ordered feature list for a group:
    time features + runtime-emitted raw features + derived features.
    """
    feats = list(DERIVED_FEATURE_MAP["_timestamp"])  # hour, day, weekend, holiday

    if "energy_consumption" in raw_features:
        feats.append("energy_consumption")
        feats += DERIVED_FEATURE_MAP["energy_consumption"]

    if "apparent_import_energy" in raw_features:
        feats.append("apparent_import_energy")

    if "voltage" in raw_features:
        feats.append("voltage")
        feats += DERIVED_FEATURE_MAP["voltage"]

    if "current" in raw_features:
        feats.append("current")
        feats += DERIVED_FEATURE_MAP["current"]

    if "power_factor" in raw_features:
        feats.append("power_factor")
        feats += DERIVED_FEATURE_MAP["power_factor"]

    # Deduplicate preserving order
    seen, ordered = set(), []
    for f in feats:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def _row_to_canonical(row: pd.Series) -> dict:
    canonical = {}
    for key, value in row.items():
        if key in {"meter_serial", "interval_timestamp"}:
            continue
        if pd.isna(value):
            continue
        canonical[key] = value
    return canonical


def _engineer_features(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Computes features using the exact runtime feature engineer.
    History is built row-by-row so rolling stats use only prior readings.
    """
    grp = grp.sort_values("interval_timestamp").copy()
    history: list[dict] = []
    engineered_rows = []

    for _, row in grp.iterrows():
        canonical = _row_to_canonical(row)
        interval_ts = row["interval_timestamp"]
        features = compute_features(
            canonical=canonical,
            interval_ts=interval_ts,
            history=history,
        )

        engineered_row = dict(canonical)
        engineered_row.update(features)
        engineered_row["meter_serial"] = row["meter_serial"]
        engineered_row["interval_timestamp"] = interval_ts
        engineered_rows.append(engineered_row)

        history.append({
            "interval_timestamp": interval_ts,
            "raw_data": canonical,
        })

    return pd.DataFrame(engineered_rows)


def _reconstruct_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Reads exact ground-truth anomaly labels from the anomaly_type
    column stored in raw_data by the dataset generator.
    normal → 0, any anomaly type string → 1.
    Falls back to zeros if column is absent (old dataset format).
    """
    if "anomaly_type" not in df.columns:
        print("        WARNING: anomaly_type column not found — using zero labels.")
        return np.zeros(len(df), dtype=int)
    return (df["anomaly_type"] != "normal").astype(int).values


def _label_breakdown(df: pd.DataFrame, y: np.ndarray) -> None:
    """Prints per-anomaly-type counts for a split."""
    if "anomaly_type" not in df.columns:
        return
    from collections import Counter
    counts = Counter(df["anomaly_type"].values)
    total  = len(df)
    print(f"        Anomaly type breakdown ({y.sum()} total anomalies):")
    for atype, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        marker = "  " if atype == "normal" else "* "
        print(f"          {marker}{atype:<28} {cnt:>5}  ({100*cnt/total:.2f}%)")


def _train_and_evaluate(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_test:  np.ndarray,
    feature_list: list[str],
    label: str,
) -> dict:
    """
    Trains one IF model, evaluates on test set, returns metrics dict.
    No NaN imputation for group models — features are always present.
    For global model NaN imputation is applied before calling.
    """
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    model    = IsolationForest(**IF_PARAMS)
    model.fit(X_tr_sc)

    preds_train = model.predict(X_tr_sc)
    preds_test  = model.predict(X_te_sc)
    scores_test = model.decision_function(X_te_sc)

    # IF returns -1=anomaly, 1=normal → convert to 1=anomaly, 0=normal
    y_pred = (preds_test == -1).astype(int)
    y_score = -scores_test   # higher = more anomalous (for ROC-AUC)

    n_train_anomalies = (preds_train == -1).sum()

    # Compute metrics (handle case where test set has no anomalies)
    has_both_classes = len(np.unique(y_test)) == 2

    metrics = {
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "n_train_anomalies": int(n_train_anomalies),
        "n_test_true_anomalies": int(y_test.sum()),
        "n_test_pred_anomalies": int(y_pred.sum()),
    }

    if has_both_classes:
        metrics["precision"] = round(precision_score(y_test, y_pred, zero_division=0), 4)
        metrics["recall"]    = round(recall_score(y_test, y_pred, zero_division=0), 4)
        metrics["f1"]        = round(f1_score(y_test, y_pred, zero_division=0), 4)
        metrics["roc_auc"]   = round(roc_auc_score(y_test, y_score), 4)
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = {
            "TN": int(cm[0,0]), "FP": int(cm[0,1]),
            "FN": int(cm[1,0]), "TP": int(cm[1,1]),
        }
    else:
        metrics["note"] = "Test set has only one class — metrics not computable"

    return model, scaler, metrics


def _print_metrics(label: str, metrics: dict, feature_list: list):
    print(f"\n  ── {label} ──")
    print(f"     Features ({len(feature_list)}): {feature_list}")
    print(f"     Train rows         : {metrics['n_train']:>6}")
    print(f"     Test rows          : {metrics['n_test']:>6}")
    print(f"     Train anomalies    : {metrics['n_train_anomalies']:>6}  "
          f"({100*metrics['n_train_anomalies']/max(metrics['n_train'],1):.1f}%)")
    print(f"     True test anomalies: {metrics['n_test_true_anomalies']:>6}")
    print(f"     Pred test anomalies: {metrics['n_test_pred_anomalies']:>6}")
    if "precision" in metrics:
        print(f"     Precision  : {metrics['precision']:.4f}")
        print(f"     Recall     : {metrics['recall']:.4f}")
        print(f"     F1 score   : {metrics['f1']:.4f}")
        print(f"     ROC-AUC    : {metrics['roc_auc']:.4f}")
        cm = metrics["confusion_matrix"]
        print(f"     Confusion  : TP={cm['TP']}  FP={cm['FP']}  "
              f"TN={cm['TN']}  FN={cm['FN']}")
    elif "note" in metrics:
        print(f"     Note: {metrics['note']}")


# =========================================================
# STEP 1 — LOAD + PARSE
# =========================================================

print("=" * 65)
print("  METER ANOMALY — ISOLATION FOREST TRAINING")
print("=" * 65)

print("\n[ 1/5 ] Loading and parsing dataset ...")
df_raw = pd.read_csv(CSV_PATH)
print(f"        {len(df_raw)} rows, {df_raw['meter_serial'].nunique()} meters")

def parse_raw(s):
    raw = json.loads(s)
    row = {}
    for k, v in raw.items():
        if k == "0.0.1.0.0.255":
            continue
        elif k == "anomaly_type":
            row["anomaly_type"] = v          # preserve label as-is
        else:
            row[OBIS_TO_CANONICAL.get(k, k)] = v
    return row

parsed        = df_raw["raw_data"].apply(parse_raw)
df_features   = pd.DataFrame(list(parsed))
df_features["meter_serial"]       = df_raw["meter_serial"].values
df_features["interval_timestamp"] = df_raw["interval_timestamp"].values

# =========================================================
# STEP 2 — FEATURE ENGINEERING PER METER
# =========================================================

print("[ 2/5 ] Engineering features per meter ...")

# Drop anomaly_type before engineering (it is a label, not a feature)
df_features_eng = df_features.drop(columns=["anomaly_type"], errors="ignore")

frames = []
for meter, grp in df_features_eng.groupby("meter_serial"):
    frames.append(_engineer_features(grp))
df_eng = pd.concat(frames, ignore_index=True)
# Re-attach anomaly_type labels aligned by original index
if "anomaly_type" in df_features.columns:
    df_eng["anomaly_type"] = df_features["anomaly_type"].values
print(f"        Engineered shape: {df_eng.shape}")

# =========================================================
# STEP 3 — TRAIN / TEST SPLIT (meter-level)
# Keeps all readings of a meter in the same split to prevent
# temporal data leakage across the boundary.
# =========================================================

print(f"[ 3/5 ] Splitting meters 80/20 (train/test) ...")

all_meters    = df_eng["meter_serial"].unique()
np.random.seed(42)
np.random.shuffle(all_meters)
n_train       = max(1, int(len(all_meters) * TRAIN_RATIO))
train_meters  = set(all_meters[:n_train])
test_meters   = set(all_meters[n_train:])

df_train = df_eng[df_eng["meter_serial"].isin(train_meters)].copy()
df_test  = df_eng[df_eng["meter_serial"].isin(test_meters)].copy()

print(f"        Train: {len(df_train)} rows ({len(train_meters)} meters)")
print(f"        Test : {len(df_test)}  rows ({len(test_meters)} meters)")

# Extract exact labels from anomaly_type column
y_test_all = _reconstruct_labels(df_test)
print(f"        Test anomaly labels: {y_test_all.sum()} / {len(y_test_all)}")
_label_breakdown(df_test, y_test_all)

# =========================================================
# STEP 4 — TRAIN PER-GROUP MODELS
# =========================================================

print("\n[ 4/5 ] Training per-capability-group models ...")

all_eval_results = {}

for group_name, raw_features in CAPABILITY_GROUPS.items():

    feature_list = _group_feature_list(raw_features)

    # ── Filter training rows: only meters whose canonical features
    #    exactly match this group's raw features ──────────────────
    def _meter_matches_group(meter_serial, group_raw):
        meter_rows = df_train[df_train["meter_serial"] == meter_serial]
        if meter_rows.empty:
            return False
        # A meter matches a group if all its group features are non-null
        # and it has no extra raw canonical features beyond the group
        present = frozenset(
            col for col in group_raw
            if col in meter_rows.columns and meter_rows[col].notna().any()
        )
        return present == group_raw

    matching_train_meters = [
        m for m in train_meters if _meter_matches_group(m, raw_features)
    ]
    matching_test_meters = [
        m for m in test_meters if _meter_matches_group(m, raw_features)
    ]

    if not matching_train_meters:
        print(f"\n  ── {group_name} ── SKIPPED (no matching training meters)")
        continue

    df_grp_train = df_train[df_train["meter_serial"].isin(matching_train_meters)]
    df_grp_test  = df_test[df_test["meter_serial"].isin(matching_test_meters)]

    # Build feature matrices — only columns in feature_list
    available_train = [f for f in feature_list if f in df_grp_train.columns]
    available_test  = [f for f in feature_list if f in df_grp_test.columns]
    feat_cols       = [f for f in feature_list if f in available_train]

    if not feat_cols:
        print(f"\n  ── {group_name} ── SKIPPED (no feature columns found)")
        continue

    X_train = df_grp_train[feat_cols].fillna(0)  # should have no NaN for matched group
    y_test_grp = np.array([])
    X_test     = pd.DataFrame()

    if not df_grp_test.empty:
        X_test     = df_grp_test[feat_cols].fillna(0)
        test_idx   = df_test[df_test["meter_serial"].isin(matching_test_meters)].index
        y_test_grp = y_test_all[df_test.index.get_indexer(test_idx)]

    if X_test.empty or len(y_test_grp) == 0:
        # Train only, no test evaluation for this group
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        model   = IsolationForest(**IF_PARAMS)
        model.fit(X_tr_sc)
        n_anom  = (model.predict(X_tr_sc) == -1).sum()
        print(f"\n  ── {group_name} ── trained only (no test meters)")
        print(f"     Features: {feat_cols}")
        print(f"     Train rows: {len(X_train)}, anomalies flagged: {n_anom}")
        metrics = {
            "n_train": len(X_train), "n_test": 0,
            "n_train_anomalies": int(n_anom), "note": "no test meters"
        }
    else:
        model, scaler, metrics = _train_and_evaluate(
            X_train, X_test, y_test_grp, feat_cols, group_name
        )
        _print_metrics(group_name, metrics, feat_cols)

    all_eval_results[group_name] = metrics

    # ── Save group artifacts ─────────────────────────────
    paths = group_model_paths(group_name)
    os.makedirs(os.path.dirname(paths["isolation_forest"]), exist_ok=True)

    joblib.dump(model,  paths["isolation_forest"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump({"features": feat_cols, "group_name": group_name,
                 "raw_features": list(raw_features)}, paths["feature_schema"])

    print(f"     Saved → models/{group_name}/")

# =========================================================
# STEP 5 — TRAIN GLOBAL FALLBACK MODEL
# Uses ALL training data with NaN imputation for optional features.
# =========================================================

print("\n[ 5/5 ] Training global fallback model ...")

for col in ALL_FEATURES:
    if col not in df_train.columns:
        df_train[col] = np.nan

X_global_train = df_train[ALL_FEATURES].copy()
impute_values  = X_global_train.median()
X_global_train = X_global_train.fillna(impute_values)

X_global_test  = df_test[ALL_FEATURES].copy().fillna(impute_values)
y_test_global  = y_test_all

global_model, global_scaler, global_metrics = _train_and_evaluate(
    X_global_train, X_global_test, y_test_global, ALL_FEATURES, "global"
)
_print_metrics("GLOBAL FALLBACK", global_metrics, ALL_FEATURES)
all_eval_results["global"] = global_metrics

# ── Save global artifacts ────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATHS["isolation_forest"]), exist_ok=True)
joblib.dump(global_model,  MODEL_PATHS["isolation_forest"])
joblib.dump(global_scaler, MODEL_PATHS["scaler"])
joblib.dump(impute_values, MODEL_PATHS["impute_values"])
joblib.dump({
    "all_features":      ALL_FEATURES,
    "core_features":     CORE_FEATURES,
    "optional_features": OPTIONAL_FEATURES,
}, MODEL_PATHS["feature_schema"])

print("\n  Saved → models/  (global fallback)")

# =========================================================
# SUMMARY TABLE
# =========================================================

print("\n" + "=" * 65)
print("  EVALUATION SUMMARY")
print("=" * 65)
print(f"  {'Model':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("  " + "-" * 53)

for name, m in all_eval_results.items():
    if "precision" in m:
        print(f"  {name:<12} {m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['roc_auc']:>9.4f}")
    else:
        note = m.get("note", "")
        print(f"  {name:<12} {'—':>10} {'—':>8} {'—':>8} {'—':>9}  ({note})")

print("\n✓ Training complete.")