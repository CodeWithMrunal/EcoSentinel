"""
dataset/generate_dataset.py
----------------------------
Generates a synthetic smart meter dataset that mirrors the real
API payload format:

  - Outer envelope: id, meterSerial, timestamp, obisCode, entryId
  - raw_data (JSONB): parsed rawValue stored as { obis_code: value }
                      (pipe-string already parsed; values keyed by OBIS code)

The dataset is saved as:
  dynamic_meter_anomaly_dataset.csv
with columns:
  id | meter_serial | received_at | profile_obis_code | entry_id |
  interval_timestamp | raw_data (JSON string, OBIS-keyed)

At training time, the canonical mapper converts OBIS keys → feature names.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta

from config.settings import (
    DATASET_CONFIG,
    OBIS_REGISTRY,
    METER_CAPABILITY_PROFILES,
)

# =========================================================
# SEED
# =========================================================

np.random.seed(DATASET_CONFIG["random_seed"])
random.seed(DATASET_CONFIG["random_seed"])

# =========================================================
# CONSTANTS
# =========================================================

NUM_METERS   = DATASET_CONFIG["num_meters"]
DAYS         = DATASET_CONFIG["days"]
FREQ_MIN     = DATASET_CONFIG["freq_minutes"]      # 30
START_TIME   = datetime.fromisoformat(DATASET_CONFIG["start_time"])

# Load-survey profile OBIS code (matches real HES payloads)
LOAD_SURVEY_OBIS = "1.0.99.1.0.255"

# Timestamp OBIS code (entry 1 of every rawValue)
TIMESTAMP_OBIS = "0.0.1.0.0.255"

# =========================================================
# HELPERS
# =========================================================

def is_holiday(date: datetime) -> bool:
    return date.weekday() == 6          # Sunday


def generate_base_consumption(hour: int, is_weekend: bool) -> float:
    base = 1.5 + 1.0 * (18 <= hour <= 23) + 0.5 * (6 <= hour <= 9)
    if is_weekend:
        base *= 0.9
    return base + np.random.normal(0, 0.2)


def build_raw_data(
    interval_ts: datetime,
    supported_obis: list[str],
    energy: float,
    voltage: float | None,
    current: float | None,
    power_factor: float | None,
    apparent_energy: float | None,
    active_export_energy: float | None,
    frequency: float | None,
) -> dict:
    """
    Builds the raw_data dict exactly as it would be stored
    after parsing the pipe-delimited rawValue string.

    Keys are OBIS codes. Only OBIS codes in supported_obis
    are included (simulates different meter capability profiles).

    Entry 1 (timestamp) is stored under its OBIS code so the
    parser/mapper can extract it correctly.
    """
    raw = {}

    # Entry 1 is always the interval timestamp
    raw[TIMESTAMP_OBIS] = interval_ts.strftime("%Y-%m-%d %H:%M:%S")

    for obis in supported_obis:
        if obis == "1.0.1.29.0.255":
            raw[obis] = round(energy, 3)

        elif obis == "1.0.12.27.0.255" and voltage is not None:
            raw[obis] = round(voltage, 3)

        elif obis == "1.0.11.27.0.255" and current is not None:
            raw[obis] = round(current, 3)

        elif obis == "1.0.13.27.0.255" and power_factor is not None:
            raw[obis] = round(power_factor, 3)

        elif obis == "1.0.9.29.0.255" and apparent_energy is not None:
            raw[obis] = round(apparent_energy, 3)

        elif obis == "1.0.2.29.0.255" and active_export_energy is not None:
            raw[obis] = round(active_export_energy, 3)

        elif obis == "1.0.14.27.0.255" and frequency is not None:
            raw[obis] = round(frequency, 3)

    return raw


# =========================================================
# MAIN GENERATION LOOP
# =========================================================

all_rows = []
global_id = 1       # simulates the auto-increment API id

for meter_idx in range(1, NUM_METERS + 1):

    meter_serial   = f"E{meter_idx:07d}"
    capability     = random.choice(METER_CAPABILITY_PROFILES)

    # --------------------------------------------------
    # Build timestamp series (30-min intervals)
    # --------------------------------------------------
    steps = int((24 * 60 / FREQ_MIN) * DAYS)
    timestamps = [
        START_TIME + timedelta(minutes=FREQ_MIN * i)
        for i in range(steps)
    ]

    # --------------------------------------------------
    # Simulate electrical values for the whole series
    # --------------------------------------------------

    # Pre-generate optional electrical parameters
    voltage_series = (
        np.random.normal(230, 5, steps)
        if "1.0.12.27.0.255" in capability else None
    )
    pf_series = (
        np.random.uniform(0.85, 1.0, steps)
        if "1.0.13.27.0.255" in capability else None
    )
    apparent_series = (
        np.random.uniform(0.5, 3.0, steps)
        if "1.0.9.29.0.255" in capability else None
    )
    export_series = (
        np.random.uniform(0.0, 0.5, steps)
        if "1.0.2.29.0.255" in capability else None
    )
    freq_series = (
        np.random.normal(50.0, 0.1, steps)
        if "1.0.14.27.0.255" in capability else None
    )
    current_multiplier = (
        np.random.uniform(0.8, 1.2)
        if "1.0.11.27.0.255" in capability else None
    )

    # --------------------------------------------------
    # Build rows
    # --------------------------------------------------

    for i, ts in enumerate(timestamps):

        hour         = ts.hour
        is_weekend   = ts.weekday() >= 5

        energy = generate_base_consumption(hour, is_weekend)

        # Inject anomalies (~2% spikes, ~0.5% negatives)
        if np.random.rand() < 0.02:
            energy *= np.random.uniform(3, 8)
        if np.random.rand() < 0.005:
            energy *= -1

        energy = round(energy, 2)

        # Current is derived from energy
        current = (
            round(abs(energy) * current_multiplier, 3)
            if current_multiplier is not None else None
        )

        raw_data = build_raw_data(
            interval_ts          = ts,
            supported_obis       = capability,
            energy               = energy,
            voltage              = float(voltage_series[i]) if voltage_series is not None else None,
            current              = current,
            power_factor         = float(pf_series[i]) if pf_series is not None else None,
            apparent_energy      = float(apparent_series[i]) if apparent_series is not None else None,
            active_export_energy = float(export_series[i]) if export_series is not None else None,
            frequency            = float(freq_series[i]) if freq_series is not None else None,
        )

        # Simulate a realistic received_at: a few seconds after interval
        received_at = ts + timedelta(seconds=random.randint(5, 30))

        all_rows.append({
            "id":                  global_id,
            "meter_serial":        meter_serial,
            "received_at":         received_at.isoformat() + "+00:00",
            "profile_obis_code":   LOAD_SURVEY_OBIS,
            "entry_id":            i + 1,
            "interval_timestamp":  ts.isoformat(),
            "raw_data":            json.dumps(raw_data),
        })

        global_id += 1

# =========================================================
# SAVE
# =========================================================

df = pd.DataFrame(all_rows)

out_path = os.path.join(os.path.dirname(__file__), "dynamic_meter_anomaly_dataset.csv")
df.to_csv(out_path, index=False)

print(f"Generated {len(df)} rows → {out_path}")
print(f"Columns : {df.columns.tolist()}")
print(f"\nSample raw_data (first row):")
print(json.dumps(json.loads(df.iloc[0]["raw_data"]), indent=2))