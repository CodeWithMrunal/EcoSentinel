"""
dataset/generate_dataset.py
----------------------------
Generates a physically plausible synthetic smart meter dataset.

Key improvements over previous version:
  1. DIURNAL LOAD PROFILE  — hour-by-hour consumption curve matching
                             real residential/commercial patterns.
                             Weekday vs weekend differ meaningfully.

  2. PHYSICS-BASED PARAMS  — all electrical parameters are derived
                             from a single load value using real
                             electrical relationships:
                               P  = load_kW
                               I  = P / (V × PF)          [current from power]
                               S  = P / PF                 [apparent power VA]
                               E  = P × (interval_hours)   [energy Wh]
                               Eapp = S × (interval_hours) [apparent energy VAh]
                             Voltage droops slightly under load.

  3. CORRELATED ANOMALIES  — when an anomaly is injected, related
                             parameters respond physically:
                             - energy spike  → current spike, voltage sag
                             - tamper        → energy low, current high (bypass)
                             - voltage event → voltage changes, PF affected
                             - PF collapse   → PF drops, apparent energy rises
                             - zero consumption → all parameters drop together

  4. ANOMALY DIVERSITY     — subtle (1.2–2×), moderate (2–4×), and
                             obvious (4–8×) ranges; 8 anomaly types.

  5. ANOMALY LABEL COLUMN  — raw_data includes "anomaly_type" key so
                             training can reconstruct precise labels
                             without heuristic guessing.
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

NUM_METERS      = DATASET_CONFIG["num_meters"]
DAYS            = DATASET_CONFIG["days"]
FREQ_MIN        = DATASET_CONFIG["freq_minutes"]   # 30
INTERVAL_HOURS  = FREQ_MIN / 60.0                  # 0.5 h
START_TIME      = datetime.fromisoformat(DATASET_CONFIG["start_time"])

LOAD_SURVEY_OBIS = "1.0.99.1.0.255"
TIMESTAMP_OBIS   = "0.0.1.0.0.255"

NOMINAL_VOLTAGE  = 230.0    # V
NOMINAL_FREQ     = 50.0     # Hz

# =========================================================
# DIURNAL LOAD PROFILES  (kW — average load in each hour)
#
# Residential weekday:
#   Night trough 0–5am, morning peak 6–9am (breakfast, shower),
#   midday dip 10am–5pm (people at work), evening peak 6–10pm.
#
# Residential weekend:
#   Later morning rise, sustained midday activity, similar evening peak.
#
# Values are mean load in kW. Per-reading noise is added later.
# =========================================================

HOURLY_LOAD_WEEKDAY = np.array([
    0.30, 0.25, 0.22, 0.20, 0.20, 0.25,   # 00–05  night trough
    0.60, 1.10, 1.20, 0.90, 0.70, 0.65,   # 06–11  morning peak + taper
    0.60, 0.55, 0.55, 0.60, 0.70, 0.85,   # 12–17  midday plateau
    1.30, 1.50, 1.40, 1.20, 0.90, 0.55,   # 18–23  evening peak
])

HOURLY_LOAD_WEEKEND = np.array([
    0.30, 0.25, 0.22, 0.20, 0.20, 0.28,   # 00–05
    0.40, 0.60, 0.90, 1.10, 1.20, 1.15,   # 06–11  later rise, higher midday
    1.10, 1.05, 1.00, 0.95, 0.90, 1.00,   # 12–17  sustained activity
    1.35, 1.50, 1.40, 1.20, 0.90, 0.55,   # 18–23  evening peak same
])

# Noise std as fraction of mean load (±15% coefficient of variation)
# This parameter controls how much randomness is added to the expected load for a given hour.
LOAD_NOISE_CV = 0.15

# =========================================================
# ELECTRICAL PARAMETERS — realistic ranges
# =========================================================

# Voltage: nominally 230V with small random walk + load droop
# Droop: each 1A of current causes ~0.05V drop (typical LV line)
VOLTAGE_BASE_STD   = 2.0    # V  — slow background variation
VOLTAGE_NOISE_STD  = 0.5    # V  — fast measurement noise
VOLTAGE_DROOP_PER_AMP = 0.05  # V/A - for calculating voltage drop under load

# Power factor: load-dependent
# Lighter loads (resistive) → higher PF; heavier loads → slightly lower
PF_BASE            = 0.92 # remaining 0.08 is reactive component
PF_LOAD_VARIATION  = 0.05   # PF range around base

# Frequency: very stable grid (±0.05 Hz typical)
FREQ_STD           = 0.04   # Hz

# =========================================================
# ANOMALY CATALOGUE
#
# Each entry defines:
#   prob       — probability per reading that this anomaly fires
#   severity   — tuple (min_multiplier, max_multiplier) OR fixed params
#   description— human-readable label stored in anomaly_type field
#
# Total injected anomaly rate ≈ sum of probs ≈ 5–6%
# =========================================================

# Anomaly severities intentionally span subtle → obvious
ANOMALY_CATALOGUE = {

    # ── Energy anomalies ─────────────────────────────────

    "subtle_energy_spike": {
        # 1.2–2× spike — hard for single-threshold rules to catch
        # Correlated: current rises proportionally, voltage sags slightly
        "prob":     0.012,
        "energy_mult": (1.2, 2.0),
    },
    "obvious_energy_spike": {
        # 4–8× spike — detectable by z-score too
        # Correlated: large current spike, clear voltage sag
        "prob":     0.008,
        "energy_mult": (4.0, 8.0),
    },
    "negative_energy": {
        # Meter register rollover or CT reversal
        # Correlated: current still positive, energy negative
        "prob":     0.004,
        "energy_mult": None,   # handled separately (sign flip)
    },
    "sustained_zero": {
        # Communication failure or disconnection — 3+ consecutive zeros
        # Handled at series level, not per-reading
        "prob":     0.003,
        "energy_mult": None,
    },

    # ── Tamper anomalies ─────────────────────────────────

    "tamper_bypass": {
        # Illegal bypass: energy reads low, but current is high
        # (current flows outside the meter CT)
        # Energy suppressed to 20–40% of expected,
        # current stays at full expected value
        "prob":     0.006,
        "energy_mult": (0.2, 0.4),
    },

    # ── Voltage anomalies ────────────────────────────────

    "voltage_sag": {
        # Voltage drops to 155–185V (heavy load or supply fault)
        # PF slightly affected, current rises (same power, lower voltage)
        "prob":     0.008,
        "voltage_range": (155.0, 185.0),
    },
    "voltage_swell": {
        # Voltage rises to 250–275V (capacitor bank switching)
        # Current drops slightly (same power, higher voltage)
        "prob":     0.005,
        "voltage_range": (250.0, 275.0),
    },

    # ── Power quality anomalies ──────────────────────────

    "pf_collapse": {
        # PF drops to 0.55–0.72 (large inductive load switch-on)
        # Apparent energy rises significantly,
        # current rises (more reactive current)
        "prob":     0.007,
        "pf_range": (0.55, 0.72),
    },
}

# =========================================================
# PHYSICS HELPERS
# =========================================================

def load_kw_for(hour: int, is_weekend: bool, meter_load_scale: float) -> float:
    """
    Returns the mean load in kW for a given hour and day type,
    scaled by a per-meter factor (simulates different household sizes).
    Adds multiplicative noise (CV = LOAD_NOISE_CV).
    """
    profile = HOURLY_LOAD_WEEKEND if is_weekend else HOURLY_LOAD_WEEKDAY
    mean_kw = profile[hour] * meter_load_scale
    noise   = np.random.normal(1.0, LOAD_NOISE_CV)
    noise   = max(0.1, noise)          # never go negative in normal operation
    return mean_kw * noise


def derive_electrical(
    load_kw:        float,
    voltage_base:   float,
    pf:             float,
    interval_hours: float,
) -> dict:
    """
    Derives all electrical parameters from a load value using
    standard single-phase AC power equations.

    P  = load_kw  (active power, kW)
    I  = P / (V × PF / 1000)          [A]  — divide by 1000 for kW→W
    S  = P / PF                        [kVA]
    Q  = sqrt(S² - P²)                 [kVAR]
    E  = P × interval_hours            [kWh → converted to Wh]
    Eapp = S × interval_hours          [kVAh → converted to VAh]

    Voltage droop: V_actual = V_base - I × VOLTAGE_DROOP_PER_AMP
    """
    P_kw = load_kw
    pf   = max(0.1, min(1.0, pf))     # clamp PF to valid range

    # Current (A) — from P = V × I × PF  →  I = P×1000 / (V × PF)
    I = (P_kw * 1000.0) / (voltage_base * pf + 1e-6)

    # Voltage with droop under load
    V = voltage_base - I * VOLTAGE_DROOP_PER_AMP
    V = max(100.0, V)   # physical floor

    # Recalculate current with actual voltage
    I = (P_kw * 1000.0) / (V * pf + 1e-6)

    # Apparent power kVA
    S_kva = P_kw / pf

    # Energy this interval
    E_wh    = P_kw * interval_hours * 1000.0     # kWh → Wh
    Eapp_vah = S_kva * interval_hours * 1000.0   # kVAh → VAh

    return {
        "voltage":  round(V + np.random.normal(0, VOLTAGE_NOISE_STD), 3),
        "current":  round(max(0.0, I + np.random.normal(0, 0.02 * I + 0.01)), 3),
        "power_factor": round(pf, 4),
        "energy_consumption": round(max(0.0, E_wh), 3),
        "apparent_energy": round(max(0.0, Eapp_vah), 3),
    }


def sample_pf(load_kw: float, base_pf: float) -> float:
    """
    Power factor is slightly load-dependent:
    lighter loads (few W) tend to be more purely resistive → higher PF.
    Heavy loads with motors/electronics → slightly lower PF.
    """
    # Normalise load against a 1.5kW reference
    load_factor = min(1.0, load_kw / 1.5)
    pf = base_pf - PF_LOAD_VARIATION * (1.0 - load_factor)
    pf = pf + np.random.normal(0, 0.01)
    return float(np.clip(pf, 0.75, 0.99))


def sample_frequency() -> float:
    """Grid frequency: very stable, narrow Gaussian around 50Hz."""
    return float(np.random.normal(NOMINAL_FREQ, FREQ_STD))


# =========================================================
# ANOMALY INJECTION
# Returns modified electrical dict + anomaly_type label.
# "normal" means no anomaly was injected this reading.
# =========================================================

def inject_anomaly(
    elec:           dict,
    load_kw:        float,
    voltage_base:   float,
    pf_base:        float,
    capability:     list[str],
) -> tuple[dict, str]:
    """
    Probabilistically injects one anomaly per reading (or none).
    Anomalies modify the electrical dict in a physically correlated way.

    Returns (modified_elec, anomaly_type_label).
    """
    roll = np.random.rand()
    cumulative = 0.0

    for anom_name, anom_cfg in ANOMALY_CATALOGUE.items():
        cumulative += anom_cfg["prob"]
        if roll < cumulative:
            return _apply_anomaly(elec, anom_name, anom_cfg,
                                  load_kw, voltage_base, pf_base, capability)

    return elec, "normal"


def _apply_anomaly(
    elec:        dict,
    anom_name:   str,
    anom_cfg:    dict,
    load_kw:     float,
    voltage_base: float,
    pf_base:     float,
    capability:  list[str],
) -> tuple[dict, str]:
    """Applies one specific anomaly type with correlated parameter changes."""

    e = dict(elec)   # copy — never mutate in-place

    has_voltage = "1.0.12.27.0.255" in capability
    has_current = "1.0.11.27.0.255" in capability
    has_pf      = "1.0.13.27.0.255" in capability
    has_app_e   = "1.0.9.29.0.255"  in capability

    # ── subtle_energy_spike ──────────────────────────────
    if anom_name == "subtle_energy_spike":
        mult = np.random.uniform(*anom_cfg["energy_mult"])
        e["energy_consumption"] = round(e["energy_consumption"] * mult, 3)
        # Current rises proportionally (same voltage, more power)
        if has_current:
            e["current"] = round(e["current"] * mult, 3)
        # Voltage sags under increased load
        if has_voltage:
            extra_I = e["current"] - elec["current"]
            e["voltage"] = round(e["voltage"] - extra_I * VOLTAGE_DROOP_PER_AMP, 3)
        # Apparent energy rises with energy
        if has_app_e and e.get("apparent_energy") is not None:
            e["apparent_energy"] = round(e["apparent_energy"] * mult, 3)

    # ── obvious_energy_spike ─────────────────────────────
    elif anom_name == "obvious_energy_spike":
        mult = np.random.uniform(*anom_cfg["energy_mult"])
        e["energy_consumption"] = round(e["energy_consumption"] * mult, 3)
        if has_current:
            e["current"] = round(e["current"] * mult, 3)
        if has_voltage:
            extra_I = e.get("current", 0) - elec.get("current", 0)
            sag = extra_I * VOLTAGE_DROOP_PER_AMP * np.random.uniform(1.0, 2.5)
            e["voltage"] = round(max(160.0, e["voltage"] - sag), 3)
        if has_app_e and e.get("apparent_energy") is not None:
            e["apparent_energy"] = round(e["apparent_energy"] * mult, 3)

    # ── negative_energy ──────────────────────────────────
    elif anom_name == "negative_energy":
        e["energy_consumption"] = round(-abs(e["energy_consumption"]), 3)
        # Current remains positive (CT still measures flow)
        # Voltage unaffected

    # ── sustained_zero (handled at series level) ─────────
    elif anom_name == "sustained_zero":
        e["energy_consumption"] = 0.0
        if has_current:
            e["current"] = 0.0
        # Voltage can remain (grid still connected, just no consumption)

    # ── tamper_bypass ────────────────────────────────────
    elif anom_name == "tamper_bypass":
        mult = np.random.uniform(*anom_cfg["energy_mult"])
        # Energy reads low (only partial load goes through meter)
        e["energy_consumption"] = round(e["energy_consumption"] * mult, 3)
        # But current stays at full expected value (bypass current not metered)
        # Current actually slightly higher due to inefficiency of bypass
        if has_current:
            e["current"] = round(elec["current"] * np.random.uniform(1.0, 1.3), 3)
        # Apparent energy is suppressed like energy (meter only sees partial)
        if has_app_e and e.get("apparent_energy") is not None:
            e["apparent_energy"] = round(e["apparent_energy"] * mult, 3)

    # ── voltage_sag ──────────────────────────────────────
    elif anom_name == "voltage_sag":
        v_anom = np.random.uniform(*anom_cfg["voltage_range"])
        if has_voltage:
            e["voltage"] = round(v_anom, 3)
        # Same power at lower voltage → higher current  (I = P / (V × PF))
        if has_current and has_voltage:
            p_watts = elec["energy_consumption"] / INTERVAL_HOURS  # W
            new_I   = p_watts / (v_anom * pf_base + 1e-6)
            e["current"] = round(max(0.0, new_I), 3)
        # PF slightly affected by voltage deviation
        if has_pf:
            e["power_factor"] = round(
                np.clip(pf_base * np.random.uniform(0.95, 1.0), 0.75, 0.99), 4
            )

    # ── voltage_swell ─────────────────────────────────────
    elif anom_name == "voltage_swell":
        v_anom = np.random.uniform(*anom_cfg["voltage_range"])
        if has_voltage:
            e["voltage"] = round(v_anom, 3)
        # Same power at higher voltage → lower current
        if has_current and has_voltage:
            p_watts = elec["energy_consumption"] / INTERVAL_HOURS
            new_I   = p_watts / (v_anom * pf_base + 1e-6)
            e["current"] = round(max(0.0, new_I), 3)

    # ── pf_collapse ───────────────────────────────────────
    elif anom_name == "pf_collapse":
        pf_anom = np.random.uniform(*anom_cfg["pf_range"])
        if has_pf:
            e["power_factor"] = round(pf_anom, 4)
        # Same active energy, but apparent energy rises (S = P / PF)
        if has_app_e and e.get("apparent_energy") is not None:
            ratio = pf_base / max(pf_anom, 0.1)
            e["apparent_energy"] = round(e["apparent_energy"] * ratio, 3)
        # Current rises (more reactive component)
        if has_current:
            ratio = pf_base / max(pf_anom, 0.1)
            e["current"] = round(e["current"] * ratio, 3)

    return e, anom_name


# =========================================================
# MAIN GENERATION LOOP
# =========================================================

all_rows  = []
global_id = 1

for meter_idx in range(1, NUM_METERS + 1):

    meter_serial = f"E{meter_idx:07d}"
    capability   = random.choice(METER_CAPABILITY_PROFILES)

    # Per-meter characteristics (fixed for the meter's lifetime)
    # Scale factor: 0.6–1.8 simulates different household sizes
    meter_load_scale = np.random.uniform(0.6, 1.8)
    # Base PF characteristic of this meter's typical load mix
    meter_pf_base    = np.random.uniform(PF_BASE - 0.04, PF_BASE + 0.03)
    # Slow voltage drift: each meter is on a slightly different
    # part of the LV network with different nominal voltage offset
    meter_voltage_offset = np.random.normal(0, 3.0)  # V

    # Slow random walk for voltage baseline (simulates supply variation)
    # AR(1) process: V[t] = 0.95 × V[t-1] + noise
    steps = int((24 * 60 / FREQ_MIN) * DAYS)
    voltage_walk = np.zeros(steps)
    voltage_walk[0] = 0.0
    for t in range(1, steps):
        voltage_walk[t] = (
            0.95 * voltage_walk[t-1]
            + np.random.normal(0, VOLTAGE_BASE_STD * 0.3)
        )

    timestamps = [
        START_TIME + timedelta(minutes=FREQ_MIN * i)
        for i in range(steps)
    ]

    # ── Per-reading generation ────────────────────────────

    # Track sustained_zero windows (inject as blocks, not isolated readings)
    zero_window_remaining = 0

    for i, ts in enumerate(timestamps):

        hour       = ts.hour
        is_weekend = ts.weekday() >= 5

        # ── Normal load + electrical parameters ──────────
        load_kw      = load_kw_for(hour, is_weekend, meter_load_scale)
        pf           = sample_pf(load_kw, meter_pf_base)
        voltage_base = (
            NOMINAL_VOLTAGE
            + meter_voltage_offset
            + voltage_walk[i]
        )

        elec = derive_electrical(load_kw, voltage_base, pf, INTERVAL_HOURS)

        # Only include parameters supported by this meter's capability
        if "1.0.14.27.0.255" in capability:
            elec["frequency"] = sample_frequency()

        if "1.0.2.29.0.255" in capability:
            # Export energy: small random amount (net-metering / solar)
            elec["active_export_energy"] = round(
                np.random.exponential(0.05), 3
            )

        # ── Anomaly injection ──────────────────────────────
        anomaly_type = "normal"

        # Handle sustained_zero as a block
        if zero_window_remaining > 0:
            elec["energy_consumption"] = 0.0
            if "1.0.11.27.0.255" in capability:
                elec["current"] = 0.0
            anomaly_type = "sustained_zero"
            zero_window_remaining -= 1
        else:
            # Check if a new sustained_zero block should start
            if np.random.rand() < ANOMALY_CATALOGUE["sustained_zero"]["prob"]:
                zero_window_remaining = np.random.randint(3, 7)  # 1.5–3.5h block
                elec["energy_consumption"] = 0.0
                if "1.0.11.27.0.255" in capability:
                    elec["current"] = 0.0
                anomaly_type = "sustained_zero"
            else:
                # All other anomaly types — single-reading
                elec, anomaly_type = inject_anomaly(
                    elec, load_kw, voltage_base, meter_pf_base, capability
                )

        # ── Build raw_data (OBIS-keyed) ───────────────────
        raw = {TIMESTAMP_OBIS: ts.strftime("%Y-%m-%d %H:%M:%S")}

        obis_map = {
            "1.0.1.29.0.255":  "energy_consumption",
            "1.0.12.27.0.255": "voltage",
            "1.0.11.27.0.255": "current",
            "1.0.13.27.0.255": "power_factor",
            "1.0.9.29.0.255":  "apparent_energy",
            "1.0.2.29.0.255":  "active_export_energy",
            "1.0.14.27.0.255": "frequency",
        }

        for obis_code, param_key in obis_map.items():
            if obis_code in capability and elec.get(param_key) is not None:
                raw[obis_code] = elec[param_key]

        # Store anomaly label in raw_data for training pseudo-label recovery
        raw["anomaly_type"] = anomaly_type

        received_at = ts + timedelta(seconds=random.randint(5, 30))

        all_rows.append({
            "id":                 global_id,
            "meter_serial":       meter_serial,
            "received_at":        received_at.isoformat() + "+00:00",
            "profile_obis_code":  LOAD_SURVEY_OBIS,
            "entry_id":           i + 1,
            "interval_timestamp": ts.isoformat(),
            "raw_data":           json.dumps(raw),
        })

        global_id += 1

# =========================================================
# SAVE + SUMMARY
# =========================================================

df = pd.DataFrame(all_rows)

out_path = os.path.join(os.path.dirname(__file__), "dynamic_meter_anomaly_dataset.csv")
df.to_csv(out_path, index=False)

# ── Anomaly distribution summary ─────────────────────────
anomaly_types = []
for raw_str in df["raw_data"]:
    raw = json.loads(raw_str)
    anomaly_types.append(raw.get("anomaly_type", "normal"))

from collections import Counter
counts = Counter(anomaly_types)
total  = len(df)

print(f"\nGenerated {total} rows → {out_path}")
print(f"\nAnomaly distribution:")
print(f"  {'Type':<25} {'Count':>6}  {'Rate':>6}")
print(f"  {'-'*40}")
for atype, count in sorted(counts.items(), key=lambda x: -x[1]):
    rate = 100 * count / total
    print(f"  {atype:<25} {count:>6}  {rate:>5.2f}%")

total_anomalies = total - counts.get("normal", total)
print(f"\n  Total anomalies : {total_anomalies} / {total}  "
      f"({100*total_anomalies/total:.2f}%)")

print(f"\nSample raw_data (first row):")
print(json.dumps(json.loads(df.iloc[0]["raw_data"]), indent=2))