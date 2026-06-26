"""
utils/seed_normal_history.py
-----------------------------
Populates meter_telemetry with realistic normal readings so that
when you send test anomalous payloads through POST /detect, the
rolling window features (rolling_mean, rolling_std, z_score,
historical_avg_same_hour) are computed against a clean baseline
rather than against empty or anomalous history.

Why this is necessary
---------------------
Without history:
  - rolling_mean   = current reading value (window of 1)
  - rolling_std    = 0
  - z_score        = 0  (no deviation possible)
  - spike_ratio    = 1.0
  Statistical layer cannot fire. Subtle anomalies are invisible.

With seeded history:
  - rolling_mean   = realistic baseline (~420 Wh for group_A)
  - rolling_std    = realistic spread (~50–80 Wh)
  - z_score        = correct deviation from baseline
  - historical_avg = per-hour average from 48h of history
  Statistical layer fires correctly. IF scores are meaningful.

Usage
-----
# Seed a group_A meter with 48 readings (24h) before a test time
python utils/seed_normal_history.py \\
    --meter TEST_NORMAL TEST_SUBTLE_SPIKE TEST_TAMPER_BYPASS \\
    --group group_A \\
    --hours 48 \\
    --before "2026-01-10 14:00:00"

# Seed multiple groups
python utils/seed_normal_history.py --meter TEST_GROUP_V --group group_V --hours 24 --before "2026-01-10 14:00:00"
python utils/seed_normal_history.py --meter TEST_GROUP_D --group group_D --hours 48 --before "2026-01-10 14:00:00"

# Seed ALL test meters for all groups in one call
python utils/seed_normal_history.py --preset all_test_meters --before "2026-01-10 14:00:00"
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from config.settings import DB_CONFIG, CAPABILITY_GROUPS

np.random.seed(99)   # separate seed from dataset generator

# =========================================================
# PHYSICS CONSTANTS  (same as dataset generator)
# =========================================================

INTERVAL_HOURS       = 0.5        # 30-minute intervals
NOMINAL_VOLTAGE      = 230.0
VOLTAGE_DROOP_PER_A  = 0.05
NOMINAL_FREQ         = 50.0

# Diurnal profile (kW) — matches generate_dataset.py
HOURLY_LOAD_WEEKDAY = np.array([
    0.30, 0.25, 0.22, 0.20, 0.20, 0.25,
    0.60, 1.10, 1.20, 0.90, 0.70, 0.65,
    0.60, 0.55, 0.55, 0.60, 0.70, 0.85,
    1.30, 1.50, 1.40, 1.20, 0.90, 0.55,
])
HOURLY_LOAD_WEEKEND = np.array([
    0.30, 0.25, 0.22, 0.20, 0.20, 0.28,
    0.40, 0.60, 0.90, 1.10, 1.20, 1.15,
    1.10, 1.05, 1.00, 0.95, 0.90, 1.00,
    1.35, 1.50, 1.40, 1.20, 0.90, 0.55,
])


# =========================================================
# PRESET — all standard test meter serials and their groups
# Add entries here whenever you add a new test meter serial.
# =========================================================

TEST_METER_PRESETS = {
    # group_A meters
    "TEST_NORMAL":          "group_A",
    "TEST_SUBTLE_SPIKE":    "group_A",
    "TEST_OBVIOUS_SPIKE":   "group_A",
    "TEST_NEGATIVE_ENERGY": "group_A",
    "TEST_TAMPER_BYPASS":   "group_A",
    "TEST_VOLTAGE_SAG":     "group_A",
    "TEST_VOLTAGE_SWELL":   "group_A",
    "TEST_VOLTAGE_SWELL_RULE": "group_A",
    "TEST_PF_COLLAPSE":     "group_A",
    "TEST_PF_INVALID":      "group_A",
    "TEST_SUSTAINED_ZERO":  "group_A",
    "BATCH_NORMAL":         "group_A",
    "BATCH_SPIKE":          "group_A",
    "BATCH_NEGATIVE":       "group_A",
    "BATCH_TAMPER":         "group_A",
    "BATCH_SAG":            "group_A",
    "BATCH_SWELL":          "group_A",
    "BATCH_PF":             "group_A",
    # group_V meters
    "TEST_GROUP_V_NORMAL":  "group_V",
    "TEST_GROUP_V_ANOM":    "group_V",
    # group_D meters
    "TEST_GROUP_D_FREQ":    "group_D",
    "BATCH_FREQ":           "group_D",
    # group_B meters
    "TEST_GRP_B_NORMAL":    "group_B",
    "TEST_GRP_B_ANOM":      "group_B",
    # group_C meters
    "TEST_GRP_C_NORMAL":    "group_C",
    "TEST_GRP_C_ANOM":      "group_C",
    # group_E meters
    "TEST_GRP_E_NORMAL":    "group_E",
    "TEST_GRP_E_SPIKE":     "group_E",
    "TEST_GRP_E_NEG":       "group_E",
}


# =========================================================
# PHYSICS HELPERS
# =========================================================

def _load_kw(hour: int, is_weekend: bool, scale: float = 1.0) -> float:
    profile = HOURLY_LOAD_WEEKEND if is_weekend else HOURLY_LOAD_WEEKDAY
    noise   = np.random.normal(1.0, 0.12)
    return max(0.05, profile[hour] * scale * noise)


def _derive(load_kw: float, v_base: float, pf: float, group: str) -> dict:
    """
    Derives canonical electrical values from load_kw.
    Returns only the parameters that the given capability group exposes.
    """
    pf  = float(np.clip(pf, 0.80, 0.99))
    I   = (load_kw * 1000.0) / (v_base * pf + 1e-6)
    V   = v_base - I * VOLTAGE_DROOP_PER_A
    V   = max(100.0, V)
    I   = (load_kw * 1000.0) / (V * pf + 1e-6)   # recalc with actual V
    S   = load_kw / pf
    E   = load_kw * INTERVAL_HOURS * 1000.0        # Wh
    Ea  = S * INTERVAL_HOURS * 1000.0              # VAh

    # Full set — will be filtered by group below
    full = {
        "energy_consumption":   round(max(0.0, E + np.random.normal(0, 0.5)), 3),
        "voltage":              round(V + np.random.normal(0, 0.4), 3),
        "current":              round(max(0.0, I + np.random.normal(0, 0.01 * I)), 3),
        "power_factor":         round(pf + np.random.normal(0, 0.005), 4),
        "apparent_import_energy": round(max(0.0, Ea + np.random.normal(0, 0.5)), 3),
        "active_export_energy": round(max(0.0, np.random.exponential(0.05)), 3),
        "frequency":            round(float(np.random.normal(NOMINAL_FREQ, 0.04)), 3),
    }

    # Group canonical feature sets (raw features only)
    group_features = CAPABILITY_GROUPS.get(group, set())

    # Filter to only the features this group sends
    result = {k: v for k, v in full.items() if k in group_features}
    return result


# =========================================================
# SEED FUNCTION
# =========================================================

def seed_meter(
    meter_serial:   str,
    group:          str,
    n_readings:     int,
    before_dt:      datetime,
    load_scale:     float = 1.0,
    overwrite:      bool  = False,
) -> int:
    """
    Inserts n_readings normal history rows into meter_telemetry
    for the given meter, ending at before_dt (exclusive).

    Parameters
    ----------
    meter_serial : meter serial number
    group        : capability group name (e.g. 'group_A')
    n_readings   : how many 30-min intervals to insert (48 = 24h)
    before_dt    : timestamp of the FIRST test reading you will send.
                   History is placed strictly before this time.
    load_scale   : per-meter load scale factor (0.6–1.8 typical)
    overwrite    : if True, delete existing history for this meter first

    Returns
    -------
    Number of rows inserted.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    try:
        if overwrite:
            cur.execute(
                "DELETE FROM meter_telemetry WHERE meter_serial = %s",
                (meter_serial,)
            )
            conn.commit()

        # Build timestamps: n_readings × 30min ending just before before_dt
        timestamps = [
            before_dt - timedelta(minutes=30 * (n_readings - i))
            for i in range(n_readings)
        ]

        # Per-meter fixed characteristics
        v_offset = np.random.normal(0, 2.0)     # network position
        pf_base  = np.random.uniform(0.88, 0.95)

        rows = []
        for ts in timestamps:
            hour       = ts.hour
            is_weekend = ts.weekday() >= 5
            load_kw    = _load_kw(hour, is_weekend, load_scale)
            v_base     = NOMINAL_VOLTAGE + v_offset + np.random.normal(0, 0.3)
            pf         = pf_base + np.random.normal(0, 0.008)
            canonical  = _derive(load_kw, v_base, pf, group)

            rows.append((
                meter_serial,
                ts,
                json.dumps(canonical),
                ts,    # received_at ≈ interval_ts for seeded data
                False, # flagged_anomalous — seeded history must stay clean
                None,  # source_raw_id — no raw record for seeded data
            ))

        sql = """
            INSERT INTO meter_telemetry
                (meter_serial, interval_timestamp, raw_data, received_at, flagged_anomalous, source_raw_id)
            VALUES %s
            ON CONFLICT ON CONSTRAINT uq_telemetry_interval DO NOTHING
        """
        execute_values(cur, sql, rows)
        inserted = cur.rowcount
        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

    return inserted


# =========================================================
# CLI
# =========================================================

def _parse_before(s: str) -> datetime:
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: '{s}'. Use 'YYYY-MM-DD HH:MM:SS'.")


def main():
    parser = argparse.ArgumentParser(
        description="Seed meter_telemetry with normal baseline history."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--meter",
        nargs="+",
        metavar="SERIAL",
        help="One or more meter serial numbers to seed"
    )
    mode.add_argument(
        "--preset",
        choices=["all_test_meters"],
        help="Seed all standard test meters from the preset list"
    )

    parser.add_argument(
        "--group",
        default="group_A",
        help="Capability group for --meter mode (default: group_A)"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=48,
        help="Hours of history to insert (default: 48 = 2 days × 48 readings)"
    )
    parser.add_argument(
        "--before",
        required=True,
        metavar="DATETIME",
        help="Insert history before this datetime. Format: 'YYYY-MM-DD HH:MM:SS'"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Load scale factor, 0.6–1.8 (default: 1.0)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing history for these meters before seeding"
    )

    args   = parser.parse_args()
    before = _parse_before(args.before)
    n_readings = args.hours * 2   # 30-min intervals

    # Build the work list: [(serial, group), ...]
    if args.preset == "all_test_meters":
        work = list(TEST_METER_PRESETS.items())
    else:
        work = [(s, args.group) for s in args.meter]

    print(f"\nSeeding {len(work)} meter(s) with {n_readings} readings "
          f"(before {before})")
    print(f"{'Meter':<30} {'Group':<12} {'Inserted':>8}")
    print("-" * 55)

    total = 0
    for serial, group in work:
        if group not in CAPABILITY_GROUPS:
            print(f"  {serial:<30} {group:<12} SKIPPED (unknown group)")
            continue
        inserted = seed_meter(
            meter_serial=serial,
            group=group,
            n_readings=n_readings,
            before_dt=before,
            load_scale=args.scale,
            overwrite=args.overwrite,
        )
        print(f"  {serial:<30} {group:<12} {inserted:>8}")
        total += inserted

    print(f"\n  ✓ Total rows inserted: {total}")
    print(f"    Meters now have proper rolling baseline before {before}")
    print(f"    You can now POST anomalous readings to /detect\n")


if __name__ == "__main__":
    main()