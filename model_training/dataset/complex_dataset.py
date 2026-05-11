import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta

# =========================================================
# CONFIG
# =========================================================

NUM_METERS = 10
DAYS = 15
FREQ_MINUTES = 60

CONSUMER_TYPES = ['residential', 'commercial', 'industrial']

LOCATIONS = [
    'Chennai',
    'Bangalore',
    'Mysore',
    'Hyderabad',
    'Mumbai'
]

np.random.seed(42)
random.seed(42)

# =========================================================
# PARAMETER NAME VARIANTS
# Simulates heterogeneous HES/vendor payloads
# =========================================================

PARAMETER_VARIANTS = {
    "energy_consumption": [
        "energy_consumption",
        "active_energy",
        "Active Energy",
        "Import Active Energy",
        "Export Active Energy",
        "kWh"
    ],

    "reactive_energy": [
        "reactive_energy",
        "Reactive Energy",
        "Import Reactive Energy",
        "Export Reactive Energy",
        "kVARh"
    ],

    "voltage": [
        "voltage",
        "Voltage",
        "Line Voltage",
        "Phase Voltage"
    ],

    "current": [
        "current",
        "Current",
        "Phase Current",
        "Line Current"
    ],

    "power_factor": [
        "power_factor",
        "Power Factor",
        "PF"
    ]
}

# =========================================================
# METER CAPABILITIES
# Different meters expose different parameters
# =========================================================

METER_CAPABILITIES = [
    [
        "energy_consumption",
        "voltage",
        "current",
        "power_factor"
    ],

    [
        "energy_consumption",
        "reactive_energy",
        "voltage"
    ],

    [
        "energy_consumption",
        "current"
    ],

    [
        "energy_consumption",
        "reactive_energy",
        "voltage",
        "current",
        "power_factor"
    ],

    [
        "energy_consumption"
    ]
]

# =========================================================
# HELPERS
# =========================================================

def random_param_name(param):
    return random.choice(PARAMETER_VARIANTS[param])


def generate_base_consumption(hour, consumer_type, is_weekend):

    if consumer_type == 'residential':
        base = 1.5 + 1.0 * (18 <= hour <= 23) + 0.5 * (6 <= hour <= 9)

    elif consumer_type == 'commercial':
        base = 3.0 if 9 <= hour <= 18 else 0.5

    else:
        base = 4.0

    if is_weekend and consumer_type == 'commercial':
        base *= 0.6

    return base + np.random.normal(0, 0.2)


def is_festival_day(date):
    return date.day in [1, 5, 10]


def is_holiday(date):
    return date.weekday() == 6


# =========================================================
# GENERATE DATA
# =========================================================

all_meter_data = []

start_time = datetime(2026, 1, 1)

for meter_id in range(1, NUM_METERS + 1):

    consumer_type = np.random.choice(CONSUMER_TYPES)
    location = np.random.choice(LOCATIONS)

    supported_params = random.choice(METER_CAPABILITIES)

    timestamps = [
        start_time + timedelta(minutes=FREQ_MINUTES * i)
        for i in range(int((24 * 60 / FREQ_MINUTES) * DAYS))
    ]

    energy_values = []

    # -----------------------------------------------------
    # GENERATE ENERGY SERIES
    # -----------------------------------------------------

    for ts in timestamps:

        hour = ts.hour
        day_of_week = ts.weekday()
        is_weekend_flag = 1 if day_of_week >= 5 else 0

        base_consumption = generate_base_consumption(
            hour,
            consumer_type,
            is_weekend_flag
        )

        # occasional spike anomaly
        if np.random.rand() < 0.02:
            base_consumption *= np.random.uniform(3, 8)

        # occasional negative anomaly
        if np.random.rand() < 0.005:
            base_consumption *= -1

        energy_values.append(round(base_consumption, 2))

    # -----------------------------------------------------
    # CREATE BASE DF
    # -----------------------------------------------------

    df_temp = pd.DataFrame({
        'timestamp': timestamps,
        'energy_consumption': energy_values
    })

    # -----------------------------------------------------
    # ELECTRICAL FEATURES
    # -----------------------------------------------------

    if "voltage" in supported_params:
        df_temp['voltage'] = np.random.normal(
            230,
            5,
            len(df_temp)
        )

    if "current" in supported_params:
        multiplier = np.random.uniform(0.8, 1.2)
        df_temp['current'] = (
            abs(df_temp['energy_consumption']) * multiplier
        )

    if "power_factor" in supported_params:
        df_temp['power_factor'] = np.random.uniform(
            0.85,
            1.0,
            len(df_temp)
        )

    if "reactive_energy" in supported_params:
        df_temp['reactive_energy'] = np.random.uniform(
            0.5,
            3.0,
            len(df_temp)
        )

    # -----------------------------------------------------
    # STATIC FEATURES
    # -----------------------------------------------------

    df_temp['consumer_type'] = consumer_type
    df_temp['location'] = location

    # -----------------------------------------------------
    # TIME FEATURES
    # -----------------------------------------------------

    df_temp['hour_of_day'] = df_temp['timestamp'].dt.hour

    df_temp['day_of_week'] = (
        df_temp['timestamp'].dt.dayofweek
    )

    df_temp['is_weekend'] = (
        df_temp['day_of_week']
        .apply(lambda x: 1 if x >= 5 else 0)
    )

    df_temp['holiday'] = (
        df_temp['timestamp']
        .apply(lambda x: 1 if is_holiday(x) else 0)
    )

    df_temp['is_festival'] = (
        df_temp['timestamp']
        .apply(lambda x: 1 if is_festival_day(x) else 0)
    )

    # -----------------------------------------------------
    # DERIVED FEATURES
    # -----------------------------------------------------

    df_temp['delta'] = (
        df_temp['energy_consumption']
        .diff()
        .fillna(0)
    )

    if "current" in supported_params:
        df_temp['current_delta'] = (
            df_temp['current']
            .diff()
            .fillna(0)
        )

    # -----------------------------------------------------
    # ROLLING FEATURES
    # -----------------------------------------------------

    df_temp['rolling_mean'] = (
        df_temp['energy_consumption']
        .rolling(window=5, min_periods=1)
        .mean()
    )

    df_temp['rolling_std'] = (
        df_temp['energy_consumption']
        .rolling(window=5, min_periods=1)
        .std()
        .fillna(0)
    )

    # -----------------------------------------------------
    # Z SCORE
    # -----------------------------------------------------

    df_temp['z_score'] = (
        (
            df_temp['energy_consumption']
            - df_temp['rolling_mean']
        )
        /
        (
            df_temp['rolling_std'] + 1e-5
        )
    )

    # -----------------------------------------------------
    # SPIKE RATIO
    # -----------------------------------------------------

    df_temp['spike_ratio'] = (
        df_temp['energy_consumption']
        /
        (
            df_temp['rolling_mean'] + 1e-5
        )
    )

    # -----------------------------------------------------
    # DEVIATIONS
    # -----------------------------------------------------

    if "voltage" in supported_params:
        df_temp['voltage_deviation'] = (
            df_temp['voltage'] - 230
        )

    if "power_factor" in supported_params:
        df_temp['power_factor_deviation'] = (
            1 - df_temp['power_factor']
        )

    # -----------------------------------------------------
    # HISTORICAL FEATURES
    # -----------------------------------------------------

    df_temp['historical_avg_same_hour'] = (
        df_temp
        .groupby('hour_of_day')['energy_consumption']
        .transform('mean')
    )

    df_temp['historical_avg_same_day_type'] = (
        df_temp
        .groupby('is_weekend')['energy_consumption']
        .transform('mean')
    )

    # =====================================================
    # BUILD DYNAMIC RAW JSON PAYLOAD
    # =====================================================

    final_rows = []

    for _, row in df_temp.iterrows():

        raw_payload = {}

        # -------------------------------------------------
        # DYNAMIC METER PARAMETERS
        # -------------------------------------------------

        raw_payload[
            random_param_name("energy_consumption")
        ] = round(row['energy_consumption'], 3)

        if "reactive_energy" in supported_params:
            raw_payload[
                random_param_name("reactive_energy")
            ] = round(row['reactive_energy'], 3)

        if "voltage" in supported_params:
            raw_payload[
                random_param_name("voltage")
            ] = round(row['voltage'], 3)

        if "current" in supported_params:
            raw_payload[
                random_param_name("current")
            ] = round(row['current'], 3)

        if "power_factor" in supported_params:
            raw_payload[
                random_param_name("power_factor")
            ] = round(row['power_factor'], 3)

        # -------------------------------------------------
        # DERIVED + CONTEXT FEATURES
        # -------------------------------------------------

        raw_payload['consumer_type'] = row['consumer_type']
        raw_payload['location'] = row['location']

        raw_payload['hour_of_day'] = int(row['hour_of_day'])
        raw_payload['day_of_week'] = int(row['day_of_week'])

        raw_payload['is_weekend'] = int(row['is_weekend'])
        raw_payload['holiday'] = int(row['holiday'])
        raw_payload['is_festival'] = int(row['is_festival'])

        raw_payload['delta'] = round(row['delta'], 3)

        if "current" in supported_params:
            raw_payload['current_delta'] = round(
                row['current_delta'],
                3
            )

        raw_payload['rolling_mean'] = round(
            row['rolling_mean'],
            3
        )

        raw_payload['rolling_std'] = round(
            row['rolling_std'],
            3
        )

        raw_payload['z_score'] = round(
            row['z_score'],
            3
        )

        raw_payload['spike_ratio'] = round(
            row['spike_ratio'],
            3
        )

        raw_payload['historical_avg_same_hour'] = round(
            row['historical_avg_same_hour'],
            3
        )

        raw_payload['historical_avg_same_day_type'] = round(
            row['historical_avg_same_day_type'],
            3
        )

        if "voltage" in supported_params:
            raw_payload['voltage_deviation'] = round(
                row['voltage_deviation'],
                3
            )

        if "power_factor" in supported_params:
            raw_payload['power_factor_deviation'] = round(
                row['power_factor_deviation'],
                3
            )

        # -------------------------------------------------
        # FINAL ROW
        # -------------------------------------------------

        final_rows.append({
            "meter_id": meter_id,
            "timestamp": row['timestamp'],
            "raw_data": json.dumps(raw_payload)
        })

    all_meter_data.extend(final_rows)

# =========================================================
# FINAL DATAFRAME
# =========================================================

df_final = pd.DataFrame(all_meter_data)

# =========================================================
# SAVE CSV
# =========================================================

df_final.to_csv(
    "dynamic_meter_anomaly_dataset.csv",
    index=False
)

print("Generated dynamic_meter_anomaly_dataset.csv")