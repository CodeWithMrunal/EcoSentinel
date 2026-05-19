import os

# =========================================================
# DATABASE
# =========================================================

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "meter_anomaly"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

# Convenience DSN string for tools that prefer it
DB_DSN = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# =========================================================
# MODEL ARTIFACTS
# =========================================================

MODEL_DIR = os.getenv(
    "MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "models")
)

MODEL_PATHS = {
    "isolation_forest": os.path.join(MODEL_DIR, "isolation_forest.joblib"),
    "scaler":           os.path.join(MODEL_DIR, "scaler.joblib"),
    "impute_values":    os.path.join(MODEL_DIR, "impute_values.joblib"),
    "feature_schema":   os.path.join(MODEL_DIR, "feature_schema.joblib"),
}

# =========================================================
# DATASET GENERATION
# =========================================================

DATASET_CONFIG = {
    "num_meters":    10,
    "days":          15,
    "freq_minutes":  30,          # 30-minute intervals to match real meter data
    "start_time":    "2026-01-01",
    "random_seed":   42,
}

# =========================================================
# OBIS CODE REGISTRY
# Maps every known OBIS code to:
#   canonical_name : used as feature name throughout the pipeline
#   description    : human-readable label
#   unit           : expected unit string from rawValue
#   is_timestamp   : True only for the clock object (extracted, not a feature)
# Add new OBIS codes here as meters expose them — nothing
# else in the codebase needs to change.
# =========================================================

OBIS_REGISTRY = {
    # ── Clock / Timestamp ────────────────────────────────
    "0.0.1.0.0.255": {
        "canonical_name": None,           # extracted as interval_timestamp
        "description":    "Interval timestamp (clock object)",
        "unit":           "",
        "is_timestamp":   True,
    },

    # ── Energy ───────────────────────────────────────────
    "1.0.1.29.0.255": {
        "canonical_name": "energy_consumption",
        "description":    "Active import energy – interval (Wh)",
        "unit":           "Wh",
        "is_timestamp":   False,
    },
    "1.0.2.29.0.255": {
        "canonical_name": "active_export_energy",
        "description":    "Active export energy – interval (Wh)",
        "unit":           "Wh",
        "is_timestamp":   False,
    },
    "1.0.9.29.0.255": {
        "canonical_name": "apparent_import_energy",
        "description":    "Apparent import energy – interval (VAh)",
        "unit":           "VAh",
        "is_timestamp":   False,
    },
    "1.0.10.29.0.255": {
        "canonical_name": "apparent_export_energy",
        "description":    "Apparent export energy – interval (VAh)",
        "unit":           "VAh",
        "is_timestamp":   False,
    },
    "1.0.3.29.0.255": {
        "canonical_name": "reactive_import_energy",
        "description":    "Reactive import energy – interval (VARh)",
        "unit":           "VARh",
        "is_timestamp":   False,
    },
    "1.0.4.29.0.255": {
        "canonical_name": "reactive_export_energy",
        "description":    "Reactive export energy – interval (VARh)",
        "unit":           "VARh",
        "is_timestamp":   False,
    },

    # ── Power ────────────────────────────────────────────
    "1.0.1.27.0.255": {
        "canonical_name": "active_import_power",
        "description":    "Active import power (W)",
        "unit":           "W",
        "is_timestamp":   False,
    },
    "1.0.2.27.0.255": {
        "canonical_name": "active_export_power",
        "description":    "Active export power (W)",
        "unit":           "W",
        "is_timestamp":   False,
    },

    # ── Voltage ──────────────────────────────────────────
    "1.0.12.27.0.255": {
        "canonical_name": "voltage",
        "description":    "Voltage (V)",
        "unit":           "V",
        "is_timestamp":   False,
    },

    # ── Current ──────────────────────────────────────────
    "1.0.11.27.0.255": {
        "canonical_name": "current",
        "description":    "Current (A)",
        "unit":           "A",
        "is_timestamp":   False,
    },

    # ── Power Factor ─────────────────────────────────────
    "1.0.13.27.0.255": {
        "canonical_name": "power_factor",
        "description":    "Power factor",
        "unit":           "",
        "is_timestamp":   False,
    },

    # ── Frequency ────────────────────────────────────────
    "1.0.14.27.0.255": {
        "canonical_name": "frequency",
        "description":    "Frequency (Hz)",
        "unit":           "Hz",
        "is_timestamp":   False,
    },
}

# =========================================================
# METER CAPABILITY PROFILES
# Each profile is a list of OBIS codes that a meter sends.
# The dataset generator picks one profile per meter.
# Add real meter profiles here as they are onboarded.
# =========================================================

METER_CAPABILITY_PROFILES = [
    # Profile A — energy + voltage + current + power factor
    [
        "1.0.1.29.0.255",   # energy_consumption
        "1.0.12.27.0.255",  # voltage
        "1.0.11.27.0.255",  # current
        "1.0.13.27.0.255",  # power_factor
    ],

    # Profile B — energy + apparent energy + voltage
    [
        "1.0.1.29.0.255",   # energy_consumption
        "1.0.9.29.0.255",   # apparent_import_energy
        "1.0.12.27.0.255",  # voltage
    ],

    # Profile C — energy + current only
    [
        "1.0.1.29.0.255",   # energy_consumption
        "1.0.11.27.0.255",  # current
    ],

    # Profile D — full set
    [
        "1.0.1.29.0.255",   # energy_consumption
        "1.0.2.29.0.255",   # active_export_energy
        "1.0.9.29.0.255",   # apparent_import_energy
        "1.0.12.27.0.255",  # voltage
        "1.0.11.27.0.255",  # current
        "1.0.13.27.0.255",  # power_factor
        "1.0.14.27.0.255",  # frequency
    ],

    # Profile E — energy only
    [
        "1.0.1.29.0.255",   # energy_consumption
    ],
]

# =========================================================
# FEATURE SCHEMA
# Fixed 19-column vector fed to the Isolation Forest.
# CORE features must be present for every reading.
# OPTIONAL features are NaN-imputed when the meter
# capability profile does not include them.
# =========================================================

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

OPTIONAL_FEATURES = [
    "voltage",
    "current",
    "power_factor",
    "apparent_import_energy",
    "current_delta",
    "voltage_deviation",
    "power_factor_deviation",
]

ALL_FEATURES = CORE_FEATURES + OPTIONAL_FEATURES

# =========================================================
# DETECTION THRESHOLDS
# =========================================================

DETECTION_CONFIG = {
    # Z-score: flag if |z| exceeds this
    "zscore_threshold": 3.0,

    # Rule-based
    "voltage_min":      180.0,    # V
    "voltage_max":      270.0,    # V
    "power_factor_min": 0.0,
    "power_factor_max": 1.0,
    "zero_consumption_window": 3, # consecutive zero readings = anomaly

    # Isolation Forest
    "if_contamination": 0.05,
}

# =========================================================
# ROLLING WINDOW
# Number of past readings fetched from DB per meter
# for computing rolling/historical features at inference.
# =========================================================

ROLLING_WINDOW_SIZE = 10