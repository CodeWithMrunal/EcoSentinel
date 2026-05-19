-- =============================================================
-- METER ANOMALY DETECTION — PostgreSQL Schema
-- =============================================================
-- Two-table design:
--
--   raw_meter_readings  → stores the API payload exactly as
--                         received, untouched. Audit trail.
--
--   meter_telemetry     → stores the parsed + canonicalized
--                         record. This is what the feature
--                         engineering and detection pipeline
--                         reads from.
-- =============================================================

-- -------------------------------------------------------------
-- Extensions
-- -------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- for gen_random_uuid()

-- -------------------------------------------------------------
-- 1. raw_meter_readings
--    One row per API record received from the HES.
--    raw_value is stored verbatim (pipe-delimited string).
-- -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS raw_meter_readings (
    id                  BIGINT          PRIMARY KEY,      -- API-supplied id
    meter_serial        VARCHAR(64)     NOT NULL,         -- e.g. "E0000002"
    received_at         TIMESTAMPTZ     NOT NULL,         -- API-level timestamp
    profile_obis_code   VARCHAR(32)     NOT NULL,         -- e.g. "1.0.99.1.0.255"
    entry_id            INTEGER         NOT NULL,         -- sequence within a batch
    raw_value           TEXT            NOT NULL,         -- verbatim pipe-string

    -- Prevent re-ingestion of the same record
    CONSTRAINT uq_raw_reading UNIQUE (meter_serial, entry_id, received_at)
);

CREATE INDEX IF NOT EXISTS idx_raw_readings_meter_serial
    ON raw_meter_readings (meter_serial);

CREATE INDEX IF NOT EXISTS idx_raw_readings_received_at
    ON raw_meter_readings (received_at DESC);

-- -------------------------------------------------------------
-- 2. meter_telemetry
--    One row per parsed load-survey interval.
--    raw_data (JSONB) holds canonical key→value pairs,
--    keyed by canonical feature names (not OBIS codes).
--    Schema-on-read: new meter parameters can appear in
--    raw_data without a schema migration.
-- -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS meter_telemetry (
    id                  BIGSERIAL       PRIMARY KEY,
    meter_serial        VARCHAR(64)     NOT NULL,

    -- Actual measurement interval timestamp from the meter clock
    -- (entry 1 of rawValue), NOT the API receive time
    interval_timestamp  TIMESTAMPTZ     NOT NULL,

    -- Canonical feature dict, e.g.:
    -- {
    --   "energy_consumption": 1.6,
    --   "voltage": 230.1,
    --   "current": 1.5,
    --   ...
    -- }
    raw_data            JSONB           NOT NULL,

    -- API receive time, for latency tracking
    received_at         TIMESTAMPTZ     NOT NULL,

    -- FK back to the source raw record (nullable: allows
    -- synthetic / backfilled rows that have no raw record)
    source_raw_id       BIGINT          REFERENCES raw_meter_readings(id)
                                        ON DELETE SET NULL,

    -- Prevent duplicate intervals per meter
    CONSTRAINT uq_telemetry_interval UNIQUE (meter_serial, interval_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_telemetry_meter_serial
    ON meter_telemetry (meter_serial);

CREATE INDEX IF NOT EXISTS idx_telemetry_interval_timestamp
    ON meter_telemetry (interval_timestamp DESC);

-- Composite index for the most common query pattern:
-- "give me last N readings for meter X ordered by time"
CREATE INDEX IF NOT EXISTS idx_telemetry_meter_time
    ON meter_telemetry (meter_serial, interval_timestamp DESC);

-- GIN index for JSONB queries (e.g. filter by feature presence)
CREATE INDEX IF NOT EXISTS idx_telemetry_raw_data_gin
    ON meter_telemetry USING GIN (raw_data);

-- -------------------------------------------------------------
-- 3. anomaly_log
--    One row per flagged anomaly, written by the detection
--    pipeline. Stores which layers fired and the IF score.
-- -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS anomaly_log (
    id                  BIGSERIAL       PRIMARY KEY,
    meter_serial        VARCHAR(64)     NOT NULL,
    interval_timestamp  TIMESTAMPTZ     NOT NULL,

    -- Which detection layers flagged this reading
    rule_based_flag     BOOLEAN         NOT NULL DEFAULT FALSE,
    zscore_flag         BOOLEAN         NOT NULL DEFAULT FALSE,
    if_flag             BOOLEAN         NOT NULL DEFAULT FALSE,

    -- Isolation Forest anomaly score (lower = more anomalous)
    if_score            FLOAT,

    -- Z-score value at time of detection
    zscore_value        FLOAT,

    -- Rule violation description (if rule layer fired)
    rule_violations     JSONB,          -- e.g. ["negative_energy", "voltage_out_of_range"]

    -- Full feature vector snapshot at time of detection
    feature_snapshot    JSONB,

    detected_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_anomaly_telemetry
        FOREIGN KEY (meter_serial, interval_timestamp)
        REFERENCES meter_telemetry (meter_serial, interval_timestamp)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_anomaly_meter_serial
    ON anomaly_log (meter_serial);

CREATE INDEX IF NOT EXISTS idx_anomaly_detected_at
    ON anomaly_log (detected_at DESC);