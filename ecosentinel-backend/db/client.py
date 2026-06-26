"""
db/client.py
------------
PostgreSQL connection pool and all query helpers used
by the detection pipeline.

Dependencies:
    pip install psycopg2-binary
"""

import os
import json
import logging
from contextlib import contextmanager
from typing import Optional
from time import perf_counter

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DB_CONFIG, ROLLING_WINDOW_SIZE

logger = logging.getLogger(__name__)

# =========================================================
# CONNECTION POOL
# Initialized once on first import.
# min/max connections are intentionally conservative —
# tune for production load.
# =========================================================

_pool: Optional[pool.SimpleConnectionPool] = None


def get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            dbname=DB_CONFIG["dbname"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
        )
        logger.info("PostgreSQL connection pool created.")
    return _pool


@contextmanager
def get_connection():
    """
    Context manager that checks out a connection from the pool,
    yields it, and returns it on exit. Rolls back on exception.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    conn = get_pool().getconn()
    try:
        logger.debug("Database connection acquired from pool.")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        get_pool().putconn(conn)
        logger.debug("Database connection returned to pool.")


# =========================================================
# SCHEMA INITIALISATION
# Run once on service startup to ensure tables exist.
# =========================================================

def init_schema():
    """
    Executes schema.sql against the configured database.
    Safe to run multiple times (all statements use IF NOT EXISTS).
    """
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        sql = f.read()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    logger.info("Schema initialised.")


# =========================================================
# WRITE HELPERS
# =========================================================

def insert_raw_reading(
    id: int,
    meter_serial: str,
    received_at: str,
    profile_obis_code: str,
    entry_id: int,
    raw_value: str,
) -> None:
    """
    Inserts one record into raw_meter_readings.
    Uses ON CONFLICT DO NOTHING to safely handle re-ingestion.
    """
    sql = """
        INSERT INTO raw_meter_readings
            (id, meter_serial, received_at, profile_obis_code, entry_id, raw_value)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT ON CONSTRAINT uq_raw_reading DO NOTHING;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                id, meter_serial, received_at,
                profile_obis_code, entry_id, raw_value,
            ))
    logger.debug(
        f"Inserted raw reading id={id} meter={meter_serial} profile={profile_obis_code}."
    )


def insert_telemetry(
    meter_serial: str,
    interval_timestamp: str,
    raw_data: dict,
    received_at: str,
    flagged_anomalous: bool = False,
    source_raw_id: Optional[int] = None,
) -> None:
    """
    Inserts one parsed + canonicalized reading into meter_telemetry.
    raw_data dict is stored as JSONB.
    Uses ON CONFLICT DO NOTHING to safely handle duplicates.
    """
    sql = """
        INSERT INTO meter_telemetry
            (meter_serial, interval_timestamp, raw_data, received_at, flagged_anomalous, source_raw_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT ON CONSTRAINT uq_telemetry_interval DO NOTHING;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                meter_serial,
                interval_timestamp,
                json.dumps(raw_data),
                received_at,
                flagged_anomalous,
                source_raw_id,
            ))
    logger.debug(
        f"Inserted telemetry for meter={meter_serial} interval_timestamp={interval_timestamp} flagged_anomalous={flagged_anomalous} source_raw_id={source_raw_id}."
    )


def insert_anomaly(
    meter_serial: str,
    interval_timestamp: str,
    rule_based_flag: bool,
    zscore_flag: bool,
    if_flag: bool,
    if_score: Optional[float],
    zscore_value: Optional[float],
    rule_violations: Optional[list],
    feature_snapshot: Optional[dict],
    explanation_status: Optional[str] = None,
) -> Optional[int]:
    """
    Writes a detected anomaly to anomaly_log.

    Parameters
    ----------
    explanation_status : if set (e.g. "pending"), marks this anomaly
                          as awaiting LLM explanation generation.
                          Leave None if the decision engine is disabled.

    Returns
    -------
    The new anomaly_log.id, or None if insert failed. Used by the API
    layer to schedule a background explanation task for this row.
    """
    sql = """
        INSERT INTO anomaly_log (
            meter_serial, interval_timestamp,
            rule_based_flag, zscore_flag, if_flag,
            if_score, zscore_value,
            rule_violations, feature_snapshot,
            explanation_status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                meter_serial,
                interval_timestamp,
                rule_based_flag,
                zscore_flag,
                if_flag,
                if_score,
                zscore_value,
                json.dumps(rule_violations) if rule_violations else None,
                json.dumps(feature_snapshot) if feature_snapshot else None,
                explanation_status,
            ))
            row = cur.fetchone()
            logger.info(
                f"Inserted anomaly log for meter={meter_serial} interval_timestamp={interval_timestamp}; explanation_status={explanation_status}."
            )
            return row[0] if row else None


def update_anomaly_explanation(
    anomaly_id: int,
    explanation: Optional[dict],
    status: str,
    error: Optional[str] = None,
) -> None:
    """
    Updates the explanation fields for an anomaly_log row.
    Called by the decision engine background task after the
    LLM call completes (successfully or not).
    """
    sql = """
        UPDATE anomaly_log
        SET explanation = %s,
            explanation_status = %s,
            explanation_generated_at = NOW(),
            explanation_error = %s
        WHERE id = %s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                json.dumps(explanation) if explanation else None,
                status,
                error,
                anomaly_id,
            ))


def get_anomaly_by_id(anomaly_id: int) -> Optional[dict]:
    """
    Fetches a single anomaly_log row by id, including explanation
    fields. Used by GET /anomalies/{id}/explanation.
    """
    sql = """
        SELECT id, meter_serial, interval_timestamp,
               rule_based_flag, zscore_flag, if_flag,
               if_score, zscore_value, rule_violations, feature_snapshot,
               detected_at,
               explanation_status, explanation,
               explanation_generated_at, explanation_error
        FROM anomaly_log
        WHERE id = %s;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (anomaly_id,))
            row = cur.fetchone()
            logger.debug(f"Fetched anomaly_log row for anomaly_id={anomaly_id}: {'found' if row else 'missing'}.")
            return dict(row) if row else None

# =========================================================
# READ HELPERS
# =========================================================

def get_last_n_readings(
    meter_serial: str,
    n: int = ROLLING_WINDOW_SIZE,
    before_timestamp: Optional[str] = None,
) -> list[dict]:
    """
    Returns the last N telemetry readings for a meter,
    ordered oldest → newest (so rolling features can be
    computed in chronological order).

    Parameters
    ----------
    meter_serial     : meter identifier
    n                : number of past readings to fetch
    before_timestamp : if provided, only fetch readings strictly
                       before this timestamp (excludes current row).
                       Pass the current interval_timestamp here
                       so the rolling window doesn't include itself.

    Returns
    -------
    List of dicts with keys: interval_timestamp, raw_data (dict)
    Ordered oldest → newest.
    """
    if before_timestamp:
        sql = """
            SELECT interval_timestamp, raw_data
            FROM meter_telemetry
            WHERE meter_serial = %s
              AND interval_timestamp < %s
                            AND flagged_anomalous = FALSE
            ORDER BY interval_timestamp DESC
            LIMIT %s;
        """
        params = (meter_serial, before_timestamp, n)
    else:
        sql = """
            SELECT interval_timestamp, raw_data
            FROM meter_telemetry
            WHERE meter_serial = %s
              AND flagged_anomalous = FALSE
            ORDER BY interval_timestamp DESC
            LIMIT %s;
        """
        params = (meter_serial, n)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    logger.debug(
        f"Fetched {len(rows)} telemetry row(s) for meter={meter_serial} before_timestamp={before_timestamp}."
    )

    # Reverse so result is oldest → newest
    rows = list(reversed(rows))

    return [
        {
            "interval_timestamp": row["interval_timestamp"],
            "raw_data": row["raw_data"],   # psycopg2 auto-parses JSONB → dict
        }
        for row in rows
    ]


def get_historical_avg_same_hour(
    meter_serial: str,
    hour: int,
    lookback_days: int = 30,
) -> Optional[float]:
    """
    Returns the average energy_consumption for this meter
    at the given hour of day, computed over the last
    `lookback_days` days. Used for historical_avg_same_hour feature.
    Returns None if no data exists.
    """
    sql = """
        SELECT AVG((raw_data->>'energy_consumption')::float)
        FROM meter_telemetry
        WHERE meter_serial = %s
          AND EXTRACT(HOUR FROM interval_timestamp) = %s
          AND interval_timestamp >= NOW() - INTERVAL '%s days'
            AND flagged_anomalous = FALSE
          AND raw_data ? 'energy_consumption';
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (meter_serial, hour, lookback_days))
            result = cur.fetchone()

    return result[0] if result and result[0] is not None else None


def get_historical_avg_same_day_type(
    meter_serial: str,
    is_weekend: int,
    lookback_days: int = 30,
) -> Optional[float]:
    """
    Returns the average energy_consumption for this meter
    on weekdays or weekends (is_weekend=0 or 1),
    computed over the last `lookback_days` days.
    Returns None if no data exists.
    """
    sql = """
        SELECT AVG((raw_data->>'energy_consumption')::float)
        FROM meter_telemetry
        WHERE meter_serial = %s
          AND (EXTRACT(DOW FROM interval_timestamp) IN (0, 6)) = %s
          AND interval_timestamp >= NOW() - INTERVAL '%s days'
            AND flagged_anomalous = FALSE
          AND raw_data ? 'energy_consumption';
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                meter_serial,
                bool(is_weekend),
                lookback_days,
            ))
            result = cur.fetchone()

    return result[0] if result and result[0] is not None else None


def close_pool():
    """Call on application shutdown to release all connections."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("PostgreSQL connection pool closed.")