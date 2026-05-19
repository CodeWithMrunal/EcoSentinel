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
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        get_pool().putconn(conn)


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


def insert_telemetry(
    meter_serial: str,
    interval_timestamp: str,
    raw_data: dict,
    received_at: str,
    source_raw_id: Optional[int] = None,
) -> None:
    """
    Inserts one parsed + canonicalized reading into meter_telemetry.
    raw_data dict is stored as JSONB.
    Uses ON CONFLICT DO NOTHING to safely handle duplicates.
    """
    sql = """
        INSERT INTO meter_telemetry
            (meter_serial, interval_timestamp, raw_data, received_at, source_raw_id)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT ON CONSTRAINT uq_telemetry_interval DO NOTHING;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                meter_serial,
                interval_timestamp,
                json.dumps(raw_data),
                received_at,
                source_raw_id,
            ))


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
) -> None:
    """
    Writes a detected anomaly to anomaly_log.
    """
    sql = """
        INSERT INTO anomaly_log (
            meter_serial, interval_timestamp,
            rule_based_flag, zscore_flag, if_flag,
            if_score, zscore_value,
            rule_violations, feature_snapshot
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
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
            ))


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
            ORDER BY interval_timestamp DESC
            LIMIT %s;
        """
        params = (meter_serial, before_timestamp, n)
    else:
        sql = """
            SELECT interval_timestamp, raw_data
            FROM meter_telemetry
            WHERE meter_serial = %s
            ORDER BY interval_timestamp DESC
            LIMIT %s;
        """
        params = (meter_serial, n)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

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