"""
utils/reset_db.py
-----------------
Resets database tables for clean testing.

Usage
-----
# Full reset — clears everything (dev/test only)
python utils/reset_db.py --all

# Reset one specific meter
python utils/reset_db.py --meter E0000001

# Reset a list of meters
python utils/reset_db.py --meter E0000001 E0000002 TEST_NORMAL TEST_TAMPER

# Preview what would be deleted without actually deleting
python utils/reset_db.py --all --dry-run
python utils/reset_db.py --meter TEST_SPIKE --dry-run
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import psycopg2
from config.settings import DB_CONFIG


# =========================================================
# CONNECTION
# =========================================================

def get_conn():
    return psycopg2.connect(**DB_CONFIG)


# =========================================================
# RESET FUNCTIONS
# =========================================================

def reset_meter(meter_serial: str, dry_run: bool = False) -> dict:
    """
    Deletes all records for a specific meter across all three tables.
    Deletion order respects FK constraints:
        anomaly_log → meter_telemetry → raw_meter_readings
    The meter_telemetry table may include flagged_anomalous rows,
    but they are deleted the same way as normal telemetry.
    """
    counts = {}

    conn = get_conn()
    cur  = conn.cursor()

    try:
        # Count before deletion
        for table in ["anomaly_log", "meter_telemetry", "raw_meter_readings"]:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE meter_serial = %s",
                        (meter_serial,))
            counts[table] = cur.fetchone()[0]

        if not dry_run:
            cur.execute("DELETE FROM anomaly_log        WHERE meter_serial = %s",
                        (meter_serial,))
            cur.execute("DELETE FROM meter_telemetry    WHERE meter_serial = %s",
                        (meter_serial,))
            cur.execute("DELETE FROM raw_meter_readings WHERE meter_serial = %s",
                        (meter_serial,))
            conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

    return counts


def reset_all(dry_run: bool = False) -> dict:
    """
    Truncates all three tables entirely.
    RESTART IDENTITY resets auto-increment sequences.
    CASCADE handles FK constraints automatically.
    """
    counts = {}

    conn = get_conn()
    cur  = conn.cursor()

    try:
        for table in ["anomaly_log", "meter_telemetry", "raw_meter_readings"]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cur.fetchone()[0]

        if not dry_run:
            cur.execute(
                "TRUNCATE anomaly_log, meter_telemetry, raw_meter_readings "
                "RESTART IDENTITY CASCADE"
            )
            conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

    return counts


def show_table_summary() -> None:
    """Prints current row counts for all tables, plus a breakdown
    of explanation_status on anomaly_log (pending/completed/failed),
    which is useful once the decision engine is in use."""
    conn = get_conn()
    cur  = conn.cursor()
    print("\n  Current table state:")
    for table in ["raw_meter_readings", "meter_telemetry", "anomaly_log"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"    {table:<25} {count:>6} rows")

    cur.execute("""
        SELECT explanation_status, COUNT(*)
        FROM anomaly_log
        GROUP BY explanation_status
        ORDER BY explanation_status
    """)
    rows = cur.fetchall()
    if rows:
        print("\n  anomaly_log.explanation_status breakdown:")
        for status_val, count in rows:
            label = status_val if status_val else "(none / engine disabled)"
            print(f"    {label:<25} {count:>6} rows")

    cur.close()
    conn.close()


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reset meter anomaly database tables."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Truncate all tables (full reset)"
    )
    group.add_argument(
        "--meter",
        nargs="+",
        metavar="SERIAL",
        help="Reset specific meter(s) by serial number"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    args = parser.parse_args()

    prefix = "[DRY RUN] " if args.dry_run else ""

    show_table_summary()

    if args.all:
        print(f"\n{prefix}Resetting ALL tables ...")
        counts = reset_all(dry_run=args.dry_run)
        print(f"\n  {'Table':<25} {'Rows deleted':>12}")
        print(f"  {'-'*40}")
        for table, count in counts.items():
            print(f"  {table:<25} {count:>12}")

        if args.dry_run:
            print("\n  (dry run — nothing was deleted)")
        else:
            print("\n  ✓ All tables cleared.")

    else:
        for serial in args.meter:
            print(f"\n{prefix}Resetting meter: {serial}")
            counts = reset_meter(serial, dry_run=args.dry_run)
            print(f"  {'Table':<25} {'Rows deleted':>12}")
            print(f"  {'-'*40}")
            for table, count in counts.items():
                print(f"  {table:<25} {count:>12}")

        if args.dry_run:
            print("\n  (dry run — nothing was deleted)")
        else:
            print(f"\n  ✓ Meter(s) cleared: {args.meter}")

    show_table_summary()


if __name__ == "__main__":
    main()