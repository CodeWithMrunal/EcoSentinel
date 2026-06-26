"""
pipeline/obis_parser.py
------------------------
Parses the raw pipe-delimited rawValue string from the HES API
into a structured dict keyed by OBIS code.

Input  (rawValue string):
    "1,0.0.1.0.0.255,2,2025-11-12 10:00:00,|2,1.0.12.27.0.255,2,225.91,V|..."

Output (parsed dict):
    {
        "interval_timestamp": "2025-11-12 10:00:00",
        "readings": {
            "1.0.12.27.0.255": {"value": 225.91, "unit": "V"},
            "1.0.1.29.0.255":  {"value": 0.0,    "unit": "Wh"},
            ...
        }
    }

This module does NOT interpret what the OBIS codes mean —
that is the job of canonical_mapper.py.
"""

import logging
from time import perf_counter
from typing import Optional

logger = logging.getLogger(__name__)

# OBIS code for the clock/timestamp object (entry 1)
TIMESTAMP_OBIS = "0.0.1.0.0.255"


class OBISParseError(Exception):
    """Raised when a rawValue string cannot be parsed."""
    pass


def _parse_entry(entry: str, entry_index: int) -> Optional[tuple]:
    """
    Parses a single pipe entry:
        "{sequence},{obis_code},{attribute},{value},{unit}"

    Returns (obis_code, value_str, unit_str) or None if entry is empty/malformed.
    Logs a warning on malformed entries rather than crashing — real HES
    payloads can contain empty trailing pipes.
    """
    entry = entry.strip()
    if not entry:
        return None

    parts = entry.split(",", 4)   # max 5 parts: seq, obis, attr, value, unit

    if len(parts) < 4:
        logger.warning(
            f"Malformed entry at index {entry_index}: '{entry}' "
            f"(expected at least 4 comma-separated fields)"
        )
        return None

    # parts[0] = sequence number  (ignored — we trust OBIS code)
    # parts[1] = obis code
    # parts[2] = attribute number (ignored — always 2=value in load profile)
    # parts[3] = value
    # parts[4] = unit             (may be absent for timestamp entry)

    obis_code  = parts[1].strip()
    value_str  = parts[3].strip()
    unit_str   = parts[4].strip() if len(parts) == 5 else ""

    return obis_code, value_str, unit_str


def _coerce_value(value_str: str, unit_str: str, obis_code: str) -> Optional[float | str]:
    """
    Attempts to coerce the value string to float.
    Returns the raw string for timestamp entries (unit is empty, value is datetime).
    Returns None and logs a warning if coercion fails for a numeric entry.
    """
    if obis_code == TIMESTAMP_OBIS:
        return value_str   # keep as string

    if value_str == "" or value_str is None:
        return None

    try:
        return float(value_str)
    except ValueError:
        logger.warning(
            f"Cannot coerce value '{value_str}' to float "
            f"for OBIS code {obis_code} (unit={unit_str})"
        )
        return None


def parse_raw_value(raw_value: str) -> dict:
    """
    Parses a full rawValue pipe-string into a structured dict.

    Parameters
    ----------
    raw_value : str
        The verbatim rawValue string from the HES API record.

    Returns
    -------
    dict with keys:
        "interval_timestamp" : str   — measurement time from clock object
        "readings"           : dict  — { obis_code: {"value": float, "unit": str} }

    Raises
    ------
    OBISParseError
        If no valid entries can be parsed, or the timestamp entry is missing.
    """
    if not raw_value or not raw_value.strip():
        raise OBISParseError("rawValue is empty or None.")

    started = perf_counter()
    logger.info(f"Parsing rawValue payload with {len(raw_value)} character(s).")

    entries = raw_value.split("|")

    interval_timestamp = None
    readings = {}

    for idx, entry in enumerate(entries):
        parsed = _parse_entry(entry, idx)
        if parsed is None:
            continue

        obis_code, value_str, unit_str = parsed
        coerced = _coerce_value(value_str, unit_str, obis_code)

        if obis_code == TIMESTAMP_OBIS:
            interval_timestamp = coerced   # string timestamp
        else:
            if coerced is not None:
                readings[obis_code] = {
                    "value": coerced,
                    "unit":  unit_str,
                }

    if interval_timestamp is None:
        raise OBISParseError(
            f"Timestamp OBIS entry ({TIMESTAMP_OBIS}) not found in rawValue. "
            f"Cannot determine interval_timestamp."
        )

    if not readings:
        raise OBISParseError(
            "No valid measurement entries found in rawValue after parsing."
        )

    logger.info(
        f"Parsed rawValue into {len(readings)} reading(s) in {(perf_counter() - started) * 1000:.1f} ms; interval_timestamp={interval_timestamp}."
    )

    return {
        "interval_timestamp": interval_timestamp,
        "readings": readings,
    }


def parse_api_record(record: dict) -> dict:
    """
    Parses a full HES API record dict (as received from the API).

    Parameters
    ----------
    record : dict
        One element of the API response list, e.g.:
        {
            "id": 449618,
            "meterSerial": "E0000002",
            "timestamp": "2025-11-12T04:38:09.523241+00:00",
            "obisCode": "1.0.99.1.0.255",
            "entryId": 5,
            "rawValue": "1,0.0.1.0.0.255,2,2025-11-12 10:00:00,|..."
        }

    Returns
    -------
    dict:
        {
            "id":                 int,
            "meter_serial":       str,
            "received_at":        str,   (API-level timestamp)
            "profile_obis_code":  str,
            "entry_id":           int,
            "interval_timestamp": str,   (from meter clock object)
            "readings":           dict   (obis_code → {value, unit})
        }

    Raises
    ------
    OBISParseError  — propagated from parse_raw_value
    KeyError        — if required envelope fields are missing
    """
    logger.info(
        f"Parsing API record id={record.get('id')} meter={record.get('meterSerial')} entry={record.get('entryId')}."
    )
    parsed_payload = parse_raw_value(record["rawValue"])
    logger.debug(
        f"API record id={record.get('id')} parsed into interval_timestamp={parsed_payload['interval_timestamp']} with OBIS codes={list(parsed_payload['readings'].keys())}."
    )

    return {
        "id":                 record["id"],
        "meter_serial":       record["meterSerial"],
        "received_at":        record["timestamp"],
        "profile_obis_code":  record["obisCode"],
        "entry_id":           record["entryId"],
        "interval_timestamp": parsed_payload["interval_timestamp"],
        "readings":           parsed_payload["readings"],
    }