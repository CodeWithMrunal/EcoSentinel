"""
pipeline/canonical_mapper.py
-----------------------------
Maps OBIS-keyed readings (output of obis_parser.py) to
canonical feature names used throughout the pipeline and
stored in meter_telemetry.raw_data (JSONB).

Input  (readings dict from obis_parser):
    {
        "1.0.12.27.0.255": {"value": 225.91, "unit": "V"},
        "1.0.1.29.0.255":  {"value": 0.0,    "unit": "Wh"},
        "1.0.11.27.0.255": {"value": 0.0,    "unit": "A"},
    }

Output (canonical dict — what gets stored in raw_data JSONB):
    {
        "voltage":            225.91,
        "energy_consumption": 0.0,
        "current":            0.0,
    }

Unknown OBIS codes are logged as warnings and skipped —
they do not break the pipeline; they will appear as NaN
in the feature matrix and be imputed.
"""

import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config.settings import OBIS_REGISTRY

logger = logging.getLogger(__name__)

# =========================================================
# BUILD LOOKUP FROM REGISTRY
# Single source of truth: config/settings.py OBIS_REGISTRY
# =========================================================

# { obis_code → canonical_name }  (timestamp entry excluded)
_OBIS_TO_CANONICAL: dict[str, str] = {
    obis: meta["canonical_name"]
    for obis, meta in OBIS_REGISTRY.items()
    if not meta["is_timestamp"] and meta["canonical_name"] is not None
}

# Set of known OBIS codes (including timestamp) for warning suppression
_KNOWN_OBIS: set[str] = set(OBIS_REGISTRY.keys())

# Track unknown codes we've already warned about to avoid log spam
_warned_unknown: set[str] = set()


def map_to_canonical(readings: dict) -> dict:
    """
    Converts an OBIS-keyed readings dict to a canonical feature dict.

    Parameters
    ----------
    readings : dict
        Output of obis_parser.parse_raw_value()["readings"].
        Format: { obis_code: {"value": float, "unit": str} }

    Returns
    -------
    dict
        Canonical feature dict: { canonical_name: float_value }
        Only OBIS codes with known canonical mappings are included.
        Unknown OBIS codes are skipped with a one-time warning.
    """
    canonical = {}

    for obis_code, payload in readings.items():

        if obis_code not in _KNOWN_OBIS:
            if obis_code not in _warned_unknown:
                logger.warning(
                    f"Unknown OBIS code encountered: '{obis_code}' "
                    f"(unit={payload.get('unit', '?')}, "
                    f"value={payload.get('value', '?')}). "
                    f"Add it to OBIS_REGISTRY in config/settings.py to map it."
                )
                _warned_unknown.add(obis_code)
            continue

        canonical_name = _OBIS_TO_CANONICAL.get(obis_code)

        if canonical_name is None:
            # Known but explicitly unmapped (e.g. timestamp entry —
            # should not appear here since parser separates it out)
            continue

        value = payload.get("value")

        if value is None:
            logger.debug(
                f"Skipping OBIS code {obis_code} "
                f"({canonical_name}): value is None."
            )
            continue

        canonical[canonical_name] = value

    return canonical


def get_canonical_name(obis_code: str) -> str | None:
    """
    Returns the canonical name for a single OBIS code,
    or None if not mapped (timestamp or unknown).
    Useful for display / debugging.
    """
    return _OBIS_TO_CANONICAL.get(obis_code)


def get_known_obis_codes() -> list[str]:
    """Returns all OBIS codes registered in settings."""
    return list(OBIS_REGISTRY.keys())


def get_canonical_feature_names() -> list[str]:
    """Returns all canonical feature names derived from the registry."""
    return list(_OBIS_TO_CANONICAL.values())