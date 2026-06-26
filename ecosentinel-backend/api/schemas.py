"""
api/schemas.py
--------------
Pydantic models for request validation and response serialization.

Request  : exactly mirrors the HES API payload structure.
Response : structured detection result per record.
"""

from pydantic import BaseModel, Field
from typing import Optional


# =========================================================
# REQUEST — single HES API record
# =========================================================

class MeterRecord(BaseModel):
    """
    One record from the HES API, as received verbatim.
    Field names match the API camelCase convention.
    """
    id:          int    = Field(..., description="API-supplied record ID")
    meterSerial: str    = Field(..., description="Meter serial number, e.g. 'E0000002'")
    timestamp:   str    = Field(..., description="API receive timestamp (ISO 8601)")
    obisCode:    str    = Field(..., description="Load survey profile OBIS code")
    entryId:     int    = Field(..., description="Sequence number within the batch")
    rawValue:    str    = Field(..., description="Pipe-delimited rawValue string")

    model_config = {"json_schema_extra": {
        "example": {
            "id": 449618,
            "meterSerial": "E0000002",
            "timestamp": "2025-11-12T04:38:09.523241+00:00",
            "obisCode": "1.0.99.1.0.255",
            "entryId": 5,
            "rawValue": (
                "1,0.0.1.0.0.255,2,2025-11-12 10:00:00,"
                "|2,1.0.12.27.0.255,2,225.91,V"
                "|3,1.0.1.29.0.255,2,1.6,Wh"
                "|4,1.0.11.27.0.255,2,1.4,A"
                "|5,1.0.13.27.0.255,2,0.92,"
            )
        }
    }}


class DetectRequest(BaseModel):
    """
    POST /detect request body.
    Accepts a batch of HES API records (1 or more).
    """
    records: list[MeterRecord] = Field(
        ...,
        min_length=1,
        description="One or more HES API records to run detection on."
    )


# =========================================================
# RESPONSE — per-record detection result
# =========================================================

class LayerResult(BaseModel):
    layer:      str
    is_anomaly: bool

class RuleLayerResult(LayerResult):
    violations: list[str]
    details:    dict

class ZScoreLayerResult(LayerResult):
    z_score:     Optional[float]
    spike_ratio: Optional[float]
    triggers:    list[str]
    details:     dict

class IFLayerResult(LayerResult):
    anomaly_score: Optional[float]
    prediction:    Optional[int]
    model_used:    Optional[str] = None
    features_used: Optional[list[str]] = None

class DetectionLayers(BaseModel):
    rule_based:       RuleLayerResult
    zscore:           ZScoreLayerResult
    isolation_forest: IFLayerResult


class DetectResponse(BaseModel):
    """
    Detection result for a single meter record.
    """
    meter_serial:        str
    interval_timestamp:  str
    is_anomaly:          bool
    layers:              Optional[DetectionLayers] = None
    features:            Optional[dict]            = None
    error:               Optional[str]             = None

    # ── Decision Engine ──────────────────────────────────
    # Set when is_anomaly=True and the decision engine is enabled.
    # The explanation itself is generated asynchronously — use
    # anomaly_id to poll GET /anomalies/{id}/explanation.
    anomaly_id:          Optional[int] = None
    explanation_status:  Optional[str] = None


class DetectBatchResponse(BaseModel):
    """
    Response for POST /detect — one result per input record.
    """
    total:      int
    anomalies:  int
    results:    list[DetectResponse]


# =========================================================
# DECISION ENGINE — explanation response
# =========================================================

class AnomalyExplanationResponse(BaseModel):
    """
    Response for GET /anomalies/{id}/explanation.
    """
    anomaly_id:               int
    meter_serial:             str
    interval_timestamp:       str
    explanation_status:       Optional[str] = None   # pending | completed | failed | None
    explanation:              Optional[dict] = None
    explanation_generated_at: Optional[str] = None
    explanation_error:        Optional[str] = None

    # Echo back detection context for convenience
    rule_violations:          Optional[list] = None
    zscore_value:             Optional[float] = None
    if_score:                 Optional[float] = None