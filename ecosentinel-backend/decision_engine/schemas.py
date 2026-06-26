"""
decision_engine/schemas.py
----------------------------
Pydantic models for:
  1. The structured JSON the LLM is asked to return
     (anomaly explanation, confidence, false-positive scenarios)
  2. The context payload assembled internally before prompting

Keeping these as explicit models means:
  - The LLM output can be validated and coerced (e.g. confidence
    must be one of a fixed set of values)
  - If the LLM returns malformed JSON, validation errors are
    caught and handled as a "failed" explanation rather than
    crashing the background task
  - Switching providers never changes these shapes — they are
    the contract between the LLM and the rest of the system
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


# =========================================================
# LLM OUTPUT SCHEMA
# This is the JSON structure the prompt instructs the LLM
# to return. Matches the format specified in requirements.
# =========================================================

ConfidenceLevel = Literal["High", "Medium", "Low"]


class AnomalyExplanation(BaseModel):
    """
    Structured explanation returned by the LLM for one anomaly.
    Persisted as JSONB in anomaly_log.explanation.
    """

    anomaly_explanation: str = Field(
        ...,
        description="Detailed explanation of why this record was flagged anomalous."
    )

    supporting_factors: list[str] = Field(
        default_factory=list,
        description="Specific observations supporting the anomaly classification."
    )

    possible_false_positive_scenarios: list[str] = Field(
        default_factory=list,
        description="Legitimate explanations that could make this a false positive."
    )

    confidence: ConfidenceLevel = Field(
        ...,
        description="LLM's confidence in its explanation: High, Medium, or Low."
    )

    limitations: Optional[str] = Field(
        default=None,
        description="Any limitations due to insufficient historical context."
    )

    # ── Provenance — added by the service, not the LLM ────
    llm_provider: Optional[str] = Field(default=None, exclude=False)
    llm_model:    Optional[str] = Field(default=None, exclude=False)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, v):
        """Coerce common variants (lowercase, 'high', 'HIGH') to canonical form."""
        if isinstance(v, str):
            v_lower = v.strip().lower()
            mapping = {"high": "High", "medium": "Medium", "low": "Low"}
            if v_lower in mapping:
                return mapping[v_lower]
        return v

    @field_validator("supporting_factors", "possible_false_positive_scenarios", mode="before")
    @classmethod
    def coerce_list(cls, v):
        """If the LLM returns a single string instead of a list, wrap it."""
        if isinstance(v, str):
            return [v]
        if v is None:
            return []
        return v


# =========================================================
# INTERNAL CONTEXT SCHEMA
# Assembled by service.py before prompt construction.
# Not sent to the LLM as JSON directly — used to render
# the prompt template.
# =========================================================

class HistoricalReading(BaseModel):
    """One historical reading for prompt context."""
    interval_timestamp: str
    values: dict   # canonical raw_data values, e.g. {"energy_consumption": 1.6, "voltage": 230.1}


class AnomalyContext(BaseModel):
    """
    Full context bundle for one anomaly, used to render the
    prompt template. Assembled from PipelineResult + DB history.
    """

    meter_serial:        str
    interval_timestamp:  str

    # The anomalous reading's raw canonical values
    current_values:      dict

    # Full feature vector (derived features) at detection time
    features:            dict

    # Historical readings preceding the anomaly (oldest -> newest)
    history:             list[HistoricalReading] = Field(default_factory=list)

    # Detection layer outputs
    rule_violations:     list[str] = Field(default_factory=list)
    zscore_value:        Optional[float] = None
    zscore_triggers:     list[str] = Field(default_factory=list)
    if_score:            Optional[float] = None
    if_model_used:       Optional[str] = None

    # Anomaly id in anomaly_log — for traceability in logs only
    anomaly_id:          Optional[int] = None