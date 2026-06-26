"""
decision_engine/prompt_builder.py
------------------------------------
Builds the prompt sent to the LLM for anomaly explanation.

Design notes
------------
- The prompt is built from an AnomalyContext (see schemas.py),
  which is assembled from the pipeline's PipelineResult plus
  historical readings from meter_telemetry.
- History is rendered as a compact table — token-efficient and
  easy for the LLM to scan for trends.
- Detection layer outputs (which rules fired, z-score, IF score)
  are given explicitly so the LLM explains *why the system flagged
  this*, not just describes the data in isolation.
- The system prompt enforces strict JSON-only output matching
  AnomalyExplanation. Output format is repeated in both system
  and user prompts because smaller local models (e.g. via Ollama)
  are less reliable at following single-instance instructions.
"""

import json
import logging
from decision_engine.schemas import AnomalyContext


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert electrical grid analyst specializing in smart meter \
anomaly investigation. You analyze flagged meter readings in the context of their \
recent history and the automated detection system's findings, then explain the \
anomaly in clear, actionable language for utility operations staff.

You must respond with ONLY a single valid JSON object, no markdown formatting, \
no code fences, no explanatory text before or after. The JSON object must have \
exactly this structure:

{
  "anomaly_explanation": "<detailed explanation of why this record was flagged>",
  "supporting_factors": ["<factor 1>", "<factor 2>", "..."],
  "possible_false_positive_scenarios": ["<scenario 1>", "<scenario 2>", "..."],
  "confidence": "High" | "Medium" | "Low",
  "limitations": "<any caveats about insufficient history or ambiguous data, or null>"
}

Guidelines:
- anomaly_explanation should reference specific numeric values and comparisons
  from the data provided (e.g. percentage deviation, which parameters moved
  together, what threshold was crossed).
- supporting_factors should be short, specific, evidence-based bullet points.
- possible_false_positive_scenarios should consider legitimate operational
  explanations: seasonal/time-of-day variation already accounted for by the
  system, meter maintenance, calibration, communication gaps, tariff changes,
  appliance/equipment changes, grid switching events.
- confidence reflects how certain you are that this is a genuine anomaly
  versus a false positive, given the evidence available. Use "Low" if the
  history window is short or the signal is ambiguous.
- limitations should note if the history window is too short, if key
  parameters are missing, or if the pattern could equally support multiple
  explanations.
- Do not invent values not present in the data. Do not output anything
  other than the JSON object."""


HISTORY_ROW_TEMPLATE = "{idx:>3} | {timestamp:<20} | {values}"


def _format_history_table(context: AnomalyContext) -> str:
    """
    Renders the historical readings as a compact aligned table.
    Each row shows the timestamp and all canonical values present.
    """
    if not context.history:
        return "(No historical readings available for this meter.)"

    lines = [f"{'#':>3} | {'Timestamp':<20} | Values"]
    lines.append("-" * 70)

    for i, reading in enumerate(context.history, 1):
        values_str = ", ".join(
            f"{k}={v}" for k, v in reading.values.items()
        )
        lines.append(
            HISTORY_ROW_TEMPLATE.format(
                idx=i,
                timestamp=str(reading.interval_timestamp),
                values=values_str,
            )
        )

    return "\n".join(lines)


def _format_detection_summary(context: AnomalyContext) -> str:
    """Renders what the automated detection layers found."""
    lines = []

    if context.rule_violations:
        lines.append(
            f"- Rule-based layer flagged: {', '.join(context.rule_violations)}"
        )
    else:
        lines.append("- Rule-based layer: no threshold violations")

    if context.zscore_triggers:
        z_val = (
            f"{context.zscore_value:.3f}"
            if context.zscore_value is not None else "N/A"
        )
        lines.append(
            f"- Statistical (z-score) layer flagged: "
            f"{', '.join(context.zscore_triggers)} (z-score = {z_val})"
        )
    else:
        z_val = (
            f"{context.zscore_value:.3f}"
            if context.zscore_value is not None else "N/A"
        )
        lines.append(f"- Statistical (z-score) layer: no triggers (z-score = {z_val})")

    if context.if_score is not None:
        model_info = (
            f" (model: {context.if_model_used})"
            if context.if_model_used else ""
        )
        lines.append(
            f"- Isolation Forest anomaly score: {context.if_score:.4f}{model_info} "
            f"(more negative = more anomalous)"
        )

    return "\n".join(lines)


def _format_current_reading(context: AnomalyContext) -> str:
    """Renders the anomalous reading's raw values and derived features."""
    raw_str = json.dumps(context.current_values, indent=2)

    # Show only the most analytically relevant derived features,
    # not the full 19-feature vector (reduces token usage and noise)
    relevant_derived = {
        k: v for k, v in context.features.items()
        if k in (
            "delta", "rolling_mean", "rolling_std", "z_score",
            "spike_ratio", "historical_avg_same_hour",
            "historical_avg_same_day_type", "voltage_deviation",
            "current_delta", "power_factor_deviation",
            "hour_of_day", "day_of_week", "is_weekend",
        ) and v is not None
    }
    derived_str = json.dumps(relevant_derived, indent=2)

    return (
        f"Raw measured values:\n{raw_str}\n\n"
        f"Computed features (relative to rolling history):\n{derived_str}"
    )


def build_messages(context: AnomalyContext) -> list[dict]:
    """
    Builds the full chat message list for the LLM call.

    Returns
    -------
    list of {"role": "system"|"user", "content": str}
    """

    user_prompt = f"""Analyze the following smart meter reading that was flagged as anomalous.

## Meter
Meter Serial: {context.meter_serial}
Timestamp of flagged reading: {context.interval_timestamp}

## Flagged Reading
{_format_current_reading(context)}

## Automated Detection Results
{_format_detection_summary(context)}

## Historical Readings (most recent {len(context.history)}, oldest to newest)
{_format_history_table(context)}

## Your Task
Based on the flagged reading, the computed features, the automated detection \
results, and the historical pattern shown above, provide your analysis as a \
JSON object with exactly the structure specified in the system prompt. \
Reference specific numbers from the data above in your explanation."""

    logger.info(
        f"Built LLM prompt for anomaly_id={context.anomaly_id}: user_prompt_length={len(user_prompt)}, history_rows={len(context.history)}."
    )
    logger.debug(
        f"LLM prompt preview for anomaly_id={context.anomaly_id}: {user_prompt[:500]}."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]