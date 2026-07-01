# Known Limitations, Technical Debt & Correctness Issues

> This file is the honest-engineer ledger for EcoSentinel. Every item is grounded in
> code that exists in the repo **right now**, with `file:line` citations. Nothing here
> is speculative behaviour — where a claim depends on runtime conditions, that condition
> is stated explicitly.
>
> Legend: **🔴 Correctness bug** (produces wrong results) · **🟠 Design gap / tech debt**
> (works, but fragile or incomplete) · **🟡 Doc/΅config drift** (misleading, low risk).
>
> Paths are relative to `ecosentinel-backend/` unless noted.

---

## 🔴 C1 — `hourly_primary_ratio` (the primary IF feature) is train/serve-skewed and near-dead at inference

**The single most important finding in the codebase.**

The Isolation Forest was deliberately redesigned around one feature, `hourly_primary_ratio`,
to avoid the diurnal-ramp problem (documented in `CLAUDE.md` "IF Feature Design" and in
`config/settings.py:288-296`). It is computed as:

```
hourly_primary_ratio = primary_value / historical_avg_same_hour
```

`historical_avg_same_hour` is built in `feature_engineer.summarize_rolling_state`
(`feature_engineer.py:201-223`) by scanning `history` for readings whose hour-of-day equals
the current reading's hour:

```python
same_hour_vals = [
    float(h["raw_data"][hist_key])
    for h in history
    if _parse_hour(h.get("interval_timestamp")) == hour_of_day
]
```

**The problem:** at inference, `history` contains only `ROLLING_WINDOW_SIZE = 5` rows
(`config/settings.py:333`, fetched in `api/main.py:_fetch_history` → `db/client.get_last_n_readings`
with `n=ROLLING_WINDOW_SIZE`). Five 30-minute readings span only 2.5 hours, so **none of them
share the current reading's hour-of-day**. Therefore `historical_avg_same_hour` is almost
always `None`, and the ratio falls back to its neutral default:

```python
hourly_primary_ratio: float = 1.0        # feature_engineer.py:229
if primary_val is not None and historical_avg_same_hour is not None and historical_avg_same_hour > 1e-3:
    hourly_primary_ratio = float(primary_val) / (historical_avg_same_hour + 1e-5)
```

**But at training time** (`train.py:_engineer_features`, lines 139-168) history is accumulated
row-by-row across the meter's **entire 30-day series**, so same-hour history is abundant and
`hourly_primary_ratio` carries real signal. The model therefore learns a distribution for a
feature that, in production, collapses to a constant `1.0`.

**Consequences:** the group IF models are effectively scoring on time features + raw absolute
values only; the feature they were designed around is inert at serving time. This is classic
**train/serve skew** and likely the biggest driver of unreliable IF verdicts on real traffic.

**Fix direction (see `04-model-training-and-locality.md`):** either (a) compute
`historical_avg_same_hour` from a persisted per-meter/per-hour baseline table instead of the
5-row window, or (b) fetch a much larger history window (e.g. last N days at this hour) for the
same-hour statistic while keeping the 5-row window for `delta`/`rolling_*`. The two windows serve
different purposes and should not share `ROLLING_WINDOW_SIZE`.

---

## 🔴 C2 — `same_hour_deviation` z-score trigger is effectively unreachable at inference

`zscore_detector.check` computes (`zscore_detector.py:114-124`):

```python
same_hour_deviation = None
if energy is not None and same_hour_avg not in (None, 0):
    same_hour_deviation = abs(energy - same_hour_avg) / abs(same_hour_avg)
    if same_hour_deviation > SAME_HOUR_DEVIATION_THRESHOLD:   # 0.40
        triggers.append("SAME_HOUR_DEVIATION_EXCEEDED")
```

It depends on `historical_avg_same_hour`, which per **C1** is ~always `None` at inference. So the
`SAME_HOUR_DEVIATION_EXCEEDED` trigger described throughout `test_data_payloads.json`
(e.g. the `same_hour_deviation` test, lines 81-98) will not fire in a real deployment even though
the payloads assert it does. Same root cause as C1.

---

## 🔴 C3 — `group_D` ("full feature set") never trains on `frequency` or `active_export_energy`

`group_D` is defined with 7 raw features including `frequency` and `active_export_energy`
(`config/settings.py:210-218`). But the training feature-list builder only ever appends
energy, apparent_import_energy, voltage, current, and power_factor (`train.py:_group_feature_list`,
lines 93-125). There is **no branch** for `frequency` or `active_export_energy`, so they are
dropped from `group_D`'s trained feature vector.

Meanwhile `if_detector._group_features_for` (`if_detector.py:79-127`) *does* contain an "other raw
features" loop (`if_detector.py:112-117`) that would add them — but that function is **dead code**:
inference builds its row from the saved `feature_schema.joblib["features"]`
(`if_detector.py:_run_group_model`, line 290-297) and routing uses `_resolve_group`
(`if_detector.py:134-169`), so `_group_features_for` is never called.

**Net effect:** the `group_D` model ignores frequency and export energy entirely, and — per
**C3.1 below** — the rule layer cannot see frequency either. So a frequency anomaly is caught by
**no layer at all**, contradicting the `frequency_anomaly` group_D test in
`test_data_payloads.json` (lines 495-512) which asserts `FREQUENCY_OUT_OF_RANGE` fires.

---

## 🔴 C3.1 — `frequency` never reaches any detector; the rule-layer frequency check is dead code

`OBIS_REGISTRY` maps `1.0.14.27.0.255` → canonical `frequency` (`config/settings.py:156-161`), so
frequency survives OBIS parsing and canonical mapping. But **`feature_engineer.compute_features`
never copies `frequency` into the emitted feature dict** (verified: no occurrence of `frequency` in
`feature_engineer.py`), and `frequency` is not in `ALL_FEATURES`
(`config/settings.py:260-278`), so the fill-missing loop (`feature_engineer.py:369-371`) doesn't add
it either.

The rule layer then reads:

```python
frequency = features.get("frequency") if "frequency" in features else None   # rule_based.py:89
```

Since `"frequency"` is never a key in `features`, this is **always `None`**, so the
`frequency_out_of_range` rule (`rule_based.py:142-149`) can never fire. Combined with **C3**
(frequency absent from the group_D model) and the fact that frequency is not the primary rolling
series, **frequency anomalies are undetectable by the rule, z-score, and IF layers alike.** This is
a genuine correctness bug, not just a design gap.

---

## 🟠 C4 — Single-phase-only design; three-phase meters are not supported (a stated future requirement)

The system is built and tested **exclusively for single-phase meters**. `OBIS_REGISTRY`
(`config/settings.py:70-162`) registers only *averaged* voltage/current codes
(`1.0.12.27.0.255`, `1.0.11.27.0.255`) plus single scalar energy/PF/frequency codes — there are **no
per-phase codes, no phase-imbalance concept, and no neutral-current concept** anywhere in config,
feature engineering (`feature_engineer.py`), the rule layer (`rule_based.py`), or the capability
groups (`config/settings.py:187-231`). The synthetic training set is likewise entirely single-phase
(`generate_dataset.derive_electrical` uses single-phase AC relations, lines 285-330).

**Three-phase support is an explicit project requirement / future scope, not a current
capability.** If a three-phase payload carrying per-phase OBIS codes were received today, those
codes would hit the "unknown OBIS" path and be silently dropped (`canonical_mapper.py:79-88`); a
pure-3φ payload could yield an empty canonical dict → `empty_canonical` error
(`pipeline/__init__.py:97-105`). This is marked 🟠 (a design gap against a stated requirement)
rather than 🔴, because it is not a defect within the single-phase scope the code currently targets.
Full change plan in `03-meter-types-and-3phase.md`.

---

## 🟠 C5 — Structural false-positive pressure (contamination + "any layer fires" + small window)

Three independent design choices compound into a high false-positive rate:

1. **`if_contamination = 0.07`** (`config/settings.py:331`). Isolation Forest's `contamination`
   sets the decision threshold so that ~7% of the *training* distribution is labelled anomalous.
   On in-distribution normal traffic the group models will therefore flag roughly 7% of readings
   as anomalies **by construction**, independent of whether anything is wrong.
2. **Verdict = OR of all layers** (`pipeline/__init__.py:159-163`): `is_anomaly` is `True` if
   rule **or** z-score **or** IF fires. This maximizes recall (intentional, per CLAUDE.md) but
   means the ~7% IF false-positive floor propagates directly to the final verdict.
3. **Z-score on a 5-reading window** (`ROLLING_WINDOW_SIZE = 5`) is itself vulnerable to the
   diurnal ramp: during the 6–9am rise, consecutive readings climb steeply, inflating `z_score`
   for perfectly normal readings. The IF was rebuilt to dodge this (C1), but the **z-score layer
   still uses the raw rolling z** (`zscore_detector.py:93-104`), so the ramp problem was never
   removed from that layer.

No layer weighting, no score fusion, no minimum-consensus rule exists to counterbalance this.

---

## 🟠 C6 — Cold-start meters are invisible to statistical/ML layers

A brand-new meter (or any meter whose telemetry has not been seeded/accumulated) has empty
`history`. In that state:

- `rolling_mean` = the single current value, `rolling_std = 0` → `z_score = 0`
  (`feature_engineer._rolling_features`, lines 115-137, and the `include_current=False` path).
- `hourly_primary_ratio` defaults to `1.0` (C1).

So the z-score and IF layers cannot fire; only the rule layer (hard thresholds) works. This is
called out honestly in `utils/seed_normal_history.py` (docstring lines 11-24) and is why the test
workflow *requires* seeding 48h of history first (`test_data_payloads.json:1-12`). In production
there is no equivalent automatic warm-up mechanism.

---

## 🟠 C7 — Anomalous readings are excluded from history, which can freeze the baseline

`get_last_n_readings` filters `AND flagged_anomalous = FALSE` (`db/client.py:307-308, 318`), and
telemetry is written with `flagged_anomalous = bool(result.is_anomaly)`
(`api/main.py:224`). This is deliberate (keep the baseline clean), but it has a side effect:
during a **legitimate sustained change** (e.g. a real step-up in consumption after new appliances),
every new reading deviates from the frozen pre-change baseline, gets flagged, and is therefore
excluded from history — so the baseline never catches up and the meter flags indefinitely until an
operator intervenes or history ages out. There is no "confirmed legitimate → re-admit to baseline"
path (related: no human-in-the-loop feedback, C13).

---

## 🟠 C8 — Detection has no per-meter or per-record concurrency; batch is a serial loop

`POST /detect` iterates records in a plain `for` loop (`api/main.py:371-452`), and each record does
its own synchronous DB round-trips (history fetch, raw insert, telemetry insert, anomaly insert,
plus **two extra** `summarize_rolling_state` baseline-logging passes and a second history fetch per
record — `api/main.py:392-420`). The connection pool is `min=1, max=10`
(`db/client.py:41-49`). This is fine for demo payloads but will not sustain streaming throughput;
see `05-realtime-streaming.md` and `06-scale-and-architecture.md`.

---

## 🟠 C9 — LLM explanation tasks are in-process and non-durable

Explanations run via FastAPI `BackgroundTasks` (`api/main.py:432-443`, `decision_engine/service.run_explanation_task`).
They execute in the same process as the API worker. Implications:

- If the process restarts while an explanation is `pending`, the task is lost and the
  `anomaly_log` row stays `pending` **forever** (nothing re-drives it). There is no queue, no
  retry-after-restart, no reaper.
- A slow local LLM (Ollama, "3–15s" per `api/main.py:587`) ties up the worker's task budget under
  load; there is no concurrency cap or backpressure on explanation generation.
- The LLM call itself has retries (`decision_engine/llm_client.py`, `num_retries`) and a
  `response_format` fallback (lines 161-179), which is good — but that only covers a live attempt,
  not a lost one.

---

## 🟠 C10 — No authentication, authorization, or tenancy on any endpoint

`api/main.py` defines `/detect`, `/health`, `/model/info`, `/model/reload`,
`/anomalies/{id}/explanation` with **no auth dependency** of any kind. CORS is open to
`http://localhost:5173` only (`api/main.py:149-154`), which is not a security control. Anyone who
can reach the port can submit readings, trigger LLM spend, reload models, and read anomaly
explanations for any meter. No notion of utility/DISCOM tenancy exists anywhere in the schema or
code. Covered in `07-production-and-ops.md`.

---

## 🟠 C11 — Hardcoded physical constants that break across localities

- `NOMINAL_VOLTAGE = 230.0` is hardcoded in `feature_engineer.py:34` and drives
  `voltage_deviation` (line 339). A 240 V or 220 V nominal region gets a systematically biased
  deviation feature.
- `FREQ_MIN/FREQ_MAX = 49.0/51.0` are hardcoded in `rule_based.py:56-57` (not in `DETECTION_CONFIG`),
  so a 60 Hz grid cannot be supported without a code change.
- `holiday` is defined as "Sunday only" (`feature_engineer._is_holiday`, lines 45-46) — not a real
  holiday calendar, and wrong for regions where Sunday is a working day.
- Voltage rule bounds (180/270) and PF bounds (0.0/1.0) are global constants in `DETECTION_CONFIG`
  with no per-locality or per-customer-class override.

---

## 🟠 C12 — Duplicate-in-batch and out-of-order readings are not handled within a request

History is fetched **once** at the top of processing for each record from the DB
(`api/main.py:377-384`). If a single `/detect` batch contains several readings for the *same*
meter, later readings do not see earlier ones from the same batch (they are only persisted after
processing). Ordering within the batch is the caller's responsibility. There is no watermarking or
reordering by `interval_timestamp` before processing.

---

## 🟠 C13 — No human-in-the-loop / labeled-feedback path

`anomaly_log` stores detector flags, feature snapshot, and LLM explanation
(`db/schema.sql:110-150`) but has **no operator disposition column** (confirmed / rejected /
false-positive) and no endpoint to record one. Consequently there is no mechanism to convert
operator judgements into labeled data, and models remain purely unsupervised with no feedback loop.
This blocks the drift/retraining story in `04-model-training-and-locality.md`.

---

## 🟡 C14 — Documentation and config drift (README/CLAUDE vs code)

The root `README.md` and `CLAUDE.md` describe an earlier version. Concrete mismatches with the
current code:

| Claim in docs | Reality in code |
|---|---|
| "10 simulated meters", "7,200 rows" (`README.md:368-370`) | `num_meters = 72`, 30 days × 48/day (`config/settings.py:55-60`, `generate_dataset.py`) |
| "19 features" / `ALL_FEATURES` = 19 (`README.md:337`) | `ALL_FEATURES` = 12 (`CORE_FEATURES` 6 + `OPTIONAL_FEATURES` 6, `config/settings.py:260-278`) |
| `if_contamination = 0.05` (`README.md:351, 981`) | `0.07` (`config/settings.py:331`) |
| `rolling_window_size = 10` (`README.md:982`) | `5` (`config/settings.py:332`) |
| `DERIVED_FEATURE_MAP["energy_consumption"]` lists delta/rolling/z_score/... (`README.md:319-327`) | Only `["hourly_primary_ratio"]` (`config/settings.py:293-295`) |
| Pseudo-labels reconstructed by heuristic `energy > 5× rolling_mean` (`README.md:432-440`) | Exact labels read from `anomaly_type` column (`train.py:_reconstruct_labels`, 171-181) |
| Backend at repo root (`config/`, `pipeline/`…) | Backend is under `ecosentinel-backend/` |
| `DB_PASSWORD` defaults to `"postgres"` (`README.md:258`) | No default; `os.getenv("DB_PASSWORD")` returns `None` (`config/settings.py:13`) |

Also `README.md` says the DB is "two-table design" in one place (`db/schema.sql:4`) but there are
three tables. The frontend `LLM_MODEL_GROUPS` lists `gemma4:latest` / "Gemma 4 9B"
(`constants/config.ts:161`) which does not correspond to a real released model name and may not
match what is installed (git history mentions renaming the gemma model to a locally-installed one).

---

## 🟡 C15 — Minor robustness / correctness nits

- **`day_of_week` and `hour_of_day` fed as raw integers** to `StandardScaler` + IF
  (`config/settings.py:261-266`, used as features). Treating cyclic time as linear means 23:00 and
  00:00 look maximally distant. Sine/cosine encoding would be more correct, though impact is small
  given IF's tree splits.
- **`_persist` does up to two extra full history reads + two `summarize_rolling_state` calls per
  record purely for logging** (`api/main.py:392-420`), doubling DB load on the hot path.
- **`get_historical_avg_same_hour` / `get_historical_avg_same_day_type`** exist in `db/client.py`
  (lines 345-402) and would fix C1/C2 if wired in — but **nothing calls them**; the feature
  engineer recomputes same-hour stats from the tiny in-request window instead. The correct plumbing
  is already half-built and simply not connected.
- **`sustained_zero` rule** requires `rolling_std < 0.01` AND `energy == 0.0`
  (`rule_based.py:102-112`); with the anomalous-exclusion behaviour (C7) and a 5-row window, the
  first zero readings may not satisfy it, so the "3 consecutive zeros" story in
  `test_data_payloads.json:207-240` depends on subtle timing.
- (Frequency handling promoted to its own finding **C3.1** above — it is a correctness bug, not a
  nit.)

---

## Cross-reference

| Item | Primarily discussed in |
|---|---|
| C1, C2, C6, C7, C13 (features, baselines, drift, feedback) | `04-model-training-and-locality.md` |
| C4 (three-phase) | `03-meter-types-and-3phase.md` |
| C5 (false positives, verdict fusion) | `01-current-state.md`, `04-model-training-and-locality.md` |
| C8, C9, C12 (throughput, durability, ordering) | `05-realtime-streaming.md`, `06-scale-and-architecture.md` |
| C10, C11, C14 (auth, locality config, drift) | `07-production-and-ops.md` |
</content>
</invoke>
