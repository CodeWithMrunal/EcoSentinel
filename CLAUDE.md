# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EcoSentinel is a **Smart Meter Anomaly Detection System** — a full-stack app with a Python FastAPI backend and a React/TypeScript frontend. The backend runs a 6-stage ML detection pipeline on DLMS meter readings and optionally generates LLM explanations for anomalies via an async decision engine.

---

## Common Commands

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic training data (one-time; writes to dataset/)
python dataset/generate_dataset.py

# Train all models (must run before starting API; rewrites models/)
python training/train.py

# Start API server (auto-reload on code changes)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Hot-reload models after retraining without restarting the server
curl -X POST http://localhost:8000/model/reload
```

### Frontend

```bash
cd ecosentinel-frontend

npm install
cp .env.example .env   # set VITE_API_BASE_URL=http://localhost:8000

npm run dev      # Vite dev server at http://localhost:5173
npm run build    # TypeScript compile + Vite production bundle
npm run lint     # ESLint (zero warnings tolerance)
npm run preview  # Serve production build locally
```

### Testing

There is no automated test suite. Testing is manual:

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Detection (JSON payload examples in test_data_payloads.json)
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d @test_data_payloads.json
```

---

## Architecture

### Backend Pipeline (`pipeline/`)

Each `POST /detect` request runs every record through **6 sequential stages**:

1. **OBIS Parser** (`obis_parser.py`) — splits pipe-delimited `rawValue` into structured readings
2. **Canonical Mapper** (`canonical_mapper.py`) — OBIS codes → canonical feature names (e.g., `voltage`, `energy_consumption`)
3. **Feature Engineer** (`feature_engineer.py`) — computes rolling stats, `hourly_primary_ratio`, and per-parameter derived features. Primary series is energy if available, else current, else voltage.
4. **Rule-Based Detection** (`rule_based.py`) — 7 deterministic rules (negative energy, voltage out-of-range, etc.)
5. **Z-Score Detection** (`zscore_detector.py`) — fires on deviations > 3σ or > 4× ratio
6. **Isolation Forest** (`if_detector.py`) — routes to one of 6 per-capability-group models or global fallback

**Verdict rule:** `is_anomaly = True` if **any** layer fires (conservative; maximizes recall).

### IF Feature Design

The Isolation Forest is fed **time-normalised features** — specifically `hourly_primary_ratio` = primary_value / historical_avg_same_hour, plus raw absolute values and time features. `z_score`, `spike_ratio`, `current_z_score`, and `current_spike_ratio` are **excluded** from the IF feature set: with a 5-reading (2.5hr) rolling window, natural morning load ramp-ups produce z_scores of 15–20 for completely normal readings, while subtle anomalies only reach z_scores of 3–6. IF would treat the normal ramp-up as more isolatable (more anomalous) → ROC-AUC inverts below 0.5. `hourly_primary_ratio` avoids this by normalising against the same-hour historical average, yielding ≈ 1.0 for any normal reading regardless of time of day.

**Same-hour baseline source (C1 fix).** `historical_avg_same_hour` must come from a *large* per-meter history so the current reading's hour-of-day is actually represented. The 5-reading rolling window (used for `delta`/`rolling_*`/`z_score`) spans only 2.5h and never contains the same hour, which would leave `hourly_primary_ratio` stuck at its neutral `1.0` default at inference (train/serve skew). At serving, the API injects a `baseline_provider` (`api/main._make_baseline_provider`) that reads the same-hour / same-day-type averages from a `SAME_HOUR_LOOKBACK_DAYS` (30-day) DB lookback via `db.client.get_historical_avg_same_hour` / `get_historical_avg_same_day_type`. `compute_features`/`summarize_rolling_state` accept this optional provider and fall back to the in-window scan when it is absent (training, or DB unavailable). Training is unchanged — it accumulates full per-meter history in memory, so it always saw a real baseline. This also reactivates the z-score `SAME_HOUR_DEVIATION_EXCEEDED` trigger. A meter with no accumulated history still defaults to `1.0` (cold-start).

### Capability Group Model Routing

The backend has 6 Isolation Forest models (group_A through group_V), each trained on a specific subset of meter features, plus a global fallback:

- **Exact match** → use that group's model
- **Subset match** → use smallest superset group model
- **No match** → global fallback with NaN imputation

**Why per-group models?** A voltage-only meter imputed with a global median energy value would produce false anomaly signals. Group models are trained on clean data only.

### Dataset & Training

The synthetic training dataset uses a deterministic **`METER_ROSTER`** (72 meters: 12 per group × 6 groups) defined in `dataset/generate_dataset.py`. The first 2–3 meters of each group use the real test serials from `test_data_payloads.json`. Training uses a **stratified 80/20 split** — 10 train / 2 test meters per group — guaranteeing evaluation metrics for every group. Anomaly injection is capability-aware: if an injected anomaly has no visible effect on the meter's tracked parameters, it is relabeled as `"normal"` to prevent phantom training labels.

### Configuration — Single Source of Truth

`config/settings.py` is where everything lives: OBIS registry, capability group definitions, feature schema, detection thresholds, `ROLLING_WINDOW_SIZE` (5-reading recent baseline) and `SAME_HOUR_LOOKBACK_DAYS` (30-day same-hour baseline window — kept separate on purpose, see IF Feature Design), Decision Engine config. **Any change to meter types, OBIS codes, or thresholds starts here.**

### Decision Engine (`decision_engine/`)

When an anomaly is flagged, an async background task calls an LLM (Ollama or OpenAI via `litellm`) to generate a human-readable explanation. This does **not** block the `/detect` response. The frontend polls `GET /anomalies/{id}/explanation` until `explanation_status` moves from `pending` → `completed` or `failed`.

### Database (`db/`)

Three PostgreSQL tables:
- `raw_meter_readings` — immutable audit trail of every raw record
- `meter_telemetry` — parsed canonical features (used for rolling-window history **and** the per-meter same-hour baseline that feeds `hourly_primary_ratio`, via `get_historical_avg_same_hour`)
- `anomaly_log` — flagged anomalies + LLM explanation fields

**DB is optional** — the API degrades gracefully if PostgreSQL is unavailable (history-based features and the same-hour baseline will be absent — `hourly_primary_ratio` falls back to `1.0` — but rule-based and IF detection still run).

### Frontend (`ecosentinel-frontend/`)

React 18 + Vite + TypeScript SPA with three pages (React Router v6):
- `/detect` — POST /detect form + results
- `/explain` — Enter anomaly ID, poll for LLM explanation
- `/ops` — Health check, model info, model reload

State is managed with **Zustand + Immer**. Components use **Radix UI** (accessible headless primitives) styled with **Tailwind CSS**.

---

## Key Paths

| Purpose | Path |
|---|---|
| API entry point | `api/main.py` |
| Pipeline orchestrator | `pipeline/__init__.py` |
| All config/thresholds | `config/settings.py` |
| DB schema | `db/schema.sql` |
| Frontend entry | `ecosentinel-frontend/src/App.tsx` |
| Frontend env template | `ecosentinel-frontend/.env.example` |
| Test payloads | `test_data_payloads.json` |
| Trained model artifacts | `models/` (gitignored; must run `training/train.py`) |
