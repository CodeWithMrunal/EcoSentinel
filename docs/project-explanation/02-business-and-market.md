# 02 — Business Context, Market Relevance & Future Features

> **Scope:** where EcoSentinel sits in the smart-meter / AMI analytics market, how it compares to
> commercial offerings, where it is genuinely differentiated versus where it lags, and a
> value/effort-framed list of future features. Marker legend as in `01-current-state.md`
> (✅ / ⚠️ / 🔲).

---

## 1. The market EcoSentinel plays in

The relevant market is **AMI analytics / meter data analytics (MDA)**, specifically the
revenue-protection and grid-edge-monitoring slices:

| Segment | What it does | Representative players |
|---|---|---|
| **Non-technical-loss / energy-theft analytics** | Detect tampering, bypass, meter-stopping, billing fraud from meter + network data | Itron (Analytics/Grid), Landis+Gyr (Advanced Grid Analytics), Siemens EnergyIP MDM, Oracle Utilities DataRaker, C3 AI Energy Management, Bidgely, Awesense |
| **Meter data management (MDM/MDMS)** | VEE (validation/estimation/editing), storage, billing determinants | Oracle Utilities, Itron Temetra, Landis+Gyr, Kaluza, Siemens |
| **Power-quality / grid-edge monitoring** | Voltage/PF/frequency events, transformer health, phase issues | Sentient Energy, Sensus (Xylem), Depsys, GridEdge vendors |
| **Horizontal anomaly/observability + LLMs** | Generic time-series anomaly detection now bolting on GenAI copilots | AWS Lookout for Metrics (retired), Azure Anomaly Detector, Datadog/Dynatrace-style, plus utility "AI copilots" |

Schneider Electric itself competes here (EcoStruxure Grid, AMI, MDM). EcoSentinel is best understood
as an **internal prototype exploring the "layered detection + LLM explanation" pattern** for that
portfolio, not a greenfield market entrant.

### Market tailwinds (why this matters)
- **AT&C / non-technical losses** in many DISCOMs (notably India, the origin of the meter-serial
  conventions in the test data) run high; even 1–2 points of loss recovery is large money.
- **Mass smart-meter rollouts** (e.g. India's RDSS / Smart Metering National Programme) are creating
  millions of new meters and a flood of interval data that no human can triage manually.
- **GenAI in operations** is a live procurement theme — utilities want explainable, analyst-friendly
  alerts, not just scores.

---

## 2. How EcoSentinel compares — differentiation vs lag

### Where it is genuinely differentiated ✅ / ⚠️

1. **LLM "decision engine" for operator-facing explanations.** ✅ The structured explanation output
   (`anomaly_explanation`, `supporting_factors`, `possible_false_positive_scenarios`, `confidence`,
   `limitations` — `decision_engine/schemas.py`) is exactly the triage artifact revenue-protection
   teams want, and the *false-positive-scenario* field is a thoughtful, differentiating touch most
   scoring products lack. Provider-agnostic via LiteLLM (Ollama/OpenAI/Azure/Anthropic) means it can
   run **fully on-prem/air-gapped** with a local model — a real selling point for utilities with data
   -sovereignty constraints. This is the strongest differentiator.

2. **Capability-group model routing.** ✅ Training a model per parameter-profile and routing on the
   exact feature set (`if_detector._resolve_group`) is a clean answer to real fleet heterogeneity —
   many products either force a common schema or degrade to one global model with imputation. This
   design scales conceptually to new meter profiles as a config change.

3. **Layered, transparent detection.** ✅ Rule + statistical + ML with per-layer attribution
   (`layers.*` in the response) is more auditable than a single black-box score — regulators and
   operators can see *which* logic fired.

4. **Schema-on-read telemetry.** ✅ JSONB `raw_data` accommodates new parameters without migrations —
   a pragmatic fit for evolving meter fleets.

### Where it lags the market ⚠️ / 🔲

| Dimension | Market norm | EcoSentinel today |
|---|---|---|
| **Phase support** | 1φ **and** 3φ, per-phase imbalance, neutral current | ⚠️ Single-phase only ([C4](./known-limitations.md)) |
| **Ingestion** | Streaming (Kafka/MQTT) at fleet scale, VEE built-in | ⚠️ Request/response `POST /detect`; no streaming |
| **Model sophistication** | Forecasting-residual, per-meter baselines, network/topology-aware theft models, cross-meter correlation | ⚠️ Isolation Forest; its key feature (per-meter same-hour ratio) now works after the [C1](./known-limitations.md) fix, but no forecasting/topology models yet |
| **Feedback / learning loop** | Analyst dispositions feed retraining; case management | 🔲 None ([C13](./known-limitations.md)) |
| **Scale** | Millions of meters, horizontal workers | ⚠️ Serial batch loop, single API process ([C8](./known-limitations.md)) |
| **Security / multi-tenancy** | AuthN/Z, tenant isolation, encryption, audit | 🔲 No auth, single-tenant ([C10](./known-limitations.md)) |
| **Grid context** | Transformer/feeder topology, weather, tariff, GIS | 🔲 Meter-in-isolation only |
| **MLOps** | Versioning, drift monitors, canary, registry | ⚠️ Only a hot-reload endpoint exists |

**Honest positioning:** EcoSentinel's *explanation layer and routing design are ahead of the
commodity "anomaly score" tier*, but its *detection accuracy, phase coverage, and operational
maturity are well behind* established MDA vendors. It is a compelling **feature prototype**, not a
competitive product — yet.

---

## 3. Competitive "moat" analysis

- **Not a moat:** the ML itself (Isolation Forest on engineered features) is commodity and, as built,
  under-performs its own design intent. Any vendor can match or beat it.
- **Potential moat:** the **combination** of (a) local/on-prem explainable LLM triage, (b) heterogeneous
  -fleet routing, and (c) Schneider's existing grid/topology data. The LLM explanation grounded in
  *network context* (transformer, feeder, neighbours) — which EcoSentinel does **not** yet have — is
  where genuine differentiation would live. Today the explanation is meter-in-isolation, which any
  competitor can replicate.

---

## 4. Future features (value / effort framed)

Effort: **S** ≈ days, **M** ≈ weeks, **L** ≈ 1–2 months, **XL** ≈ quarter+. Value is business impact.

### Tier 1 — high value, correctness/foundation (do first)
| Feature | Value | Effort | Notes |
|---|---|---|---|
| ✅ Fix `hourly_primary_ratio` (same-hour baseline from per-meter DB lookback) | Very high | M | **Done** — unlocked the ML layer ([C1](./known-limitations.md)); `db/client.get_historical_avg_same_hour` now wired via a `baseline_provider`. Follow-up: persist baselines in a store to also cover cold-start ([C6](./known-limitations.md)) |
| Three-phase support (OBIS, canonical, features, rules, groups) | Very high | L | Stated future requirement; system is single-phase only ([C4](./known-limitations.md)); see `03-...` |
| Operator feedback loop + case disposition | High | M | Turns alerts into labeled data; enables supervised uplift ([C13](./known-limitations.md)) |
| Streaming ingestion (Kafka/HES-pull consumer) | High | L | Moves from demo to continuous operation; see `05-...` |

### Tier 2 — high value, differentiation
| Feature | Value | Effort | Notes |
|---|---|---|---|
| Network-context-aware explanations (transformer/feeder/neighbour comparison) | Very high | L | The real moat; needs topology data |
| Forecasting-residual detection (Prophet/LSTM/quantile) per meter/segment | High | L | Better accuracy than IF for consumption; see `04-...` |
| Cross-meter / feeder-level theft correlation (energy balance) | Very high | XL | Detects theft invisible at a single meter (sum of meters vs feeder head) |
| Locality/customer-class segmentation of models & thresholds | High | M | Residential vs commercial vs industrial baselines ([C11](./known-limitations.md)) |
| Analyst case-management UI (queue, assign, resolve, SLA) | High | L | Productizes the operator workflow |

### Tier 3 — valuable extensions
| Feature | Value | Effort | Notes |
|---|---|---|---|
| Weather/tariff/calendar covariates | Medium | M | Reduces seasonal false positives |
| Transformer/asset health & DER (solar/EV) detection | Medium–High | L | New anomaly classes; export-energy already partly modeled |
| Severity scoring & prioritization (not just boolean) | Medium | S | Rank alerts by expected revenue/risk |
| Drift monitors + auto-retrain triggers | Medium | M | Ops maturity; see `04-...` |
| Multi-tenancy for serving multiple DISCOMs | Medium | L | Commercial enabler; see `07-...` |
| Batch backfill / replay tooling | Medium | M | Re-score history after model/OBIS changes |

### Tier 4 — advanced / research
| Feature | Value | Effort | Notes |
|---|---|---|---|
| Graph/topology-aware GNN theft models | High | XL | State-of-the-art NTL detection |
| Agentic investigation (LLM tool-use over meter/GIS/CRM) | Medium–High | XL | Auto-gather evidence, draft field-work orders |
| Federated / privacy-preserving learning across utilities | Medium | XL | Data-sovereignty play |

---

## 5. Bottom line

EcoSentinel is **strategically well-aimed** — layered detection plus on-prem explainable LLM triage
for a heterogeneous meter fleet is exactly where the market is heading, and it aligns with Schneider's
grid portfolio. Its **near-term value depends on fixing the ML foundation and adding three-phase +
streaming** (Tier 1); its **long-term differentiation depends on network-context-aware detection and
explanation** (Tier 2), which is the one thing incumbents can't trivially copy because it leans on
grid-topology data. As it stands, treat it as a promising internal prototype whose explanation UX is
its crown jewel and whose detection engine needs a rebuild before any accuracy claims can be made.
</content>
