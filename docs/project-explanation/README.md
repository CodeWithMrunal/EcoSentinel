# EcoSentinel — Project Explanation & Technical Reference

> A comprehensive, **code-grounded** reference for the EcoSentinel Smart Meter Anomaly Detection
> System: what it is today, how it actually works, where it breaks, and a realistic path from the
> current prototype to a production-grade, real-time, multi-meter, multi-locality deployment.
>
> Every "how it works today" claim in these documents is traceable to code in this repository, cited
> as `file:line` (paths under `ecosentinel-backend/` unless noted). Where behaviour depends on runtime
> conditions, that condition is stated explicitly. This is written to be trusted precisely *because it
> does not oversell* — gaps, bugs, and risks are called out plainly.

---

## Executive summary

EcoSentinel is a **full-stack, single-phase smart-meter anomaly detector**: a FastAPI backend runs a
6-stage pipeline (OBIS parse → canonical map → feature engineering → rule-based → z-score → group-routed
Isolation Forest), flags a reading if **any** layer fires, persists results to PostgreSQL, and
asynchronously generates a **human-readable LLM explanation** (provider-agnostic via LiteLLM, local
Ollama by default). A React/TypeScript SPA drives detection, explanation polling, and ops.

**What's genuinely strong:** a clean single-source-of-truth config, capability-group model routing for
heterogeneous meters, graceful degradation everywhere, schema-on-read telemetry, and a differentiated,
on-prem-capable explanation layer.

**Recently fixed:**
- ✅ The Isolation Forest's primary feature (`hourly_primary_ratio`) was **train/serve-skewed and inert
  at inference**; it now reads its same-hour baseline from a per-meter DB lookback and carries real
  signal in production. The same fix reactivates the `same_hour_deviation` z-score trigger.
  ([C1](./known-limitations.md)/[C2](./known-limitations.md)).

**What's still broken or missing (honest headline):**
- 🔴 **Frequency anomalies are undetectable by any layer**; `group_D` silently ignores frequency and
  export energy.
- ⚠️ **Three-phase support does not exist** — the pipeline is single-phase only, while three-phase is
  an explicit future requirement of the project.
- ⚠️ Structural false-positive pressure (contamination 0.07 + OR-verdict + tiny z-score window); no
  streaming ingestion; no auth; non-durable async LLM tasks; no feedback loop; no drift/MLOps beyond a
  hot-reload endpoint.

It is a **well-architected prototype trained on synthetic data**, production-*shaped* but not
production-*ready*. The roadmap sequences correctness → three-phase/real data → production real-time →
scale → advanced differentiation.

---

## How to read these documents

Three markers are applied consistently so current reality is never confused with a recommendation:

| Marker | Meaning |
|---|---|
| ✅ | **Implemented** — exists and works in the repo today |
| ⚠️ | **Partial / caveated** — exists but incomplete, fragile, or has a known issue |
| 🔲 | **Not yet / Recommended** — a proposal for the future, not current reality |

Correctness issues are tagged `C1`–`C15` in [`known-limitations.md`](./known-limitations.md) and
referenced from every other file by their tag.

---

## Table of contents

| # | File | What it covers |
|---|---|---|
| — | **[known-limitations.md](./known-limitations.md)** | 🔴 The honest-engineer ledger — every bug, tech-debt item, and gap with `file:line` citations (`C1`–`C15`). Start here for the truth. |
| 01 | **[01-current-state.md](./01-current-state.md)** | End-to-end code-grounded walkthrough; everything the system handles today (meter groups, OBIS, layers, endpoints, DB, frontend); the business need; where it breaks down. |
| 02 | **[02-business-and-market.md](./02-business-and-market.md)** | AMI/theft-detection market, competitors, where EcoSentinel is differentiated vs lagging, and future features framed by value/effort. |
| 03 | **[03-meter-types-and-3phase.md](./03-meter-types-and-3phase.md)** | Adding new meter types/parameters (config-only vs code); the full **three-phase** change plan per pipeline stage. |
| 04 | **[04-model-training-and-locality.md](./04-model-training-and-locality.md)** | Synthetic data & training internals and realism; is Isolation Forest the right long-term model; locality/customer-class modeling; cold-start; drift; MLOps; human-in-the-loop feedback. |
| 05 | **[05-realtime-streaming.md](./05-realtime-streaming.md)** | Kafka push vs HES pull; streaming re-architecture; post-detection workflow (alert/ticket/SLA/feedback); late/duplicate/malformed/cold-window problems with concrete solutions. |
| 06 | **[06-scale-and-architecture.md](./06-scale-and-architecture.md)** | Behaviour at thousands→millions of meters; stateless vs stateful scaling; DB/caching/LLM bottlenecks; service topology at scale. |
| 07 | **[07-production-and-ops.md](./07-production-and-ops.md)** | Deployment topology; CI/CD & safe model rollout; missing production scripts; observability, security/privacy, multi-tenancy, DR, and cost. |
| 08 | **[08-roadmap.md](./08-roadmap.md)** | Prioritized roadmap (fix → productionize → scale → advanced) and candid future analysis with the key decisions and risks. |

**Suggested reading order:** `known-limitations.md` → `01` for grounding, then `02`–`08` in order.
For a specific concern: three-phase → `03`; ML/accuracy → `04`; streaming → `05`; scale → `06`;
deploy/security → `07`; what to do next → `08`.

---

## The things that matter most (from `08-roadmap.md`)

0. ✅ **`hourly_primary_ratio` is fixed** — same-hour baseline now read from a per-meter DB lookback;
   this unlocked the ML layer and reactivated the same-hour z-score trigger. [C1](./known-limitations.md)/[C2](./known-limitations.md)
1. **Wire frequency + make features data-driven** — stop dropping parameters. [C3](./known-limitations.md)/[C3.1](./known-limitations.md)
2. **Three-phase support** (rules-first) — a stated future requirement; the system is single-phase only. [C4](./known-limitations.md)
3. **Operator feedback loop** — gives the system its first real ground truth. [C13](./known-limitations.md)
4. **AuthN/Z + security** — the service currently trusts anyone who can reach it. [C10](./known-limitations.md)

---

*Deliverable note: these documents are the only artifact of this effort — no application code, config,
or pipeline behaviour was modified. Bugs discovered while reading are recorded in
[`known-limitations.md`](./known-limitations.md), not fixed.*
</content>
