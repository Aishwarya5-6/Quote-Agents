# InsurTech AI Pipeline — Engineering Documentation

> **4-Agent Auto Insurance Quote Engine**  
> FastAPI · LangGraph · XGBoost · Groq LLM · Next.js 14

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [System Architecture & Full-Stack Handshake](#2-system-architecture--full-stack-handshake)
3. [LangGraph Orchestrator — The 4-Agent DAG](#3-langgraph-orchestrator--the-4-agent-dag)
4. [Agent Deep Dives](#4-agent-deep-dives)
5. [The Storytelling UI Approach](#5-the-storytelling-ui-approach)
6. [Core Technologies & Libraries](#6-core-technologies--libraries)
7. [Edge Cases & Resilience Handling](#7-edge-cases--resilience-handling)
8. [Deployment Strategy](#8-deployment-strategy)
9. [API Contract Reference](#9-api-contract-reference)
10. [Local Development Setup](#10-local-development-setup)
11. [Environment Variables](#11-environment-variables)

---

## 1. System Overview

An insurance customer fills in a **quote form** — driver age, experience, accident history, citations, annual mileage, vehicle usage, and policy details. The system routes that quote through **four specialised AI agents in sequence**, each reading the upstream context before adding its own output. Within seconds, the result is a full underwriting verdict with SHAP-level explanations at every stage.

```
Customer Form → [Agent 1: Risk] → [Agent 2: Conversion] → [Agent 3: Advisor] → [Agent 4: Router]
                                                                                        ↓
                                              AUTO_APPROVE | MANUAL_REVIEW | REJECT + LLM reason
```

### Why decompose into four agents instead of one model?

| Concern | Monolithic Model | 4-Agent Pipeline |
|---|---|---|
| Maintainability | One change retrains everything | Each agent retrained independently |
| Explainability | Black-box output | SHAP drivers at every stage |
| Cross-domain context | Implicit | Explicit: Agent 2 receives Agent 1's tier as a feature |
| Failure isolation | Total failure on bad input | OOD gate, confidence gate, LLM fallbacks per node |
| Business logic | Hard to encode | Rule engines in Agents 3 & 4 are human-readable |

---

## 2. System Architecture & Full-Stack Handshake

### 2.1 High-Level Topology

```
┌───────────────────────────────────────────────────────────────────┐
│                    FRONTEND  (Next.js 14)                         │
│              Vercel  ·  https://your-app.vercel.app               │
│                         localhost:3000 (dev)                      │
│                                                                   │
│   QuoteForm → POST /api/v1/full-analysis → Storytelling Cards     │
└────────────────────────┬──────────────────────────────────────────┘
                         │  HTTP POST  (JSON)
                         │  Content-Type: application/json
                         ▼
┌───────────────────────────────────────────────────────────────────┐
│                    BACKEND  (FastAPI + Uvicorn)                    │
│              Render  ·  https://your-api.onrender.com             │
│                         localhost:8001 (dev)                      │
│                                                                   │
│  ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Agent 1 │→ │  Agent 2  │→ │ Agent 3  │→ │    Agent 4       │  │
│  │  Risk   │  │Conversion │  │ Advisor  │  │ Decision Router  │  │
│  └─────────┘  └───────────┘  └──────────┘  └──────────────────┘  │
│                                                                   │
│  ─────── LangGraph StateGraph (linear DAG) ───────────────────── │
│  [START] → node_risk → node_conversion → node_advisor →           │
│            node_router → [END]                                    │
└───────────────────────────────────────────────────────────────────┘
```

### 2.2 The Request Lifecycle

**Step 1 — Form Submission (Frontend)**  
The user clicks "Analyse Quote" in `QuoteForm.tsx`. The Next.js client-side handler fires a `fetch()` call targeting `${NEXT_PUBLIC_API_URL}/api/v1/full-analysis` with a 90-second `AbortController` timeout to accommodate Render cold-start latency.

```typescript
// frontend/app/page.tsx
const res = await fetch(`${API_BASE}/api/v1/full-analysis`, {
  method:  "POST",
  headers: { "Content-Type": "application/json" },
  body:    JSON.stringify(input),
  signal:  controller.signal,   // aborts at 90s
});
```

**Step 2 — CORS & Validation (Backend)**  
FastAPI's `CORSMiddleware` is configured to allow `http://localhost:3000` in development and any additional domains set via the `FRONTEND_ORIGINS` environment variable in production. The request body is immediately parsed against the `QuoteRequest` Pydantic model — if any field fails validation (wrong type, out-of-range, or the cross-field logic check `Driving_Exp ≤ Driver_Age − 16`), FastAPI returns `HTTP 422` before the LangGraph graph is ever invoked.

**Step 3 — LangGraph Invocation (Backend)**  
The validated quote dict is placed into the initial `AgentState` as `input_data` and passed to `_pipeline.invoke()`. The pipeline is a pre-compiled `StateGraph` singleton loaded at startup — there is zero recompilation cost per request. The invocation is wrapped in `asyncio.wait_for(..., timeout=90.0)` and run via `asyncio.to_thread()` to keep the FastAPI event loop unblocked.

**Step 4 — Response Serialisation**  
After the graph terminates, the final `AgentState` is unpacked into a `PipelineResponse` Pydantic model with typed sub-objects for each agent. A `transaction_id` (UUID4) is attached for distributed tracing. The response is returned as JSON.

**Step 5 — Progressive Reveal (Frontend)**  
While the fetch is in-flight, a `setInterval` timer fires every **1,200 ms**, advancing each agent card from `hidden → loading → revealed` sequentially. When the real response lands, the timer is cleared and all cards transition to their final data-driven state simultaneously. This is detailed in [Section 5](#5-the-storytelling-ui-approach).

### 2.3 API Endpoint Summary

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/full-analysis` | Primary endpoint — full 4-agent LangGraph pipeline |
| `POST` | `/api/v1/quote` | Identical to above — named alias |
| `POST` | `/api/process_quote` | Legacy alias retained for backwards compatibility |
| `GET`  | `/api/health` | Liveness check + artifact inventory |
| `GET`  | `/docs` | Auto-generated Swagger UI |

---

## 3. LangGraph Orchestrator — The 4-Agent DAG

### 3.1 Why LangGraph?

LangGraph provides a `StateGraph` abstraction that compiles a typed directed acyclic graph of node functions. For this pipeline, two properties are critical:

- **Typed shared state** — `AgentState` is a `TypedDict` that every node reads from and writes to. The type annotations enforce that no node can accidentally write to the wrong field or consume a field that hasn't been populated yet.
- **Compile-once, invoke-many** — the graph is compiled at module import time (`_pipeline = _build_graph()`) and reused across every HTTP request. There is no per-request graph construction overhead.

### 3.2 AgentState — The Shared Nerve System

```python
# backend/main.py
class AgentState(TypedDict, total=False):
    input_data:              Dict[str, Any]   # READ ONLY — raw validated quote
    risk_results:            Dict[str, Any]   # Written by node_risk
    conversion_results:      Dict[str, Any]   # Written by node_conversion
    advisor_pitch:           Dict[str, Any]   # Written by node_advisor
    final_decision:          Dict[str, Any]   # Written by node_router (raw)
    final_routing_decision:  str              # Written by node_router (canonical)
```

`input_data` is the only field populated before `invoke()` is called. Each subsequent node reads all upstream fields it needs and writes exactly one new field. No node ever mutates `input_data` — this is an enforced convention across all four node functions.

### 3.3 Graph Topology & Compilation

```python
# backend/main.py — _build_graph()
builder = StateGraph(AgentState)
builder.add_node("node_risk",       node_risk)
builder.add_node("node_conversion", node_conversion)
builder.add_node("node_advisor",    node_advisor)
builder.add_node("node_router",     node_router)
builder.add_edge(START,             "node_risk")
builder.add_edge("node_risk",       "node_conversion")
builder.add_edge("node_conversion", "node_advisor")
builder.add_edge("node_advisor",    "node_router")
builder.add_edge("node_router",     END)
return builder.compile()
```

The graph is strictly linear — no conditional branching, no loops. This is intentional: all routing logic lives inside `agent4.route_decision()` as business rules. The graph's job is orchestration and state propagation, not control flow.

### 3.4 Node-by-Node Data Flow

```
node_risk        reads: input_data
                 writes: risk_results
                   └─ predicted_tier, confidence, class_probabilities,
                      top_3_features (SHAP), status (OOD flag or "OK")

node_conversion  reads: input_data, risk_results["predicted_tier"]
                 writes: conversion_results
                   └─ bind_probability, sales_status,
                      distance_to_conversion, conversion_score

node_advisor     reads: input_data, conversion_results["conversion_score"]
                 writes: advisor_pitch
                   └─ premium_flag, adjustment, recommended_premium,
                      original_premium, reason (LLM-enriched or rule fallback)

node_router      reads: risk_results, conversion_results,
                        advisor_pitch, input_data
                 writes: final_decision, final_routing_decision
                   └─ decision, reason, human_required,
                      priority, action_items,
                      final_routing_decision (AUTO_APPROVE | MANUAL_REVIEW | REJECT)
```

### 3.5 Cross-Agent Context: The Key Design Choice

Agent 2 is not an independent model. It receives `predicted_tier` from Agent 1 as an explicit input feature. This means the conversion model has learned that a `High`-risk customer has a different baseline bind propensity than a `Low`-risk one — a fact that raw demographic features alone can't capture. Similarly, Agent 4 synthesises all three upstream outputs into a single routing decision, making it a true second-order reasoner over the pipeline's accumulated knowledge.

---

## 4. Agent Deep Dives

### 4.1 Agent 1 — Risk Profiler

**File:** `backend/agents/agent1_risk_profiler.py`  
**Model:** XGBoost → `CalibratedClassifierCV` (isotonic regression)  
**Output:** `predicted_tier` (Low / Medium / High), `confidence`, SHAP top-3 drivers

#### Training Pipeline

The dataset (`insurance_data.csv`, 146,259 rows) contains no pre-existing risk label. Labels are synthetically generated using an actuarial scoring formula:

| Signal | Score Contribution |
|--------|--------------------|
| Prior accident | +4 per accident |
| Prior citation | +2 per citation |
| Driving experience ≤ 3 yrs | +3 |
| Driving experience ≤ 7 yrs | +2 |
| Driving experience ≤ 15 yrs | +1 |
| Driver age < 22 | +2 |
| Driver age < 26 | +1 |
| Annual miles > 45K | +2 |
| Annual miles > 25K | +1 |
| Business vehicle usage | +1 |

Gaussian noise (σ = 2.0) is added to the raw scores before thresholding into tiers. This is critical: without noise, the XGBoost model would memorise the formula rather than learning to generalise from patterns in the features. The noise forces the model to fit calibrated probabilities over a softened label boundary.

#### Interaction Features

Three derived features capture risk-density relationships that raw inputs miss:

```
Miles_Per_Exp   = Annual_Miles / (Driving_Exp + 1)  — exposure density
Total_Incidents = Prev_Accidents + Prev_Citations    — combined incident burden
Age_Exp_Gap     = Driver_Age − Driving_Exp − 16     — delayed licensing signal
```

#### Cost-Sensitive Weighting

In insurance, a False Low (classifying a High-risk driver as Low-risk) results in an under-priced policy and future claims losses. This asymmetry is encoded directly into the model:

```
High risk class weight  = 3× baseline
Medium risk class weight = 2× baseline
Low risk class weight   = 1× baseline
```

This prevents the model from achieving high accuracy by defaulting to the majority class.

#### Calibration

Raw XGBoost softmax outputs are not well-calibrated probabilities. A `CalibratedClassifierCV` with `method='isotonic'` is fitted on a held-out 16% calibration set. The resulting probabilities satisfy the requirement that "a 70% confidence prediction should be correct ~70% of the time" — a regulatory necessity for insurance applications.

#### SHAP Explanations

`shap.TreeExplainer` is instantiated on the **base** XGBoost estimator (not the calibrated wrapper) using `feature_perturbation='interventional'` to compute causal SHAP values. The top-3 features by absolute SHAP magnitude are returned with their direction (`↑ increases risk` / `↓ decreases risk`) and magnitude label (`HIGH` / `MEDIUM` / `LOW`).

---

### 4.2 Agent 2 — Conversion Predictor

**File:** `backend/agents/agent2_conversion_predictor.py`  
**Model:** XGBoost → `CalibratedClassifierCV` (isotonic regression) + SMOTE  
**Output:** `bind_probability`, `sales_status`, `distance_to_conversion`

The target variable `Policy_Bind` has a ~22% bind rate — a severe class imbalance. SMOTE (Synthetic Minority Oversampling Technique) is applied **in the training split only** to balance the classes to a 1:1 ratio. The hold-out test and calibration sets remain unbalanced to reflect real-world distribution.

The classification threshold is not the default 0.5. It is found by maximising F1 score on the precision-recall curve, yielding a threshold of approximately **0.3370**. This lower threshold acknowledges that in a sales context, a false negative (missing a buyer) is more costly than a false positive (pursuing a non-buyer).

`sales_status` buckets are derived from the distance to this threshold:

| Status | Meaning |
|--------|---------|
| `HIGH_PROPENSITY` | Well above threshold — strong buying signal |
| `NEAR_MISS_FOR_ADVISOR` | Just below threshold — small incentive could close the deal |
| `LOW_PROB` | Far below threshold — unlikely to bind without major intervention |
| `UNCERTAIN` | Borderline — advisor judgement recommended |

---

### 4.3 Agent 3 — Premium Advisor

**File:** `backend/agents/agent3.py`  
**Logic:** Rule engine + Groq LLM (`llama-3.1-8b-instant`)  
**Activation Gate:** Only fires when `conversion_score < 40`

Agent 3 has an explicit activation gate. If Agent 2 reports a conversion score ≥ 40, no premium adjustment is generated — the customer is already likely to bind and a discount would erode margin unnecessarily.

When active, five business rules are evaluated independently. Where multiple discount rules fire, the **largest** discount wins (most customer-friendly). Coverage downgrade rules are additive on top of the winning discount:

| Rule | Condition | Action |
|------|-----------|--------|
| R1 | Premium > $800 AND salary band ≤ 1 | −15% premium |
| R2 | Premium > $700 AND salary band = 0 | −10% premium |
| R3 | Enhanced coverage AND salary band ≤ 1 | Advisory: downgrade to Balanced |
| R4 | Re-quote flag = 1 | −10% premium (price-sensitive signal) |
| R5 | Cheap vehicle AND Enhanced coverage | Advisory: downgrade to Basic |

The rule output is then passed to a Groq LLM call (Llama 3.1 8B, max_tokens=150, temperature=0.7) to generate a 2-sentence customer-facing explanation. The prompt instructs the model to be empathetic, professional, and to never mention AI or algorithms — the output reads like advice from a human insurance advisor.

---

### 4.4 Agent 4 — Decision Router

**File:** `backend/agents/agent4.py`  
**Logic:** 3-stage rule engine + Groq LLM (`llama-3.1-8b-instant`)  
**Output:** `AUTO_APPROVE` | `MANUAL_REVIEW` (via Escalate/Follow-Up) | `REJECT`

The routing logic is a strict priority ladder evaluated top-to-bottom. A case that satisfies a higher-priority condition never reaches a lower one:

**Stage 0 — Hard Rejection** *(evaluated before escalation)*  
Condition: `High risk tier AND conversion_score < 20 AND prev_accidents ≥ 2`  
Rationale: The risk-reward profile is so unfavourable that escalating to an underwriter would waste expert time. The application is automatically rejected.

**Stage 1 — Escalation to Underwriter** *(any single rule fires → escalate)*

| Rule | Trigger |
|------|---------|
| E1 | `risk_tier == High` — mandatory underwriter sign-off |
| E2 | `conversion_score < 30` — critically low buying intent |
| E3 | `prev_accidents > 0 AND conversion_score < 50` — unfavourable risk-reward |
| E4 | `prev_citations > 0` — citation pattern requires underwriter judgement regardless of tier |

**Stage 2 — Auto Approval** *(all conditions must be true)*  
`Low risk AND conversion_score ≥ 50 AND no premium flag AND no accidents`

**Stage 3 — Agent Follow-Up** *(catch-all)*  
Everything that didn't qualify for rejection, escalation, or auto-approval is assigned to an agent for human-assisted closure, with priority (`High` / `Medium` / `Low`) based on the premium flag and conversion score.

The decision label is then mapped to the canonical `final_routing_decision` string before being written to `AgentState`:

```python
_DEC_TO_CANONICAL = {
    "Auto Approve":            "AUTO_APPROVE",
    "Reject Application":      "REJECT",
    "Escalate to Underwriter": "MANUAL_REVIEW",
    "Agent Follow-Up":         "MANUAL_REVIEW",
}
```

---

## 5. The Storytelling UI Approach

### 5.1 The Problem with Standard Dashboards

A conventional approach would be to display all four agent results simultaneously once the API response arrives — four panels rendered in a single render cycle. This creates two problems for AI-powered interfaces:

1. **Cognitive overload** — the user is confronted with four panels of data, probability scores, SHAP charts, and LLM text all at once. There is no natural reading order, no sense of cause and effect, and no moment for the user to understand one result before moving to the next.
2. **Black-box perception** — when everything appears simultaneously, the AI feels like a magic oracle rather than a reasoned process. The user cannot see the *thinking* — the chain from risk → conversion → pricing → decision.

### 5.2 The Sequential Reveal Strategy

The core insight is that the four agents have a natural narrative structure: each one answers a question that leads to the next. The UI is designed to make that narrative visible and paced.

```
               t=0ms    t=1200ms   t=2400ms   t=3600ms   t=API_DONE
Agent 1 Risk:  loading  revealed   revealed   revealed   ← real data
Agent 2 Conv:  hidden   loading    revealed   revealed   ← real data
Agent 3 Adv:   hidden   hidden     loading    revealed   ← real data
Agent 4 Route: hidden   hidden     hidden     loading    ← real data
```

Each agent card transitions through three states — `hidden → loading → revealed` — on a 1,200 ms interval timer. The timer runs **independently** of the actual API call. When the real response arrives early, all in-flight timers are cleared and the cards jump to their final data-driven state. When the response arrives after all four timers have fired, the cards upgrade their placeholder content with real data invisibly.

The `AGENT_REVEAL_DELAY_MS = 1200` constant in `frontend/app/page.tsx` is the single knob that controls the entire reveal cadence.

### 5.3 State Machine Architecture

The page-level state is a discriminated union — an explicit state machine that makes illegal states unrepresentable:

```typescript
type AppState =
  | { kind: "idle" }
  | { kind: "running"; input: QuoteInput; cardStates: [CardState, CardState, CardState, CardState]; data: PipelineResponse | null; }
  | { kind: "complete"; input: QuoteInput; data: PipelineResponse; elapsed: number; }
  | { kind: "ood_error"; error: OodErrorResponse }
  | { kind: "error"; title: string; message: string };
```

The `running` state carries both the current `cardStates` array and a nullable `data` field. Before the API responds, `data` is `null` and cards render skeleton placeholders. After the response, `data` is populated and cards render real content — all within the same `running→complete` transition, with no intermediate re-renders or race conditions.

### 5.4 Framer Motion — State-Driven Animations

Every card mount and unmount is wrapped in `<AnimatePresence>` from Framer Motion. Cards enter with `{ opacity: 0, y: 20 }` and animate to `{ opacity: 1, y: 0 }` over 400ms with an `easeOut` curve. The stagger delay is implicit — because each card is revealed 1,200 ms after the previous one, the animation start times are naturally offset without explicit `staggerChildren` configuration.

The `HandoffConnector` component uses Framer Motion's `animate={{ height: active ? "100%" : "0%" }}` to grow a vertical line between agent cards as each one completes. A pulse dot animates in at the midpoint with a `scale: [0, 1.3, 1]` spring sequence, providing a visual "handshake" that reinforces the agent-to-agent flow.

```typescript
// frontend/components/HandoffConnector.tsx
<motion.div
  className={`absolute inset-x-0 top-0 rounded-full ${color}`}
  initial={{ height: "0%" }}
  animate={{ height: active ? "100%" : "0%" }}
  transition={{ duration: 0.5, ease: "easeOut" }}
/>
```

### 5.5 Skeleton Loaders — Perceived Performance

While an agent card is in its `loading` state, `SkeletonLoader` renders a pulsing placeholder that mirrors the exact shape of the real card — a header bar, a title block, several content lines of decreasing width, and a row of pill badges. Each line animates in with a staggered opacity transition (8ms per line) to create a subtle "painting" effect.

The use of agent-specific accent colors (emerald for Risk, sky for Conversion, violet for Advisor, amber for Router) on the skeleton itself means the placeholder is already brand-correct before the data arrives. The user perceives loading time as intentional pacing rather than system latency.

### 5.6 The Collapsed Header

Once the form is submitted, `QuoteForm` is replaced by `CollapsedHeader` — a slim summary bar showing the key input parameters. This achieves two things: it preserves the user's context (they can see what they submitted without scrolling up) and it clears vertical space for the four agent cards to expand downward. The form can be re-expanded via an "Edit" button that resets the state machine to `idle`.

---

## 6. Core Technologies & Libraries

### 6.1 Backend

| Technology | Version | Role |
|---|---|---|
| **Python** | 3.11+ | Core runtime |
| **FastAPI** | latest | ASGI web framework with automatic OpenAPI schema generation |
| **Uvicorn** | latest | High-performance ASGI server (serves FastAPI) |
| **LangGraph** | ≥ 0.2 | `StateGraph` DAG — compiles the 4-node pipeline at startup |
| **XGBoost** | 2.1.4 | Gradient-boosted trees for Agents 1 & 2 |
| **scikit-learn** | 1.6.1 | `CalibratedClassifierCV`, `IsolationForest`, `OneHotEncoder` |
| **SHAP** | 0.49.1 | `TreeExplainer` — interventional SHAP values for Agent 1 |
| **imbalanced-learn** | latest | SMOTE oversampling for Agent 2 training |
| **Groq** | latest | LLM API client — Llama 3.1 8B inference for Agents 3 & 4 |
| **Pydantic** | ≥ 2.0 | Input validation (`QuoteRequest`) and response schema enforcement |
| **joblib** | latest | Compressed `.pkl` artifact serialisation |
| **python-dotenv** | latest | `GROQ_API_KEY` loading from `backend/.env` |

**FastAPI Async Architecture**  
The FastAPI app is fully asynchronous. Route handlers are declared `async`, and the synchronous LangGraph `_pipeline.invoke()` call is dispatched via `asyncio.to_thread()` to avoid blocking the event loop. This allows the server to handle other requests (e.g., `/api/health`) during graph execution.

**LangGraph State Machines**  
LangGraph's `StateGraph` provides a compile-time check that all node connections are valid — unconnected nodes and missing edges are caught at startup, not at request time. The `recursion_limit=50` invocation parameter is a safety cap against any future accidental cycles (far above what the current 4-node linear graph needs).

**Groq LLM Inference**  
Groq's inference API provides sub-second LLM response times by running Llama 3.1 8B on dedicated LPU hardware. The client is configured with `timeout=5.0, max_retries=0` — fail-fast settings that prevent a slow LLM call from stalling the pipeline. If a call fails, the rule-generated fallback string is returned transparently.

**XGBoost**  
XGBoost was chosen over alternatives (Random Forest, LightGBM) for three reasons: native SHAP support via `TreeExplainer` for interpretable explanations, `scale_pos_weight` / `sample_weight` support for cost-sensitive training, and mature `scikit-learn` compatibility for `CalibratedClassifierCV` wrapping.

### 6.2 Frontend

| Technology | Version | Role |
|---|---|---|
| **Next.js** | 14.2 | App Router, SSR/SSG, production build optimisation |
| **React** | 18.3 | Component library |
| **TypeScript** | 5 | Type safety — enforced against `api-contract.ts` |
| **Tailwind CSS** | 3 | Utility-first styling with a dark slate/zinc theme |
| **Framer Motion** | 11.2 | State-driven animations — card reveal, accordion, connectors |
| **Lucide React** | 0.378 | Icon library: `Shield` (Risk), `BarChart2` (Conversion), `MessageSquare` (Advisor), `Gavel` (Router) |

**Next.js App Router**  
The App Router (`app/` directory) is used throughout. The dashboard page is a single `"use client"` component — all rendering is client-side after the initial HTML shell loads, which is appropriate for a form-driven interactive application with no SEO requirements.

**Contract-First API Types**  
`frontend/lib/api-contract.ts` is the single source of truth for all TypeScript types. Every interface (`RiskAssessment`, `ConversionMetrics`, `AdvisorStrategy`, `FinalRouting`, `PipelineResponse`) mirrors the corresponding Pydantic model in `backend/main.py` exactly. This ensures that any backend schema change that is not reflected in the frontend contract is caught as a TypeScript compile error.

**Tailwind CSS Theming**  
Each agent has a dedicated accent color applied consistently across its card border, header text, skeleton shimmer, SHAP bar, and handoff connector:

| Agent | Accent |
|-------|--------|
| Agent 1 — Risk | `emerald-500` |
| Agent 2 — Conversion | `sky-500` |
| Agent 3 — Advisor | `violet-500` |
| Agent 4 — Router | `amber-500` |

These colors provide an immediate visual identity for each agent that persists from the skeleton loader through to the final verdict badge.

---

## 7. Edge Cases & Resilience Handling

### 7.1 OOD (Out-of-Distribution) Anomaly Detection

**The Problem**  
ML models return confident predictions on inputs that look nothing like their training data. A quote with `Driver_Age = -5` or `Annual_Miles = 9,999,999` will pass basic Pydantic range validation if the bounds are set generously, but will produce a statistically meaningless probability output. The model has no way to signal that the input is corrupt.

**The Solution: IsolationForest Gate**  
An `IsolationForest` detector (200 trees) is trained on the Agent 1 training data's raw feature space. Rather than using `contamination`-based `predict()` — which would incorrectly flag rare-but-valid profiles (e.g., a young driver with an accident) — the detector uses `score_samples()` and a hard threshold set at the **0.01th percentile** of training anomaly scores. Only inputs whose anomaly score is more extreme than 99.99% of all real training quotes are flagged.

```python
# backend/agents/agent1_risk_profiler.py
OOD_SCORE_PERCENTILE = 0.01  # flag if below 0.01th percentile of training distribution
OOD_FLAG             = "ACTION_REQUIRED: DATA_ANOMALY"
```

**Why Raw Features, Not Interaction Features?**  
The OOD detector is trained on the 6 raw features only, explicitly excluding the 3 interaction features (`Miles_Per_Exp`, `Total_Incidents`, `Age_Exp_Gap`). If a corrupt record has `Annual_Miles = 9,999,999`, the derived `Miles_Per_Exp = 9,999,999 / 1 = 9,999,999` would saturate the IsolationForest's path-length metric and ironically make the record appear "average" (because tree depth caps out at the extreme value). The raw `Annual_Miles = 9,999,999` is cleanly extreme and correctly triggers the anomaly detector.

**Deterministic Physics Checks**  
On top of the statistical detector, a deterministic layer of "physics" rules catches logically impossible values that the model may not flag:

- `Driver_Age ≥ 16` (minimum legal driving age)
- `Annual_Miles ≥ 0` (mileage cannot be negative)
- `Driving_Exp ≤ Driver_Age − 16` (cross-field constraint enforced in Pydantic and again in the predictor)

**Frontend Handling (HTTP 422)**  
When the OOD gate fires, the backend returns `HTTP 422` with `status: "ACTION_REQUIRED: DATA_ANOMALY_ESCALATE"`. The frontend detects this status string and renders the `ErrorBanner` component with a specific OOD message, rather than a generic validation error. The app state transitions to `{ kind: "ood_error" }`, which is a distinct branch of the state machine.

```typescript
// frontend/app/page.tsx
if (res.status === 422) {
  const json = await res.json();
  if (json.status && String(json.status).includes("DATA_ANOMALY")) {
    setState({ kind: "ood_error", error: json as OodErrorResponse });
    return;
  }
}
```

### 7.2 LLM Fallback Logic

Both Agent 3 and Agent 4 make Groq API calls to generate human-readable explanations. These calls can fail for several reasons: missing API key, invalid key (HTTP 400), network timeout, rate limiting, or the `groq` package not being installed.

The fallback pattern is identical in both agents:

```python
# backend/agents/agent3.py  &  agent4.py
try:
    client = _Groq(api_key=api_key, timeout=5.0, max_retries=0)
    # ... build prompt, call API ...
    return response.choices[0].message.content.strip()
except Exception as exc:
    log.warning("Agent X │ Groq API call failed — %s: %s", type(exc).__name__, exc)
    return _LLM_UI_FALLBACK
```

`_LLM_UI_FALLBACK` is a pre-written professional fallback string defined as a module-level constant:

- **Agent 3 fallback:** `"Premium is competitive for this risk profile; no manual adjustment needed."`
- **Agent 4 fallback:** `"Referred for underwriter review based on standard risk-routing rules."`

These are not error messages — they are plausible, professional sentences that fit naturally into the UI's storytelling cards. A user who receives a fallback response will see a coherent, well-formatted card rather than an empty field or a stack trace.

The `timeout=5.0, max_retries=0` client configuration is deliberate: fail fast and return the fallback rather than waiting for a retry that compounds latency. The LLM enrichment is additive — the pipeline's correctness does not depend on it.

### 7.3 Confidence Gate

Agent 1's XGBoost model outputs a class probability for the `predicted_tier`. If the highest class probability is below **60%** (`CONFIDENCE_GATE = 0.60`), the pipeline still returns a prediction but marks the `status` as `LOW_CONFIDENCE_ESCALATE` and populates `escalation_reason` with a human-readable message:

```python
if confidence < CONFIDENCE_GATE:
    pipeline_status   = LOW_CONF_STATUS
    escalation_reason = (
        f"Model confidence {confidence:.1%} is below the "
        f"{CONFIDENCE_GATE:.0%} threshold. Route to human underwriter."
    )
```

This prevents borderline predictions from being automatically routed without a human check.

### 7.4 Manual Review — "Near Miss" Amber Routing

The `NEAR_MISS_FOR_ADVISOR` sales status in Agent 2 and the `Agent Follow-Up` decision in Agent 4 together define the "amber" zone — cases that are neither clean auto-approvals nor clear rejections. These cases require human judgment, and the frontend renders them with the amber (`amber-500`) accent color and a `MANUAL_REVIEW` verdict badge.

Agent 4's follow-up path includes a `priority` field (`High` / `Medium` / `Low`) that the handling team can use for queue prioritisation. The `action_items` list is a structured set of next-step recommendations, also generated by the routing logic.

### 7.5 HTTP Error Handling on the Frontend

The frontend handles every HTTP status code explicitly:

| Status | Frontend behaviour |
|--------|--------------------|
| `200` | Parse `PipelineResponse`, transition to `complete` |
| `422` (OOD) | Detect `DATA_ANOMALY` in body, transition to `ood_error` |
| `422` (validation) | Parse `detail` array, transition to `error` with field messages |
| `503` | Service unavailable — artifacts not loaded |
| `504` / `502` | Detect `TIMEOUT` status in body — display cold-start message |
| Abort (90s) | `DOMException.AbortError` — display timeout retry message |
| Network failure | Generic connection error |

The Render cold-start case (`502`/`504` with a non-`TIMEOUT` body) receives a specifically worded message: *"The server is starting up after a period of inactivity. This takes up to 60 seconds on the free tier — please wait a moment and try again."* This sets accurate expectations rather than displaying a confusing generic error.

---

## 8. Deployment Strategy

### 8.1 Architecture: Vercel (Frontend) + Render (Backend)

The frontend and backend are deployed as two independent services. This split is deliberate:

- **Vercel** is optimised for Next.js deployments — zero-config build, global CDN, automatic HTTPS, and instant rollbacks. The frontend has no ML dependencies and no `models/` directory; it is a pure static + client-side application.
- **Render** runs the Docker-containerised FastAPI backend with the full Python ML stack, including the trained `.pkl` artifacts. Render's Docker support allows the exact environment from local development to be reproduced in production.

### 8.2 Render Configuration

The backend deployment is fully specified in `render.yaml`:

```yaml
# render.yaml
services:
  - type: web
    name: quote-agents-backend
    runtime: docker
    dockerfilePath: ./backend/Dockerfile
    dockerContext: ./backend
    branch: main
    plan: free
    healthCheckPath: /api/health
    envVars:
      - key: GROQ_API_KEY
        sync: false            # entered securely on first deploy
      - key: FRONTEND_ORIGINS
        sync: false            # set to Vercel URL after frontend deploys
      - key: PORT
        value: 8001
```

The `healthCheckPath: /api/health` tells Render to poll the health endpoint before marking the deployment as live, ensuring traffic is only routed to a fully-initialised instance where all ML artifacts are loaded.

### 8.3 Cold Start Handling

Render's free tier spins down instances after 15 minutes of inactivity. When a new request arrives after a spin-down, the container must restart, deserialise the ML `.pkl` artifacts from disk, compile the LangGraph graph, and warm the Python interpreter — a process that takes 30–60 seconds.

This latency is handled at two levels:

**Backend — 90-second pipeline timeout:**  
The `asyncio.wait_for(..., timeout=90.0)` wrapper ensures the server returns a clean `504` JSON response rather than hanging indefinitely if a cold start coincides with a compute-heavy first request.

**Frontend — Polling on cold start:**  
The `fetch()` call carries a 90-second `AbortController` signal. Before the user submits their first quote, a lightweight ping to `/api/health` can be used to detect a sleeping instance and display a warm-up indicator. The client-side error handling for `502`/`504` responses provides a user-friendly "server is waking up" message with instructions to wait and retry.

**Production recommendation:** For a production deployment that cannot tolerate cold starts, Render's paid "Starter" tier keeps instances warm permanently and should be used alongside the polling strategy.

### 8.4 CORS Configuration

The backend's CORS policy is configured at startup and reads additional allowed origins from the `FRONTEND_ORIGINS` environment variable:

```python
# backend/main.py
_extra_origins = [
    o.strip() for o in os.environ.get("FRONTEND_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        *_extra_origins,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

After deploying the frontend to Vercel, set `FRONTEND_ORIGINS=https://your-app.vercel.app` in the Render environment variables. Multiple origins can be comma-separated.

### 8.5 Docker Build

```dockerfile
# backend/Dockerfile (multi-stage)
# Stage 1 — Python dependencies
# Stage 2 — Copy application code + models/
# HEALTHCHECK — polls /api/health every 30s
```

The `models/` directory (containing the trained `.pkl` artifacts) must either be:
1. **Baked into the Docker image** at build time (simplest — rebuild image when models are retrained), or
2. **Mounted as a volume** from persistent storage (allows model updates without image rebuilds).

The current `Dockerfile` uses option 1. The `.dockerignore` file excludes `.venv/`, `__pycache__/`, and raw data files to keep image size minimal.

---

## 9. API Contract Reference

### 9.1 Request Schema

```typescript
// POST /api/v1/full-analysis
interface QuoteInput {
  // Required — core risk features
  Driver_Age:        number;   // 16–100
  Driving_Exp:       number;   // 0–84  (≤ Driver_Age − 16)
  Prev_Accidents:    number;   // 0–20
  Prev_Citations:    number;   // 0–20
  Annual_Miles:      number;   // 0–200,000
  Veh_Usage:         "Business" | "Commute" | "Pleasure";

  // Optional — Agent 3 pricing inputs
  Quoted_Premium?:    number;  // current quoted premium ($)
  Sal_Range?:         number;  // salary band 0–4
  Coverage?:          number;  // coverage level 0=Basic, 1=Balanced, 2=Enhanced
  Vehicl_Cost_Range?: number;  // vehicle cost band 0–4
  Re_Quote?:          number;  // 1 if re-quoting, 0 otherwise
}
```

### 9.2 Response Schema

```typescript
interface PipelineResponse {
  transaction_id:         string;           // UUID4
  status:                 string;           // "OK" | "LOW_CONFIDENCE_ESCALATE"
  final_routing_decision: RoutingDecision;  // "AUTO_APPROVE" | "MANUAL_REVIEW" | "REJECT"
  escalation_reason:      string | null;    // populated on LOW_CONFIDENCE_ESCALATE
  risk_assessment:        RiskAssessment;
  conversion_metrics:     ConversionMetrics;
  advisor_strategy:       AdvisorStrategy;
  final_routing:          FinalRouting;
}

interface RiskAssessment {
  predicted_tier:      "Low" | "Medium" | "High";
  confidence_score:    number;                      // 0.0–1.0
  ood_flag:            string;                      // "OK"
  class_probabilities: Record<string, number>;      // { "Low": 0.6, "Medium": 0.3, "High": 0.1 }
  top_shap_drivers:    ShapDriver[];                // top 3 by absolute value
}

interface ConversionMetrics {
  bind_probability:       number | null;   // 0.0–1.0
  sales_status:           SalesStatus | null;
  distance_to_conversion: number | null;  // ≥ 0
}

interface AdvisorStrategy {
  premium_flag:            boolean;
  suggested_discount_pct:  string | null;  // "-15%" | "none"
  recommended_premium:     number | null;
  original_premium:        number | null;
  customer_facing_message: string | null;  // LLM-enriched reason
}

interface FinalRouting {
  decision:               AgentDecision | null;
  reason:                 string | null;    // LLM-enriched 2-sentence justification
  human_required:         boolean;
  priority:               "High" | "Medium" | "Low" | null;
  action_items:           string[];
  final_routing_decision: RoutingDecision | null;
}
```

### 9.3 Error Responses

```typescript
// HTTP 422 — OOD Anomaly
interface OodErrorResponse {
  transaction_id: string;
  status:         "ACTION_REQUIRED: DATA_ANOMALY_ESCALATE";
  message:        string;
  input:          Record<string, unknown>;
}

// HTTP 504 — Pipeline Timeout
{
  "transaction_id": "<uuid>",
  "status": "TIMEOUT",
  "detail": "Pipeline took longer than 90 seconds. Please try again."
}

// HTTP 503 — Artifacts not loaded
{
  "status": "ERROR",
  "detail": "Agent 1 not loaded. Run agent1_risk_profiler.py first."
}
```

---

## 10. Local Development Setup

### 10.1 Prerequisites

- Python 3.11+
- Node.js 18+
- A Groq API key (free tier available at [console.groq.com](https://console.groq.com))

### 10.2 Backend

```bash
# 1. Create and activate virtual environment
cd backend
python -m venv ../.venv
source ../.venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here

# 4. Train Agent 1 and generate all ML artifacts
python agents/agent1_risk_profiler.py

# 5. Train Agent 2
python agents/agent2_conversion_predictor.py

# 6. Start the FastAPI server
uvicorn main:app --reload --port 8001
```

The server will be available at `http://localhost:8001`. Visit `http://localhost:8001/docs` for the auto-generated Swagger UI.

### 10.3 Frontend

```bash
# In a separate terminal
cd frontend

# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.local.example .env.local
# .env.local already points to http://localhost:8001 by default

# 3. Start the development server
npm run dev
```

The dashboard will be available at `http://localhost:3000`.

### 10.4 Running End-to-End Tests

```bash
cd backend
python test_pipeline.py
```

The test suite runs four profiles through the full pipeline:
1. **Safe driver** — expects `LOW` risk, `AUTO_APPROVE`
2. **High-risk driver** — expects `HIGH` risk, `MANUAL_REVIEW`
3. **Near-miss driver** — expects `MEDIUM` risk, `MANUAL_REVIEW` with premium advice
4. **OOD corrupt profile** — expects `DATA_ANOMALY` flag, no prediction

---

## 11. Environment Variables

### Backend (`backend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM calls in Agents 3 & 4 |
| `FRONTEND_ORIGINS` | Production only | Comma-separated allowed CORS origins (e.g. `https://your-app.vercel.app`) |
| `PORT` | No | Server port (default: `8001`) |

### Frontend (`frontend/.env.local`)

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Yes | Backend base URL (default: `http://localhost:8001`) |

---

*Documentation version: 1.0 — March 2026*  
*Pipeline version: 2.0.0*
