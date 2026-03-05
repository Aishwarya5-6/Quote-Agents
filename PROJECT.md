# InsurTech AI Pipeline — Complete Project Documentation

### 4-Agent Auto Insurance Quote Engine · Built from Scratch

> A production-grade multi-agent AI system that evaluates auto insurance quotes
> through **four specialised agents** orchestrated by **LangGraph**, served via
> **FastAPI**, and visualised through a **Next.js** storytelling dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture at a Glance](#2-architecture-at-a-glance)
3. [Project Structure](#3-project-structure)
4. [Tech Stack](#4-tech-stack)
5. [How We Built This — Step by Step](#5-how-we-built-this--step-by-step)
6. [The Dataset](#6-the-dataset)
7. [Agent 1 — Risk Profiler (Deep Dive)](#7-agent-1--risk-profiler-deep-dive)
8. [Agent 2 — Conversion Predictor (Deep Dive)](#8-agent-2--conversion-predictor-deep-dive)
9. [Agent 3 — Premium Advisor (Deep Dive)](#9-agent-3--premium-advisor-deep-dive)
10. [Agent 4 — Decision Router (Deep Dive)](#10-agent-4--decision-router-deep-dive)
11. [LangGraph Orchestration — main.py](#11-langgraph-orchestration--mainpy)
12. [API Contract & Endpoints](#12-api-contract--endpoints)
13. [Frontend — Storytelling Dashboard](#13-frontend--storytelling-dashboard)
14. [End-to-End Pipeline Flow](#14-end-to-end-pipeline-flow)
15. [Getting Started — Local Development](#15-getting-started--local-development)
16. [Docker Deployment](#16-docker-deployment)
17. [Environment Variables](#17-environment-variables)
18. [Testing](#18-testing)
19. [Key Design Decisions](#19-key-design-decisions)
20. [Troubleshooting](#20-troubleshooting)

---

## 1. Project Overview

### What Does This System Do?

An insurance customer fills in a **quote form** (age, driving experience, accidents, citations, annual miles, vehicle usage, premium, etc.). The system runs the quote through **four AI agents in sequence**, and within seconds delivers:

| Step | Agent | What It Tells You |
|------|-------|--------------------|
| 1 | **Risk Profiler** | Is this driver Low / Medium / High risk? Why? (SHAP explanations) |
| 2 | **Conversion Predictor** | Will this customer actually buy the policy? (bind probability) |
| 3 | **Premium Advisor** | Should we adjust the premium to increase conversion? How much? |
| 4 | **Decision Router** | Final routing: Auto-Approve, Manual Review, or Reject? |

### Why Multi-Agent?

Traditional insurance quoting is a single monolithic model. Our approach **decomposes** the problem into four specialised agents, each an expert in one domain:

- **Separation of concerns** — each agent can be improved, retrained, or replaced independently
- **Cross-agent context** — Agent 2 uses Agent 1's risk tier; Agent 4 uses all three upstream outputs
- **Safety layers** — OOD detection, confidence gates, physics checks, and LLM fallbacks ensure no agent can silently produce a bad result
- **Transparency** — SHAP explanations, rule audit trails, and LLM-generated reasons at every stage

---

## 2. Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js 14)                        │
│   Quote Form → Storytelling Cards → Progressive Reveal → Details    │
│                        localhost:3000                                │
└─────────────────────┬───────────────────────────────────┬───────────┘
                      │  POST /api/v1/full-analysis       │
                      ▼                                   ▲  JSON
┌─────────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI + LangGraph)                    │
│                        localhost:8001                                │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────┐ │
│  │  Agent 1  │──▶│   Agent 2    │──▶│   Agent 3   │──▶│ Agent 4  │ │
│  │   Risk    │   │  Conversion  │   │   Premium   │   │ Decision │ │
│  │ Profiler  │   │  Predictor   │   │   Advisor   │   │  Router  │ │
│  └──────────┘   └──────────────┘   └─────────────┘   └──────────┘ │
│       │                │                  │                │        │
│    XGBoost          XGBoost           Rules +          Rules +     │
│    + SHAP           + SMOTE          Groq LLM         Groq LLM    │
│    + OOD            + Calib.                                       │
│                                                                     │
│  ─────────── LangGraph StateGraph (DAG) ──────────────────────────  │
│  [START] → node_risk → node_conversion → node_advisor → node_router → [END]
│                                                                     │
│  AgentState = { input_data, risk_results, conversion_results,       │
│                 advisor_pitch, final_decision, final_routing }      │
└─────────────────────────────────────────────────────────────────────┘
          │                                              │
          ▼                                              ▼
    backend/models/                              backend/data/raw/
    (*.pkl artifacts)                         (insurance_data.csv)
```

---

## 3. Project Structure

```
Quote-Agents/
├── backend/                          ← Python ML pipeline + FastAPI
│   ├── agents/
│   │   ├── agent1_risk_profiler.py   ← 1,453 lines — XGBoost + SHAP + OOD
│   │   ├── agent2_conversion_predictor.py ← 1,153 lines — SMOTE + Calibration
│   │   ├── agent3.py                 ← 385 lines — Rule engine + Groq LLM
│   │   └── agent4.py                 ← 415 lines — Routing rules + Groq LLM
│   ├── data/
│   │   ├── raw/
│   │   │   └── insurance_data.csv    ← 146,259 real insurance quotes
│   │   └── processed/               ← auto-generated by Agent 1 training
│   ├── models/                       ← trained .pkl artifacts (gitignored)
│   │   ├── calibrated_risk_profiler.pkl
│   │   ├── shap_explainer.pkl
│   │   ├── ohe_encoder.pkl
│   │   ├── label_encoder.pkl
│   │   ├── feature_names.pkl
│   │   ├── ood_detector.pkl
│   │   ├── ood_threshold.pkl
│   │   ├── training_stats.pkl
│   │   ├── agent2_conversion_model_v2.pkl
│   │   ├── agent2_feature_names_v2.pkl
│   │   ├── agent2_ohe_encoder_v2.pkl
│   │   ├── agent2_shap_explainer_v2.pkl
│   │   ├── agent2_threshold_v2.pkl
│   │   ├── agent2_tier_encoder_v2.pkl
│   │   ├── agent2_metadata_v2.json
│   │   └── manifest.json
│   ├── main.py                       ← 716 lines — LangGraph DAG + FastAPI
│   ├── app.py                        ← Legacy single-agent API
│   ├── test_pipeline.py              ← End-to-end pipeline tests
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   └── .env.example
│
├── frontend/                         ← Next.js 14 + React 18 + Tailwind
│   ├── app/
│   │   ├── page.tsx                  ← 425 lines — Storytelling state machine
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── QuoteForm.tsx             ← 11-field insurance quote form
│   │   ├── StorytellingCard.tsx      ← Generic agent card with accordion
│   │   ├── CollapsedHeader.tsx       ← Slim summary after form submission
│   │   ├── SkeletonLoader.tsx        ← Pulsing placeholder during loading
│   │   ├── VerdictBadge.tsx          ← Color-coded verdict pill
│   │   ├── HandoffConnector.tsx      ← Animated vertical connector
│   │   ├── RiskDetails.tsx           ← Agent 1 accordion body
│   │   ├── ConversionDetails.tsx     ← Agent 2 accordion body
│   │   ├── AdvisorDetails.tsx        ← Agent 3 accordion body
│   │   ├── DecisionDetails.tsx       ← Agent 4 accordion body
│   │   └── ErrorBanner.tsx           ← OOD / validation error display
│   ├── lib/
│   │   └── api-contract.ts           ← TypeScript types matching Pydantic
│   ├── package.json
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── FRONTEND.md
│   └── .env.local.example
│
├── .env                              ← Secrets (gitignored)
├── .gitignore
├── .venv/                            ← Python virtual environment
└── README.md
```

---

## 4. Tech Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Core language |
| **FastAPI** | latest | REST API framework with auto-generated Swagger docs |
| **LangGraph** | ≥ 0.2 | StateGraph DAG orchestration for the 4-agent pipeline |
| **XGBoost** | 2.1.4 | Gradient-boosted trees for risk & conversion models |
| **scikit-learn** | 1.6.1 | CalibratedClassifierCV, IsolationForest, preprocessing |
| **SHAP** | 0.49.1 | TreeExplainer for interpretable feature-importance |
| **imbalanced-learn** | latest | SMOTE oversampling for Agent 2 |
| **Groq** | latest | LLM API client (Llama 3 8B) for Agents 3 & 4 |
| **Pydantic** | ≥ 2.0 | Input validation & response schema enforcement |
| **uvicorn** | latest | ASGI server |
| **joblib** | latest | Model serialisation (compressed .pkl) |

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Next.js** | 14.2 | App Router, SSR/SSG, production build |
| **React** | 18.3 | Component library |
| **TypeScript** | 5 | Type safety |
| **Tailwind CSS** | 3 | Utility-first styling, dark theme |
| **Framer Motion** | 11.2 | Animations: card reveal, accordion, transitions |
| **Lucide React** | 0.378 | Icon library (Shield, BarChart2, Gavel, etc.) |

---

## 5. How We Built This — Step by Step

This section walks through the **development journey** from zero to production.

### Phase 1: Data Exploration & Understanding

1. **Acquired dataset** — `insurance_data.csv` with 146,259 real auto insurance quotes from a quoting platform
2. **Explored columns** — 25 columns including demographics (age, gender, marital status, education), driving history (accidents, citations, experience), policy details (premium, coverage, vehicle cost), and the target `Policy_Bind` (Yes/No)
3. **Identified key challenge** — the `Policy_Bind` target is imbalanced (~22% bind rate), and there's no pre-existing "risk tier" label, so we need to generate one

### Phase 2: Agent 1 — Risk Profiler (the Foundation)

4. **Designed synthetic risk labels** — created an actuarial scoring formula based on insurance domain knowledge:
   - +4 per prior accident (strongest predictor of future claims)
   - +2 per prior citation
   - +3/2/1 for low driving experience (≤3 / ≤7 / ≤15 years)
   - +2/1 for young drivers (<22 / <26 years)
   - +2/1 for high mileage (>45K / >25K miles/year)
   - +1 for business vehicle usage
5. **Added Gaussian noise** (σ=2.0) to prevent the model from memorising the formula — forces it to learn calibrated probabilities
6. **Engineered interaction features**:
   - `Miles_Per_Exp` = Annual_Miles / (Driving_Exp + 1) — exposure density
   - `Total_Incidents` = Prev_Accidents + Prev_Citations — combined burden
   - `Age_Exp_Gap` = Driver_Age − Driving_Exp − 16 — delayed licensing signal
7. **Trained XGBoost** with RandomizedSearchCV (50 iterations × 5 folds), scoring by `neg_log_loss` to penalise confidently wrong predictions
8. **Applied cost-sensitive weights** — High risk 3×, Medium 2×, Low 1× to prevent the model from ignoring minority classes
9. **Calibrated probabilities** — CalibratedClassifierCV (isotonic regression) on a held-out 16% calibration set
10. **Built SHAP TreeExplainer** — interventional SHAP values on the base XGBoost for regulatory-grade explanations
11. **Trained IsolationForest OOD detector** — flags corrupt/impossible quotes before they reach the model
12. **Added deterministic physics checks** — hard rules like "Driver_Age must be ≥ 16", "Annual_Miles cannot be negative"
13. **Red-teamed with adversarial examples** — 5 deliberately impossible profiles to verify the safety gate

### Phase 3: Agent 2 — Conversion Predictor

14. **Generated bind labels** — used risk tier from Agent 1 to create realistic bind probabilities (Low→55%, Medium→28%, High→12% base rates, blurred with noise)
15. **Applied SMOTE** — oversampled the minority Bind class to 1:1 ratio in training only
16. **Trained second XGBoost** + CalibratedClassifierCV for calibrated bind probabilities
17. **Found optimal F1 threshold** via precision-recall curve analysis (0.3370)
18. **Created sales status buckets** — HIGH_PROPENSITY / NEAR_MISS / LOW_PROB / UNCERTAIN based on distance to threshold
19. **Implemented cross-agent context** — Agent 2 receives `Risk_Tier` from Agent 1 as an input feature

### Phase 4: Agent 3 — Premium Advisor

20. **Designed activation gate** — only activates when Agent 2's conversion_score < 40 (no adjustment needed for likely buyers)
21. **Built rule engine** with 5 business rules:
    - Rule 1: Premium >$800 + low salary → 15% reduction
    - Rule 2: Premium >$700 + very low salary → 10% reduction
    - Rule 3: Enhanced coverage + low salary → coverage downgrade advisory
    - Rule 4: Re-quote detected → 10% reduction (price-sensitive signal)
    - Rule 5: Cheap vehicle + Enhanced coverage → Basic coverage advisory
22. **Integrated Groq LLM** (Llama 3 8B) to generate customer-facing 2-sentence explanations from the rule output
23. **Built silent fallback** — any LLM failure (timeout, bad key, import error) falls back to the rule-generated reason string

### Phase 5: Agent 4 — Decision Router

24. **Designed 3-stage routing logic**:
    - Stage 1 — **Escalation rules** (evaluated first, strict priority): High risk tier, critically low conversion (<30), accident + low conversion, citation + medium/high risk
    - Stage 2 — **Auto-approval** (only if no escalation): Low risk + score ≥60 + no premium flag + clean record
    - Stage 3 — **Agent follow-up** (catch-all): everything else, with priority based on premium flag
25. **Integrated Groq LLM** for underwriter-facing 2-sentence justifications with the same silent fallback pattern

### Phase 6: LangGraph Orchestration

26. **Defined `AgentState` TypedDict** — shared mutable state with fields for each agent's output
27. **Created 4 node functions** — each reads upstream state and writes its own field
28. **Compiled StateGraph** — `START → node_risk → node_conversion → node_advisor → node_router → END`
29. **Built FastAPI app** with CORS, health checks, input validation (Pydantic), and response schemas
30. **Added safety layers** — 30-second timeout, OOD gate, confidence gate (60% threshold), global exception handler
31. **Loaded ML singletons at startup** — zero reload cost per request

### Phase 7: Frontend — Storytelling Dashboard

32. **Created Next.js 14 app** with App Router, TypeScript, Tailwind CSS dark theme
33. **Built QuoteForm** — 11-field form with validation, defaults, and number/select inputs
34. **Designed contract-first API types** — `api-contract.ts` mirrors the Pydantic schemas exactly
35. **Built Storytelling UI** — sequential card reveal with state machine:
    - `idle → running → complete` (or `ood_error` / `error`)
    - Progressive reveal: each agent card transitions `hidden → loading → revealed` on a 1.2s timer
    - Collapsed header replaces the form after submission
    - Handoff connectors animate between agent cards
    - Accordion details panels with verdict badges
36. **Created per-agent detail components** — each with custom visualisations:
    - RiskDetails: SVG risk gauge + SHAP waterfall bars
    - ConversionDetails: bind probability tiles + threshold bar
    - AdvisorDetails: 3-column premium comparison + LLM chat bubble
    - DecisionDetails: decision banner + terminal log + action items

### Phase 8: Testing, Polish & Deployment

37. **Wrote end-to-end tests** — `test_pipeline.py` with 4 profiles covering safe/high-risk/near-miss/OOD
38. **Created Docker deployment** — multi-stage Dockerfile with healthcheck
39. **Set up environment configs** — `.env.example`, `.env.local.example`
40. **Fixed all type errors** — Pyright/Pylance clean across all Python and TypeScript files
41. **Created documentation** — `FRONTEND.md`, `README.md`, and this `PROJECT.md`

---

## 6. The Dataset

**File:** `backend/data/raw/insurance_data.csv`
**Records:** 146,259 real auto insurance quotes

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `Driver_Age` | int | Driver's age (16–100) |
| `Driving_Exp` | int | Years of driving experience |
| `Prev_Accidents` | int | Number of prior at-fault accidents |
| `Prev_Citations` | int | Number of prior traffic citations |
| `Annual_Miles_Range` | string | Mileage band (e.g., "> 15 K & <= 25 K") — converted to numeric midpoint |
| `Veh_Usage` | string | Business / Commute / Pleasure |
| `Quoted_Premium` | float | Current quoted premium in dollars |
| `Sal_Range` | string | Salary band (encoded 0–4) |
| `Coverage` | string | Basic / Balanced / Enhanced (encoded 0–2) |
| `Vehicl_Cost_Range` | string | Vehicle cost band (encoded 0–4) |
| `Re_Quote` | string | Whether this is a re-quote (Yes/No → 0/1) |
| `Policy_Bind` | string | **Target** — did the customer buy? (Yes/No) |

### Data Pipeline

```
insurance_data.csv (146K rows, 25 cols)
        │
        ▼  Agent 1: load_and_prepare_data()
  • Convert Annual_Miles_Range → numeric midpoint
  • Impute nulls (median for numeric, mode for categorical)
        │
        ▼  Agent 1: generate_risk_labels()
  • Compute actuarial score per row
  • Add Gaussian noise (σ=2.0)
  • Bucket into Low / Medium / High
        │
        ▼  Agent 1: add_interaction_features()
  • Miles_Per_Exp, Total_Incidents, Age_Exp_Gap
        │
        ▼  Agent 1: encode_features()
  • OneHotEncode Veh_Usage → 3 binary columns
  • LabelEncode Risk_Tier → 0/1/2
  • Result: 11-feature matrix (5 numeric + 3 interaction + 3 OHE)
        │
        ▼  Saved as data/processed/cleaned_agent1_data.csv
        │
        ▼  Agent 2: reads processed CSV + adds Risk_Tier as ordinal feature
```

---

## 7. Agent 1 — Risk Profiler (Deep Dive)

**File:** `backend/agents/agent1_risk_profiler.py` (1,453 lines)

### Purpose

Classifies each driver into **Low / Medium / High** risk using a calibrated XGBoost model with SHAP explanations and an OOD safety gate.

### Training Pipeline (11 Steps)

```
Step 0:  Clean Slate ─── wipe models/ and data/processed/
Step 1:  Load & Clean ─── CSV → impute nulls → convert mileage ranges
Step 2:  Noisy Labels ─── actuarial score + Gaussian noise → risk tier
Step 3a: Interaction Features ─── Miles_Per_Exp, Total_Incidents, Age_Exp_Gap
Step 3b: Encoding ─── OHE Veh_Usage + LabelEncode Risk_Tier → 11 features
Step 4a: OOD Detector ─── IsolationForest (200 trees, 0.01th percentile threshold)
Step 4b: Sample Weights ─── Cost-sensitive: High=3×, Medium=2×, Low=1×
Step 5:  Hyperparameter Search ─── RandomizedSearchCV (50 iter × 5 folds, neg_log_loss)
Step 6:  Calibration ─── CalibratedClassifierCV (isotonic, cv='prefit')
Step 7:  Evaluation ─── Accuracy, balanced accuracy, log-loss, confusion matrix
Step 8:  SHAP ─── TreeExplainer (interventional, 500-row background)
Step 9:  Demo Predictions ─── 4 profiles + adversarial red-team test
Step 11: Export ─── 8 .pkl artifacts + manifest.json + processed CSV
```

### Inference Pipeline (3 Stages)

```
┌──────────────────────────────────────────────────────────────────┐
│ Stage 1 │ Physics Check → Build 11-feature vector               │
│ Stage 2 │ OOD Gate (IsolationForest score < threshold?)         │
│         │   Anomaly → {"status": "ACTION_REQUIRED: DATA_ANOMALY"} │
│         │   Normal  → continue                                    │
│ Stage 3 │ CalibratedXGBoost.predict_proba() + SHAP explanation  │
└──────────────────────────────────────────────────────────────────┘
```

### Artifacts Produced (8 files)

| Artifact | What It Stores |
|----------|---------------|
| `calibrated_risk_profiler.pkl` | CalibratedClassifierCV wrapping XGBoost |
| `shap_explainer.pkl` | TreeExplainer bound to base XGBoost |
| `ohe_encoder.pkl` | Fitted OneHotEncoder for Veh_Usage |
| `label_encoder.pkl` | LabelEncoder (High/Low/Medium ↔ 0/1/2) |
| `feature_names.pkl` | Ordered list of 11 feature names |
| `ood_detector.pkl` | IsolationForest anomaly detector |
| `ood_threshold.pkl` | Score threshold (0.01th percentile) |
| `training_stats.pkl` | Feature means for drift monitoring |

### Output Schema

```json
{
  "status": "OK",
  "predicted_tier": "Medium",
  "confidence": 0.6234,
  "class_probabilities": { "High": 0.12, "Low": 0.25, "Medium": 0.63 },
  "top_3_features": [
    { "feature": "Prev_Accidents", "shap_value": 0.182, "direction": "↑ increases risk", "magnitude": "HIGH" },
    { "feature": "Miles_Per_Exp",  "shap_value": 0.089, "direction": "↑ increases risk", "magnitude": "MEDIUM" },
    { "feature": "Driving_Exp",    "shap_value": -0.041, "direction": "↓ decreases risk", "magnitude": "LOW" }
  ]
}
```

### Safety Features

- **Deterministic Physics Check** — hard rules that catch logically impossible inputs (age < 16, negative miles, exp > age - 15)
- **IsolationForest OOD Gate** — statistical anomaly detection blocks corrupt/extreme values
- **Adversarial Red-Team** — 5 deliberately impossible profiles tested on every training run
- **Drift Monitor** — compares incoming Annual_Miles mean against training distribution, alerts at >10% shift
- **Counterfactual Advisor** — "What-If" analysis that tells High/Medium risk customers how to reach Low tier

---

## 8. Agent 2 — Conversion Predictor (Deep Dive)

**File:** `backend/agents/agent2_conversion_predictor.py` (1,153 lines)

### Purpose

Predicts the **probability a customer will buy** (bind) the insurance policy. Uses cross-agent context from Agent 1's risk tier.

### Key Innovations

| Technique | Why |
|-----------|-----|
| **SMOTE** (k=5) | Oversamples the ~22% minority Bind class to 1:1 in training only |
| **Cross-agent context** | Risk_Tier from Agent 1 is label-encoded as an ordinal integer feature |
| **CalibratedClassifierCV** | Isotonic calibration produces reliable bind probabilities |
| **Optimal F1 threshold** | Precision-recall curve analysis finds the best decision boundary (0.3370) |
| **Sales status buckets** | Segments customers for actionable sales workflows |

### Sales Status Buckets

| Status | Condition | Action |
|--------|-----------|--------|
| `HIGH_PROPENSITY` | bind_prob ≥ threshold + 0.10 | Auto-process, no intervention needed |
| `NEAR_MISS_FOR_ADVISOR` | threshold − 0.10 ≤ bind_prob < threshold | Send to Agent 3 for premium adjustment |
| `LOW_PROB` | bind_prob < threshold − 0.10 | Low priority, unlikely to convert |
| `UNCERTAIN` | Near the threshold boundary | Needs human review |

### Artifacts Produced (7 files, all prefixed `agent2_*_v2`)

| Artifact | Purpose |
|----------|---------|
| `agent2_conversion_model_v2.pkl` | CalibratedClassifierCV for bind prediction |
| `agent2_feature_names_v2.pkl` | Feature name ordering |
| `agent2_ohe_encoder_v2.pkl` | OneHotEncoder for categorical features |
| `agent2_shap_explainer_v2.pkl` | SHAP explainer for conversion drivers |
| `agent2_threshold_v2.pkl` | Optimal F1 threshold (0.3370) |
| `agent2_tier_encoder_v2.pkl` | LabelEncoder for Risk_Tier |
| `agent2_metadata_v2.json` | Training metadata and performance metrics |

---

## 9. Agent 3 — Premium Advisor (Deep Dive)

**File:** `backend/agents/agent3.py` (385 lines)

### Purpose

When a customer is **unlikely to buy** (conversion score < 40), recommends a premium adjustment to increase conversion. Generates customer-facing explanations via Groq LLM.

### Activation Gate

```
conversion_score ≥ 40  →  No adjustment (customer already likely to bind)
conversion_score < 40  →  Activate rule engine
```

### Business Rules

| Rule | Condition | Reduction |
|------|-----------|-----------|
| **R1** | Premium > $800 AND salary ≤ Low | 15% |
| **R2** | Premium > $700 AND salary = Very Low (≤$25K) | 10% |
| **R3** | Enhanced coverage AND salary ≤ Low | 0% + coverage downgrade advisory |
| **R4** | Re-quote detected (price-sensitive customer) | 10% |
| **R5** | Cheap vehicle (≤$20K) AND Enhanced coverage | 0% + Basic coverage advisory |

**Resolution:** When multiple rules fire, the **largest reduction** wins.

### LLM Layer

- **Model:** Groq `llama3-8b-8192` (max_tokens=150, temperature=0.7)
- **Prompt:** Takes the rule-derived adjustment and generates a 2-sentence customer-facing explanation
- **Fallback:** If the LLM call fails for ANY reason (missing key, timeout, import error), the rule-generated reason string is used silently — the function never raises

### Output Schema

```json
{
  "premium_flag": true,
  "recommended_premium": 637.50,
  "adjustment": "-15%",
  "reason": "Based on your driving profile, we'd like to offer you a more competitive rate...",
  "original_premium": 750.00
}
```

---

## 10. Agent 4 — Decision Router (Deep Dive)

**File:** `backend/agents/agent4.py` (415 lines)

### Purpose

Makes the **final routing decision** by consuming outputs from all three upstream agents. Routes each quote to one of three outcomes.

### 3-Stage Decision Logic

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: ESCALATION RULES (evaluated first, strict priority) │
│   E1: High risk tier → Escalate                              │
│   E2: Conversion score < 30 → Escalate                       │
│   E3: Prior accident + score < 50 → Escalate                 │
│   E4: Citation + Medium/High risk → Escalate                 │
│   ANY match → immediately route to underwriter                │
├─────────────────────────────────────────────────────────────┤
│ Stage 2: AUTO-APPROVAL (ALL conditions must be true)          │
│   A1: Low risk tier                                           │
│   A2: Conversion score ≥ 60                                   │
│   A3: No premium adjustment recommended                       │
│   A4: Zero prior accidents                                    │
│   ALL true → process policy automatically                     │
├─────────────────────────────────────────────────────────────┤
│ Stage 3: AGENT FOLLOW-UP (catch-all)                          │
│   Everything else → sales agent engagement                    │
│   Priority = Medium if premium flag, Low otherwise            │
└─────────────────────────────────────────────────────────────┘
```

### Routing Outcomes

| Decision | Canonical Label | Human Required | Priority |
|----------|----------------|---------------|----------|
| Escalate to Underwriter | `MANUAL_REVIEW` | Yes | High |
| Auto Approve | `AUTO_APPROVE` | No | Low |
| Agent Follow-Up | `MANUAL_REVIEW` | Yes | Medium/Low |

### LLM Layer

Same pattern as Agent 3 — Groq `llama3-8b-8192` generates a 2-sentence underwriter-facing justification, with silent rule-based fallback.

---

## 11. LangGraph Orchestration — main.py

**File:** `backend/main.py` (716 lines)

### How LangGraph Wires the Agents

LangGraph's `StateGraph` provides a typed, inspectable DAG that threads a shared `AgentState` through all four agents:

```python
class AgentState(TypedDict, total=False):
    input_data:              Dict[str, Any]   # raw quote — READ ONLY
    risk_results:            Dict[str, Any]   # Agent 1 output
    conversion_results:      Dict[str, Any]   # Agent 2 output
    advisor_pitch:           Dict[str, Any]   # Agent 3 output
    final_decision:          Dict[str, Any]   # Agent 4 output
    final_routing_decision:  str              # canonical: AUTO_APPROVE | MANUAL_REVIEW
```

### Node Function Pattern

Each node follows the same pattern:
1. Read upstream state fields
2. Call the agent's core function
3. Write results to a new state field
4. Return the updated state

```python
def node_risk(state: AgentState) -> AgentState:
    result = _risk_engine.predict_and_explain(state["input_data"])
    return {**state, "risk_results": result}
```

### Graph Compilation

```python
builder = StateGraph(AgentState)
builder.add_node("node_risk",       node_risk)
builder.add_node("node_conversion", node_conversion)
builder.add_node("node_advisor",    node_advisor)
builder.add_node("node_router",     node_router)
builder.add_edge(START, "node_risk")
builder.add_edge("node_risk", "node_conversion")
builder.add_edge("node_conversion", "node_advisor")
builder.add_edge("node_advisor", "node_router")
builder.add_edge("node_router", END)
_pipeline = builder.compile()  # compiled once, reused for every request
```

### Startup — ML Singleton Loading

```python
@app.on_event("startup")
async def load_agents():
    _risk_engine = RiskProfilerPredictor.from_artifacts(MODELS_DIR)
    _conv_engine = ConversionPredictor.from_artifacts(MODELS_DIR)
    # Agents 3 & 4 are stateless — no artifact loading needed
```

### Safety Layers in main.py

| Layer | Purpose |
|-------|---------|
| **Pydantic Validation** | `QuoteRequest` schema with field constraints (age 16–100, etc.) and cross-field validation (exp ≤ age − 16) |
| **30s Timeout** | `asyncio.wait_for()` wraps the pipeline so the browser never hangs |
| **OOD Gate** | If Agent 1 returns `OOD_FLAG`, HTTP 422 is returned immediately |
| **Confidence Gate** | If Agent 1's confidence < 60%, status is set to `LOW_CONFIDENCE_ESCALATE` |
| **Global Exception Handler** | Catches any unhandled error and returns a clean JSON error |

---

## 12. API Contract & Endpoints

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/quote` | Primary endpoint — runs all 4 agents |
| `POST` | `/api/v1/full-analysis` | Same as above (named alias) |
| `POST` | `/api/process_quote` | Legacy alias for backwards compatibility |
| `GET` | `/api/health` | Liveness check + artifact inventory |
| `GET` | `/docs` | Auto-generated Swagger UI |

### Request Schema

```json
{
  "Driver_Age": 34,
  "Driving_Exp": 12,
  "Prev_Accidents": 0,
  "Prev_Citations": 1,
  "Annual_Miles": 22000,
  "Veh_Usage": "Pleasure",
  "Quoted_Premium": 750.0,
  "Sal_Range": 1,
  "Coverage": 1,
  "Vehicl_Cost_Range": 2,
  "Re_Quote": 0
}
```

### Response Schema

```json
{
  "transaction_id": "uuid-string",
  "status": "OK",
  "final_routing_decision": "AUTO_APPROVE",
  "escalation_reason": null,
  "risk_assessment": {
    "predicted_tier": "Low",
    "confidence_score": 0.82,
    "ood_flag": "OK",
    "class_probabilities": { "High": 0.05, "Low": 0.82, "Medium": 0.13 },
    "top_shap_drivers": [ /* ShapDriver[] */ ]
  },
  "conversion_metrics": {
    "bind_probability": 0.67,
    "sales_status": "HIGH_PROPENSITY",
    "distance_to_conversion": 0.33
  },
  "advisor_strategy": {
    "premium_flag": false,
    "suggested_discount_pct": "none",
    "recommended_premium": 750.0,
    "original_premium": 750.0,
    "customer_facing_message": "No adjustment needed..."
  },
  "final_routing": {
    "decision": "Auto Approve",
    "reason": "LLM-generated 2-sentence justification...",
    "human_required": false,
    "priority": "Low",
    "action_items": ["Process policy automatically", "No human review needed"],
    "final_routing_decision": "AUTO_APPROVE"
  }
}
```

---

## 13. Frontend — Storytelling Dashboard

### UI Concept

Instead of a traditional grid layout showing all results at once, the frontend uses a **storytelling metaphor** — results appear sequentially, like chapters in a story:

```
┌─────────────────────────────────────┐
│  📝 Quote Form (11 fields)          │  ← User fills in and clicks "Run Analysis"
└─────────────────────────────────────┘
                  │
                  ▼  Form collapses into a slim summary header
┌─────────────────────────────────────┐
│  Age: 34 · Exp: 12y · Acc: 0 · ... │  [✎ Edit]
└─────────────────────────────────────┘
                  │
                  ▼  Agent 1 card appears with skeleton loader, then reveals
┌─────────────────────────────────────┐
│  🛡 Agent 1 · Risk Profiler         │
│  [Low Risk]  "Clean driving record" │
│  ▶ View Details (accordion)         │
│    └─ SVG gauge + SHAP waterfall    │
└─────────────────────────────────────┘
                  │  ← animated handoff connector
                  ▼
┌─────────────────────────────────────┐
│  📊 Agent 2 · Conversion Engine     │
│  [High Propensity]  "67% bind prob" │
│  ▶ View Details (accordion)         │
│    └─ Probability tiles + threshold │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  💬 Agent 3 · AI Premium Advisor    │
│  [No Adjustment]  "Score sufficient"│
│  ▶ View Details (accordion)         │
│    └─ Premium comparison + LLM msg  │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  ⚖ Agent 4 · Underwriting Router    │
│  [Auto-Approved]  "Clean case"      │
│  ▼ View Details (open by default)   │
│    └─ Decision banner + action items│
└─────────────────────────────────────┘
```

### State Machine

```
             ┌───────────────────────┐
             │        idle           │  ← initial state, form visible
             └──────────┬────────────┘
                        │ user clicks "Run Analysis"
                        ▼
             ┌───────────────────────┐
             │      running          │  ← form collapses, cards reveal
             │                       │    on 1.2s timer intervals
             │  Timer: hidden →      │
             │    loading → revealed │
             │  Fetch: POST to API   │
             └──────────┬────────────┘
                        │ response arrives
               ┌────────┼────────┐
               ▼        ▼        ▼
          ┌─────────┐ ┌────────┐ ┌──────────┐
          │complete │ │ood_err │ │  error   │
          │ all 4   │ │ anomaly│ │ network/ │
          │ revealed│ │ banner │ │ HTTP err │
          └─────────┘ └────────┘ └──────────┘
```

### Color System

| Agent | Accent Color | Hex |
|-------|-------------|-----|
| Agent 1 — Risk Profiler | Emerald | `#34d399` |
| Agent 2 — Conversion Engine | Sky | `#38bdf8` |
| Agent 3 — Premium Advisor | Violet | `#a78bfa` |
| Agent 4 — Decision Router | Amber | `#fbbf24` |

### Verdict Badges

| Variant | Color | Usage Example |
|---------|-------|---------------|
| `positive` | Green | "Low Risk", "High Propensity", "Auto-Approved" |
| `caution` | Amber | "Medium Risk", "Near Miss", "Manual Review" |
| `negative` | Red | "High Risk", "Low Probability", "Rejected" |
| `info` | Blue | "-15% Discount", "Uncertain" |
| `neutral` | Gray | "No Adjustment" |

### Component Architecture

| Component | Role |
|-----------|------|
| `QuoteForm` | 11-field form with defaults, validation, number/select inputs |
| `CollapsedHeader` | Slim strip showing key inputs + Edit button after form collapses |
| `StorytellingCard` | Generic wrapper: skeleton → reveal → header + verdict + accordion |
| `SkeletonLoader` | Per-agent colored pulse animation during loading |
| `VerdictBadge` | Color-coded pill (positive/caution/negative/info/neutral) |
| `HandoffConnector` | Growing vertical line + pulse dot between agent cards |
| `RiskDetails` | SVG risk gauge, SHAP waterfall bars, magnitude badges |
| `ConversionDetails` | Bind probability tiles, threshold progress bar |
| `AdvisorDetails` | 3-column premium comparison, LLM-generated chat bubble |
| `DecisionDetails` | Decision banner, terminal-style log, action item checklist |
| `ErrorBanner` | Displays OOD anomaly or HTTP error messages |

---

## 14. End-to-End Pipeline Flow

Here's what happens when a user clicks **"Run Analysis"**:

### Timeline

```
t=0.0s  User clicks "Run Analysis"
        │
        ├─ Frontend: Form collapses → CollapsedHeader appears
        ├─ Frontend: Agent 1 card → "loading" (skeleton pulse)
        ├─ Frontend: POST /api/v1/full-analysis fires
        │
t=1.2s  Frontend: Agent 1 → "revealed" (placeholder), Agent 2 → "loading"
t=2.4s  Frontend: Agent 2 → "revealed", Agent 3 → "loading"
t=3.6s  Frontend: Agent 3 → "revealed", Agent 4 → "loading"
t=4.8s  Frontend: Agent 4 → "revealed"
        │
t=~2-5s Backend response arrives (actual ML computation)
        │
        ├─ If response arrives BEFORE timer finishes:
        │    All remaining cards instantly → "revealed" with real data
        │
        ├─ If response arrives AFTER timer finishes:
        │    Cards already showing placeholders; replaced with real data
        │
        └─ Final state: all 4 cards revealed with real verdicts + details
```

### Backend Processing (what happens inside the 2-5 seconds)

```
1. FastAPI receives QuoteRequest → Pydantic validates all fields
2. Pydantic cross-validation: Driving_Exp ≤ Driver_Age - 16

3. LangGraph invokes _pipeline.invoke(initial_state)

4. node_risk (Agent 1):
   ├─ Physics check (age ≥ 16, miles ≥ 0, etc.)
   ├─ Build 11-feature vector
   ├─ IsolationForest OOD gate
   │   └─ If OOD: abort entire pipeline → HTTP 422
   ├─ CalibratedXGBoost.predict_proba()
   ├─ SHAP TreeExplainer → top 3 features
   ├─ Drift monitor check
   └─ Write risk_results to AgentState

5. node_conversion (Agent 2):
   ├─ Read risk_results.predicted_tier from AgentState
   ├─ Build feature vector (including Risk_Tier)
   ├─ CalibratedXGBoost.predict_proba()
   ├─ Apply optimal F1 threshold (0.3370)
   ├─ Compute sales_status bucket
   └─ Write conversion_results to AgentState

6. node_advisor (Agent 3):
   ├─ Read conversion_results.conversion_score
   ├─ Check activation gate (score < 40?)
   ├─ Evaluate 5 business rules
   ├─ Select largest reduction
   ├─ Call Groq LLM for customer-facing reason
   │   └─ Fallback: use rule-generated string
   └─ Write advisor_pitch to AgentState

7. node_router (Agent 4):
   ├─ Read risk_results + conversion_results + advisor_pitch
   ├─ Translate to agent4 input format
   ├─ Stage 1: Check escalation rules
   ├─ Stage 2: Check auto-approval conditions
   ├─ Stage 3: Default to agent follow-up
   ├─ Call Groq LLM for justification
   └─ Write final_decision + final_routing_decision to AgentState

8. Post-pipeline checks:
   ├─ OOD gate (if Agent 1 flagged anomaly)
   ├─ Confidence gate (if confidence < 60% → LOW_CONFIDENCE_ESCALATE)
   └─ Assemble PipelineResponse

9. Return JSON response to frontend
```

---

## 15. Getting Started — Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- A Groq API key (free at [console.groq.com](https://console.groq.com/keys))

### Step 1: Clone & Set Up Python Environment

```bash
cd Quote-Agents
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install -r backend/requirements.txt
```

### Step 2: Configure Environment

```bash
# Backend secrets
cp backend/.env.example backend/.env
# Edit backend/.env and add your GROQ_API_KEY

# Frontend config (optional — defaults to localhost:8001)
cp frontend/.env.local.example frontend/.env.local
```

### Step 3: Train the Models

```bash
cd backend

# Train Agent 1 (generates risk model + processed CSV for Agent 2)
python agents/agent1_risk_profiler.py

# Train Agent 2 (reads Agent 1's processed CSV)
python agents/agent2_conversion_predictor.py
```

This will populate `backend/models/` with all `.pkl` artifacts.

### Step 4: Start the Backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001
```

Verify: visit http://localhost:8001/docs for Swagger UI or:

```bash
curl http://localhost:8001/api/health | python3 -m json.tool
```

### Step 5: Start the Frontend

```bash
cd frontend
npm install
npm run dev        # development mode → http://localhost:3000
# or
npm run build && npm start   # production mode
```

### Step 6: Use the Application

1. Open http://localhost:3000
2. Fill in the quote form (defaults are pre-populated)
3. Click "Run Analysis"
4. Watch the 4 agent cards appear sequentially
5. Click "View Details" on any card for deep-dive visualisations

---

## 16. Docker Deployment

### Build & Run (Backend Only)

```bash
# Ensure models/ directory has trained artifacts
docker build -t quote-agents-backend ./backend
docker run --env-file backend/.env -p 8001:8001 quote-agents-backend
```

### Dockerfile Features

- Multi-stage Python 3.11-slim build
- No `.pyc` files, unbuffered stdout
- Built-in healthcheck (pings `/api/health` every 30s)
- Exposes port 8001

### Important Notes

- `backend/models/*.pkl` files are gitignored — you must either:
  1. Train models locally before building the Docker image
  2. Use Git LFS for the `.pkl` files
  3. Add a training step to your CI pipeline

---

## 17. Environment Variables

### Backend (`backend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | API key for Groq LLM (Agents 3 & 4) |
| `FRONTEND_ORIGINS` | No | `""` | Comma-separated CORS origins for production |
| `HOST` | No | `0.0.0.0` | Server bind address |
| `PORT` | No | `8001` | Server port |

### Frontend (`frontend/.env.local`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | No | `http://localhost:8001` | Backend API base URL |

---

## 18. Testing

### Backend Tests

```bash
cd backend
python test_pipeline.py
```

Runs 4 end-to-end test profiles:

| Profile | Purpose | Expected Result |
|---------|---------|-----------------|
| Safe Driver | Clean record, high exp | Low risk, Auto Approve |
| High-Risk Driver | Young, 3 accidents | High risk, Escalate |
| Near Miss | Moderate, near threshold | Tests NEAR_MISS bucket |
| Corrupt Input | Age = -5 | OOD gate blocks immediately |

### Frontend Build Check

```bash
cd frontend
npm run build    # TypeScript compilation + Next.js production build
```

Zero errors = ✅

### Manual API Test

```bash
curl -s -X POST http://localhost:8001/api/v1/full-analysis \
  -H 'Content-Type: application/json' \
  -d '{
    "Driver_Age": 34,
    "Driving_Exp": 12,
    "Prev_Accidents": 0,
    "Prev_Citations": 1,
    "Annual_Miles": 22000,
    "Veh_Usage": "Pleasure"
  }' | python3 -m json.tool
```

---

## 19. Key Design Decisions

### Why LangGraph Instead of Simple Function Calls?

| Benefit | Explanation |
|---------|-------------|
| **Typed State** | `AgentState` TypedDict ensures every node reads/writes the right fields |
| **Inspectable DAG** | Graph can be visualised, debugged, and audited |
| **Future-proof** | Easy to add conditional edges, parallel branches, or retry logic |
| **Production patterns** | Built-in support for timeouts, error handling, and state persistence |

### Why XGBoost + Calibration (Not Deep Learning)?

| Factor | XGBoost + Isotonic | Deep Learning |
|--------|--------------------|---------------|
| Data size | 146K rows — XGBoost excels | Needs >>100K to outperform |
| Interpretability | SHAP TreeExplainer gives exact feature importance | Black box |
| Calibration | Isotonic regression → reliable probabilities | Requires temperature scaling |
| Training time | ~2 minutes on CPU | Hours with GPU |
| Deployment | Single .pkl file, no GPU needed | Requires model server |

### Why Groq LLM for Explanations?

- **Speed** — Groq's LPU delivers <1s inference for Llama 3 8B
- **Cost** — Free tier is sufficient for development
- **Safety** — Silent fallback to rule strings means the system **never breaks** if the LLM is unavailable
- **Simplicity** — 2-sentence explanations don't need GPT-4-class models

### Why Gaussian Noise on Labels?

Without noise, the risk label is a **deterministic function** of the raw features — XGBoost achieves 100% accuracy by memorising the formula. That's useless. Adding σ=2.0 noise:
- Forces the model to learn **calibrated probabilities** (borderline cases genuinely land in either tier)
- Makes the model **generalisable** (not just a lookup table)
- Mimics real-world label noise (human underwriters disagree on borderline cases)

### Why Cost-Sensitive Weights (High=3×)?

In insurance, the cost of errors is asymmetric:
- **False Low** (miss a High-risk driver) → underpriced policy → claims losses → $$$ 
- **False High** (over-flag a Safe driver) → customer declines the quote → $ lost premium

The 3× multiplier on High-risk makes the model prioritise **not missing dangerous drivers** over perfect overall accuracy.

---

## 20. Troubleshooting

### "Agent 1 not loaded"

```bash
# Train the models first
cd backend
python agents/agent1_risk_profiler.py
python agents/agent2_conversion_predictor.py
```

### "Connection Failed" on Frontend

```bash
# Make sure backend is running on port 8001
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001
```

### GROQ_API_KEY Not Working

1. Get a key at https://console.groq.com/keys
2. Add to `backend/.env`: `GROQ_API_KEY=gsk_your_key_here`
3. Restart the backend
4. Note: LLM calls will silently fall back to rule strings if the key is invalid

### Port Already in Use

```bash
# Kill processes on ports
lsof -ti :8001 | xargs kill -9   # backend
lsof -ti :3000 | xargs kill -9   # frontend
```

### Pyright/Pylance Warnings

The ML libraries (pandas, sklearn, imbalanced-learn) have incomplete type stubs. Some `# type: ignore` comments are necessary. CSS `@tailwind` warnings are suppressed via `.vscode/settings.json`.

---

## Summary

This project demonstrates how to build a **production-grade multi-agent AI system** from scratch:

1. **Start with data** — understand the domain, explore the dataset
2. **Build specialised agents** — each expert in one aspect of the problem
3. **Wire them with a DAG** — LangGraph provides typed, inspectable orchestration
4. **Add safety everywhere** — OOD detection, physics checks, confidence gates, LLM fallbacks
5. **Serve via API** — FastAPI with Pydantic validation and auto-docs
6. **Visualise beautifully** — Next.js storytelling UI with progressive disclosure
7. **Test comprehensively** — adversarial profiles, end-to-end tests, build checks

The result is a system where every piece is **explainable**, **auditable**, and **safe** — exactly what insurance regulators and customers expect.

---

*Built with ❤️ using Python, LangGraph, XGBoost, SHAP, FastAPI, Groq, Next.js, React, Tailwind CSS, and Framer Motion.*
