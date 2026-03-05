# InsurTech AI Pipeline — Frontend Documentation

> **4-Agent LangGraph Quote Engine · v3.0**
> A Storytelling UI that walks the user through each agent's decision, one step at a time.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
5. [What the User Sees](#what-the-user-sees)
   - [1. Idle State](#1-idle-state)
   - [2. Quote Input Form](#2-quote-input-form)
   - [3. Processing — Sequential Agent Reveal](#3-processing--sequential-agent-reveal)
   - [4. Agent 1 — Risk Profiler Card](#4-agent-1--risk-profiler-card)
   - [5. Agent 2 — Conversion Engine Card](#5-agent-2--conversion-engine-card)
   - [6. Agent 3 — AI Premium Advisor Card](#6-agent-3--ai-premium-advisor-card)
   - [7. Agent 4 — Underwriting Router Card](#7-agent-4--underwriting-router-card)
   - [8. Error States](#8-error-states)
6. [UI Flow & State Machine](#ui-flow--state-machine)
7. [Component Architecture](#component-architecture)
8. [API Contract](#api-contract)
9. [Environment Variables](#environment-variables)

---

## Overview

The frontend is a **single-page Next.js application** that serves as the user interface for the InsurTech 4-agent AI pipeline. Instead of showing all results at once in a grid, it uses a **Storytelling UI pattern** — progressive disclosure where each agent's output is revealed sequentially with animations, verdict badges, plain-English summaries, and collapsible detail accordions.

**Key design principles:**
- **Progressive disclosure** — don't overwhelm; reveal one agent at a time
- **Plain-English verdicts** — every agent card shows a one-liner verdict badge before any charts
- **Accordion details** — technical charts/data are behind a "View Details" toggle (closed by default)
- **Handoff animations** — growing vertical lines + pulse dots bridge agent cards, visualizing the pipeline flow
- **Collapsing input** — the form shrinks to a slim summary strip after submission so the story takes center stage

---

## Tech Stack

| Technology       | Version  | Purpose                              |
| ---------------- | -------- | ------------------------------------ |
| Next.js          | 14.2     | React framework (App Router)         |
| React            | 18.3     | UI library                           |
| TypeScript       | 5.x      | Type safety                          |
| Tailwind CSS     | 3.x      | Utility-first styling                |
| Framer Motion    | 11.2     | Animations & transitions             |
| Lucide React     | 0.378    | Icon set                             |

---

## Project Structure

```
frontend/
├── app/
│   ├── page.tsx                  # Main dashboard — storytelling state machine
│   ├── layout.tsx                # Root layout with metadata
│   └── globals.css               # Tailwind directives + custom styles
│
├── components/
│   ├── QuoteForm.tsx             # 11-field input form with defaults
│   ├── CollapsedHeader.tsx       # Slim summary strip (post-submission)
│   ├── StorytellingCard.tsx      # Wrapper: skeleton → verdict → accordion
│   ├── SkeletonLoader.tsx        # Pulsing placeholder per agent color
│   ├── VerdictBadge.tsx          # Color-coded verdict pill
│   ├── HandoffConnector.tsx      # Animated vertical line between cards
│   ├── ErrorBanner.tsx           # OOD anomaly + general error display
│   │
│   ├── RiskDetails.tsx           # Agent 1 accordion body (gauge + SHAP)
│   ├── ConversionDetails.tsx     # Agent 2 accordion body (probability bar)
│   ├── AdvisorDetails.tsx        # Agent 3 accordion body (premium comparison)
│   ├── DecisionDetails.tsx       # Agent 4 accordion body (terminal log)
│   │
│   ├── RiskPanel.tsx             # (Legacy) standalone risk card
│   ├── ConversionPanel.tsx       # (Legacy) standalone conversion card
│   ├── AdvisorPanel.tsx          # (Legacy) standalone advisor card
│   ├── DecisionBanner.tsx        # (Legacy) standalone decision banner
│   └── PipelineStepper.tsx       # (Legacy) vertical/horizontal stepper
│
├── lib/
│   └── api-contract.ts           # TypeScript types matching backend schemas
│
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── postcss.config.js
```

---

## Getting Started

```bash
# Install dependencies
cd frontend
npm install

# Development mode (HMR — hot reload)
npm run dev            # → http://localhost:3000

# Production mode (recommended for demo)
npm run build          # creates .next/ output
npx next start --port 3000
```

**Requires:** Backend running on port 8001 (`cd backend && uvicorn main:app --port 8001`)

---

## What the User Sees

### 1. Idle State

When the page first loads, the user sees:

- **Header bar** — "InsurTech AI Pipeline" title with a CPU icon and version label
- **Quote Input Form** — a dark-themed card with 11 input fields pre-filled with default values
- **Empty state message** — "No pipeline run yet" with instructions to fill the form and click "Run Analysis"

The page has a dark slate-950 background with a centered single-column layout (`max-width: 768px`).

---

### 2. Quote Input Form

A card titled **"Quote Input"** with 11 fields in a 2-column grid:

| Field              | Type     | Default     | Required | Description                         |
| ------------------ | -------- | ----------- | -------- | ----------------------------------- |
| Driver Age         | Number   | 34          | ✅       | Age of the driver (16–100)          |
| Driving Experience | Number   | 12          | ✅       | Years of driving experience (0–84)  |
| Prior Accidents    | Number   | 0           | ✅       | Number of previous accidents (0–20) |
| Prior Citations    | Number   | 1           | ✅       | Number of previous citations (0–20) |
| Annual Miles       | Number   | 22,000      | ✅       | Annual miles driven (full-width)    |
| Vehicle Usage      | Dropdown | Pleasure    | ✅       | Pleasure / Commute / Business       |
| Quoted Premium ($) | Number   | 750         |          | Current quoted premium              |
| Salary Range       | Dropdown | $40K–$60K   |          | Income bracket (0–4 encoded)        |
| Coverage Level     | Dropdown | Balanced    |          | Basic / Balanced / Enhanced         |
| Vehicle Cost       | Dropdown | $20K–$30K   |          | Vehicle value bracket (0–4 encoded) |
| Re-Quote?          | Dropdown | No          |          | Whether this is a re-quote          |

**Button:** A full-width violet "Run Analysis" button at the bottom. It shows "Processing…" and disables while the pipeline runs.

---

### 3. Processing — Sequential Agent Reveal

After clicking "Run Analysis":

1. **Form collapses** — animates out and is replaced by a **Collapsed Header** strip showing:
   - Age, experience, accident/citation count, annual miles, vehicle usage, quoted premium
   - An "Edit Quote" button to re-expand the form

2. **Agents appear one-by-one** (1.2-second intervals):
   - Each card starts as a **Skeleton Loader** — pulsing placeholder bars in the agent's accent color
   - A **Handoff Connector** grows between cards — a vertical line with a pulse dot
   - When data arrives, all cards transition to their **Revealed** state

3. **Transaction metadata** appears in the top-right corner:
   - `tx: a1b2c3d4…` — truncated transaction ID
   - `2.3s` — total pipeline execution time

---

### 4. Agent 1 — Risk Profiler Card

**Accent color:** Emerald (green)
**Icon:** Shield

**What the user sees at a glance:**

- **Verdict Badge** — colored pill showing one of:
  - 🟢 `Low Risk` (emerald)
  - 🟡 `Medium Risk` (amber)
  - 🔴 `High Risk` (rose)
- **Summary sentence** — e.g., *"This driver's profile is 94% likely to be low risk. The strongest factor is 'Driving_Exp'."*

**Expandable "View Details" accordion (collapsed by default):**

| Visual Element       | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| Confidence Gauge     | SVG semicircle arc showing calibrated probability (e.g., 94%)               |
| Predicted Tier       | Large bold text: "LOW RISK" / "MEDIUM RISK" / "HIGH RISK"                  |
| SHAP Feature Impact  | Center-baseline horizontal waterfall chart for each feature                 |
|                      | — Green bars = reduces risk, Red bars = increases risk                      |
|                      | — SHAP values shown numerically (e.g., +0.347, −0.129)                     |
| Driver Magnitude     | Row of rounded pills: `Driving_Exp: HIGH`, `Driver_Age: MEDIUM`, etc.      |
| OOD Status           | Small badge: "OOD ✓ Clear" (emerald) or "⚠ OOD Flag" (rose)               |

---

### 5. Agent 2 — Conversion Engine Card

**Accent color:** Sky (blue)
**Icon:** Bar Chart

**What the user sees at a glance:**

- **Verdict Badge** — one of:
  - 🟢 `High Propensity` (emerald) — strong buying signal
  - 🟡 `Near Miss` (amber) — just below conversion threshold
  - ⚪ `Low Probability` (slate) — unlikely to convert
  - 🔵 `Uncertain` (sky) — borderline signal
- **Summary sentence** — e.g., *"Strong buying signal — 78% likelihood this customer will bind. Pursue confidently."*

**Expandable "View Details" accordion:**

| Visual Element         | Description                                                               |
| ---------------------- | ------------------------------------------------------------------------- |
| Bind Probability tile  | Large percentage in the agent's accent color (e.g., 78%)                  |
| Gap to Convert tile    | Percentage distance to the 0.3370 conversion threshold                    |
| Conversion Proximity   | Horizontal progress bar with gradient fill                                |
|                        | — White vertical marker at the 0.3370 threshold                          |
|                        | — Scale from 0% to 100% with threshold label                             |

---

### 6. Agent 3 — AI Premium Advisor Card

**Accent color:** Violet (purple)
**Icon:** Message Square

**What the user sees at a glance:**

- **Verdict Badge** — one of:
  - 🟣 `-15% Discount` (violet) — discount applied
  - ⚪ `No Adjustment` (slate) — premium stays the same
- **Summary sentence** — e.g., *"Premium reduced from $750 to $638 (saving $112) to improve conversion chances."*

**Expandable "View Details" accordion:**

| Visual Element             | Description                                                          |
| -------------------------- | -------------------------------------------------------------------- |
| Premium Comparison (3-col) | When discount active:                                                |
|                            | — **Original**: crossed-out price ($750)                             |
|                            | — **Discount**: arrow-down icon + percentage (−15%) + savings        |
|                            | — **Recommended**: final price in emerald ($638)                     |
| No Discount fallback       | Single tile: "Premium holds at $750"                                 |
| Customer-Facing Message    | Chat-bubble styled box with LLM-generated personalized message       |
|                            | — Typewriter font, violet accent                                     |
|                            | — Footer: "Generated by LLM · Agent 3"                              |

---

### 7. Agent 4 — Underwriting Router Card

**Accent color:** Amber (yellow/orange)
**Icon:** Gavel
**This card's accordion opens by default** (it contains the final decision).

**What the user sees at a glance:**

- **Verdict Badge** — one of:
  - 🟢 `Auto-Approved` (emerald) — policy can be issued automatically
  - 🟡 `Manual Review` (amber) — human underwriter must review
  - 🔴 `Rejected` (rose) — policy declined
- **Summary sentence** — e.g., *"This quote has been automatically approved. Low risk profile with strong conversion signal."*

**Expanded "View Details" accordion (open by default):**

| Visual Element           | Description                                                            |
| ------------------------ | ---------------------------------------------------------------------- |
| Decision Banner          | Large icon (ShieldCheck / ShieldAlert / ShieldX) + decision text       |
|                          | — Shows `AgentDecision` text (e.g., "Auto Approve")                   |
|                          | — "Human review required" flag if applicable                           |
|                          | — Priority badge: High (rose) / Medium (amber) / Low (emerald)        |
| Routing Justification    | Terminal-style log box with macOS window chrome:                       |
|                          | — `›` prompt + LLM-generated 2-sentence justification                 |
|                          | — `routing_decision: AUTO_APPROVE` in accent color                    |
|                          | — Blinking cursor at the bottom                                       |
| Action Items             | Bullet list of recommended next steps (colored dots)                   |

---

### 8. Error States

#### Data Anomaly (OOD — Out-of-Distribution)

When the backend's OOD gate (Isolation Forest) flags the input as anomalous (HTTP 422):

- **Red error banner** with AlertTriangle icon
- **Title:** "Data Anomaly Detected"
- **Message:** Explanation from the backend
- **Input Data:** All submitted values displayed as labeled chips
- **Footer:** "This quote has been flagged for human underwriter review."
- **Dismiss button** returns to the form

#### General Errors

| Error Type          | Title               | When it appears                              |
| ------------------- | ------------------- | -------------------------------------------- |
| Validation Error    | "Validation Error"  | Pydantic rejects input fields (HTTP 422)     |
| Service Unavailable | "Service Unavailable"| Model artifacts not loaded (HTTP 503)       |
| HTTP Error          | "HTTP {status}"     | Any other non-200 response                   |
| Connection Failed   | "Connection Failed" | Backend not running / network error           |

All errors show a dismiss button that returns the user to the form.

---

## UI Flow & State Machine

```
                    ┌─────────┐
                    │  IDLE   │  ← Form visible, empty state message
                    └────┬────┘
                         │ User clicks "Run Analysis"
                         ▼
                  ┌──────────────┐
                  │   RUNNING    │  ← Form collapses to header strip
                  │              │    Agents reveal sequentially:
                  │ card[0]: ███ │    hidden → loading → revealed
                  │ card[1]: ░░░ │    (1.2s intervals)
                  │ card[2]: ─── │
                  │ card[3]: ─── │
                  └──────┬───────┘
                         │ API response arrives
                         ▼
                  ┌──────────────┐
                  │  COMPLETE    │  ← All 4 cards revealed with real data
                  │              │    Transaction ID + elapsed time shown
                  │  tx: a1b2... │    "Run New Analysis" button appears
                  │  2.3s        │
                  └──────┬───────┘
                         │ User clicks "Run New Analysis" or "Edit Quote"
                         ▼
                    ┌─────────┐
                    │  IDLE   │  ← Form re-expands
                    └─────────┘

        ── Error paths ──

        RUNNING ──▶ OOD_ERROR    (HTTP 422 + DATA_ANOMALY status)
        RUNNING ──▶ ERROR        (HTTP 422/503/other, network failure)
        OOD_ERROR ──▶ IDLE       (user dismisses)
        ERROR ──▶ IDLE           (user dismisses)
```

---

## Component Architecture

```
page.tsx (DashboardPage)
│
├─ QuoteForm                          ← 11-field form, emits QuoteInput
│
├─ CollapsedHeader                    ← Slim strip showing key inputs + "Edit"
│
├─ StorytellingCard [×4]              ← Wrapper for each agent
│  ├─ SkeletonLoader                  ← Shown during "loading" state
│  ├─ VerdictBadge                    ← Colored pill with verdict text
│  ├─ Summary text                    ← Plain-English sentence
│  └─ Accordion body (children)       ← Detailed panel, toggle to expand
│     ├─ RiskDetails                  ← Agent 1: SVG gauge + SHAP waterfall
│     ├─ ConversionDetails            ← Agent 2: probability bar + threshold
│     ├─ AdvisorDetails               ← Agent 3: premium comparison + message
│     └─ DecisionDetails              ← Agent 4: terminal log + action items
│
├─ HandoffConnector [×3]              ← Animated lines between cards
│
└─ ErrorBanner                        ← OOD anomaly or general error display
```

**Verdict helpers** — Each detail component exports a `get*Verdict()` function that derives the verdict badge label, color variant, and summary sentence from the raw API data:

| Function                | Returns                                          |
| ----------------------- | ------------------------------------------------ |
| `getRiskVerdict()`      | `"Low Risk"` / `"Medium Risk"` / `"High Risk"`  |
| `getConversionVerdict()`| `"High Propensity"` / `"Near Miss"` / etc.       |
| `getAdvisorVerdict()`   | `"-15% Discount"` / `"No Adjustment"`            |
| `getDecisionVerdict()`  | `"Auto-Approved"` / `"Manual Review"` / `"Rejected"` |

---

## API Contract

**Endpoint:** `POST /api/v1/full-analysis`
**Backend:** FastAPI on port 8001

### Request Body (`QuoteInput`)

```json
{
  "Driver_Age": 34,
  "Driving_Exp": 12,
  "Prev_Accidents": 0,
  "Prev_Citations": 1,
  "Annual_Miles": 22000,
  "Veh_Usage": "Pleasure",
  "Quoted_Premium": 750,
  "Sal_Range": 2,
  "Coverage": 1,
  "Vehicl_Cost_Range": 2,
  "Re_Quote": 0
}
```

### Response Body (`PipelineResponse`)

```json
{
  "transaction_id": "a1b2c3d4-...",
  "status": "OK",
  "final_routing_decision": "AUTO_APPROVE",
  "escalation_reason": null,
  "risk_assessment": {
    "predicted_tier": "Low",
    "confidence_score": 0.94,
    "ood_flag": "OK",
    "class_probabilities": { "Low": 0.94, "Medium": 0.04, "High": 0.02 },
    "top_shap_drivers": [
      { "feature": "Driving_Exp", "shap_value": 0.347, "direction": "↓ decreases risk", "magnitude": "HIGH" }
    ]
  },
  "conversion_metrics": {
    "bind_probability": 0.78,
    "sales_status": "HIGH_PROPENSITY",
    "distance_to_conversion": 0.0
  },
  "advisor_strategy": {
    "premium_flag": false,
    "suggested_discount_pct": "none",
    "recommended_premium": 750,
    "original_premium": 750,
    "customer_facing_message": "Your clean driving record qualifies you for our best rate..."
  },
  "final_routing": {
    "decision": "Auto Approve",
    "reason": "Low risk profile with high conversion probability. No adverse indicators detected.",
    "human_required": false,
    "priority": "Low",
    "action_items": ["Issue policy", "Send welcome email"],
    "final_routing_decision": "AUTO_APPROVE"
  }
}
```

### Error Responses

| Status | Type             | Shape                                                    |
| ------ | ---------------- | -------------------------------------------------------- |
| 422    | OOD Anomaly      | `{ transaction_id, status: "..DATA_ANOMALY..", message, input }` |
| 422    | Validation Error | `{ detail: [{ msg, loc }] }`                            |
| 503    | Service Down     | `{ detail: "Agent artifacts not loaded..." }`            |

---

## Environment Variables

| Variable              | File                   | Default                    | Description            |
| --------------------- | ---------------------- | -------------------------- | ---------------------- |
| `NEXT_PUBLIC_API_URL`  | `.env.local`          | `http://localhost:8001`    | Backend API base URL   |

Create a `.env.local` file from the template:

```bash
cp .env.local.example .env.local
```

---

## Color System

Each agent has a consistent accent color used across all components:

| Agent   | Name                 | Accent Color | Tailwind Class   | Hex       |
| ------- | -------------------- | ------------ | ---------------- | --------- |
| Agent 1 | Risk Profiler        | Emerald      | `emerald-400/500`| `#10b981` |
| Agent 2 | Conversion Engine    | Sky          | `sky-400/500`    | `#38bdf8` |
| Agent 3 | AI Premium Advisor   | Violet       | `violet-400/500` | `#8b5cf6` |
| Agent 4 | Underwriting Router  | Amber        | `amber-400/500`  | `#f59e0b` |

**Verdict badge colors:**

| Variant    | Color   | Used for                                      |
| ---------- | ------- | --------------------------------------------- |
| `positive` | Emerald | Low Risk, Auto-Approved, High Propensity      |
| `caution`  | Amber   | Medium Risk, Manual Review, Near Miss         |
| `negative` | Rose    | High Risk, Rejected, Low Probability          |
| `info`     | Violet  | Discount Applied, Uncertain                   |
| `neutral`  | Slate   | No Adjustment, fallback                       |
