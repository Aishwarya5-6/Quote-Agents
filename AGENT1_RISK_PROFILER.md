# Agent 1 – Risk Profiler v3
### Auto Insurance Multi-Agent Pipeline

> **Status:** ✅ Deployment Ready — 42/42 checks passed (March 5, 2026)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture — The Six Layers](#2-architecture--the-six-layers)
3. [Project Structure](#3-project-structure)
4. [Prerequisites & Installation](#4-prerequisites--installation)
5. [Training the Model](#5-training-the-model)
6. [Model Performance](#6-model-performance)
7. [Feature Engineering](#7-feature-engineering)
8. [Safety System](#8-safety-system)
9. [Artifact Reference](#9-artifact-reference)
10. [The Production API](#10-the-production-api)
11. [API Endpoint Reference](#11-api-endpoint-reference)
12. [Response Contract](#12-response-contract)
13. [Observability & Drift Monitoring](#13-observability--drift-monitoring)
14. [Counterfactual What-If Analyzer](#14-counterfactual-what-if-analyzer)
15. [Configuration Reference](#15-configuration-reference)
16. [Key Design Decisions](#16-key-design-decisions)

---

## 1. Overview

Agent 1 is the first stage of a multi-agent insurance quoting pipeline. It receives a raw customer quote and returns a calibrated **Risk Tier** (High / Medium / Low) together with a SHAP explanation, a drift signal, and a counterfactual recommendation — all before any downstream pricing or underwriting agent ever sees the quote.

**Core technology stack:**

| Component | Library | Version |
|---|---|---|
| Gradient Boosting | XGBoost `XGBClassifier` | 2.1.4 |
| Probability Calibration | scikit-learn `CalibratedClassifierCV` | 1.6.1 |
| Anomaly Detection | scikit-learn `IsolationForest` | 1.6.1 |
| SHAP Explanation | `TreeExplainer` | 0.49.1 |
| Production API | FastAPI + Uvicorn | latest |
| Input Validation | Pydantic v2 | 2.12.5 |
| Python | CPython | 3.9.6 |

---

## 2. Architecture — The Six Layers

Every prediction passes through six defensive layers in strict order. No layer is skippable.

```
Incoming Quote (raw dict)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Layer 0 │ Clean Slate (training only)                        │
│          │ shutil.rmtree on models/ and data/processed/       │
│          │ Prevents stale artifact contamination              │
└──────────────────────────────┬────────────────────────────────┘
                               │
        ▼ (inference path)
┌───────────────────────────────────────────────────────────────┐
│  Layer 1 │ Pydantic Gateway  (app.py — API layer only)        │
│          │ QuoteRequest schema: type checks, range bounds,    │
│          │ Literal Veh_Usage enum, cross-field validator      │
│          │ FAIL → HTTP 422 Unprocessable Entity               │
└──────────────────────────────┬────────────────────────────────┘
                               │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Layer 2 │ Deterministic Physics Check                        │
│          │ Hardcoded domain rules — no ML needed:             │
│          │   • Driver_Age < 16                                │
│          │   • Annual_Miles < 0                               │
│          │   • Prev_Accidents < 0 or Prev_Citations < 0       │
│          │   • Driving_Exp > (Driver_Age − 15)                │
│          │ FAIL → {"status": "ACTION_REQUIRED: DATA_ANOMALY"} │
└──────────────────────────────┬────────────────────────────────┘
                               │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Layer 3 │ IsolationForest OOD Gate                           │
│          │ Trained on 8 raw features (no interaction feats)   │
│          │ Threshold: 0.01th percentile of training scores    │
│          │ FAIL → {"status": "ACTION_REQUIRED: DATA_ANOMALY"} │
└──────────────────────────────┬────────────────────────────────┘
                               │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Layer 4 │ XGBoost Prediction + SHAP                          │
│          │ 11 features (5 raw + 3 interaction + 3 OHE)        │
│          │ CalibratedClassifierCV(isotonic, prefit)           │
│          │ High=3× · Medium=2× cost-sensitive weights         │
│          │ TreeExplainer → Top-3 SHAP drivers                 │
└──────────────────────────────┬────────────────────────────────┘
                               │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Layer 5 │ Confidence Gate  (API layer only)                  │
│          │ If max(class_probability) < 0.60:                  │
│          │   status → "LOW_CONFIDENCE_ESCALATE"               │
│          │ Prediction still returned for reviewer context     │
└──────────────────────────────┬────────────────────────────────┘
                               │
        ▼
   Final Response JSON
```

---

## 3. Project Structure

```
Quote-Agents/
├── agents/
│   └── agent1_risk_profiler.py     ← Full training + inference script
├── app.py                          ← FastAPI production service
├── data/
│   ├── raw/
│   │   └── insurance_data.csv      ← 146,259 row source dataset (25 cols)
│   └── processed/
│       └── cleaned_agent1_data.csv ← 146,259 rows, 11 features + Risk_Tier
├── models/                         ← 8 serialised artifacts (auto-generated)
│   ├── calibrated_risk_profiler.pkl
│   ├── shap_explainer.pkl
│   ├── ohe_encoder.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   ├── ood_detector.pkl
│   ├── ood_threshold.pkl
│   ├── training_stats.pkl
│   └── manifest.json
├── AGENT1_RISK_PROFILER.md         ← This file
└── .venv/                          ← Python virtual environment
```

---

## 4. Prerequisites & Installation

### System requirements

- macOS (Apple Silicon or Intel) — tested on macOS with Apple M-series
- Python 3.9.6
- `libomp` (required by XGBoost on macOS): `brew install libomp`

### Environment setup

```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate

pip install xgboost scikit-learn shap pandas numpy joblib \
            fastapi "uvicorn[standard]" pydantic
```

### Required data file

Place the raw dataset at:
```
data/raw/insurance_data.csv
```
Expected: **146,259 rows × 25 columns** including `Prev_Accidents`, `Prev_Citations`,
`Driving_Exp`, `Driver_Age`, `Annual_Miles_Range`, `Veh_Usage`.

---

## 5. Training the Model

Run the full training pipeline with one command:

```bash
cd /path/to/Quote-Agents
.venv/bin/python agents/agent1_risk_profiler.py
```

**What happens during training (in order):**

| Step | Action |
|---|---|
| 0 | `shutil.rmtree` wipes `models/` and `data/processed/` entirely |
| 1 | Load CSV, map mileage ranges to numeric midpoints, fill nulls |
| 2 | Generate noisy Risk_Tier labels (Gaussian noise σ=2.0 on actuarial score) |
| 3a | Add 3 interaction features (`Miles_Per_Exp`, `Total_Incidents`, `Age_Exp_Gap`) |
| 3b | OneHotEncode `Veh_Usage` → 11-column feature matrix |
| 4 | Three-way stratified split: 64% subtrain / 16% calibration / 20% test |
| 4a | Train `IsolationForest` on 8 raw features; compute 0.01th-pct threshold |
| 4a+ | Capture `training_stats` (feature means) for drift monitoring |
| 4b | Compute cost-sensitive sample weights (High=3×, Medium=2×, Low=1×) |
| 5 | `RandomizedSearchCV` — 50 iterations, 5-fold StratifiedKFold, `neg_log_loss` |
| 6 | `CalibratedClassifierCV(isotonic, prefit)` on calibration split |
| 7 | Evaluate on held-out test set (accuracy, balanced accuracy, log-loss, F1) |
| 8 | Build SHAP `TreeExplainer` on base `XGBClassifier` (500-row background) |
| 9a | Demo predictions through the full pipeline |
| 9b | Adversarial red-team test on 5 impossible profiles |
| 11 | Export 8 artifacts + manifest.json + processed CSV |

**Expected runtime:** ~4 minutes on Apple M-series CPU (250 total fits).

---

## 6. Model Performance

> Results from the last training run — March 5, 2026.

### Overall metrics

| Metric | Value |
|---|---|
| **Test Accuracy** | **79.10%** |
| **Balanced Accuracy** | 55.02% |
| **Log-Loss (calibrated)** | **0.4944** |
| **Training set size** | 93,605 rows |
| **Calibration set size** | 23,402 rows |
| **Test set size** | 29,252 rows |

### Per-class performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **High** | 0.470 | **0.651** | 0.546 | 1,763 |
| **Low** | 0.821 | **0.987** | 0.897 | 22,211 |
| **Medium** | 0.575 | 0.012 | 0.024 | 5,278 |
| *macro avg* | 0.622 | 0.550 | 0.489 | 29,252 |
| *weighted avg* | 0.756 | 0.791 | 0.718 | 29,252 |

### Confusion matrix

```
                Predicted
Actual    High    Low   Medium
High      1148    590       25
Low        263  21925       23
Medium    1032   4181       65
```

### Best hyperparameters

Found by `RandomizedSearchCV(n_iter=50, scoring="neg_log_loss", cv=StratifiedKFold(5))`.

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 7 |
| `learning_rate` | 0.01 |
| `subsample` | 0.8 |
| `colsample_bytree` | 1.0 |
| `min_child_weight` | 5 |
| `gamma` | 0.3 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.5 |
| `max_delta_step` | 1 |

### Label distribution (training data)

| Class | Count | % |
|---|---|---|
| Low | 111,057 | 75.9% |
| Medium | 26,388 | 18.0% |
| High | 8,814 | 6.0% |

---

## 7. Feature Engineering

### Raw input features (6)

These are the fields a caller must supply:

| Feature | Type | Description |
|---|---|---|
| `Prev_Accidents` | int | Prior at-fault accidents |
| `Prev_Citations` | int | Prior traffic citations |
| `Driving_Exp` | int | Years of driving experience |
| `Driver_Age` | int | Age in years |
| `Annual_Miles` | int | Estimated annual mileage |
| `Veh_Usage` | str | `"Business"`, `"Commute"`, or `"Pleasure"` |

### Interaction features (3) — computed automatically

These are derived at inference time from the raw inputs. The caller never supplies them.

| Feature | Formula | Actuarial Meaning |
|---|---|---|
| `Miles_Per_Exp` | `Annual_Miles / (Driving_Exp + 1)` | Exposure density — high miles with low experience signals elevated risk |
| `Total_Incidents` | `Prev_Accidents + Prev_Citations` | Combined incident load — single number capturing overall record severity |
| `Age_Exp_Gap` | `Driver_Age − Driving_Exp − 16` | Delayed licensing signal — gap between legal driving age and when the driver actually started |

### One-Hot Encoded feature (3 columns)

`Veh_Usage` is expanded into three binary columns by `OneHotEncoder(sparse_output=False, handle_unknown="ignore")`:

- `Veh_Usage_Business`
- `Veh_Usage_Commute`
- `Veh_Usage_Pleasure`

### Full feature vector (11 columns in model order)

```python
["Prev_Accidents", "Prev_Citations", "Driving_Exp", "Driver_Age", "Annual_Miles",
 "Miles_Per_Exp", "Total_Incidents", "Age_Exp_Gap",
 "Veh_Usage_Business", "Veh_Usage_Commute", "Veh_Usage_Pleasure"]
```

### Label generation (training only)

Labels are **not read from the CSV**. They are generated synthetically to prevent data leakage from a deterministic formula:

1. Compute a deterministic actuarial score from the 6 raw features.
2. Add Gaussian noise: `score += rng.normal(0.0, σ=2.0)`.
3. Bucket into tiers: High (score ≥ 7), Medium (score ≥ 4), Low (score < 4).

The noise forces the model to learn probability distributions rather than memorising a formula.

---

## 8. Safety System

### Layer 2 — Deterministic Physics Check

Applied **before** any ML scoring. Rejects inputs that violate physical or legal constraints:

```python
if Driver_Age < 16:            → BLOCKED  # below legal driving age
if Annual_Miles < 0:           → BLOCKED  # negative mileage impossible
if Prev_Accidents < 0:         → BLOCKED  # negative count impossible
if Prev_Citations < 0:         → BLOCKED  # negative count impossible
if Driving_Exp > (Driver_Age - 15): → BLOCKED  # more exp than lifetime allows
```

Return:
```json
{
  "status": "ACTION_REQUIRED: DATA_ANOMALY",
  "message": "Deterministic Physics Check Failed: Logically impossible driver inputs detected. Driver_Age=-5.0 < 16 (below legal driving age)"
}
```

### Layer 3 — IsolationForest OOD Gate

- Trained on **8 raw features only** (5 numeric + 3 OHE columns from `Veh_Usage`).
- Interaction features are **excluded** from OOD training — extreme values like `Miles_Per_Exp = 9,999,999 / 1` saturate tree-path depth and make corrupt data appear normal.
- Threshold: **0.01th percentile** of training anomaly scores = `−0.710535`.
- Only quotes whose score falls below 99.99% of all training quotes are blocked.

```
Normal quote  → score > −0.710535 → passes to XGBoost
Anomaly quote → score < −0.710535 → blocked, returns OOD_FLAG
```

### Adversarial Red-Team Test

Runs automatically at the end of every training run. Tests 5 adversarial profiles:

| Profile | Attack Type |
|---|---|
| `age=18, exp=40` | Impossible experience for age |
| `Annual_Miles=−500` | Negative mileage |
| `age=−5, miles=9,999,999` | Data entry corruption |
| `Driver_Age=0` | Pre-birth driver |
| `Prev_Accidents=−3` | Negative incident count |

The Physics Check (Layer 2) handles sign violations and impossible age/experience gaps.
The IsolationForest (Layer 3) handles statistically extreme outliers.

---

## 9. Artifact Reference

All 8 artifacts are written to `models/` on every training run. The directory is wiped first, so no stale files can persist.

| File | Size | Contents |
|---|---|---|
| `calibrated_risk_profiler.pkl` | 2.2 MB | `CalibratedClassifierCV` wrapping the best `XGBClassifier` |
| `shap_explainer.pkl` | 4.1 MB | `TreeExplainer` bound to the base `XGBClassifier` |
| `ohe_encoder.pkl` | < 1 KB | Fitted `OneHotEncoder` for `Veh_Usage` |
| `label_encoder.pkl` | < 1 KB | Fitted `LabelEncoder` mapping High/Low/Medium ↔ 0/1/2 |
| `feature_names.pkl` | < 1 KB | Ordered list of 11 feature names |
| `ood_detector.pkl` | 951 KB | Fitted `IsolationForest(n_estimators=200)` |
| `ood_threshold.pkl` | < 1 KB | Float: `−0.710535` (0.01th percentile of training scores) |
| `training_stats.pkl` | < 1 KB | Feature means dict for drift monitoring |
| `manifest.json` | 1.5 KB | Human-readable pipeline record — version, params, artifact list |

### Loading artifacts (Python)

```python
from agents.agent1_risk_profiler import RiskProfilerPredictor

predictor = RiskProfilerPredictor.from_artifacts("models/")
result    = predictor.predict_and_explain({
    "Driver_Age": 30, "Driving_Exp": 8,
    "Prev_Accidents": 0, "Prev_Citations": 1,
    "Annual_Miles": 28_000, "Veh_Usage": "Commute",
})
```

---

## 10. The Production API

`app.py` wraps the predictor in a FastAPI service with full defensive programming.

### Starting the server

```bash
# Requires models/ to be populated first (run the training script once)
cd /path/to/Quote-Agents

# Production
uvicorn app:app --host 0.0.0.0 --port 8000

# Development (auto-reload)
uvicorn app:app --reload --port 8000

# Or directly
.venv/bin/python app.py
```

The server loads all 8 artifacts once at startup via FastAPI's `lifespan` context manager. Artifact loading happens **once** — all requests share the same singleton predictor.

### Interactive API docs

Once running, visit:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 11. API Endpoint Reference

### `GET /health`

Returns model version and artifact inventory from `manifest.json`.

**Response (200 OK):**
```json
{
  "status": "OK",
  "agent": "Agent 1 – Risk Profiler v3",
  "model_type": "CalibratedClassifierCV(isotonic, cv=prefit) → XGBClassifier",
  "n_features": 11,
  "classes": ["High", "Low", "Medium"],
  "ood_detector": "IsolationForest(n_estimators=200)",
  "artifacts": ["calibrated_risk_profiler.pkl", "..."]
}
```

**Response (503):** `manifest.json` not found — training script has not been run.

---

### `POST /predict/risk`

Full five-layer inference pipeline.

**Request body (`QuoteRequest`):**

```json
{
  "Driver_Age":     30,
  "Driving_Exp":     8,
  "Prev_Accidents":  0,
  "Prev_Citations":  1,
  "Annual_Miles":  28000,
  "Veh_Usage":    "Commute"
}
```

**Field constraints (Pydantic v2):**

| Field | Type | Constraint |
|---|---|---|
| `Driver_Age` | `int` | `16 ≤ age ≤ 100` |
| `Driving_Exp` | `int` | `0 ≤ exp ≤ 84` and `exp ≤ Driver_Age − 16` |
| `Prev_Accidents` | `int` | `0 ≤ value ≤ 20` |
| `Prev_Citations` | `int` | `0 ≤ value ≤ 20` |
| `Annual_Miles` | `int` | `0 ≤ value ≤ 200,000` |
| `Veh_Usage` | `Literal` | `"Business"`, `"Commute"`, or `"Pleasure"` |

**Validation failure (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "Driving_Exp"],
      "msg": "Value error, Driving_Exp (40) cannot exceed Driver_Age − 16 = 1."
    }
  ]
}
```

---

## 12. Response Contract

### Status values

| `status` | Meaning | HTTP Code |
|---|---|---|
| `OK` | Normal prediction — confidence ≥ 0.60 | 200 |
| `LOW_CONFIDENCE_ESCALATE` | Prediction returned but confidence < 0.60 — route to human underwriter | 200 |
| `ACTION_REQUIRED: DATA_ANOMALY_ESCALATE` | Physics Check or IsolationForest blocked the input | 422 |
| `ERROR` | Unhandled server exception | 500 |

### Success response (`OK` or `LOW_CONFIDENCE_ESCALATE`)

```json
{
  "status": "OK",
  "predicted_tier": "Low",
  "confidence": 0.9444,
  "class_probabilities": {
    "High": 0.0001,
    "Low":  0.9444,
    "Medium": 0.0554
  },
  "top_3_features": [
    {
      "feature":    "Driving_Exp",
      "shap_value": 0.3561,
      "direction":  "↑ increases risk",
      "magnitude":  "HIGH"
    },
    {
      "feature":    "Annual_Miles",
      "shap_value": 0.2025,
      "direction":  "↑ increases risk",
      "magnitude":  "HIGH"
    },
    {
      "feature":    "Total_Incidents",
      "shap_value": 0.1432,
      "direction":  "↑ increases risk",
      "magnitude":  "MEDIUM"
    }
  ],
  "escalation_reason": null,
  "dashboard_metrics": {
    "drift_status": {
      "status":        "OK: NO_DRIFT_DETECTED",
      "feature":       "Annual_Miles",
      "training_mean": 23269.07,
      "incoming_mean": 28000.0,
      "pct_shift_pct": 20.33,
      "n_samples":     1
    },
    "counterfactual_advice": null
  }
}
```

### Low-confidence response

```json
{
  "status": "LOW_CONFIDENCE_ESCALATE",
  "predicted_tier": "Medium",
  "confidence": 0.4821,
  "escalation_reason": "Model confidence 48.2% is below the 60% threshold. This is a borderline case — route to a human underwriter for manual review.",
  "dashboard_metrics": { "..." }
}
```

### OOD / anomaly response (422)

```json
{
  "status":  "ACTION_REQUIRED: DATA_ANOMALY_ESCALATE",
  "message": "Deterministic Physics Check Failed: Logically impossible driver inputs detected. Driver_Age=-5.0 < 16 (below legal driving age)",
  "input":   { "Driver_Age": -5, "..." }
}
```

### SHAP magnitude guide

| `magnitude` | Condition |
|---|---|
| `HIGH` | `|SHAP| > 0.15` |
| `MEDIUM` | `0.05 < |SHAP| ≤ 0.15` |
| `LOW` | `|SHAP| ≤ 0.05` |

---

## 13. Observability & Drift Monitoring

`calculate_feature_drift(incoming_batch, training_stats)` monitors the `Annual_Miles` distribution of live traffic against the training baseline.

**Reference baseline (from training):**

| Feature | Training Mean |
|---|---|
| `Annual_Miles` | 23,269 mi/yr |
| `Driver_Age` | 41.51 yrs |
| `Driving_Exp` | 24.51 yrs |
| `Prev_Accidents` | 0.111 |
| `Prev_Citations` | 0.111 |

**Alert logic:**

```
|  (incoming_mean - training_mean) / training_mean  | > 10%
    → status: "SYSTEM_ALERT: DATA_DRIFT_DETECTED"
```

**Drift response fields:**

```json
{
  "status":        "SYSTEM_ALERT: DATA_DRIFT_DETECTED",
  "feature":       "Annual_Miles",
  "training_mean": 23269.07,
  "incoming_mean": 50000.0,
  "pct_shift_pct": 114.9,
  "n_samples":     10
}
```

Every `POST /predict/risk` call includes a `drift_status` object inside `dashboard_metrics`. A monitoring system can poll for `"SYSTEM_ALERT"` in the response to trigger a retraining workflow.

---

## 14. Counterfactual What-If Analyzer

`RiskProfilerPredictor.generate_counterfactual_advice(quote_data, current_tier)` finds the minimum change that would improve a High or Medium risk customer to Low.

**Algorithm — two levers probed in sequence:**

**Lever A — Mileage Reduction:**
Iterates `Annual_Miles` down in 1,000 mi steps (floor: 1,000 mi, max 200 iterations) until `predicted_tier == "Low"`.

**Lever B — Incident Record Clearing:**
Sets both `Prev_Accidents` and `Prev_Citations` to zero and re-predicts.

**Fallback:**
If neither lever achieves a tier improvement, generic guidance is returned. The dashboard always has actionable text.

**Example outputs:**

```
"Reducing annual mileage from 40,000 to 12,000 miles would likely
 transition this profile to a Low Risk tier."

"Clearing the 2 incident record(s) (accidents + citations) would
 likely transition this profile to a Low Risk tier."

"This profile is currently classified as High Risk. Sustained
 incident-free driving and a gradual reduction in annual mileage
 are the strongest levers for improving the risk tier over time."
```

---

## 15. Configuration Reference

All constants are defined at the top of `agents/agent1_risk_profiler.py`.

### Training constants

| Constant | Value | Description |
|---|---|---|
| `RANDOM_STATE` | `42` | Global random seed |
| `TEST_SIZE` | `0.20` | 20% held-out test split |
| `CALIB_SIZE` | `0.20` | 20% of train-val for isotonic calibration |
| `NOISE_SCALE` | `2.0` | Gaussian noise σ on actuarial score |

### Economic layer

| Constant | Value | Description |
|---|---|---|
| `HIGH_RISK_WEIGHT_MULTIPLIER` | `3.0` | Sample weight multiplier for High class |
| `MEDIUM_RISK_WEIGHT_MULTIPLIER` | `2.0` | Sample weight multiplier for Medium class |

### OOD detector

| Constant | Value | Description |
|---|---|---|
| `OOD_N_ESTIMATORS` | `200` | IsolationForest tree count |
| `OOD_SCORE_PERCENTILE` | `0.01` | 0.01th percentile threshold (blocks only extreme outliers) |
| `OOD_FLAG` | `"ACTION_REQUIRED: DATA_ANOMALY"` | Status string on anomaly detection |

### Drift monitor

| Constant | Value | Description |
|---|---|---|
| `DRIFT_THRESHOLD_PCT` | `0.10` | 10% mean shift triggers alert |
| `DRIFT_ALERT_STATUS` | `"SYSTEM_ALERT: DATA_DRIFT_DETECTED"` | Alert status string |

### Counterfactual search

| Constant | Value | Description |
|---|---|---|
| `CF_MILES_STEP` | `1,000` | Mileage reduction step size per iteration |
| `CF_MAX_ITER` | `200` | Maximum search iterations (safety cap) |

### API (app.py)

| Constant | Value | Description |
|---|---|---|
| `CONFIDENCE_GATE` | `0.60` | Predictions below this → `LOW_CONFIDENCE_ESCALATE` |
| `LOW_CONF_STATUS` | `"LOW_CONFIDENCE_ESCALATE"` | Status string for borderline cases |
| `OOD_ESCALATE_STATUS` | `"ACTION_REQUIRED: DATA_ANOMALY_ESCALATE"` | API-layer OOD status |

---

## 16. Key Design Decisions

### Why Gaussian noise on labels?
A deterministic label formula produces 100% training accuracy — the model memorises the formula instead of learning insurance risk. Adding `σ=2.0` Gaussian noise forces the model to learn probability distributions, producing realistic calibrated outputs (e.g., 72% High rather than 100%).

### Why `neg_log_loss` for hyperparameter search?
Accuracy rewards confident correct predictions but ignores calibration quality. Log-loss penalises *confidently wrong* predictions more heavily, which is exactly what an insurance model needs — a wrong but confident tier assignment is more damaging than an uncertain one.

### Why `CalibratedClassifierCV(isotonic, prefit)`?
Raw XGBoost probabilities are not reliably calibrated on imbalanced data. Isotonic regression (non-parametric) on a **held-out** calibration split corrects the probability estimates without touching the model's decision boundaries.

### Why train IsolationForest on raw features only?
Interaction features like `Miles_Per_Exp = Annual_Miles / (Driving_Exp + 1)` amplify corrupt inputs — `9,999,999 / 1 = 9,999,999`. This saturates IsolationForest's path-length depth, making the corrupt record look statistically "average". Training on the 8 raw features preserves the extreme signal in individual values like `Annual_Miles = 9,999,999`.

### Why a two-layer safety gate?
- The **Physics Check** (Layer 2) handles violations detectable by logic alone — negative values, impossible age/experience combinations. It requires no training data and cannot be fooled.
- The **IsolationForest** (Layer 3) handles statistically anomalous inputs that look syntactically valid — unusual but physically plausible combinations that sit far outside the training distribution.

### Why `shutil.rmtree` instead of per-file deletion?
Listing specific files to delete requires the cleanup function to be updated every time an artifact is added or renamed. `rmtree` guarantees that no renamed or orphaned file from any previous version can persist and be silently loaded.

### Why a singleton predictor in the API?
Each `joblib.load` of the 6+ MB SHAP explainer takes ~1 second. Loading on every request would cap throughput at ~1 req/s. The `lifespan` context manager loads everything once at startup; all requests share a single in-memory predictor instance.
