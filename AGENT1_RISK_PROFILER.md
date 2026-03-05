# Agent 1 – Risk Profiler
### Auto Insurance Multi-Agent Pipeline

> **Script:** `agent1_risk_profiler.py`
> **Role in Pipeline:** First agent. Receives a raw quote, assigns a real-time **Risk Tier (Low / Medium / High)**, and returns an explainable prediction to downstream agents.
> **Date Trained:** March 5, 2026
> **Dataset:** `Autonomous QUOTE AGENTS.csv` — 146,259 quotes

---

## Table of Contents

1. [Overview](#1-overview)
2. [Directory Structure](#2-directory-structure)
3. [Dependencies](#3-dependencies)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Step-by-Step Breakdown](#5-step-by-step-breakdown)
   - [Step 1 – Data Loading & Cleaning](#step-1--data-loading--cleaning)
   - [Step 2 – Risk Tier Label Generation](#step-2--risk-tier-label-generation)
   - [Step 3 – Feature Engineering](#step-3--feature-engineering)
   - [Step 4 – Encoding](#step-4--encoding)
   - [Step 5 – Sample Weights](#step-5--sample-weights)
   - [Step 6 – Hyperparameter Tuning](#step-6--hyperparameter-tuning)
   - [Step 7 – Evaluation](#step-7--evaluation)
   - [Step 8 – SHAP Explainer](#step-8--shap-explainer)
   - [Step 9 – `explain_risk_prediction()`](#step-9--explain_risk_prediction)
   - [Step 10 – `RiskProfilerPredictor` Class](#step-10--riskprofilerpreditor-class)
   - [Step 11 – Artifact Export](#step-11--artifact-export)
6. [Model Results](#6-model-results)
7. [Exported Artifacts](#7-exported-artifacts)
8. [Feature Reference](#8-feature-reference)
9. [Inference API Reference](#9-inference-api-reference)
10. [CrewAI Integration Guide](#10-crewai-integration-guide)
11. [Configuration Constants](#11-configuration-constants)

---

## 1. Overview

Agent 1 is the **risk classification layer** of the multi-agent quoting pipeline. It takes the six core driver/vehicle features from an insurance quote and outputs:

- A **Risk Tier** label: `Low`, `Medium`, or `High`
- A **confidence score** (softmax probability of the predicted class)
- **Per-class probabilities** for all three tiers
- A **SHAP-based explanation** — the top 3 features that most influenced the specific decision, with direction and magnitude

Since the raw dataset does not contain a pre-existing `Risk_Tier` column, the agent generates ground-truth labels using a **domain-weighted actuarial scoring function** before training.

---

## 2. Directory Structure

```
Quote-Agents/
├── Autonomous QUOTE AGENTS.csv     ← Raw dataset (146,259 rows)
├── agent1_risk_profiler.py         ← This script (training + inference)
├── AGENT1_RISK_PROFILER.md         ← This documentation
└── models/                         ← All exported artifacts
    ├── xgb_risk_profiler.pkl       ← Trained XGBClassifier
    ├── shap_explainer.pkl          ← SHAP TreeExplainer
    ├── ohe_encoder.pkl             ← OneHotEncoder (Veh_Usage)
    ├── label_encoder.pkl           ← LabelEncoder (Risk Tier ↔ int)
    ├── feature_names.pkl           ← Ordered feature list (14 features)
    └── manifest.json               ← Human-readable artifact manifest
```

---

## 3. Dependencies

| Package | Version Used | Purpose |
|---|---|---|
| `pandas` | 2.3.3 | Data loading, manipulation, feature engineering |
| `numpy` | — | Numerical operations, sample weight arrays |
| `xgboost` | 2.1.4 | Core gradient boosting classifier |
| `scikit-learn` | 1.6.1 | Encoding, splitting, cross-validation, metrics |
| `shap` | 0.49.1 | SHAP TreeExplainer for prediction explainability |
| `joblib` | — | Serialization of all model artifacts |

> **macOS note:** XGBoost requires OpenMP. Install it with `brew install libomp` before running the script.

Install all Python dependencies:
```bash
pip install pandas numpy xgboost scikit-learn shap joblib
```

---

## 4. Pipeline Architecture

```
CSV Dataset
    │
    ▼
load_and_prepare_data()        ← Step 1: load, map miles range → numeric, fill nulls
    │
    ▼
assign_risk_tier()             ← Step 2: generate Risk_Tier labels (actuarial scoring)
    │
    ▼
engineer_features()            ← Step 3: create 6 new interaction/ratio features
    │
    ▼
encode_features()              ← Step 4: OHE for Veh_Usage, LabelEncode target
    │
    ▼
compute_sample_weights()       ← Step 5: inverse-frequency weights for imbalance
    │
    ▼
train_model()                  ← Step 6: RandomizedSearchCV (50 iter, 5-fold StratKFold)
    │
    ▼
evaluate_model()               ← Step 7: classification report + confusion matrix
    │
    ▼
build_shap_explainer()         ← Step 8: TreeExplainer (interventional, 500-row BG)
    │
    ▼
export_artifacts()             ← Step 11: save .pkl + manifest.json to ./models/
    │
    ▼
RiskProfilerPredictor          ← Step 10: CrewAI-ready inference wrapper
```

---

## 5. Step-by-Step Breakdown

### Step 1 – Data Loading & Cleaning

**Function:** `load_and_prepare_data(path)`

The raw dataset uses range strings for annual mileage (e.g. `"> 25 K & <= 35 K"`). These are mapped to numeric midpoint values for use in calculations:

| Range String | Midpoint Used |
|---|---|
| `<= 7.5 K` | 7,500 |
| `> 7.5 K & <= 15 K` | 11,250 |
| `> 15 K & <= 25 K` | 20,000 |
| `> 25 K & <= 35 K` | 30,000 |
| `> 35 K & <= 45 K` | 40,000 |
| `> 45 K & <= 55 K` | 50,000 |
| `> 55 K` | 62,500 |

**Null handling strategy:**
- All numeric columns: coerce to numeric, fill nulls with **column median** (robust to outliers)
- `Veh_Usage` (categorical): fill nulls with `"Commute"` (most common value)

---

### Step 2 – Risk Tier Label Generation

**Functions:** `_score_row(row)` → `assign_risk_tier(row)`

Since no `Risk_Tier` column exists in the raw data, ground-truth labels are generated using a **weighted actuarial scoring model** inspired by auto-insurance loss-ratio research.

#### Scoring Weights

| Factor | Condition | Points |
|---|---|---|
| Prior accident | `Prev_Accidents = 1` | **+4** |
| Prior citation | `Prev_Citations = 1` | **+2** |
| Driving experience | ≤ 3 years | **+3** |
| Driving experience | ≤ 7 years | **+2** |
| Driving experience | ≤ 15 years | **+1** |
| Driver age | < 22 years | **+2** |
| Driver age | < 26 years | **+1** |
| Annual mileage | > 45,000 mi/yr | **+2** |
| Annual mileage | > 25,000 mi/yr | **+1** |
| Vehicle usage | `Business` | **+1** |

#### Tier Thresholds

| Score | Risk Tier | Rationale |
|---|---|---|
| ≥ 7 | **High** | ≥2 major risk factors, or 1 incident + aggravating conditions |
| ≥ 4 | **Medium** | 1 moderate risk factor, or compounding exposure |
| < 4 | **Low** | Clean record, experienced, low-exposure driver |

#### Dataset Distribution (146,259 rows)

| Tier | Count | Share |
|---|---|---|
| Low | 111,223 | 76.0% |
| Medium | 28,531 | 19.5% |
| High | 6,505 | 4.4% |

> The dataset is **significantly imbalanced** — this is handled via inverse-frequency sample weights in Step 5.

---

### Step 3 – Feature Engineering

**Function:** `engineer_features(df)`

Six new features are created to expose non-linear interactions and risk ratios that the base columns alone cannot express:

| Feature | Formula | Rationale |
|---|---|---|
| `Total_Incidents` | `Prev_Accidents + Prev_Citations` | Simple incident count; useful as standalone and interaction base |
| `Incident_Score` | `Prev_Accidents × 2 + Prev_Citations` | Actuarially weighted severity: accidents 2× more costly than citations |
| `Miles_Per_Exp_Year` | `Annual_Miles / (Driving_Exp + 1)` | Exposure-normalised driving load; high value = high-risk lifestyle |
| `Risk_Exposure_Index` | `(Incident_Score + 1) × Annual_Miles / (Driving_Exp + 1)` | Compound effect of bad record × heavy use × novice driver |
| `Young_Inexperienced` | `1 if (Driver_Age < 25 AND Driving_Exp < 5) else 0` | Binary flag for high-claim teen/early-adult novice drivers |
| `Age_Exp_Gap` | `Driver_Age - Driving_Exp - 16` | Deviation from expected experience at age (US licensing age = 16); positive gap → late start or suspended licence |

---

### Step 4 – Encoding

**Function:** `encode_features(df, engineered, fit, ohe, le)`

| Column | Treatment | Notes |
|---|---|---|
| `Veh_Usage` | **One-Hot Encoding** | Produces: `Veh_Usage_Business`, `Veh_Usage_Commute`, `Veh_Usage_Pleasure`. `handle_unknown="ignore"` for safe inference on unseen categories. |
| `Risk_Tier` | **Label Encoding** | `High=0, Low=1, Medium=2` (alphabetical) |

The function accepts `fit=False` + pre-fitted encoders to safely transform new inference data without re-fitting.

**Final feature set (14 features):**
```
Prev_Accidents, Prev_Citations, Driving_Exp, Driver_Age, Annual_Miles,
Total_Incidents, Incident_Score, Miles_Per_Exp_Year, Risk_Exposure_Index,
Young_Inexperienced, Age_Exp_Gap,
Veh_Usage_Business, Veh_Usage_Commute, Veh_Usage_Pleasure
```

---

### Step 5 – Sample Weights

**Function:** `compute_sample_weights(y)`

Because the dataset is highly imbalanced (76% Low, 19.5% Medium, 4.4% High), each training sample is assigned an **inverse-frequency weight**:

$$w_c = \frac{N}{K \times n_c}$$

Where $N$ = total samples, $K$ = number of classes, $n_c$ = samples in class $c$.

This ensures the model treats every Risk Tier as equally important, preventing it from ignoring the rare `High` class.

**Resulting weight range:** `0.4383 – 7.4947`

---

### Step 6 – Hyperparameter Tuning

**Function:** `train_model(X_train, y_train, n_classes, sample_weights, n_iter=50)`

**Search strategy:** `RandomizedSearchCV` — samples 50 random parameter combinations, each evaluated with **5-fold StratifiedKFold** cross-validation (stratification preserves Risk Tier ratios in every fold).

**Search space:**

| Parameter | Values Searched | Role |
|---|---|---|
| `n_estimators` | 200, 300, 400, 500, 600 | Number of boosting rounds |
| `max_depth` | 4, 5, 6, 7, 8 | Tree depth (complexity vs. overfit) |
| `learning_rate` | 0.01, 0.05, 0.08, 0.10, 0.15, 0.20 | Shrinkage per step |
| `subsample` | 0.70, 0.80, 0.90, 1.00 | Row sub-sampling per tree |
| `colsample_bytree` | 0.70, 0.80, 0.90, 1.00 | Feature sub-sampling per tree |
| `min_child_weight` | 1, 3, 5, 7 | Min sum of instance weights in a leaf |
| `gamma` | 0, 0.05, 0.10, 0.20, 0.30 | Min loss reduction for a split |
| `reg_alpha` | 0, 0.01, 0.05, 0.10, 0.50 | L1 regularisation |
| `reg_lambda` | 0.5, 1.0, 1.5, 2.0, 3.0 | L2 regularisation |
| `max_delta_step` | 0, 1, 5 | Convergence aid for imbalanced data |

**Fixed settings:** `objective="multi:softprob"`, `eval_metric="mlogloss"`, `tree_method="hist"`, `device="cpu"`

**Best parameters found:**
```json
{
    "subsample": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0,
    "n_estimators": 500,
    "min_child_weight": 1,
    "max_depth": 8,
    "max_delta_step": 0,
    "learning_rate": 0.08,
    "gamma": 0,
    "colsample_bytree": 1.0
}
```

**Search time:** ~195 seconds (250 total fits: 50 candidates × 5 folds)

---

### Step 7 – Evaluation

**Function:** `evaluate_model(model, X_test, y_test, le)`

**Train/test split:** 80% train (117,007 rows) / 20% test (29,252 rows), stratified.

#### Results on Held-Out Test Set

| Metric | Score |
|---|---|
| **Test Accuracy** | **100.00%** |
| **Balanced Accuracy** | **100.00%** |

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| High | 1.0000 | 1.0000 | 1.0000 | 1,301 |
| Low | 1.0000 | 1.0000 | 1.0000 | 22,245 |
| Medium | 1.0000 | 1.0000 | 1.0000 | 5,706 |

**Confusion Matrix:**

| | Pred: High | Pred: Low | Pred: Medium |
|---|---|---|---|
| **Actual: High** | 1,301 | 0 | 0 |
| **Actual: Low** | 0 | 22,245 | 0 |
| **Actual: Medium** | 0 | 0 | 5,706 |

> **Note on 100% accuracy:** The labels were generated by a deterministic scoring function applied to the same feature columns the model trains on. XGBoost with sufficient depth can perfectly learn this deterministic mapping, making 100% accuracy on both train and test expected and valid. In production, if real-world claim outcomes are available, retraining on those labels is recommended.

---

### Step 8 – SHAP Explainer

**Function:** `build_shap_explainer(model, X_background)`

- **Method:** `shap.TreeExplainer` with `feature_perturbation="interventional"`
- **Background dataset:** Random sample of 500 training rows (capped for inference speed)
- **Output shape:** `(n_samples, n_features, n_classes)` — one SHAP value per feature per class per sample

Interventional perturbation breaks feature dependencies by marginalising over the background distribution, producing causally-interpretable SHAP values rather than correlation-driven ones.

---

### Step 9 – `explain_risk_prediction()`

**Signature:**
```python
explain_risk_prediction(
    quote_data: Dict[str, Any],
    *,
    model, explainer, ohe, le, feature_names, engineered_features
) -> Dict[str, Any]
```

**Input `quote_data` keys:**

| Key | Type | Description |
|---|---|---|
| `Prev_Accidents` | `int` | 0 or 1 — prior accident on record |
| `Prev_Citations` | `int` | 0 or 1 — prior traffic citation on record |
| `Driving_Exp` | `int` | Years of licensed driving experience |
| `Driver_Age` | `int` | Driver's age in years |
| `Annual_Miles` | `int` | Estimated annual mileage (**numeric**, not a range string) |
| `Veh_Usage` | `str` | `"Commute"` \| `"Pleasure"` \| `"Business"` |

**Output dictionary:**

| Key | Type | Description |
|---|---|---|
| `predicted_tier` | `str` | `"Low"`, `"Medium"`, or `"High"` |
| `predicted_class_id` | `int` | Encoded class index (0=High, 1=Low, 2=Medium) |
| `confidence` | `float` | Softmax probability of the predicted class (0.0–1.0) |
| `class_probabilities` | `dict` | `{"High": p, "Low": p, "Medium": p}` for all 3 classes |
| `top_3_features` | `list` | Top 3 SHAP-ranked features (see structure below) |
| `all_shap_values` | `dict` | `{feature: shap_value}` for all 14 features |

**`top_3_features` item structure:**

```json
{
  "feature":    "Risk_Exposure_Index",
  "shap_value": 3.9642,
  "direction":  "↑ increases risk",
  "magnitude":  "HIGH"
}
```

**Magnitude thresholds:**

| Magnitude | Condition |
|---|---|
| `HIGH` | `|SHAP| > 0.15` |
| `MEDIUM` | `0.05 < |SHAP| ≤ 0.15` |
| `LOW` | `|SHAP| ≤ 0.05` |

---

### Step 10 – `RiskProfilerPredictor` Class

A self-contained wrapper that loads all `.pkl` artifacts from disk and exposes a single clean inference method — designed for drop-in use as a **CrewAI custom tool**.

```python
class RiskProfilerPredictor:
    @classmethod
    def from_artifacts(cls, model_dir: str = "./models/") -> "RiskProfilerPredictor"
    def predict_and_explain(self, quote_data: Dict[str, Any]) -> Dict[str, Any]
```

Internally reconstructs all 6 engineered features from the raw input dict, applies the saved OHE encoder, assembles the 14-feature vector in the exact training order, and calls `explain_risk_prediction()`.

---

### Step 11 – Artifact Export

**Function:** `export_artifacts(model, explainer, ohe, le, feature_names, model_dir)`

All objects are serialized with `joblib.dump(obj, path, compress=3)` (zlib level-3 compression — good balance of size vs. speed).

---

## 6. Model Results

| Metric | Value |
|---|---|
| Best CV Accuracy (5-fold) | **1.0000** |
| Test Accuracy | **100.00%** |
| Balanced Accuracy | **100.00%** |
| Search Duration | ~195 seconds |
| Search Candidates | 50 iterations × 5 folds = 250 fits |
| Training Rows | 117,007 |
| Test Rows | 29,252 |

### Demo Predictions

**Customer 1 — High Risk** *(accident + citation, young, business use)*
```
Predicted Tier  : High
Confidence      : 100.00%
Top 3 Drivers:
  • Total_Incidents       SHAP = +4.7667  [↑ increases risk]  [HIGH]
  • Risk_Exposure_Index   SHAP = +3.9642  [↑ increases risk]  [HIGH]
  • Driving_Exp           SHAP = +3.6475  [↑ increases risk]  [HIGH]
```

**Customer 2 — Low Risk** *(clean record, 20 yrs experience, pleasure driver)*
```
Predicted Tier  : Low
Confidence      : 100.00%
Top 3 Drivers:
  • Driving_Exp           SHAP = +1.6854  [↑ increases risk]  [HIGH]
  • Prev_Accidents        SHAP = +1.1204  [↑ increases risk]  [HIGH]
  • Risk_Exposure_Index   SHAP = +0.9851  [↑ increases risk]  [HIGH]
```

**Customer 3 — Medium Risk** *(one citation, commuter, moderate mileage)*
```
Predicted Tier  : Medium
Confidence      : 100.00%
Top 3 Drivers:
  • Risk_Exposure_Index   SHAP = +3.7997  [↑ increases risk]  [HIGH]
  • Total_Incidents       SHAP = +3.1529  [↑ increases risk]  [HIGH]
  • Driving_Exp           SHAP = +0.7892  [↑ increases risk]  [HIGH]
```

---

## 7. Exported Artifacts

All files are saved to `./models/`:

| File | Size | Description |
|---|---|---|
| `xgb_risk_profiler.pkl` | 796 KB | Trained `XGBClassifier` (best params from RandomizedSearch) |
| `shap_explainer.pkl` | 1.5 MB | `shap.TreeExplainer` bound to the model + background data |
| `ohe_encoder.pkl` | < 1 KB | Fitted `OneHotEncoder` for `Veh_Usage` |
| `label_encoder.pkl` | < 1 KB | Fitted `LabelEncoder` mapping Risk Tier ↔ int |
| `feature_names.pkl` | < 1 KB | Python list of 14 feature names in training order |
| `manifest.json` | < 1 KB | Human-readable metadata about the pipeline |

---

## 8. Feature Reference

Complete list of the 14 features the model receives (in order):

| # | Feature | Source | Type |
|---|---|---|---|
| 1 | `Prev_Accidents` | Raw dataset | Binary int (0/1) |
| 2 | `Prev_Citations` | Raw dataset | Binary int (0/1) |
| 3 | `Driving_Exp` | Raw dataset | Int (years) |
| 4 | `Driver_Age` | Raw dataset | Int (years) |
| 5 | `Annual_Miles` | Derived from `Annual_Miles_Range` | Int (midpoint miles) |
| 6 | `Total_Incidents` | Engineered | Int |
| 7 | `Incident_Score` | Engineered | Int |
| 8 | `Miles_Per_Exp_Year` | Engineered | Float |
| 9 | `Risk_Exposure_Index` | Engineered | Float |
| 10 | `Young_Inexperienced` | Engineered | Binary int (0/1) |
| 11 | `Age_Exp_Gap` | Engineered | Int |
| 12 | `Veh_Usage_Business` | OHE from `Veh_Usage` | Binary float (0.0/1.0) |
| 13 | `Veh_Usage_Commute` | OHE from `Veh_Usage` | Binary float (0.0/1.0) |
| 14 | `Veh_Usage_Pleasure` | OHE from `Veh_Usage` | Binary float (0.0/1.0) |

---

## 9. Inference API Reference

### Standalone (no CrewAI)

```python
from agent1_risk_profiler import explain_risk_prediction, RiskProfilerPredictor
import joblib

# Load artifacts manually
model         = joblib.load("./models/xgb_risk_profiler.pkl")
explainer     = joblib.load("./models/shap_explainer.pkl")
ohe           = joblib.load("./models/ohe_encoder.pkl")
le            = joblib.load("./models/label_encoder.pkl")
feature_names = joblib.load("./models/feature_names.pkl")

quote = {
    "Prev_Accidents": 1,
    "Prev_Citations":  0,
    "Driving_Exp":     5,
    "Driver_Age":     24,
    "Annual_Miles": 32_000,
    "Veh_Usage":   "Commute",
}

result = explain_risk_prediction(
    quote,
    model=model,
    explainer=explainer,
    ohe=ohe,
    le=le,
    feature_names=feature_names,
    engineered_features=[
        "Total_Incidents", "Incident_Score", "Miles_Per_Exp_Year",
        "Risk_Exposure_Index", "Young_Inexperienced", "Age_Exp_Gap",
    ],
)

print(result["predicted_tier"])        # "High"
print(result["confidence"])            # 0.9987
print(result["top_3_features"])        # [{feature, shap_value, direction, magnitude}, ...]
```

### Via `RiskProfilerPredictor` (recommended)

```python
from agent1_risk_profiler import RiskProfilerPredictor

agent = RiskProfilerPredictor.from_artifacts("./models/")

result = agent.predict_and_explain({
    "Prev_Accidents": 0,
    "Prev_Citations":  0,
    "Driving_Exp":    20,
    "Driver_Age":     42,
    "Annual_Miles": 10_000,
    "Veh_Usage":   "Pleasure",
})

print(result)
```

---

## 10. CrewAI Integration Guide

Wrap `RiskProfilerPredictor` in a CrewAI `BaseTool` subclass:

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from agent1_risk_profiler import RiskProfilerPredictor

# Load once at module level — avoids reloading on every call
_agent = RiskProfilerPredictor.from_artifacts("./models/")


class RiskProfilerInput(BaseModel):
    Prev_Accidents: int   = Field(..., description="0 or 1 — prior accident on record")
    Prev_Citations: int   = Field(..., description="0 or 1 — prior citation on record")
    Driving_Exp:    int   = Field(..., description="Years of licensed driving experience")
    Driver_Age:     int   = Field(..., description="Driver age in years")
    Annual_Miles:   int   = Field(..., description="Estimated annual mileage (numeric)")
    Veh_Usage:      str   = Field(..., description="Commute | Pleasure | Business")


class RiskProfilerTool(BaseTool):
    name:        str = "risk_profiler"
    description: str = (
        "Classifies a driver as Low / Medium / High risk and returns "
        "a SHAP-based explanation of the top 3 factors driving the decision."
    )
    args_schema: type[BaseModel] = RiskProfilerInput

    def _run(self, **kwargs) -> dict:
        return _agent.predict_and_explain(kwargs)
```

Then assign the tool to your Agent 1 in your CrewAI crew definition:

```python
from crewai import Agent

risk_profiler_agent = Agent(
    role="Risk Profiler",
    goal="Assess the risk tier of every incoming auto insurance quote",
    backstory="Expert actuary with 20 years in auto insurance risk classification",
    tools=[RiskProfilerTool()],
    verbose=True,
)
```

---

## 11. Configuration Constants

All key settings are defined at the top of [agent1_risk_profiler.py](agent1_risk_profiler.py) for easy modification:

| Constant | Value | Description |
|---|---|---|
| `DATA_PATH` | `./Autonomous QUOTE AGENTS.csv` | Path to the raw dataset |
| `MODEL_DIR` | `./models/` | Output directory for all artifacts |
| `RANDOM_STATE` | `42` | Global random seed for reproducibility |
| `TEST_SIZE` | `0.20` | Fraction of data reserved for evaluation |
| `NUMERIC_FEATURES` | 5 base features | Features taken directly from the CSV |
| `CAT_FEATURES` | `["Veh_Usage"]` | Features requiring One-Hot Encoding |
| `MILES_MAP` | Dict of 7 entries | Maps range strings → numeric midpoints |

To retrain with different settings (e.g. more search iterations), modify `n_iter` in the `main()` call to `train_model()`.

---

*Agent 1 – Risk Profiler | Auto Insurance Multi-Agent Pipeline | Trained March 5, 2026*
