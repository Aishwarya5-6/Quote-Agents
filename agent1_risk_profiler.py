#!/usr/bin/env python3
"""
=============================================================================
  Agent 1 – Risk Profiler  │  Auto Insurance Multi-Agent Pipeline
=============================================================================
Trains an XGBoost multi-class classifier to assign a real-time Risk Tier
(Low / Medium / High) to every incoming auto insurance quote.

Outputs (saved to ./models/):
    xgb_risk_profiler.pkl  – Trained XGBClassifier (best from RandomizedSearch)
    shap_explainer.pkl     – SHAP TreeExplainer bound to the model
    ohe_encoder.pkl        – OneHotEncoder for Veh_Usage
    label_encoder.pkl      – LabelEncoder mapping Risk_Tier ↔ int
    feature_names.pkl      – Ordered list of features the model expects

Usage (standalone training):
    python agent1_risk_profiler.py

Usage (inference inside a CrewAI custom tool):
    from agent1_risk_profiler import RiskProfilerPredictor
    agent  = RiskProfilerPredictor.from_artifacts("./models/")
    result = agent.predict_and_explain(quote_data_dict)
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
)

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RiskProfiler")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATA_PATH    = BASE_DIR / "Autonomous QUOTE AGENTS.csv"
MODEL_DIR    = BASE_DIR / "models"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Base numeric features extracted from the raw dataset
NUMERIC_FEATURES: List[str] = [
    "Prev_Accidents",
    "Prev_Citations",
    "Driving_Exp",
    "Driver_Age",
    "Annual_Miles",          # converted from Annual_Miles_Range
]

# Categorical features requiring OHE
CAT_FEATURES: List[str] = ["Veh_Usage"]

# Map ordinal mileage-range strings → numeric midpoints (miles / year)
MILES_MAP: Dict[str, int] = {
    "<= 7.5 K":             7_500,
    "> 7.5 K & <= 15 K":   11_250,
    "> 15 K & <= 25 K":    20_000,
    "> 25 K & <= 35 K":    30_000,
    "> 35 K & <= 45 K":    40_000,
    "> 45 K & <= 55 K":    50_000,
    "> 55 K":              62_500,
}

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 – DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load CSV, convert Annual_Miles_Range to a numeric midpoint,
    and defensively fill any nulls in the modelling columns.
    """
    log.info("Loading dataset → %s", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("Raw shape: %s rows × %s cols", *df.shape)

    # ── Annual mileage: range string → numeric midpoint ────────────────────
    df["Annual_Miles"] = df["Annual_Miles_Range"].map(MILES_MAP)

    # ── Null handling (defensive) ─────────────────────────────────────────
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Commute")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 – RISK TIER LABELLING  (creates ground truth)
# ─────────────────────────────────────────────────────────────────────────────
def _score_row(row: pd.Series) -> int:
    """
    Compute an actuarial risk score using domain-weighted factors.

    Scoring rationale (based on auto-insurance loss-ratio research):
        +4   prior accident      — strongest single predictor of future claims
        +2   prior citation      — moderate predictor of risky behaviour
        +3/+2/+1  experience penalty  (≤3 / ≤7 / ≤15 yrs driving)
        +2/+1     young-driver penalty (<22 / <26 yrs of age)
        +2/+1     mileage exposure    (>45 K / >25 K miles per year)
        +1        business-use uplift (commercial exposure, higher liability)
    """
    score = 0

    # Incident history ──────────────────────────────────────────────────────
    score += int(row["Prev_Accidents"]) * 4
    score += int(row["Prev_Citations"]) * 2

    # Experience penalty ────────────────────────────────────────────────────
    exp = row["Driving_Exp"]
    if   exp <= 3:  score += 3
    elif exp <= 7:  score += 2
    elif exp <= 15: score += 1

    # Young-driver penalty ──────────────────────────────────────────────────
    age = row["Driver_Age"]
    if   age < 22: score += 2
    elif age < 26: score += 1

    # Mileage exposure ──────────────────────────────────────────────────────
    miles = row["Annual_Miles"]
    if   miles > 45_000: score += 2
    elif miles > 25_000: score += 1

    # Vehicle usage ─────────────────────────────────────────────────────────
    if row["Veh_Usage"] == "Business":
        score += 1

    return score


def assign_risk_tier(row: pd.Series) -> str:
    """
    Map numeric risk score → Low / Medium / High tier.

    Calibrated thresholds:
        score ≥ 7  → High   (≥2 major risk factors, or 1 incident + aggravating)
        score ≥ 4  → Medium (1 moderate risk factor, or exposure accumulation)
        score < 4  → Low    (clean record, experienced, low-exposure driver)
    """
    score = _score_row(row)
    if   score >= 7: return "High"
    elif score >= 4: return "Medium"
    else:            return "Low"


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 – FEATURE ENGINEERING  (6 new signals)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create 6 engineered features that expose non-linear interactions
    and risk ratios the base columns alone cannot express.

    Returns the mutated DataFrame and the list of new column names.
    """
    # 1. Total incident count  ─────────────────────────────────────────────
    #    Simple combination; useful standalone and as interaction base
    df["Total_Incidents"] = df["Prev_Accidents"] + df["Prev_Citations"]

    # 2. Weighted incident score  ──────────────────────────────────────────
    #    Accidents are 2× more severe in actuarial tables than citations
    df["Incident_Score"] = df["Prev_Accidents"] * 2 + df["Prev_Citations"]

    # 3. Miles per year of experience  ────────────────────────────────────
    #    Exposure-normalised driving load; high value = high-risk lifestyle
    df["Miles_Per_Exp_Year"] = df["Annual_Miles"] / (df["Driving_Exp"] + 1)

    # 4. Risk Exposure Index  ──────────────────────────────────────────────
    #    Incidents amplified by mileage and dampened by experience —
    #    captures the compound effect of "bad record + heavy use + novice"
    df["Risk_Exposure_Index"] = (
        (df["Incident_Score"] + 1) * df["Annual_Miles"]
        / (df["Driving_Exp"] + 1)
    )

    # 5. Young & inexperienced flag  ───────────────────────────────────────
    #    Non-linear interaction: teens/early-twenties with <5 yrs experience
    #    have disproportionately high claim rates
    df["Young_Inexperienced"] = (
        ((df["Driver_Age"] < 25) & (df["Driving_Exp"] < 5)).astype(int)
    )

    # 6. Age – experience gap  ─────────────────────────────────────────────
    #    Expected exp ≈ Driver_Age − 16 (typical licensing age in the US).
    #    Positive gap → driver started late or had licence suspended;
    #    both scenarios correlate with elevated risk.
    df["Age_Exp_Gap"] = df["Driver_Age"] - df["Driving_Exp"] - 16

    engineered = [
        "Total_Incidents",
        "Incident_Score",
        "Miles_Per_Exp_Year",
        "Risk_Exposure_Index",
        "Young_Inexperienced",
        "Age_Exp_Gap",
    ]
    log.info("Engineered %d new features: %s", len(engineered), engineered)
    return df, engineered


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 – ENCODING
# ─────────────────────────────────────────────────────────────────────────────
def encode_features(
    df: pd.DataFrame,
    engineered: List[str],
    fit: bool = True,
    ohe: Optional[OneHotEncoder] = None,
    le:  Optional[LabelEncoder]  = None,
) -> Tuple[pd.DataFrame, pd.Series, OneHotEncoder, LabelEncoder, List[str]]:
    """
    One-Hot Encode categorical features and label-encode the target.
    Pass fit=False + pre-fitted ohe/le to transform new data at inference time.
    """
    all_numeric = NUMERIC_FEATURES + engineered

    if fit:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        le  = LabelEncoder()

    # Narrow Optional types — callers must pass ohe/le when fit=False
    assert ohe is not None, "ohe must be provided when fit=False"
    assert le  is not None, "le must be provided when fit=False"

    X_cat = pd.DataFrame(df[CAT_FEATURES])   # explicit ctor → guaranteed DataFrame
    veh_enc  = ohe.fit_transform(X_cat) if fit else ohe.transform(X_cat)
    veh_cols = list(ohe.get_feature_names_out(CAT_FEATURES))
    df_ohe   = pd.DataFrame(np.array(veh_enc), columns=veh_cols, index=df.index)

    X = pd.concat([df[all_numeric], df_ohe], axis=1)

    risk_tier_arr = df["Risk_Tier"].to_numpy()
    if fit:
        y = pd.Series(le.fit_transform(risk_tier_arr), index=df.index)
    else:
        y = pd.Series(le.transform(risk_tier_arr), index=df.index)

    final_features = all_numeric + veh_cols
    return X, y, ohe, le, final_features


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 – SAMPLE WEIGHTS  (handle class imbalance)
# ─────────────────────────────────────────────────────────────────────────────
def compute_sample_weights(y: pd.Series) -> np.ndarray:
    """
    Inverse-frequency weights so the model sees each class as equally
    important during training — crucial for imbalanced insurance datasets.
    """
    counts  = Counter(y)
    total   = len(y)
    n_cls   = len(counts)
    w_map   = {cls: total / (n_cls * cnt) for cls, cnt in counts.items()}
    return np.array([w_map[yi] for yi in y])


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 – HYPERPARAMETER TUNING  (RandomizedSearchCV)
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    X_train:        pd.DataFrame,
    y_train:        pd.Series,
    n_classes:      int,
    sample_weights: np.ndarray,
    n_iter:         int = 50,
) -> Tuple[XGBClassifier, Dict]:
    """
    Tune XGBClassifier with RandomizedSearchCV over a wide parameter grid.
    Uses StratifiedKFold to preserve Risk Tier ratios in every fold.

    Parameters tuned
    ────────────────
    n_estimators      – number of boosting rounds
    max_depth         – tree depth (controls complexity vs. overfit)
    learning_rate     – shrinkage per step (lower = more robust, slower)
    subsample         – row sub-sampling per tree (reduces variance)
    colsample_bytree  – feature sub-sampling per tree (reduces variance)
    min_child_weight  – minimum sum of instance weights in a leaf
    gamma             – min loss-reduction to make a further partition
    reg_alpha         – L1 regularisation (feature selection pressure)
    reg_lambda        – L2 regularisation (weight magnitude control)
    max_delta_step    – helps convergence on imbalanced data
    """
    log.info(
        "Starting RandomizedSearchCV  n_iter=%d │ 5-fold StratifiedKFold …",
        n_iter,
    )

    param_dist = {
        "n_estimators":     [200, 300, 400, 500, 600],
        "max_depth":        [4, 5, 6, 7, 8],
        "learning_rate":    [0.01, 0.05, 0.08, 0.10, 0.15, 0.20],
        "subsample":        [0.70, 0.80, 0.90, 1.00],
        "colsample_bytree": [0.70, 0.80, 0.90, 1.00],
        "min_child_weight": [1, 3, 5, 7],
        "gamma":            [0, 0.05, 0.10, 0.20, 0.30],
        "reg_alpha":        [0, 0.01, 0.05, 0.10, 0.50],
        "reg_lambda":       [0.5, 1.0, 1.5, 2.0, 3.0],
        "max_delta_step":   [0, 1, 5],
    }

    base_xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",         # fastest exact method for tabular data
        device="cpu",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    t0 = time.time()
    search.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time.time() - t0

    log.info("Search completed in %.1f s", elapsed)
    log.info("Best CV Accuracy : %.4f", search.best_score_)
    log.info("Best params      :\n%s", json.dumps(search.best_params_, indent=4))

    return search.best_estimator_, search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 – EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model:  XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    le:     LabelEncoder,
) -> None:
    """Print a full classification report and confusion matrix."""
    y_pred   = model.predict(X_test)
    acc      = accuracy_score(y_test, y_pred)
    bal_acc  = balanced_accuracy_score(y_test, y_pred)

    print("\n" + "═" * 60)
    print("  EVALUATION RESULTS  (held-out test set)")
    print("═" * 60)
    print(f"  Test Accuracy          : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Balanced Accuracy      : {bal_acc:.4f}  ({bal_acc * 100:.2f}%)")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=4,
    ))
    print("  Confusion Matrix  (rows = Actual, cols = Predicted)")
    cm    = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 8 – SHAP EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────
def build_shap_explainer(
    model:        XGBClassifier,
    X_background: pd.DataFrame,
) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer with an interventional background dataset.
    The background sample is capped at 500 rows to keep inference fast.
    """
    log.info("Building SHAP TreeExplainer …")
    bg = X_background.sample(
        min(500, len(X_background)), random_state=RANDOM_STATE
    )
    explainer = shap.TreeExplainer(
        model,
        data=bg,
        feature_perturbation="interventional",
    )
    log.info("SHAP explainer ready.")
    return explainer


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 9 – PREDICT & EXPLAIN  (core inference function)
# ─────────────────────────────────────────────────────────────────────────────
def explain_risk_prediction(
    quote_data: Dict[str, Any],
    *,
    model:               XGBClassifier,
    explainer:           shap.TreeExplainer,
    ohe:                 OneHotEncoder,
    le:                  LabelEncoder,
    feature_names:       List[str],
    engineered_features: List[str],
) -> Dict[str, Any]:
    """
    Predict the Risk Tier for ONE customer quote and return a full
    SHAP-based explanation of the decision.

    Parameters
    ──────────
    quote_data : dict
        Required keys:
            Prev_Accidents  (int)   0 or 1
            Prev_Citations  (int)   0 or 1
            Driving_Exp     (int)   years of licensed driving
            Driver_Age      (int)   driver's age in years
            Annual_Miles    (int)   estimated annual mileage  ← numeric, NOT range string
            Veh_Usage       (str)   "Commute" | "Pleasure" | "Business"

    Returns
    ───────
    dict
        predicted_tier      (str)   "Low" | "Medium" | "High"
        predicted_class_id  (int)   encoded class index (0/1/2)
        confidence          (float) softmax probability of predicted class
        class_probabilities (dict)  {tier: probability} for all 3 classes
        top_3_features      (list)  top-3 SHAP drivers with direction & magnitude
        all_shap_values     (dict)  feature → SHAP value for the predicted class
    """
    # ── Reconstruct the same engineered features used during training ──────
    row = dict(quote_data)

    row["Total_Incidents"]     = row["Prev_Accidents"] + row["Prev_Citations"]
    row["Incident_Score"]      = row["Prev_Accidents"] * 2 + row["Prev_Citations"]
    row["Miles_Per_Exp_Year"]  = row["Annual_Miles"] / (row["Driving_Exp"] + 1)
    row["Risk_Exposure_Index"] = (
        (row["Incident_Score"] + 1) * row["Annual_Miles"] / (row["Driving_Exp"] + 1)
    )
    row["Young_Inexperienced"] = int(
        row["Driver_Age"] < 25 and row["Driving_Exp"] < 5
    )
    row["Age_Exp_Gap"] = row["Driver_Age"] - row["Driving_Exp"] - 16

    # ── One-Hot Encode Veh_Usage ───────────────────────────────────────────
    veh_df      = pd.DataFrame([[row["Veh_Usage"]]], columns=["Veh_Usage"])
    veh_enc_arr = np.array(ohe.transform(veh_df))   # guarantee dense ndarray
    veh_cols    = list(ohe.get_feature_names_out(["Veh_Usage"]))
    for col, val in zip(veh_cols, veh_enc_arr[0]):
        row[col] = val

    # ── Assemble feature vector in the exact training order ───────────────
    X_input = pd.DataFrame([row])[feature_names].astype(float)

    # ── Predict ───────────────────────────────────────────────────────────
    pred_class = int(model.predict(X_input)[0])
    pred_proba = model.predict_proba(X_input)[0]
    pred_tier  = le.inverse_transform([pred_class])[0]

    # ── SHAP values  ──────────────────────────────────────────────────────
    # explainer(X) returns an Explanation object:
    #   .values shape → (n_samples, n_features, n_classes)
    shap_exp   = explainer(X_input, check_additivity=False)
    shap_vals  = np.array(shap_exp.values)              # cast → ndarray for 3-D indexing
    sv         = shap_vals[0, :, pred_class]            # (n_features,)
    shap_dict = {feat: float(sv[i]) for i, feat in enumerate(feature_names)}

    # ── Top-3 features ranked by |SHAP| ──────────────────────────────────
    top3 = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    top3_output = [
        {
            "feature":    feat,
            "shap_value": round(val, 6),
            "direction":  "↑ increases risk" if val > 0 else "↓ decreases risk",
            "magnitude":  (
                "HIGH"   if abs(val) > 0.15 else
                "MEDIUM" if abs(val) > 0.05 else
                "LOW"
            ),
        }
        for feat, val in top3
    ]

    # ── Class probability breakdown ───────────────────────────────────────
    class_probs = {
        str(le.inverse_transform([i])[0]): round(float(p), 4)
        for i, p in enumerate(pred_proba)
    }

    return {
        "predicted_tier":       pred_tier,
        "predicted_class_id":   pred_class,
        "confidence":           round(float(pred_proba[pred_class]), 4),
        "class_probabilities":  class_probs,
        "top_3_features":       top3_output,
        "all_shap_values":      {k: round(v, 6) for k, v in shap_dict.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 10 – CREWAI-READY PREDICTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────
class RiskProfilerPredictor:
    """
    Wraps all trained artifacts into a single object that can be dropped
    directly into a CrewAI custom tool with two lines:

        from agent1_risk_profiler import RiskProfilerPredictor

        agent  = RiskProfilerPredictor.from_artifacts("./models/")
        result = agent.predict_and_explain({
            "Prev_Accidents": 1, "Prev_Citations": 0,
            "Driving_Exp": 5,   "Driver_Age": 24,
            "Annual_Miles": 32_000, "Veh_Usage": "Commute",
        })
    """

    _ENGINEERED = [
        "Total_Incidents",
        "Incident_Score",
        "Miles_Per_Exp_Year",
        "Risk_Exposure_Index",
        "Young_Inexperienced",
        "Age_Exp_Gap",
    ]

    def __init__(
        self,
        model:         XGBClassifier,
        explainer:     shap.TreeExplainer,
        ohe:           OneHotEncoder,
        le:            LabelEncoder,
        feature_names: List[str],
    ) -> None:
        self.model         = model
        self.explainer     = explainer
        self.ohe           = ohe
        self.le            = le
        self.feature_names = feature_names

    # ── Class constructor ─────────────────────────────────────────────────
    @classmethod
    def from_artifacts(cls, model_dir: str = "./models/") -> "RiskProfilerPredictor":
        """Load all .pkl artifacts from disk and return a ready predictor."""
        p = Path(model_dir)
        log.info("Loading artifacts from %s", p.resolve())
        return cls(
            model         = joblib.load(p / "xgb_risk_profiler.pkl"),
            explainer     = joblib.load(p / "shap_explainer.pkl"),
            ohe           = joblib.load(p / "ohe_encoder.pkl"),
            le            = joblib.load(p / "label_encoder.pkl"),
            feature_names = joblib.load(p / "feature_names.pkl"),
        )

    # ── Main inference method ─────────────────────────────────────────────
    def predict_and_explain(self, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept a customer quote dict and return a fully explained prediction.
        This is the method to call from your CrewAI tool's _run() method.
        """
        return explain_risk_prediction(
            quote_data,
            model=self.model,
            explainer=self.explainer,
            ohe=self.ohe,
            le=self.le,
            feature_names=self.feature_names,
            engineered_features=self._ENGINEERED,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 11 – ARTIFACT EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export_artifacts(
    model:         XGBClassifier,
    explainer:     shap.TreeExplainer,
    ohe:           OneHotEncoder,
    le:            LabelEncoder,
    feature_names: List[str],
    model_dir:     Path = MODEL_DIR,
) -> None:
    """
    Serialize all pipeline components with joblib (compress=3).
    Also writes a human-readable manifest.json for documentation.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "xgb_risk_profiler.pkl": model,
        "shap_explainer.pkl":    explainer,
        "ohe_encoder.pkl":       ohe,
        "label_encoder.pkl":     le,
        "feature_names.pkl":     feature_names,
    }

    for fname, obj in artifacts.items():
        out = model_dir / fname
        joblib.dump(obj, out, compress=3)
        log.info("  Saved  %-30s  %d KB", fname, out.stat().st_size // 1024)

    # Human-readable manifest
    manifest = {
        "agent":          "Agent 1 – Risk Profiler",
        "model_type":     "XGBClassifier",
        "objective":      "multi:softprob (3-class)",
        "classes":        list(le.classes_),
        "n_features":     len(feature_names),
        "feature_names":  feature_names,
        "artifacts":      list(artifacts.keys()),
        "shap_method":    "TreeExplainer (interventional)",
    }
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("  Saved  manifest.json")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN  – orchestrates the full training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "═" * 60)
    print("  AGENT 1 – RISK PROFILER  │  Training Pipeline")
    print("═" * 60 + "\n")

    # ── 1. Load & clean ───────────────────────────────────────────────────
    df = load_and_prepare_data()

    # ── 2. Create ground-truth Risk_Tier labels ───────────────────────────
    log.info("Generating Risk_Tier labels from actuarial scoring function …")
    df["Risk_Tier"] = df.apply(assign_risk_tier, axis=1)
    dist = df["Risk_Tier"].value_counts()
    print("\n  Risk Tier distribution (full dataset):")
    for tier, cnt in dist.items():
        pct = cnt / len(df) * 100
        print(f"    {tier:<8}  {cnt:>7,}  ({pct:.1f}%)")

    # ── 3. Feature engineering ────────────────────────────────────────────
    df, engineered = engineer_features(df)

    # ── 4. Encode features & target ───────────────────────────────────────
    X, y, ohe, le, feature_names = encode_features(df, engineered)
    n_classes = len(le.classes_)
    log.info(
        "Classes: %s  │  Total features: %d",
        list(le.classes_), len(feature_names),
    )

    # ── 5. Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log.info(
        "Split → Train: %d rows | Test: %d rows", len(X_train), len(X_test)
    )

    # ── 6. Compute sample weights ─────────────────────────────────────────
    sw_train = compute_sample_weights(y_train)
    log.info(
        "Sample weight range: %.4f – %.4f",
        sw_train.min(), sw_train.max(),
    )

    # ── 7. Hyperparameter search ──────────────────────────────────────────
    best_model, best_params = train_model(
        X_train, y_train, n_classes, sw_train, n_iter=50
    )

    # ── 8. Evaluate on held-out test set ──────────────────────────────────
    evaluate_model(best_model, X_test, y_test, le)

    # ── 9. Build SHAP explainer ───────────────────────────────────────────
    explainer = build_shap_explainer(best_model, X_train)

    # ── 10. Demo inference  (3 representative customers) ──────────────────
    demo_customers = [
        {
            "label": "High-Risk  │  accident + citation, young & inexperienced, business use",
            "data": {
                "Prev_Accidents": 1, "Prev_Citations": 1,
                "Driving_Exp":    3, "Driver_Age":    22,
                "Annual_Miles": 40_000, "Veh_Usage": "Business",
            },
        },
        {
            "label": "Low-Risk   │  clean record, experienced, low-mileage pleasure driver",
            "data": {
                "Prev_Accidents": 0, "Prev_Citations": 0,
                "Driving_Exp":   20, "Driver_Age":    42,
                "Annual_Miles": 10_000, "Veh_Usage": "Pleasure",
            },
        },
        {
            "label": "Medium-Risk│  one citation, moderate mileage, commuter",
            "data": {
                "Prev_Accidents": 0, "Prev_Citations": 1,
                "Driving_Exp":    8, "Driver_Age":    30,
                "Annual_Miles": 28_000, "Veh_Usage": "Commute",
            },
        },
    ]

    print("\n" + "─" * 60)
    print("  DEMO: explain_risk_prediction()")
    print("─" * 60)

    for demo in demo_customers:
        result = explain_risk_prediction(
            demo["data"],
            model=best_model,
            explainer=explainer,
            ohe=ohe,
            le=le,
            feature_names=feature_names,
            engineered_features=engineered,
        )
        print(f"\n  🧑  {demo['label']}")
        print(f"  ⚑   Predicted Tier      : {result['predicted_tier']}")
        print(f"  📊  Confidence          : {result['confidence']:.2%}")
        print(f"  📈  Class Probabilities : {result['class_probabilities']}")
        print("  🔑  Top 3 SHAP Drivers  :")
        for feat in result["top_3_features"]:
            print(
                f"       • {feat['feature']:<25}  "
                f"SHAP = {feat['shap_value']:+.4f}  "
                f"[{feat['direction']}]  [{feat['magnitude']}]"
            )

    # ── 11. Export all artifacts ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  EXPORTING ARTIFACTS  →  ./models/")
    print("─" * 60)
    export_artifacts(best_model, explainer, ohe, le, feature_names)

    print("\n" + "═" * 60)
    print("  ✅  Agent 1 – Risk Profiler : Training Complete!")
    print(f"  📁  Artifacts saved to : {MODEL_DIR}/")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
