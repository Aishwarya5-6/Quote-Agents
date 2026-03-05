#!/usr/bin/env python3
"""
=============================================================================
  Agent 1 – Risk Profiler  │  Auto Insurance Multi-Agent Pipeline
  v2 — Production-Ready (No Data Leakage)

  ✓ Gaussian-noisy synthetic labels   → model learns probabilities, not a formula
  ✓ Strict 6-feature input only       → zero engineered features in X
  ✓ neg_log_loss tuning               → penalises confidently wrong predictions
  ✓ CalibratedClassifierCV (isotonic) → statistically sound output probabilities
  ✓ SHAP TreeExplainer                → base XGBoost extracted from calibrated wrapper
  ✓ Artifact cleanup on every run     → guaranteed clean slate before training
=============================================================================
  Script location : agents/agent1_risk_profiler.py
  Data input      : ../data/raw/insurance_data.csv
  Models output   : ../models/
  Processed data  : ../data/processed/cleaned_agent1_data.csv
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import logging
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import shap
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
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
AGENTS_DIR   = Path(__file__).resolve().parent           # …/agents/
BASE_DIR     = AGENTS_DIR.parent                          # …/Quote-Agents/
DATA_PATH    = BASE_DIR / "data" / "raw" / "insurance_data.csv"
MODEL_DIR    = BASE_DIR / "models"
PROC_DIR     = BASE_DIR / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE    = 0.20    # 20 % held-out test
CALIB_SIZE   = 0.20    # 20 % of train-val used for isotonic calibration
NOISE_SCALE  = 2.0     # Gaussian noise σ for synthetic label generation

# ── Strict 6-feature input (NO engineered features in X) ─────────────────────
NUMERIC_FEATURES: List[str] = [
    "Prev_Accidents",
    "Prev_Citations",
    "Driving_Exp",
    "Driver_Age",
    "Annual_Miles",
]
CAT_FEATURES: List[str] = ["Veh_Usage"]

# Mileage-range string → numeric midpoint (miles / year)
MILES_MAP: Dict[str, int] = {
    "<= 7.5 K":            7_500,
    "> 7.5 K & <= 15 K":  11_250,
    "> 15 K & <= 25 K":   20_000,
    "> 25 K & <= 35 K":   30_000,
    "> 35 K & <= 45 K":   40_000,
    "> 45 K & <= 55 K":   50_000,
    "> 55 K":             62_500,
}

# All artifacts written by this agent — cleaned up on every run
AGENT1_MODEL_FILES = [
    "calibrated_risk_profiler.pkl",
    "shap_explainer.pkl",
    "ohe_encoder.pkl",
    "label_encoder.pkl",
    "feature_names.pkl",
    "manifest.json",
    "xgb_risk_profiler.pkl",       # v1 legacy — removed if present
]
AGENT1_PROCESSED_FILE = "cleaned_agent1_data.csv"


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 – CLEANUP PREVIOUS ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
def cleanup_previous_artifacts() -> None:
    """
    Delete all .pkl / .json / .csv artifacts from previous Agent 1 runs.
    Creates target directories if they don't yet exist.
    Guarantees a completely clean slate before training begins.
    """
    log.info("── Step 0: Cleaning up previous Agent 1 artifacts ──────────")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    removed = 0

    for fname in AGENT1_MODEL_FILES:
        artifact = MODEL_DIR / fname
        if artifact.exists():
            artifact.unlink()
            log.info("  🗑  Removed  models/%s", fname)
            removed += 1

    processed_csv = PROC_DIR / AGENT1_PROCESSED_FILE
    if processed_csv.exists():
        processed_csv.unlink()
        log.info("  🗑  Removed  data/processed/%s", AGENT1_PROCESSED_FILE)
        removed += 1

    if removed == 0:
        log.info("  ✓  No previous artifacts found — fresh slate confirmed.")
    else:
        log.info("  ✓  Cleaned %d artifact(s).", removed)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 – DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load CSV, convert Annual_Miles_Range → numeric midpoint,
    and fill nulls across all modelling columns.
    """
    log.info("── Step 1: Loading data → %s ────────────────────────────────", path.name)
    df = pd.read_csv(path, low_memory=False)
    log.info("  Raw shape  : %d rows × %d cols", *df.shape)

    # Convert range-string mileage column → numeric midpoint
    if "Annual_Miles_Range" in df.columns and "Annual_Miles" not in df.columns:
        df["Annual_Miles"] = df["Annual_Miles_Range"].map(MILES_MAP)

    # Null handling: numeric → column median (robust to outliers)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Null handling: categorical → column mode
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    null_report = df[NUMERIC_FEATURES + CAT_FEATURES].isnull().sum().to_dict()
    log.info("  Null check : %s", null_report)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 – NOISY RISK TIER GENERATION  (synthetic ground truth)
# ─────────────────────────────────────────────────────────────────────────────
def _base_actuarial_score(row: pd.Series) -> float:
    """
    Deterministic actuarial score from raw inputs.

    Weights are grounded in auto-insurance loss-ratio research:
        +4      prior accident      (strongest single predictor of future claims)
        +2      prior citation      (moderate behavioural signal)
        +3/2/1  experience penalty  (≤3 / ≤7 / ≤15 years of licensed driving)
        +2/1    young-driver penalty (<22 / <26 years of age)
        +2/1    high-mileage penalty (>45 K / >25 K miles per year)
        +1      business-use uplift (higher liability exposure)
    """
    score: float = 0.0

    score += float(row["Prev_Accidents"]) * 4.0
    score += float(row["Prev_Citations"]) * 2.0

    exp = float(row["Driving_Exp"])
    if   exp <= 3:  score += 3.0
    elif exp <= 7:  score += 2.0
    elif exp <= 15: score += 1.0

    age = float(row["Driver_Age"])
    if   age < 22: score += 2.0
    elif age < 26: score += 1.0

    miles = float(row["Annual_Miles"])
    if   miles > 45_000: score += 2.0
    elif miles > 25_000: score += 1.0

    if row["Veh_Usage"] == "Business":
        score += 1.0

    return score


def _assign_noisy_risk_tier(
    row:         pd.Series,
    noise_scale: float,
    rng:         np.random.Generator,
) -> str:
    """
    CRITICAL ANTI-LEAKAGE STEP:
    Add Gaussian noise to the actuarial score before bucketing into tiers.

    Without noise the label is a perfect deterministic function of the
    features — XGBoost trivially memorises it (100 % accuracy).
    With noise (σ=2.0), the boundary is blurred realistically:
    a driver near a tier boundary has a genuine probability of landing
    in either adjacent tier, so the model is forced to learn probabilities
    rather than a rule table.

    Thresholds applied to NOISY score:
        noisy ≥ 7  → High
        noisy ≥ 4  → Medium
        noisy < 4  → Low
    """
    noisy = _base_actuarial_score(row) + rng.normal(0.0, noise_scale)
    if   noisy >= 7: return "High"
    elif noisy >= 4: return "Medium"
    else:            return "Low"


def generate_risk_labels(
    df:          pd.DataFrame,
    noise_scale: float = NOISE_SCALE,
) -> pd.DataFrame:
    """Apply noisy labelling with a seeded RNG for full reproducibility."""
    log.info(
        "── Step 2: Generating noisy Risk_Tier labels (σ=%.1f) ──────────",
        noise_scale,
    )
    rng = np.random.default_rng(RANDOM_STATE)
    df  = df.copy()
    df["Risk_Tier"] = df.apply(
        lambda row: _assign_noisy_risk_tier(row, noise_scale, rng),
        axis=1,
    )
    dist = df["Risk_Tier"].value_counts()
    log.info("  Label distribution after noise injection:")
    for tier, cnt in dist.items():
        log.info("    %-8s  %7d  (%.1f%%)", tier, cnt, cnt / len(df) * 100)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 – ENCODING  (strict 6-feature input — ZERO engineered features)
# ─────────────────────────────────────────────────────────────────────────────
def encode_features(
    df:  pd.DataFrame,
    fit: bool                    = True,
    ohe: Optional[OneHotEncoder] = None,
    le:  Optional[LabelEncoder]  = None,
) -> Tuple[pd.DataFrame, pd.Series, OneHotEncoder, LabelEncoder, List[str]]:
    """
    Build X (8 columns after OHE) and encoded y from the dataframe.

    Input features (strict, no engineering):
        Numeric   : Prev_Accidents, Prev_Citations, Driving_Exp,
                    Driver_Age, Annual_Miles
        Categorical (OHE): Veh_Usage
            → Veh_Usage_Business, Veh_Usage_Commute, Veh_Usage_Pleasure
        ─────────────────────────────────────────────────────────────
        Total     : 8 features

    Pass fit=False + pre-fitted encoders for inference-time transforms.
    """
    if fit:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        le  = LabelEncoder()

    assert ohe is not None, "ohe must be provided when fit=False"
    assert le  is not None, "le  must be provided when fit=False"

    # OneHotEncode Veh_Usage
    X_cat    = pd.DataFrame(df[CAT_FEATURES])
    veh_enc  = ohe.fit_transform(X_cat) if fit else ohe.transform(X_cat)
    veh_cols = list(ohe.get_feature_names_out(CAT_FEATURES))
    df_ohe   = pd.DataFrame(np.array(veh_enc), columns=veh_cols, index=df.index)

    # Combine numeric + OHE columns (the complete, leak-free feature matrix)
    X             = pd.concat([df[NUMERIC_FEATURES], df_ohe], axis=1)
    feature_names = NUMERIC_FEATURES + veh_cols

    # Label-encode target
    risk_arr = df["Risk_Tier"].to_numpy()
    y = pd.Series(
        le.fit_transform(risk_arr) if fit else le.transform(risk_arr),
        index=df.index,
    )

    log.info(
        "  Feature matrix : %d rows × %d cols  |  classes : %s",
        *X.shape, list(le.classes_),
    )
    return X, y, ohe, le, feature_names


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 – SAMPLE WEIGHTS  (compensate for class imbalance)
# ─────────────────────────────────────────────────────────────────────────────
def compute_sample_weights(y: pd.Series) -> np.ndarray:
    """
    Inverse-frequency weights:  w_c = N / (K × n_c)

    Ensures every Risk Tier is equally important during training
    regardless of its natural prevalence in the dataset.
    """
    counts = Counter(y.tolist())
    total  = len(y)
    n_cls  = len(counts)
    w_map  = {cls: total / (n_cls * cnt) for cls, cnt in counts.items()}
    return np.array([w_map[int(yi)] for yi in y])


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 – HYPERPARAMETER SEARCH  (scoring = neg_log_loss)
# ─────────────────────────────────────────────────────────────────────────────
def run_hyperparameter_search(
    X_subtrain: pd.DataFrame,
    y_subtrain: pd.Series,
    n_classes:  int,
    sw:         np.ndarray,
    n_iter:     int = 50,
) -> Tuple[XGBClassifier, Dict]:
    """
    RandomizedSearchCV across 10 hyperparameters, scored by neg_log_loss.

    neg_log_loss is the correct objective here because:
        • It penalises confident wrong predictions heavily (log diverges → ∞)
        • It rewards well-calibrated probability distributions
        • It aligns with the downstream calibration step

    5-fold StratifiedKFold preserves Risk Tier ratios in every fold.
    """
    log.info(
        "── Step 5: RandomizedSearchCV  n_iter=%d │ 5-fold │ neg_log_loss ──",
        n_iter,
    )

    param_dist = {
        "n_estimators":     [200, 300, 400, 500, 600],
        "max_depth":        [3, 4, 5, 6, 7],
        "learning_rate":    [0.01, 0.05, 0.08, 0.10, 0.15],
        "subsample":        [0.70, 0.80, 0.90, 1.00],
        "colsample_bytree": [0.60, 0.70, 0.80, 0.90, 1.00],
        "min_child_weight": [1, 3, 5, 7, 10],
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
        tree_method="hist",
        device="cpu",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_log_loss",     # ← penalise confident wrong predictions
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,                 # refit best params on full X_subtrain
        return_train_score=True,
    )

    t0 = time.time()
    search.fit(X_subtrain, y_subtrain, sample_weight=sw)
    elapsed = time.time() - t0

    log.info("  Search done in %.1f s", elapsed)
    log.info("  Best CV neg_log_loss : %.6f", search.best_score_)
    log.info("  Best params:\n%s", json.dumps(search.best_params_, indent=4))

    return search.best_estimator_, search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 – PROBABILITY CALIBRATION  (isotonic regression, cv='prefit')
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_model(
    base_xgb: XGBClassifier,
    X_calib:  pd.DataFrame,
    y_calib:  pd.Series,
) -> CalibratedClassifierCV:
    """
    Wrap the already-fitted XGBoost in CalibratedClassifierCV.

    method='isotonic'  – non-parametric monotone mapping:
                         raw scores → true posterior probabilities.
                         More flexible than Platt scaling (sigmoid),
                         recommended for n_calib > ~1 000 samples.

    cv='prefit'        – the base estimator is already fitted on
                         X_subtrain; we only fit the isotonic layer
                         on the separate held-out calibration set.
                         This avoids any information leakage between
                         fitting and calibrating.

    After .fit(), calibrated.estimator  → the base XGBClassifier
                  calibrated.classes_   → class labels [0, 1, 2]
    """
    log.info("── Step 6: Isotonic probability calibration ─────────────────")
    calibrated = CalibratedClassifierCV(
        base_xgb,
        method="isotonic",
        cv="prefit",
    )
    calibrated.fit(X_calib, y_calib)
    log.info(
        "  Calibrated.  classes_ = %s  |  calibration rows = %d",
        list(calibrated.classes_), len(X_calib),
    )
    return calibrated


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 – EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model:  Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    le:     LabelEncoder,
) -> None:
    """Classification report + confusion matrix + log-loss on held-out test set."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc     = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    ll      = log_loss(y_test, y_proba)

    print("\n" + "═" * 64)
    print("  EVALUATION RESULTS  (held-out 20 % test set)")
    print("═" * 64)
    print(f"  Test Accuracy          : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Balanced Accuracy      : {bal_acc:.4f}  ({bal_acc * 100:.2f}%)")
    print(f"  Log-Loss               : {ll:.6f}")
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
    print("═" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 8 – SHAP EXPLAINER  (base XGBoost only — not the calibrated wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def build_shap_explainer(
    base_xgb:     XGBClassifier,
    X_background: pd.DataFrame,
) -> shap.TreeExplainer:
    """
    Build SHAP TreeExplainer on the raw XGBoost estimator.

    Why the BASE estimator and not the calibrated wrapper?
    CalibratedClassifierCV wraps XGBoost in isotonic regressors; SHAP's
    TreeExplainer cannot traverse the isotonic layer.  The base XGBoost
    produces identical feature-importance rankings (calibration only
    adjusts probability magnitudes, not the rank order of predictions).

    Background: 500-row sample for fast interventional SHAP computation.
    """
    log.info("── Step 8: Building SHAP TreeExplainer ─────────────────────")
    bg = X_background.sample(
        min(500, len(X_background)), random_state=RANDOM_STATE
    )
    explainer = shap.TreeExplainer(
        base_xgb,
        data=bg,
        feature_perturbation="interventional",
    )
    log.info("  SHAP explainer ready (background rows: %d).", len(bg))
    return explainer


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 9 – PREDICT & EXPLAIN  (core inference helper)
# ─────────────────────────────────────────────────────────────────────────────
def explain_risk_prediction(
    quote_data:    Dict[str, Any],
    *,
    model:         Any,                  # CalibratedClassifierCV at runtime
    explainer:     shap.TreeExplainer,
    ohe:           OneHotEncoder,
    le:            LabelEncoder,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Predict Risk Tier for ONE customer quote and explain the decision via SHAP.

    Input  (quote_data keys)
    ─────────────────────────────────────────────────────────────────────
    Prev_Accidents  int    0 or 1  — prior at-fault accident on record
    Prev_Citations  int    0 or 1  — prior traffic citation on record
    Driving_Exp     int    years of licensed driving experience
    Driver_Age      int    driver's age in years
    Annual_Miles    int    estimated annual mileage (numeric, not range string)
    Veh_Usage       str    "Commute" | "Pleasure" | "Business"

    Output (returned dict)
    ─────────────────────────────────────────────────────────────────────
    predicted_tier      str    "Low" | "Medium" | "High"
    predicted_class_id  int    encoded class index  (High=0, Low=1, Medium=2)
    confidence          float  calibrated probability of predicted class
    class_probabilities dict   {tier: calibrated_prob} for all 3 tiers
    top_3_features      list   top-3 SHAP drivers with value, direction, magnitude
    all_shap_values     dict   {feature: shap_value} for all 8 model features
    """
    row = dict(quote_data)

    # One-Hot Encode Veh_Usage
    veh_df      = pd.DataFrame([[row["Veh_Usage"]]], columns=["Veh_Usage"])
    veh_enc_arr = np.array(ohe.transform(veh_df))
    veh_cols    = list(ohe.get_feature_names_out(["Veh_Usage"]))
    for col, val in zip(veh_cols, veh_enc_arr[0]):
        row[col] = val

    # Assemble feature vector in exact training order
    X_input = pd.DataFrame([row])[feature_names].astype(float)

    # ── Predict with calibrated model (sound probabilities) ───────────────
    pred_class = int(model.predict(X_input)[0])
    pred_proba = model.predict_proba(X_input)[0]
    pred_tier  = str(le.inverse_transform([pred_class])[0])

    # ── SHAP values from base XGBoost TreeExplainer ───────────────────────
    # .values shape → (1, n_features, n_classes)
    shap_exp  = explainer(X_input, check_additivity=False)
    shap_vals = np.array(shap_exp.values)          # cast → ndarray for 3-D indexing
    sv        = shap_vals[0, :, pred_class]         # (n_features,) for predicted class
    shap_dict = {feat: float(sv[i]) for i, feat in enumerate(feature_names)}

    # ── Top-3 features by |SHAP| ──────────────────────────────────────────
    top3 = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    top3_out = [
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

    class_probs = {
        str(le.inverse_transform([i])[0]): round(float(p), 4)
        for i, p in enumerate(pred_proba)
    }

    return {
        "predicted_tier":      pred_tier,
        "predicted_class_id":  pred_class,
        "confidence":          round(float(pred_proba[pred_class]), 4),
        "class_probabilities": class_probs,
        "top_3_features":      top3_out,
        "all_shap_values":     {k: round(v, 6) for k, v in shap_dict.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 10 – CREWAI-READY PREDICTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────
class RiskProfilerPredictor:
    """
    Self-contained wrapper for all trained artifacts.
    Designed as a drop-in for a CrewAI BaseTool._run() method.

    Usage
    ─────
        from agent1_risk_profiler import RiskProfilerPredictor

        agent  = RiskProfilerPredictor.from_artifacts("../models/")
        result = agent.predict_and_explain({
            "Prev_Accidents": 1, "Prev_Citations": 0,
            "Driving_Exp":    5, "Driver_Age":    24,
            "Annual_Miles": 32_000, "Veh_Usage": "Commute",
        })
    """

    def __init__(
        self,
        model:         CalibratedClassifierCV,
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

    @classmethod
    def from_artifacts(
        cls, model_dir: str = "../models/"
    ) -> "RiskProfilerPredictor":
        """Load all serialised artifacts from disk and return a ready predictor."""
        p = Path(model_dir)
        log.info("Loading artifacts from %s", p.resolve())
        return cls(
            model         = joblib.load(p / "calibrated_risk_profiler.pkl"),
            explainer     = joblib.load(p / "shap_explainer.pkl"),
            ohe           = joblib.load(p / "ohe_encoder.pkl"),
            le            = joblib.load(p / "label_encoder.pkl"),
            feature_names = joblib.load(p / "feature_names.pkl"),
        )

    def predict_and_explain(self, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Entry point for CrewAI tool's _run() method."""
        return explain_risk_prediction(
            quote_data,
            model=self.model,
            explainer=self.explainer,
            ohe=self.ohe,
            le=self.le,
            feature_names=self.feature_names,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 11 – ARTIFACT EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export_artifacts(
    calibrated:   CalibratedClassifierCV,
    explainer:    shap.TreeExplainer,
    ohe:          OneHotEncoder,
    le:           LabelEncoder,
    feature_names: List[str],
    df_processed: pd.DataFrame,
) -> None:
    """
    Serialize all pipeline components with joblib (compress=3) and write
    the processed CSV for Agent 2.
    """
    log.info("── Step 11: Exporting artifacts ─────────────────────────────")

    pkl_artifacts = {
        "calibrated_risk_profiler.pkl": calibrated,
        "shap_explainer.pkl":            explainer,
        "ohe_encoder.pkl":               ohe,
        "label_encoder.pkl":             le,
        "feature_names.pkl":             feature_names,
    }

    for fname, obj in pkl_artifacts.items():
        out = MODEL_DIR / fname
        joblib.dump(obj, out, compress=3)
        log.info("  💾  Saved  models/%-38s %d KB", fname, out.stat().st_size // 1024)

    # Manifest
    manifest = {
        "agent":         "Agent 1 – Risk Profiler v2",
        "model_type":    "CalibratedClassifierCV(isotonic, cv=prefit) → XGBClassifier",
        "objective":     "multi:softprob (3-class)",
        "tuning_metric": "neg_log_loss",
        "noise_scale":   NOISE_SCALE,
        "classes":       list(le.classes_),
        "n_features":    len(feature_names),
        "feature_names": feature_names,
        "shap_method":   "TreeExplainer (interventional) on base XGBClassifier",
        "artifacts":     list(pkl_artifacts.keys()),
    }
    manifest_path = MODEL_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("  💾  Saved  models/manifest.json")

    # Processed CSV for Agent 2
    csv_out = PROC_DIR / AGENT1_PROCESSED_FILE
    df_processed.to_csv(csv_out, index=False)
    log.info(
        "  💾  Saved  data/processed/%-30s %d rows",
        AGENT1_PROCESSED_FILE, len(df_processed),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN  – full training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "═" * 64)
    print("  AGENT 1 – RISK PROFILER v2  │  Training Pipeline")
    print("═" * 64 + "\n")

    # ── Step 0: Clean slate ───────────────────────────────────────────────
    cleanup_previous_artifacts()

    # ── Step 1: Load & clean ──────────────────────────────────────────────
    df = load_and_prepare_data()

    # ── Step 2: Generate noisy Risk_Tier labels ───────────────────────────
    df = generate_risk_labels(df, noise_scale=NOISE_SCALE)

    # ── Step 3: Encode (strict 6-feature input, no engineering) ──────────
    log.info("── Step 3: Encoding features ────────────────────────────────")
    X, y, ohe, le, feature_names = encode_features(df)
    n_classes = len(le.classes_)

    # ── Step 4: Three-way split  64 % subtrain │ 16 % calib │ 20 % test ──
    log.info("── Step 4: Three-way stratified split ───────────────────────")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_subtrain, X_calib, y_subtrain, y_calib = train_test_split(
        X_trainval, y_trainval,
        test_size=CALIB_SIZE,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )
    log.info(
        "  Subtrain : %d  |  Calibration : %d  |  Test : %d",
        len(X_subtrain), len(X_calib), len(X_test),
    )

    # Sample weights for the subtrain set
    sw_subtrain = compute_sample_weights(y_subtrain)
    log.info(
        "  Sample weight range : %.4f – %.4f",
        sw_subtrain.min(), sw_subtrain.max(),
    )

    # ── Step 5: Hyperparameter search on subtrain ─────────────────────────
    best_xgb, best_params = run_hyperparameter_search(
        X_subtrain, y_subtrain, n_classes, sw_subtrain, n_iter=50
    )

    # ── Step 6: Calibrate on held-out calibration set ─────────────────────
    calibrated = calibrate_model(best_xgb, X_calib, y_calib)

    # ── Step 7: Evaluate calibrated model on test set ─────────────────────
    evaluate_model(calibrated, X_test, y_test, le)

    # ── Step 8: Build SHAP on base XGBoost ───────────────────────────────
    # calibrated.estimator is the XGBClassifier fitted on X_subtrain
    explainer = build_shap_explainer(calibrated.estimator, X_subtrain)

    # ── Step 9: Demo predictions with SHAP explanations ──────────────────
    print("─" * 64)
    print("  DEMO: explain_risk_prediction()")
    print("─" * 64)

    demo_customers = [
        {
            "label": "High-Risk   │ accident + citation, young novice, business use",
            "data":  {
                "Prev_Accidents": 1, "Prev_Citations": 1,
                "Driving_Exp":    3, "Driver_Age":    22,
                "Annual_Miles": 40_000, "Veh_Usage": "Business",
            },
        },
        {
            "label": "Low-Risk    │ clean record, experienced, low-mileage pleasure",
            "data":  {
                "Prev_Accidents": 0, "Prev_Citations": 0,
                "Driving_Exp":   20, "Driver_Age":    42,
                "Annual_Miles": 10_000, "Veh_Usage": "Pleasure",
            },
        },
        {
            "label": "Medium-Risk │ one citation, moderate mileage, commuter",
            "data":  {
                "Prev_Accidents": 0, "Prev_Citations": 1,
                "Driving_Exp":    8, "Driver_Age":    30,
                "Annual_Miles": 28_000, "Veh_Usage": "Commute",
            },
        },
    ]

    for demo in demo_customers:
        result = explain_risk_prediction(
            demo["data"],
            model=calibrated,
            explainer=explainer,
            ohe=ohe,
            le=le,
            feature_names=feature_names,
        )
        print(f"\n  🧑  {demo['label']}")
        print(f"  ⚑   Predicted Tier         : {result['predicted_tier']}")
        print(f"  📊  Confidence (calibrated) : {result['confidence']:.2%}")
        print(f"  📈  Class Probabilities     : {result['class_probabilities']}")
        print("  🔑  Top 3 SHAP Drivers      :")
        for feat in result["top_3_features"]:
            print(
                f"       • {feat['feature']:<22}  "
                f"SHAP = {feat['shap_value']:+.4f}  "
                f"[{feat['direction']}]  [{feat['magnitude']}]"
            )

    # ── Step 11: Export all artifacts ─────────────────────────────────────
    print("\n" + "─" * 64)
    print("  EXPORTING ARTIFACTS")
    print("─" * 64)

    # Build processed dataframe (raw + derived columns + noisy Risk_Tier)
    keep_cols = (
        NUMERIC_FEATURES
        + CAT_FEATURES
        + ["Risk_Tier"]
        + (["Annual_Miles_Range"] if "Annual_Miles_Range" in df.columns else [])
    )
    df_processed = df[[c for c in keep_cols if c in df.columns]].copy()

    export_artifacts(calibrated, explainer, ohe, le, feature_names, df_processed)

    print("\n" + "═" * 64)
    print("  ✅  Agent 1 – Risk Profiler v2 : Training Complete!")
    print(f"  📁  Models  → {MODEL_DIR}/")
    print(f"  📄  Data    → {PROC_DIR / AGENT1_PROCESSED_FILE}")
    print("═" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
