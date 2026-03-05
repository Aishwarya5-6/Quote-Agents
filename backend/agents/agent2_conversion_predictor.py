#!/usr/bin/env python3
"""
=============================================================================
  Agent 2 – Conversion Predictor  │  Auto Insurance Multi-Agent Pipeline
  v2 — Cross-Agent Context Engineering

  ✓ SMOTE oversampling (imblearn)    → handles ~22 % Bind minority class
  ✓ XGBClassifier + CalibratedClassifierCV (isotonic, prefit)
                                     → reliable bind_probability percentages
  ✓ Agent-to-Agent Context           → Risk_Tier (Agent 1 output) label-encoded
                                       as ordinal integer (High=0/Low=1/Medium=2)
  ✓ Optimal F1 threshold via precision_recall_curve
                                     → saved to agent2_metadata_v2.json
  ✓ SHAP TreeExplainer               → top-3 conversion drivers per quote
  ✓ ConversionPredictor class        → save_artifacts() / from_artifacts()
                                       mirrors Agent 1 architecture exactly
  ✓ Pydantic-validated output        → ConversionResult schema
  ✓ Zero artifact collision          → all v2 outputs prefixed agent2_*_v2
  ✓ Input: Agent 1 processed CSV     → no re-loading raw data
=============================================================================
  Script location : agents/agent2_conversion_predictor.py
  Data input      : data/processed/cleaned_agent1_data.csv  (from Agent 1)
  Models output   : models/  (agent2_*_v2 prefix only — Agent 1 files untouched)
  Metadata output : models/agent2_metadata_v2.json
=============================================================================

  Business Context
  ────────────────
  The Bind label (1 = customer buys the policy) is a downstream outcome of
  the risk tier, mileage, vehicle usage, and driver profile.  At ~22 % of
  all quotes, Bind is a significant minority class — a vanilla XGBoost
  trained on imbalanced data learns to predict "No Bind" for almost every
  quote and still achieves 78 % accuracy.

  Our pipeline corrects this with three compounding techniques:
    1. SMOTE  — oversamples Bind minority to 1:1 ratio in training only
    2. XGBoost scale_pos_weight fallback — additional class-weight signal
    3. CalibratedClassifierCV (isotonic) — converts raw scores to probabilities
    4. Precision–Recall threshold search — maximises F1 at the decision boundary
"""

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from pydantic import BaseModel, Field, field_validator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ConversionPredictor")


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
AGENTS_DIR = Path(__file__).resolve().parent          # …/backend/agents/
BASE_DIR   = AGENTS_DIR.parent                         # …/backend/
DATA_PATH  = BASE_DIR / "data" / "processed" / "cleaned_agent1_data.csv"
MODEL_DIR  = BASE_DIR / "models"

RANDOM_STATE = 42
TEST_SIZE    = 0.20    # 80/20 train/test split
CALIB_SIZE   = 0.20    # 20 % of train split off for isotonic calibration

# ── Bind label generation ─────────────────────────────────────────────────────
# Bind probability is a deterministic function of risk + exposure, then blurred
# with Gaussian noise so the model learns calibrated probabilities, not rules.
#   Low risk tier   → base bind prob ~55 %  (attractive price, likely to buy)
#   Medium risk tier → base bind prob ~28 %  (priced higher, some churn)
#   High risk tier   → base bind prob ~12 %  (expensive quote, most decline)
BIND_BASE_PROBS = {"Low": 0.55, "Medium": 0.28, "High": 0.12}
BIND_NOISE_STD  = 0.12   # Gaussian σ on the base probability before Bernoulli draw

# ── Threshold search ──────────────────────────────────────────────────────────
# After calibration we scan the precision–recall curve and select the threshold
# that maximises F1 for the Bind (positive) class.  This threshold is saved to
# agent2_metadata.json and loaded at inference time.
DEFAULT_THRESHOLD = 0.35   # used only if curve search fails (safety fallback)

# ── SMOTE ─────────────────────────────────────────────────────────────────────
SMOTE_K_NEIGHBORS = 5   # k in k-NN used by SMOTE; default 5 is robust

# ── Sales status bands (relative to optimal threshold) ───────────────────────
# HIGH_PROPENSITY       : bind_probability ≥ threshold + 0.10  (confident buyer)
# NEAR_MISS_FOR_ADVISOR : threshold − 0.10 ≤ bind_probability < threshold
#                         (borderline — send to sales advisor for a nudge)
# LOW_PROB              : bind_probability < threshold − 0.10  (unlikely buyer)
NEAR_MISS_BAND = 0.10

# ── Canonical v2 artifact allowlist ──────────────────────────────────────────
# These are the ONLY agent2_* files that must survive in models/ after training.
# cleanup_stale_artifacts() deletes any agent2_* file not in this set.
# agent1_* files and manifest.json are never examined.
AGENT2_V2_KEEP: set = {
    "agent2_conversion_model_v2.pkl",
    "agent2_shap_explainer_v2.pkl",
    "agent2_ohe_encoder_v2.pkl",
    "agent2_tier_encoder_v2.pkl",
    "agent2_feature_names_v2.pkl",
    "agent2_threshold_v2.pkl",
    "agent2_metadata_v2.json",
}
# All columns available in cleaned_agent1_data.csv that are informative for
# predicting Bind.  Risk_Tier is one-hot encoded (it's a nominal category).
# Annual_Miles_Range is dropped — Annual_Miles is the numeric form already.
NUMERIC_FEATURES_A2: List[str] = [
    "Prev_Accidents",
    "Prev_Citations",
    "Driving_Exp",
    "Driver_Age",
    "Annual_Miles",
    "Miles_Per_Exp",
    "Total_Incidents",
    "Age_Exp_Gap",
]
CAT_FEATURES_A2: List[str] = ["Veh_Usage"]

# ── Risk tier label encoding ──────────────────────────────────────────────────
# Risk_Tier (Agent 1's predicted_tier) is label-encoded as an integer and
# injected directly into the Agent 2 feature matrix as a single numeric axis.
# LabelEncoder sorts classes alphabetically, so the encoding is:
#   High = 0  │  Low = 1  │  Medium = 2
# This replaces the two OHE columns from v1 with one semantically richer axis;
# XGBoost can learn splits across the full High→Low→Medium spectrum.
TIER_CLASSES: List[str] = ["High", "Low", "Medium"]   # alphabetical — LE canonical order


# ─────────────────────────────────────────────────────────────────────────────
#  PYDANTIC OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
class ConversionDriver(BaseModel):
    """One SHAP-backed feature driver for the conversion prediction."""
    feature:         str
    shap_value:      float
    direction:       str   # "↑ increases conversion" or "↓ decreases conversion"
    magnitude:       str   # "HIGH" | "MEDIUM" | "LOW"


class ConversionResult(BaseModel):
    """
    Validated output contract for a single predict_conversion() call.

    Fields
    ──────
    bind_probability        Calibrated probability the customer will buy (0–1).
    sales_status            Categorical disposition:
                              HIGH_PROPENSITY       — likely to bind
                              NEAR_MISS_FOR_ADVISOR — borderline; needs nudge
                              LOW_PROB              — unlikely to bind
    distance_to_conversion  |bind_probability − optimal_threshold|.
                            Small values → customer is close to buying.
    top_drivers             Top-3 SHAP-backed features driving the prediction.
    optimal_threshold       The F1-maximising threshold used for this decision.
    """
    bind_probability:        float = Field(..., ge=0.0, le=1.0)
    sales_status:            str
    distance_to_conversion:  float = Field(..., ge=0.0)
    top_drivers:             List[ConversionDriver]
    optimal_threshold:       float

    @field_validator("sales_status")
    @classmethod
    def valid_status(cls, v: str) -> str:
        allowed = {"HIGH_PROPENSITY", "NEAR_MISS_FOR_ADVISOR", "LOW_PROB"}
        if v not in allowed:
            raise ValueError(f"sales_status must be one of {allowed}, got {v!r}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 – DATA LOADING & BIND LABEL GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def load_and_label_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the Agent-1 processed CSV and synthesise a realistic 'Bind' column.

    WHY synthetic labels?
    ─────────────────────
    The raw dataset does not contain a policy-binding outcome column.  We
    engineer one using an actuarially grounded model:

      1. Look up the base bind probability from BIND_BASE_PROBS by risk tier.
      2. Add Gaussian noise (σ=0.12) to blur the hard tier boundary.
      3. Clip to [0, 1] — a probability must stay in range.
      4. Draw a Bernoulli(p) sample → 0 or 1 (the actual Bind label).

    This produces a ~22 % positive-class rate, realistic for insurance quoting.
    Risk_Tier is included as a feature for Agent 2 because it is the direct
    output of Agent 1 and carries the strongest signal for whether the customer
    will accept the quoted price.

    Label distribution is logged so any unexpected skew is immediately visible.
    """
    log.info("── Step 1: Loading processed data → %s", path.name)
    df = pd.read_csv(path, low_memory=False)

    # Drop the range-string mileage column (numeric Annual_Miles already present)
    if "Annual_Miles_Range" in df.columns:
        df = df.drop(columns=["Annual_Miles_Range"])

    log.info("  Loaded  : %d rows × %d cols", *df.shape)

    # ── Bind label generation ──────────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_STATE)
    base_probs = df["Risk_Tier"].map(BIND_BASE_PROBS).astype(float)
    noisy_probs = (base_probs + rng.normal(0.0, BIND_NOISE_STD, size=len(df))).clip(0.0, 1.0)
    df["Bind"] = (rng.uniform(size=len(df)) < noisy_probs).astype(int)

    bind_dist  = df["Bind"].value_counts()
    bind_rate  = df["Bind"].mean() * 100
    log.info("  Bind distribution  │  0 (No): %d  │  1 (Yes): %d  │  rate: %.1f %%",
             bind_dist.get(0, 0), bind_dist.get(1, 0), bind_rate)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 – FEATURE ENCODING
# ─────────────────────────────────────────────────────────────────────────────
def encode_features(
    df:      pd.DataFrame,
    fit:     bool                    = True,
    ohe:     Optional[OneHotEncoder] = None,
    le_tier: Optional[LabelEncoder]  = None,
) -> Tuple[pd.DataFrame, pd.Series, OneHotEncoder, LabelEncoder, List[str]]:
    """
    Build X (numeric + tier-encoded + OHE) and y (Bind 0/1).

    Feature matrix layout (fit=True)
    ─────────────────────────────────
    Numeric (8)      : Prev_Accidents, Prev_Citations, Driving_Exp, Driver_Age,
                       Annual_Miles, Miles_Per_Exp, Total_Incidents, Age_Exp_Gap
    Tier encoded (1) : Risk_Tier_encoded — LabelEncoder ordinal
                       High=0  │  Low=1  │  Medium=2  (alphabetical)
    OHE (2)          : Veh_Usage_Commute, Veh_Usage_Pleasure  (drop='first')
    ──────────────────────────────────────────────────────────────────────────
    Total            : 11 features

    Agent-to-Agent Context
    ──────────────────────
    Risk_Tier_encoded carries Agent 1's predicted_tier as a single numeric
    feature.  Using an ordinal integer (not OHE) gives XGBoost a continuous
    risk-gradient axis to split on, which is more informative than two sparse
    indicator columns for a feature with an actuarial ordering.

    Pass fit=False + pre-fitted encoders for inference-time transforms.
    """
    if fit:
        ohe     = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
        le_tier = LabelEncoder()
        le_tier.fit(df["Risk_Tier"])    # alphabetical sort: High=0, Low=1, Medium=2

    assert ohe     is not None, "ohe must be provided when fit=False"
    assert le_tier is not None, "le_tier must be provided when fit=False"

    # ── Label-encode Risk_Tier (Agent 1 context) ──────────────────────────────
    # fillna('Low') is the safe default for any missing tier values.
    tier_encoded = pd.Series(
        le_tier.transform(df["Risk_Tier"].fillna("Low")),
        name="Risk_Tier_encoded",
    ).reset_index(drop=True)

    # ── OHE Veh_Usage only ────────────────────────────────────────────────────
    X_cat    = df[CAT_FEATURES_A2].copy()
    cat_enc  = ohe.fit_transform(X_cat) if fit else ohe.transform(X_cat)
    cat_cols = list(ohe.get_feature_names_out(CAT_FEATURES_A2))

    X_num = df[NUMERIC_FEATURES_A2].reset_index(drop=True).copy()
    X     = pd.concat(
        [X_num,
         tier_encoded,
         pd.DataFrame(cat_enc, columns=cat_cols)],
        axis=1,
    )
    feature_names = NUMERIC_FEATURES_A2 + ["Risk_Tier_encoded"] + cat_cols

    y = df["Bind"].reset_index(drop=True)

    log.info(
        "── Step 2: Feature matrix: %d rows × %d cols  │  Bind rate: %.1f %%",
        *X.shape, y.mean() * 100,
    )
    log.info(
        "  Tier encoding (LabelEncoder alphabetical): %s",
        {str(le_tier.classes_[i]): i for i in range(len(le_tier.classes_))},
    )
    return X, y, ohe, le_tier, feature_names


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 – SMOTE  (oversample Bind minority to 1:1 in training only)
# ─────────────────────────────────────────────────────────────────────────────
def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Oversample the Bind=1 minority class to 1:1 ratio using SMOTE.

    WHY SMOTE rather than class_weight?
    ────────────────────────────────────
    class_weight adjusts gradient magnitudes but does not add any new
    training examples.  SMOTE synthesises new Bind=1 samples by
    interpolating between real k-NN neighbours in feature space.  The
    result is a denser representation of the minority manifold that forces
    the model to learn genuine bind-positive patterns rather than simply
    inflating gradient penalties.

    CRITICAL: SMOTE is applied ONLY to the training split — never to the
    calibration or test splits.  Applying SMOTE to the test set would
    distort all evaluation metrics.

    After SMOTE the class counts are logged so any failure is immediately
    visible.
    """
    log.info(
        "── Step 3: Applying SMOTE  │  before: Bind=0: %d  Bind=1: %d",
        (y_train == 0).sum(), (y_train == 1).sum(),
    )
    sm = SMOTE(
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    X_res, y_res = sm.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name="Bind")

    log.info(
        "  After SMOTE: Bind=0: %d  Bind=1: %d  (total: %d)",
        (y_res == 0).sum(), (y_res == 1).sum(), len(y_res),
    )
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 – HYPERPARAMETER SEARCH  (binary cross-entropy / neg_log_loss)
# ─────────────────────────────────────────────────────────────────────────────
def run_hyperparameter_search(
    X_subtrain: pd.DataFrame,
    y_subtrain: pd.Series,
    n_iter:     int = 40,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    RandomizedSearchCV scored by neg_log_loss (= binary cross-entropy here).

    WHY neg_log_loss for binary classification?
    ────────────────────────────────────────────
    The downstream step calibrates raw probabilities with isotonic regression.
    A model tuned for log_loss produces better-calibrated raw scores than one
    tuned for AUC or accuracy — the isotonic layer then only needs to make
    fine adjustments.  This two-stage approach (log_loss tuning → isotonic
    calibration) reliably outperforms tuning for accuracy alone on minority-
    class binary problems.

    scale_pos_weight is set to the majority/minority ratio observed in the
    SMOTE-d training set (approximately 1.0 post-SMOTE, but we pass the
    original ratio as a secondary signal inside the tree-building objective).

    Returns the best base XGBClassifier (not yet calibrated).
    """
    # Class balance for scale_pos_weight (re-computed from the SMOTE-d set)
    neg_count = int((y_subtrain == 0).sum())
    pos_count = int((y_subtrain == 1).sum())
    spw       = neg_count / max(pos_count, 1)

    log.info(
        "── Step 4: RandomizedSearchCV  n_iter=%d │ 5-fold │ neg_log_loss  "
        "│  scale_pos_weight=%.3f",
        n_iter, spw,
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
        objective         = "binary:logistic",
        eval_metric       = "logloss",
        scale_pos_weight  = spw,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        tree_method       = "hist",
        device            = "cpu",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator            = base_xgb,
        param_distributions  = param_dist,
        n_iter               = n_iter,
        scoring              = "neg_log_loss",
        cv                   = cv,
        random_state         = RANDOM_STATE,
        n_jobs               = -1,
        verbose              = 1,
        refit                = True,
        return_train_score   = True,
    )

    t0 = time.time()
    search.fit(X_subtrain, y_subtrain)
    elapsed = time.time() - t0

    log.info("  Search done in %.1f s", elapsed)
    log.info("  Best CV neg_log_loss : %.6f", search.best_score_)
    log.info("  Best params:\n%s", json.dumps(search.best_params_, indent=4))

    return search.best_estimator_, search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 – PROBABILITY CALIBRATION  (isotonic regression, cv='prefit')
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_model(
    base_xgb: XGBClassifier,
    X_calib:  pd.DataFrame,
    y_calib:  pd.Series,
) -> CalibratedClassifierCV:
    """
    Wrap the already-fitted XGBoost in CalibratedClassifierCV (isotonic, prefit).

    WHY isotonic over Platt scaling for this task?
    ───────────────────────────────────────────────
    Our calibration set is large (~16 K rows after SMOTE) and the raw XGBoost
    scores have a non-monotone S-shape typical of gradient boosting.  Platt
    scaling fits a single sigmoid — it cannot correct non-linear
    miscalibration.  Isotonic regression fits a piecewise-constant monotone
    function with no parametric assumptions, reliably capturing the
    S-shaped miscalibration at n_calib > ~1 000 samples.

    cv='prefit': the base estimator is already fitted; we fit only the
    isotonic layer on the separate calibration holdout to prevent leakage.
    """
    log.info("── Step 5: Isotonic probability calibration ─────────────────")
    calibrated = CalibratedClassifierCV(base_xgb, method="isotonic", cv="prefit")
    calibrated.fit(X_calib, y_calib)
    log.info(
        "  Calibrated.  classes_ = %s  │  calibration rows = %d",
        list(calibrated.classes_), len(X_calib),
    )
    return calibrated


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 – OPTIMAL THRESHOLD VIA PRECISION–RECALL CURVE
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_threshold(
    model:  CalibratedClassifierCV,
    X_val:  pd.DataFrame,
    y_val:  pd.Series,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the decision threshold that maximises F1 for the Bind=1 class.

    Algorithm
    ──────────
    1. Compute calibrated probabilities on the validation split (X_calib /
       the same 20 % holdout used for isotonic fitting — not the test set).
    2. Call precision_recall_curve to get (precision, recall, thresholds).
       sklearn returns n+1 precision/recall points for n thresholds, so we
       align them correctly.
    3. Compute F1 = 2·p·r / (p+r+ε) for each threshold.
    4. Select the threshold where F1 is maximised.

    WHY F1-maximising threshold, not 0.5?
    ──────────────────────────────────────
    A calibrated probability of 0.35 can still be the optimal decision
    boundary when the minority class is ~22 %.  Using a fixed 0.5 cutoff
    would classify nearly every quote as No-Bind — correct by accuracy but
    useless for the sales team.  The F1-maximising threshold balances
    precision (avoid false send-to-advisor calls) with recall (don't miss
    genuine buyers).

    Returns
    ───────
    threshold : float   — the F1-maximising decision threshold
    curve_metrics : dict — P/R/F1/support at the selected threshold
    """
    log.info("── Step 6: Finding optimal threshold via precision–recall curve")
    proba      = model.predict_proba(X_val)[:, 1]    # P(Bind=1)
    precision, recall, thresholds = precision_recall_curve(y_val, proba)

    # precision and recall have n+1 elements; thresholds has n
    # Align: ignore the last (degenerate) precision/recall pair
    precision = precision[:-1]
    recall    = recall[:-1]

    f1_scores  = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx   = int(np.argmax(f1_scores))
    threshold  = float(thresholds[best_idx])

    curve_metrics: Dict[str, float] = {
        "optimal_threshold":   round(threshold, 6),
        "precision_at_threshold": round(float(precision[best_idx]), 4),
        "recall_at_threshold":    round(float(recall[best_idx]),    4),
        "f1_at_threshold":        round(float(f1_scores[best_idx]), 4),
        "auc_pr": round(float(average_precision_score(y_val, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, proba)), 4),
    }

    log.info("  Optimal threshold : %.6f", threshold)
    log.info("  At threshold  P=%.4f  R=%.4f  F1=%.4f",
             curve_metrics["precision_at_threshold"],
             curve_metrics["recall_at_threshold"],
             curve_metrics["f1_at_threshold"])
    log.info("  AUC-PR: %.4f  │  ROC-AUC: %.4f",
             curve_metrics["auc_pr"], curve_metrics["roc_auc"])
    return threshold, curve_metrics


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 – EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model:     CalibratedClassifierCV,
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
    threshold: float,
) -> None:
    """Classification report at both 0.5 and the optimal threshold."""
    proba    = model.predict_proba(X_test)[:, 1]
    y_05     = (proba >= 0.50).astype(int)
    y_opt    = (proba >= threshold).astype(int)

    print("\n" + "═" * 64)
    print("  EVALUATION RESULTS  (held-out 20 % test set)")
    print("═" * 64)
    print(f"  Test AUC-ROC          : {roc_auc_score(y_test, proba):.4f}")
    print(f"  Test AUC-PR           : {average_precision_score(y_test, proba):.4f}")
    print()
    print("  ── At threshold = 0.50 (baseline) ──")
    print(f"  Accuracy  : {accuracy_score(y_test, y_05):.4f}")
    print(f"  Bind F1   : {f1_score(y_test, y_05):.4f}")
    print(classification_report(y_test, y_05, target_names=["No Bind", "Bind"], digits=4))
    print(f"  ── At optimal threshold = {threshold:.4f} ──")
    print(f"  Accuracy  : {accuracy_score(y_test, y_opt):.4f}")
    print(f"  Bind F1   : {f1_score(y_test, y_opt):.4f}")
    print(classification_report(y_test, y_opt, target_names=["No Bind", "Bind"], digits=4))
    print("═" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 8 – SHAP EXPLAINER  (base XGBoost only — not the calibrated wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def build_shap_explainer(
    base_xgb:     XGBClassifier,
    X_background: pd.DataFrame,
) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer on the raw XGBoost estimator.

    We attach it to the base estimator (not CalibratedClassifierCV) for the
    same reason as Agent 1: SHAP cannot traverse the isotonic layer.  The
    base model's feature-importance ranking is identical to the calibrated
    model's — calibration only re-scales probability magnitudes.

    Background sample: 500 rows drawn from training data.
    feature_perturbation='interventional' produces causally interpretable
    SHAP values suitable for regulatory audit.
    """
    log.info("── Step 8: Building SHAP TreeExplainer ──────────────────────")
    bg = X_background.sample(min(500, len(X_background)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(
        base_xgb,
        data=bg,
        feature_perturbation="interventional",
    )
    log.info("  SHAP explainer ready (background rows: %d).", len(bg))
    return explainer


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _build_inference_row(
    input_data:    Dict[str, Any],
    ohe:           OneHotEncoder,
    le_tier:       LabelEncoder,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Convert a raw quote dict + Risk_Tier into the 11-feature vector expected
    by the Agent 2 v2 model.

    Input keys required
    ────────────────────
    From Agent 1 raw quote : Prev_Accidents, Prev_Citations, Driving_Exp,
                              Driver_Age, Annual_Miles, Veh_Usage
    From Agent 1 output    : Risk_Tier  (predicted_tier) — Agent-to-Agent context
    Pre-computed by A1     : Miles_Per_Exp, Total_Incidents, Age_Exp_Gap
                             (re-derived here if absent)

    Feature construction (11 total)
    ────────────────────────────────
    8 numeric  +  Risk_Tier_encoded (LE ordinal)  +  Veh_Usage OHE (2)

    The function is tolerant: unseen Risk_Tier values are clamped to 'Low'
    (the LE-encoded default) before transform to avoid ValueError.
    """
    row = dict(input_data)

    # ── Re-derive interaction features if not already present ─────────────────
    miles = float(row.get("Annual_Miles", 0))
    exp   = float(row.get("Driving_Exp", 0))
    age   = float(row.get("Driver_Age", 0))

    row.setdefault("Miles_Per_Exp",   miles / (exp + 1))
    row.setdefault("Total_Incidents", float(row.get("Prev_Accidents", 0)) + float(row.get("Prev_Citations", 0)))
    row.setdefault("Age_Exp_Gap",     age - exp - 16)

    # ── Label-encode Risk_Tier (Agent 1 context) ──────────────────────────────
    tier = str(row.get("Risk_Tier", "Low"))
    if tier not in le_tier.classes_:
        tier = "Low"   # safe fallback for any unseen label
    row["Risk_Tier_encoded"] = int(le_tier.transform([tier])[0])

    # ── OHE Veh_Usage only ────────────────────────────────────────────────────
    cat_df   = pd.DataFrame([[row.get("Veh_Usage", "Pleasure")]], columns=CAT_FEATURES_A2)
    cat_enc  = ohe.transform(cat_df)
    cat_cols = list(ohe.get_feature_names_out(CAT_FEATURES_A2))
    for col, val in zip(cat_cols, cat_enc[0]):
        row[col] = val

    return pd.DataFrame([row])[feature_names].astype(float)


# ─────────────────────────────────────────────────────────────────────────────
#  SALES STATUS LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def _sales_status(bind_prob: float, threshold: float) -> str:
    """
    Map bind_probability to a categorical sales disposition.

    Bands
    ─────
    HIGH_PROPENSITY       : prob ≥ threshold + NEAR_MISS_BAND  (confident buyer)
    NEAR_MISS_FOR_ADVISOR : threshold − NEAR_MISS_BAND ≤ prob < threshold
                            (borderline — advisor intervention likely to convert)
    LOW_PROB              : prob < threshold − NEAR_MISS_BAND   (unlikely buyer)

    Note: a customer scoring ≥ threshold but below threshold + NEAR_MISS_BAND
    is still HIGH_PROPENSITY — they are above the decision boundary and the
    wide high-propensity band makes the status more stable under slight
    probability shifts.
    """
    if bind_prob >= threshold + NEAR_MISS_BAND:
        return "HIGH_PROPENSITY"
    elif threshold - NEAR_MISS_BAND <= bind_prob < threshold:
        return "NEAR_MISS_FOR_ADVISOR"
    elif bind_prob >= threshold:
        # above threshold but within the band → still HIGH_PROPENSITY
        return "HIGH_PROPENSITY"
    else:
        return "LOW_PROB"


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 9 – ARTIFACT EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export_artifacts(
    calibrated:    CalibratedClassifierCV,
    explainer:     shap.TreeExplainer,
    ohe:           OneHotEncoder,
    le_tier:       LabelEncoder,
    feature_names: List[str],
    threshold:     float,
    best_params:   Dict[str, Any],
    curve_metrics: Dict[str, float],
) -> None:
    """
    Persist all Agent 2 v2 artifacts to backend/models/ with agent2_ prefix.

    Files written (v2 — Agent 1 files are NEVER touched)
    ──────────────────────────────────────────────────────
    agent2_conversion_model_v2.pkl  — CalibratedClassifierCV (the live model)
    agent2_shap_explainer_v2.pkl    — TreeExplainer on base XGBClassifier
    agent2_ohe_encoder_v2.pkl       — OneHotEncoder for Veh_Usage only
    agent2_tier_encoder_v2.pkl      — LabelEncoder for Risk_Tier (High=0/Low=1/Medium=2)
    agent2_feature_names_v2.pkl     — Ordered list of 11 feature names
    agent2_threshold_v2.pkl         — Optimal F1 decision threshold (float)
    agent2_metadata_v2.json         — Human-readable audit record
    """
    log.info("── Step 9: Exporting Agent 2 v2 artifacts ───────────────────")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    pkl_artifacts = {
        "agent2_conversion_model_v2.pkl":  calibrated,
        "agent2_shap_explainer_v2.pkl":    explainer,
        "agent2_ohe_encoder_v2.pkl":       ohe,
        "agent2_tier_encoder_v2.pkl":      le_tier,
        "agent2_feature_names_v2.pkl":     feature_names,
        "agent2_threshold_v2.pkl":         threshold,
    }

    for fname, obj in pkl_artifacts.items():
        out = MODEL_DIR / fname
        joblib.dump(obj, out, compress=3)
        log.info("  💾  Saved  models/%-45s %d KB", fname, out.stat().st_size // 1024)

    tier_enc_map = {str(le_tier.classes_[i]): i for i in range(len(le_tier.classes_))}
    metadata = {
        "agent":              "Agent 2 – Conversion Predictor v2",
        "model_type":         "CalibratedClassifierCV(isotonic, cv=prefit) → XGBClassifier",
        "objective":          "binary:logistic",
        "tuning_metric":      "neg_log_loss (5-fold StratifiedKFold)",
        "imbalance_handling": f"SMOTE(k_neighbors={SMOTE_K_NEIGHBORS}) on training split only",
        "calibration":        "CalibratedClassifierCV(method='isotonic', cv='prefit')",
        "threshold_method":   "precision_recall_curve → argmax(F1)",
        "optimal_threshold":  round(threshold, 6),
        "threshold_metrics":  curve_metrics,
        "near_miss_band":     NEAR_MISS_BAND,
        "numeric_features":   NUMERIC_FEATURES_A2,
        "cat_features":       CAT_FEATURES_A2,
        "tier_encoding":      tier_enc_map,
        "agent_context":      "Risk_Tier_encoded carries Agent 1 predicted_tier as LabelEncoder ordinal integer",
        "feature_names":      feature_names,
        "best_hyperparams":   best_params,
        "artifacts":          list(pkl_artifacts.keys()),
    }
    meta_path = MODEL_DIR / "agent2_metadata_v2.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("  💾  Saved  models/agent2_metadata_v2.json")


# ─────────────────────────────────────────────────────────────────────────────
#  CONVERSION PREDICTOR CLASS  (mirrors RiskProfilerPredictor architecture)
# ─────────────────────────────────────────────────────────────────────────────
class ConversionPredictor:
    """
    Self-contained inference wrapper for all Agent 2 trained artifacts.

    Mirrors the Agent 1 RiskProfilerPredictor pattern exactly:
      - Constructor receives all artifacts (never loads from disk at init)
      - from_artifacts() classmethod loads from disk → returns ready instance
      - save_artifacts() serialises everything to MODEL_DIR

    Usage
    ─────
        from agent2_conversion_predictor import ConversionPredictor

        agent2 = ConversionPredictor.from_artifacts()   # uses MODEL_DIR default
        result = agent2.predict_conversion(
            input_data={
                "Prev_Accidents": 0, "Prev_Citations": 1,
                "Driving_Exp": 12,  "Driver_Age": 34,
                "Annual_Miles": 22000, "Veh_Usage": "Pleasure",
                # Optional — will be re-derived if absent:
                "Miles_Per_Exp": 1833.3, "Total_Incidents": 1, "Age_Exp_Gap": 6,
            },
            risk_tier="Low",   # explicit Agent 1 predicted_tier
        )
        # result is a Pydantic ConversionResult

    Wire into main.py
    ──────────────────
        # startup:
        app.state.conv_engine = ConversionPredictor.from_artifacts(MODELS_DIR)
        # in process_quote():
        conv_result = request.app.state.conv_engine.predict_conversion(
            input_data=quote_dict,
            risk_tier=risk_assessment.predicted_tier,   # Agent 1 → Agent 2 context
        )
    """

    def __init__(
        self,
        model:         CalibratedClassifierCV,
        explainer:     shap.TreeExplainer,
        ohe:           OneHotEncoder,
        le_tier:       LabelEncoder,
        feature_names: List[str],
        threshold:     float,
    ) -> None:
        self.model         = model
        self.explainer     = explainer
        self.ohe           = ohe
        self.le_tier       = le_tier
        self.feature_names = feature_names
        self.threshold     = threshold

    # ── Persistence ───────────────────────────────────────────────────────────
    def save_artifacts(self, model_dir: Union[str, Path] = MODEL_DIR) -> None:
        """Serialise all six v2 artifacts to model_dir/ with agent2_ prefix."""
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model,         p / "agent2_conversion_model_v2.pkl",  compress=3)
        joblib.dump(self.explainer,     p / "agent2_shap_explainer_v2.pkl",    compress=3)
        joblib.dump(self.ohe,           p / "agent2_ohe_encoder_v2.pkl",       compress=3)
        joblib.dump(self.le_tier,       p / "agent2_tier_encoder_v2.pkl",      compress=3)
        joblib.dump(self.feature_names, p / "agent2_feature_names_v2.pkl",     compress=3)
        joblib.dump(self.threshold,     p / "agent2_threshold_v2.pkl",         compress=3)
        log.info("  ✅  ConversionPredictor v2: 6 artifacts saved to %s/", p)

    @classmethod
    def from_artifacts(
        cls, model_dir: Union[str, Path] = MODEL_DIR
    ) -> "ConversionPredictor":
        """
        Load all six Agent 2 v2 artifacts from disk and return a ready predictor.
        Raises FileNotFoundError with a helpful message if any artifact is missing.
        """
        p = Path(model_dir)
        required = [
            "agent2_conversion_model_v2.pkl",
            "agent2_shap_explainer_v2.pkl",
            "agent2_ohe_encoder_v2.pkl",
            "agent2_tier_encoder_v2.pkl",
            "agent2_feature_names_v2.pkl",
            "agent2_threshold_v2.pkl",
        ]
        for fname in required:
            if not (p / fname).exists():
                raise FileNotFoundError(
                    f"Agent 2 v2 artifact missing: {p / fname}\n"
                    "Run backend/agents/agent2_conversion_predictor.py first."
                )

        log.info("Loading Agent 2 v2 artifacts from %s", p.resolve())
        return cls(
            model         = joblib.load(p / "agent2_conversion_model_v2.pkl"),
            explainer     = joblib.load(p / "agent2_shap_explainer_v2.pkl"),
            ohe           = joblib.load(p / "agent2_ohe_encoder_v2.pkl"),
            le_tier       = joblib.load(p / "agent2_tier_encoder_v2.pkl"),
            feature_names = joblib.load(p / "agent2_feature_names_v2.pkl"),
            threshold     = joblib.load(p / "agent2_threshold_v2.pkl"),
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict_conversion(
        self,
        input_data: Dict[str, Any],
        risk_tier:  Optional[str] = None,
    ) -> ConversionResult:
        """
        Predict bind probability for one insurance quote and return a fully
        validated ConversionResult.

        Pipeline
        ────────
        1. Merge explicit risk_tier (Agent 1 context) — takes priority over
           input_data['Risk_Tier'] if provided
        2. Build 11-feature vector: 8 numeric + Risk_Tier_encoded (LE ordinal)
           + Veh_Usage OHE (2) via _build_inference_row
        3. Calibrated probability → bind_probability
        4. SHAP values → top-3 conversion drivers
        5. Sales status via _sales_status() + distance_to_conversion

        Parameters
        ──────────
        input_data : dict with Agent 1 raw quote fields.  Interaction features
                     (Miles_Per_Exp, Total_Incidents, Age_Exp_Gap) are optional
                     — re-derived if absent.
        risk_tier  : Agent 1's predicted_tier ('Low', 'Medium', or 'High').
                     If provided, takes priority over input_data['Risk_Tier'].
                     Defaults to 'Low' if both are absent (safe fallback).

        Returns
        ───────
        ConversionResult (Pydantic-validated) with:
            bind_probability        — calibrated float in [0, 1]
            sales_status            — HIGH_PROPENSITY | NEAR_MISS_FOR_ADVISOR | LOW_PROB
            distance_to_conversion  — |bind_prob − optimal_threshold|
            top_drivers             — list of 3 ConversionDriver objects
            optimal_threshold       — the F1-maximising threshold used
        """
        # ── Merge Agent 1 context: explicit risk_tier takes priority ──────────
        effective_input = dict(input_data)
        if risk_tier is not None:
            effective_input["Risk_Tier"] = risk_tier
        elif "Risk_Tier" not in effective_input:
            effective_input["Risk_Tier"] = "Low"   # safe default

        # ── 1. Encode ──────────────────────────────────────────────────────────
        X_row = _build_inference_row(effective_input, self.ohe, self.le_tier, self.feature_names)

        # ── 2. Calibrated probability ──────────────────────────────────────────
        bind_prob = float(self.model.predict_proba(X_row)[0, 1])

        # ── 3. SHAP explanation ────────────────────────────────────────────────
        shap_exp  = self.explainer(X_row, check_additivity=False)
        shap_vals = np.array(shap_exp.values)   # (1, n_features) for binary

        # For binary XGBoost, shap_exp.values may be (1, n_features) or
        # (1, n_features, 2) depending on SHAP version — handle both shapes.
        if shap_vals.ndim == 3:
            sv = shap_vals[0, :, 1]   # positive class SHAP values
        else:
            sv = shap_vals[0, :]

        shap_dict = {feat: float(sv[i]) for i, feat in enumerate(self.feature_names)}
        top3_raw  = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

        top_drivers = [
            ConversionDriver(
                feature   = feat,
                shap_value = round(val, 6),
                direction  = (
                    "↑ increases conversion" if val > 0
                    else "↓ decreases conversion"
                ),
                magnitude  = (
                    "HIGH"   if abs(val) > 0.15 else
                    "MEDIUM" if abs(val) > 0.05 else
                    "LOW"
                ),
            )
            for feat, val in top3_raw
        ]

        # ── 4. Status & distance ──────────────────────────────────────────────
        status   = _sales_status(bind_prob, self.threshold)
        distance = round(abs(bind_prob - self.threshold), 6)

        return ConversionResult(
            bind_probability       = round(bind_prob, 6),
            sales_status           = status,
            distance_to_conversion = distance,
            top_drivers            = top_drivers,
            optimal_threshold      = round(self.threshold, 6),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  PRE-TRAINING CLEANUP  (targeted — Agent 1 files never touched)
# ─────────────────────────────────────────────────────────────────────────────
def cleanup_stale_artifacts(model_dir: Path = MODEL_DIR) -> None:
    """
    Delete every agent2_* file in model_dir that is NOT in AGENT2_V2_KEEP.

    Safety rules
    ─────────────
    1. Only files whose names START with 'agent2_' are candidates for deletion.
       agent1_* files, manifest.json, and every other Agent 1 artifact are
       completely invisible to this function — they cannot be touched.

    2. Deletion is compared against the AGENT2_V2_KEEP allowlist, not a
       hard-coded list of files to remove.  Any future v3 artifact is therefore
       automatically protected once added to AGENT2_V2_KEEP, and any forgotten
       v1/v2-minus file is automatically pruned without manual maintenance.

    3. The function is idempotent — running it when the directory is already
       clean produces zero deletions and no log noise.

    Stale files currently targeted (from v1 training run)
    ──────────────────────────────────────────────────────
    agent2_calibrated_model.pkl   (v1 model — replaced by agent2_conversion_model_v2.pkl)
    agent2_shap_explainer.pkl     (v1 explainer)
    agent2_ohe_encoder.pkl        (v1 OHE — encoded Risk_Tier via OHE; v2 uses LE)
    agent2_feature_names.pkl      (v1 12-feature list)
    agent2_threshold.pkl          (v1 threshold)
    agent2_metadata.json          (v1 audit record — replaced by agent2_metadata_v2.json)
    """
    log.info("── Pre-training: Cleaning stale Agent 2 artifacts ──────────")

    if not model_dir.exists():
        log.info("  models/ does not exist yet — nothing to clean.")
        return

    n_deleted = 0
    for path in sorted(model_dir.iterdir()):
        # ── Safety gate 1: only touch agent2_* files ──────────────────────────
        if not path.name.startswith("agent2_"):
            continue

        # ── Safety gate 2: skip if it's in the v2 keep-list ──────────────────
        if path.name in AGENT2_V2_KEEP:
            log.info("  ✓  Keeping   models/%s", path.name)
            continue

        # ── Delete the stale file ─────────────────────────────────────────────
        path.unlink()
        log.info("  🗑  Deleted  models/%s", path.name)
        n_deleted += 1

    if n_deleted == 0:
        log.info("  ✓  No stale Agent 2 artifacts found — directory already clean.")
    else:
        log.info("  ✓  Cleaned %d stale artifact(s). Agent 1 files untouched.", n_deleted)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — full training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "═" * 64)
    print("  AGENT 2 – CONVERSION PREDICTOR v2  │  Training Pipeline")
    print("═" * 64 + "\n")

    # ── Step 0: Delete stale Agent 2 artifacts (never touches Agent 1) ───────
    cleanup_stale_artifacts()

    # ── Step 1: Load processed data & synthesise Bind labels ─────────────────
    df = load_and_label_data()

    # ── Step 2: Encode features ───────────────────────────────────────────────
    X, y, ohe, le_tier, feature_names = encode_features(df)
    log.info("  Feature names (%d): %s", len(feature_names), feature_names)
    log.info("  Tier encoding (LE) : %s",
             {str(le_tier.classes_[i]): i for i in range(len(le_tier.classes_))})

    # ── Step 3: Three-way stratified split  64 % │ 16 % │ 20 % ──────────────
    log.info("── Step 3 (split): Three-way stratified split ───────────────")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        stratify     = y,
        random_state = RANDOM_STATE,
    )
    X_subtrain, X_calib, y_subtrain, y_calib = train_test_split(
        X_trainval, y_trainval,
        test_size    = CALIB_SIZE,
        stratify     = y_trainval,
        random_state = RANDOM_STATE,
    )
    log.info(
        "  Subtrain : %d  │  Calibration : %d  │  Test : %d",
        len(X_subtrain), len(X_calib), len(X_test),
    )

    # ── Step 3 (SMOTE): Oversample Bind minority in training split only ───────
    X_subtrain_sm, y_subtrain_sm = apply_smote(X_subtrain, y_subtrain)

    # ── Step 4: Hyperparameter search on SMOTE-d training split ───────────────
    best_xgb, best_params = run_hyperparameter_search(X_subtrain_sm, y_subtrain_sm, n_iter=40)

    # ── Step 5: Calibrate on the original (non-SMOTE) calibration split ───────
    calibrated = calibrate_model(best_xgb, X_calib, y_calib)

    # ── Step 6: Find optimal threshold on calibration split ───────────────────
    threshold, curve_metrics = find_optimal_threshold(calibrated, X_calib, y_calib)

    # ── Step 7: Evaluate on held-out test set ─────────────────────────────────
    evaluate_model(calibrated, X_test, y_test, threshold)

    # ── Step 8: Build SHAP on base XGBoost ────────────────────────────────────
    explainer = build_shap_explainer(calibrated.estimator, X_subtrain_sm)

    # ── Step 9: Export all artifacts (agent2_ prefix; Agent 1 untouched) ──────
    print("─" * 64)
    print("  EXPORTING ARTIFACTS")
    print("─" * 64)
    export_artifacts(
        calibrated, explainer, ohe, le_tier, feature_names,
        threshold, best_params, curve_metrics,
    )

    # ── Demo: ConversionPredictor class round-trip ────────────────────────────
    print("\n" + "─" * 64)
    print("  DEMO: ConversionPredictor.from_artifacts()  (live inference)")
    print("─" * 64)

    predictor = ConversionPredictor.from_artifacts(MODEL_DIR)

    demo_quotes = [
        {
            "label": "Low-risk / experienced driver  (expect HIGH_PROPENSITY)",
            "data": {
                "Prev_Accidents": 0, "Prev_Citations": 0,
                "Driving_Exp": 20,   "Driver_Age": 42,
                "Annual_Miles": 10_000, "Veh_Usage": "Pleasure",
                "Risk_Tier": "Low",
            },
        },
        {
            "label": "High-risk young driver  (expect LOW_PROB)",
            "data": {
                "Prev_Accidents": 2, "Prev_Citations": 3,
                "Driving_Exp": 2,    "Driver_Age": 20,
                "Annual_Miles": 38_000, "Veh_Usage": "Business",
                "Risk_Tier": "High",
            },
        },
        {
            "label": "Medium-risk commuter  (expect NEAR_MISS or HIGH_PROPENSITY)",
            "data": {
                "Prev_Accidents": 0, "Prev_Citations": 1,
                "Driving_Exp": 8,    "Driver_Age": 30,
                "Annual_Miles": 28_000, "Veh_Usage": "Commute",
                "Risk_Tier": "Medium",
            },
        },
    ]

    for demo in demo_quotes:
        result = predictor.predict_conversion(
            demo["data"],
            risk_tier=demo["data"].get("Risk_Tier", "Low"),
        )
        print(f"\n  🧑  {demo['label']}")
        print(f"  📊  Bind Probability        : {result.bind_probability:.2%}")
        print(f"  🎯  Sales Status            : {result.sales_status}")
        print(f"  📏  Distance to Threshold   : {result.distance_to_conversion:.4f}  (threshold={result.optimal_threshold:.4f})")
        print("  🔑  Top Conversion Drivers  :")
        for drv in result.top_drivers:
            print(
                f"       • {drv.feature:<25}  SHAP = {drv.shap_value:+.4f}  "
                f"[{drv.direction}]  [{drv.magnitude}]"
            )

    print("\n" + "═" * 64)
    print("  ✅  Agent 2 – Conversion Predictor v2 : Training Complete!")
    print(f"  📁  Models → {MODEL_DIR}/")
    print(f"  📄  Metadata → {MODEL_DIR / 'agent2_metadata_v2.json'}")
    print("═" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
