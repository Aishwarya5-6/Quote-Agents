#!/usr/bin/env python3
"""
=============================================================================
  Agent 1 – Risk Profiler  │  Auto Insurance Multi-Agent Pipeline
  v3 — World-Class Engineering

  ✓ Gaussian-noisy synthetic labels      → model learns probabilities, not a formula
  ✓ Interaction features (reasoning)     → Miles_Per_Exp, Total_Incidents, Age_Exp_Gap
  ✓ OOD / Anomaly detector (safety)     → IsolationForest gate; flags corrupt quotes
  ✓ Cost-sensitive weights (economic)   → High-risk class 3× multiplier for False Lows
  ✓ neg_log_loss tuning                 → penalises confidently wrong predictions
  ✓ CalibratedClassifierCV (isotonic)   → statistically sound output probabilities
  ✓ SHAP TreeExplainer                  → base XGBoost; causal interventional SHAP
  ✓ Clean-slate management              → models/ and data/processed/ wiped entirely
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
import shutil
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import shap
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
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
AGENTS_DIR  = Path(__file__).resolve().parent           # …/agents/
BASE_DIR    = AGENTS_DIR.parent                          # …/Quote-Agents/
DATA_PATH   = BASE_DIR / "data" / "raw" / "insurance_data.csv"
MODEL_DIR   = BASE_DIR / "models"
PROC_DIR    = BASE_DIR / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE    = 0.20    # 20 % held-out test set
CALIB_SIZE   = 0.20    # 20 % of train-val used for isotonic calibration
NOISE_SCALE  = 2.0     # Gaussian noise σ for synthetic label generation

# ── Economic Layer ─────────────────────────────────────────────────────────
# In insurance, a False Low (missing a High-risk driver) leads to an
# underpriced policy and eventual claims losses — roughly 3× more costly
# than a False High (over-flagged Safe driver who simply declines the quote).
# A missed Medium-risk (priced as Low) is less severe but still costly —
# 2× multiplier forces the model to maintain a Medium recall floor.
HIGH_RISK_WEIGHT_MULTIPLIER:   float = 3.0
MEDIUM_RISK_WEIGHT_MULTIPLIER: float = 2.0

# ── Raw input features ───────────────────────────────────────────────────────
NUMERIC_FEATURES: List[str] = [
    "Prev_Accidents",
    "Prev_Citations",
    "Driving_Exp",
    "Driver_Age",
    "Annual_Miles",
]
CAT_FEATURES: List[str] = ["Veh_Usage"]

# ── Interaction / Reasoning Features ─────────────────────────────────────────
# Derived ONLY from raw inputs — not from the label formula.
# Capture risk-density patterns that raw features alone may not expose.
INTERACTION_FEATURES: List[str] = [
    "Miles_Per_Exp",    # Annual_Miles / (Driving_Exp + 1) — exposure density
    "Total_Incidents",  # Prev_Accidents + Prev_Citations  — combined incident load
    "Age_Exp_Gap",      # Driver_Age − Driving_Exp − 16   — delayed licensing signal
]

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

# ── Features for the OOD detector (raw inputs only) ─────────────────────────
# The OOD detector runs on the 6 raw features only, NOT the interaction features.
# Reason: interaction features like Miles_Per_Exp = 9,999,999 / 1 = 9,999,999
# saturate IsolationForest's path-length metric, making the corrupt record look
# 'average' because tree depth caps out.  Raw values like Annual_Miles=9,999,999
# are cleanly extreme and correctly trigger the anomaly detector.
RAW_FEATURES_FOR_OOD: List[str] = NUMERIC_FEATURES + ["Veh_Usage_Business", "Veh_Usage_Commute", "Veh_Usage_Pleasure"]

# ── OOD Detector config ───────────────────────────────────────────────────────
OOD_N_ESTIMATORS = 200     # More trees → more stable anomaly scores
# We use score_samples() and a hard percentile threshold instead of the
# contamination parameter.  contamination-based predict() labels the lowest N %
# of the training set as outliers — which incorrectly flags rare-but-valid
# profiles (e.g., young driver + accident).  A score threshold set at the
# 0.01th percentile of training scores only blocks quotes whose anomaly score
# is more extreme than 99.99 % of all real training quotes — i.e., only truly
# impossible or corrupted data.  The deterministic Physics Check (in
# explain_risk_prediction) handles logically impossible values that the
# statistical detector may not catch.
OOD_SCORE_PERCENTILE = 0.01  # flag if below 0.01th pct of training score dist
OOD_FLAG             = "ACTION_REQUIRED: DATA_ANOMALY"

# ── Observability & Drift Monitor ────────────────────────────────────────────
DRIFT_THRESHOLD_PCT  = 0.10                         # 10 % mean shift → alert
DRIFT_ALERT_STATUS   = "SYSTEM_ALERT: DATA_DRIFT_DETECTED"
DRIFT_OK_STATUS      = "OK: NO_DRIFT_DETECTED"

# ── Counterfactual 'What-If' search ──────────────────────────────────────────
CF_MILES_STEP        = 1_000   # iterate Annual_Miles down in 1 000 mi steps
CF_MAX_ITER          = 200     # safety cap — never loop more than 200 times

AGENT1_PROCESSED_FILE = "cleaned_agent1_data.csv"


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 – CLEAN SLATE  (wipe entire output directories)
# ─────────────────────────────────────────────────────────────────────────────
def cleanup_previous_artifacts() -> None:
    """
    Completely wipe and recreate models/ and data/processed/ before training.

    WHY wipe the entire directory (not just a listed set of files)?
    ───────────────────────────────────────────────────────────────
    Any future artifact added to the pipeline is automatically cleaned up
    with no manual maintenance of a file-name list.  Stale files from
    previous versions (e.g., a renamed pkl) can never silently persist
    and be loaded by downstream agents.  Guarantees an absolutely clean
    slate before every training run.
    """
    log.info("── Step 0: Wiping output directories for clean slate ────────")
    for target_dir in (MODEL_DIR, PROC_DIR):
        if target_dir.exists():
            shutil.rmtree(target_dir)
            log.info("  🗑  Removed  %s/", target_dir.relative_to(BASE_DIR))
        target_dir.mkdir(parents=True, exist_ok=True)
        log.info("  ✓  Re-created %s/", target_dir.relative_to(BASE_DIR))
    log.info("  ✓  Clean slate confirmed.")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 – DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load CSV, convert Annual_Miles_Range → numeric midpoint,
    and fill nulls across all modelling columns.

    Imputation strategy
    ───────────────────
    Numeric   → column median  (robust to outliers — a single bad entry
                                won't skew the fill value)
    Categorical → column mode  (most common usage pattern as neutral default)
    """
    log.info("── Step 1: Loading data → %s ────────────────────────────────", path.name)
    df = pd.read_csv(path, low_memory=False)
    log.info("  Raw shape  : %d rows × %d cols", *df.shape)

    # Convert range-string mileage column → numeric midpoint
    if "Annual_Miles_Range" in df.columns and "Annual_Miles" not in df.columns:
        df["Annual_Miles"] = df["Annual_Miles_Range"].map(MILES_MAP)

    # Null handling: numeric → column median
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

    Weights grounded in auto-insurance loss-ratio research:
        +4      prior accident      (strongest single predictor of future claims)
        +2      prior citation      (moderate behavioural signal)
        +3/2/1  experience penalty  (≤3 / ≤7 / ≤15 years of licensed driving)
        +2/1    young-driver penalty (<22 / <26 years of age)
        +2/1    high-mileage penalty (>45 K / >25 K miles per year)
        +1      business-use uplift  (higher liability exposure)
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
    ANTI-LEAKAGE: add Gaussian noise before bucketing into tiers.

    Without noise, the label is a perfect deterministic function of the
    raw features — XGBoost memorises it (100 % accuracy, zero real learning).
    Noise σ=2.0 blurs tier boundaries realistically: a borderline driver
    has a genuine probability of landing in either adjacent tier, forcing
    the model to learn calibrated probabilities rather than hard rules.

    Thresholds (applied to NOISY score):
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
#  STEP 3a – INTERACTION / REASONING FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer three interaction features that capture risk-density patterns
    the raw inputs alone may not fully expose to the model.

    Miles_Per_Exp  =  Annual_Miles / (Driving_Exp + 1)
        WHY: A 3-year driver doing 50 K miles/yr is far riskier than a
             20-year driver doing the same.  Normalising by experience
             quantifies per-year exposure density — a key actuarial metric.
             (+1 avoids division-by-zero for brand-new drivers.)

    Total_Incidents  =  Prev_Accidents + Prev_Citations
        WHY: Captures the combined incident burden in a single axis that
             XGBoost can split on.  Even though both components are in X,
             their sum gives the model an explicit "total history" feature
             without the model having to learn that combination itself.

    Age_Exp_Gap  =  Driver_Age − Driving_Exp − 16
        WHY: 16 is the minimum legal driving age in most US states.
             A gap of 0 means the driver started at 16 (normal).
             A gap of 10 means they started at 26 — a delayed licensing
             history that correlates with licence suspensions, DUI periods,
             or extended periods abroad.  Raw age and experience alone
             cannot expose this signal; their combination does.
    """
    log.info(
        "── Step 3a: Adding interaction features  "
        "[Miles_Per_Exp | Total_Incidents | Age_Exp_Gap]"
    )
    df = df.copy()
    df["Miles_Per_Exp"]   = df["Annual_Miles"] / (df["Driving_Exp"] + 1)
    df["Total_Incidents"] = df["Prev_Accidents"] + df["Prev_Citations"]
    df["Age_Exp_Gap"]     = df["Driver_Age"] - df["Driving_Exp"] - 16
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3b – ENCODING  (11 features: 5 numeric + 3 interaction + 3 OHE)
# ─────────────────────────────────────────────────────────────────────────────
def encode_features(
    df:  pd.DataFrame,
    fit: bool                    = True,
    ohe: Optional[OneHotEncoder] = None,
    le:  Optional[LabelEncoder]  = None,
) -> Tuple[pd.DataFrame, pd.Series, OneHotEncoder, LabelEncoder, List[str]]:
    """
    Build X (11 columns after OHE) and encoded y.

    Feature matrix layout
    ─────────────────────
    Numeric (5)     : Prev_Accidents, Prev_Citations, Driving_Exp,
                      Driver_Age, Annual_Miles
    Interaction (3) : Miles_Per_Exp, Total_Incidents, Age_Exp_Gap
    OHE (3)         : Veh_Usage_Business, Veh_Usage_Commute, Veh_Usage_Pleasure
    ─────────────────────────────────────────────────────────────────────────
    Total           : 11 features

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

    # Assemble: 5 numeric + 3 interaction + 3 OHE
    all_numeric   = NUMERIC_FEATURES + INTERACTION_FEATURES
    X             = pd.concat([df[all_numeric], df_ohe], axis=1)
    feature_names = all_numeric + veh_cols

    # Label-encode target
    risk_arr = df["Risk_Tier"].to_numpy()
    y = pd.Series(
        le.fit_transform(risk_arr) if fit else le.transform(risk_arr),
        index=df.index,
    )

    log.info(
        "── Step 3b: Feature matrix : %d rows × %d cols  │  classes : %s",
        *X.shape, list(le.classes_),
    )
    return X, y, ohe, le, feature_names


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4a – OOD DETECTOR  (IsolationForest safety layer)
# ─────────────────────────────────────────────────────────────────────────────
def train_ood_detector(
    X_train: pd.DataFrame,
) -> Tuple[IsolationForest, float]:
    """
    Train an IsolationForest anomaly detector and compute a score threshold.

    HOW IsolationForest WORKS
    ─────────────────────────
    It builds random trees that randomly partition the feature space.
    Normal points sit deep in the distribution and need many splits to
    isolate (high score → inlier).  Anomalies are isolated in few splits
    (low score → outlier).  score_samples() returns a continuous anomaly
    score rather than a binary label.

    WHY score_samples() + percentile threshold instead of contamination?
    ────────────────────────────────────────────────────────────────────
    The contamination parameter labels the lowest N % of the training set
    as outliers — so even a rare-but-valid profile (young driver + accident)
    can be flagged simply for being statistically uncommon.  Instead, we
    compute the 0.1th percentile of training scores and use it as the
    rejection boundary.  This means only quotes whose anomaly score falls
    below 99.9 % of all seen training quotes are blocked — truly impossible
    or corrupted inputs like Driver_Age=-5 or Annual_Miles=9,999,999.

    Returns
    ───────
    ood        : fitted IsolationForest
    threshold  : float score below which a new quote is flagged as OOD
    """
    log.info("── Step 4a: Training OOD detector (IsolationForest) ─────────")
    ood = IsolationForest(
        n_estimators=OOD_N_ESTIMATORS,
        contamination="auto",         # only used internally; we override threshold
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    X_raw = X_train[RAW_FEATURES_FOR_OOD]
    ood.fit(X_raw)

    # Compute anomaly scores on training data and store the percentile threshold
    train_scores = ood.score_samples(X_raw)
    threshold    = float(np.percentile(train_scores, OOD_SCORE_PERCENTILE))

    log.info(
        "  IsolationForest fitted on %d raw features  "
        "│  training rows=%d",
        len(RAW_FEATURES_FOR_OOD), len(X_raw),
    )
    log.info(
        "  OOD threshold (%.2f%% pct of train scores) : %.6f",
        OOD_SCORE_PERCENTILE, threshold,
    )
    return ood, threshold


# ─────────────────────────────────────────────────────────────────────────────
#  ADVERSARIAL TEST GENERATOR  (Red-Team OOD Safety Verification)
# ─────────────────────────────────────────────────────────────────────────────
# A hard-coded suite of logically impossible or physically absurd insurance
# quotes.  Each example probes a specific failure mode: domain violation,
# sign error, impossible cross-field relationship, or extreme data corruption.
ADVERSARIAL_PROFILES: List[Dict[str, Any]] = [
    {
        "label": "AGE_EXP_IMPOSSIBLE  │ age=18  but exp=40 (more exp than lifetime)",
        "data":  {"Driver_Age": 18,  "Driving_Exp": 40, "Prev_Accidents": 0,
                  "Prev_Citations": 0, "Annual_Miles": 15_000, "Veh_Usage": "Commute"},
    },
    {
        "label": "NEGATIVE_MILEAGE   │ Annual_Miles=-500 (physically impossible)",
        "data":  {"Driver_Age": 30,  "Driving_Exp": 8,  "Prev_Accidents": 0,
                  "Prev_Citations": 0, "Annual_Miles": -500, "Veh_Usage": "Pleasure"},
    },
    {
        "label": "CORRUPT_ENTRY      │ age=−5, miles=9,999,999 (data entry corruption)",
        "data":  {"Driver_Age": -5,  "Driving_Exp": 0,  "Prev_Accidents": 0,
                  "Prev_Citations": 0, "Annual_Miles": 9_999_999, "Veh_Usage": "Pleasure"},
    },
    {
        "label": "ZERO_AGE           │ Driver_Age=0 (pre-birth driver)",
        "data":  {"Driver_Age": 0,   "Driving_Exp": 0,  "Prev_Accidents": 5,
                  "Prev_Citations": 10, "Annual_Miles": 50_000, "Veh_Usage": "Business"},
    },
    {
        "label": "NEGATIVE_INCIDENTS │ Prev_Accidents=−3 (invalid sign flip)",
        "data":  {"Driver_Age": 35,  "Driving_Exp": 10, "Prev_Accidents": -3,
                  "Prev_Citations": 0, "Annual_Miles": 20_000, "Veh_Usage": "Commute"},
    },
]


def generate_adversarial_examples(
    ood_detector:  IsolationForest,
    ood_threshold: float,
    ohe:           OneHotEncoder,
    feature_names: List[str],
) -> None:
    """
    Red-team the IsolationForest OOD gate with a battery of adversarial quotes.

    Each profile is deliberately impossible or corrupt.  The function:
      1. Builds the full feature vector for each adversarial input.
      2. Scores the RAW features (no interaction features) against the detector.
      3. Logs "✅ BLOCKED" when the gate correctly refuses to predict, and
         "❌ MISSED"  when a corrupt input slips through uncaught.

    This test runs automatically at the end of every training run so that any
    future model or threshold change that weakens the safety gate is immediately
    visible in the training log.
    """
    log.info("── Adversarial OOD Gate Test ──────────────────────────────────")
    n_caught = 0
    for profile in ADVERSARIAL_PROFILES:
        X_adv   = _build_inference_row(profile["data"], ohe, feature_names)
        score   = float(ood_detector.score_samples(X_adv[RAW_FEATURES_FOR_OOD])[0])
        blocked = score < ood_threshold
        if blocked:
            n_caught += 1
            log.info(
                "  ✅  BLOCKED  [score=%.4f < threshold=%.4f]  %s",
                score, ood_threshold, profile["label"],
            )
        else:
            log.warning(
                "  ❌  MISSED   [score=%.4f ≥ threshold=%.4f]  %s  "
                "— OOD gate did NOT catch this adversarial input.",
                score, ood_threshold, profile["label"],
            )
    log.info(
        "  Adversarial Gate Test : %d / %d anomalies correctly BLOCKED.",
        n_caught, len(ADVERSARIAL_PROFILES),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  OBSERVABILITY & DRIFT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def calculate_feature_drift(
    incoming_batch: List[Dict[str, Any]],
    training_stats: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compare a batch of incoming live quotes against the training distribution.

    Algorithm
    ─────────
    1. Compute the mean of Annual_Miles across the incoming batch.
    2. Compare to the training mean stored in training_stats.
    3. If |pct_shift| > DRIFT_THRESHOLD_PCT (10 %), return DRIFT_ALERT_STATUS.

    Parameters
    ──────────
    incoming_batch   : list of raw quote dicts (same keys as the 6 raw inputs)
    training_stats   : {"Annual_Miles_mean": float, ...} saved at training time

    Returns
    ───────
    JSON-serialisable dict with:
      status          — DRIFT_ALERT_STATUS or DRIFT_OK_STATUS
      feature         — monitored feature name
      training_mean   — reference mean from training
      incoming_mean   — mean of the current batch
      pct_shift       — percentage shift (positive = higher than training)
      n_samples       — batch size
    """
    if not incoming_batch:
        return {"status": DRIFT_OK_STATUS, "detail": "Empty batch — no drift check performed."}

    train_mean = training_stats.get("Annual_Miles_mean", 0.0)
    batch_mean = float(
        np.mean([float(q.get("Annual_Miles", train_mean)) for q in incoming_batch])
    )
    pct_shift = (batch_mean - train_mean) / (train_mean + 1e-9)
    drifted   = abs(pct_shift) > DRIFT_THRESHOLD_PCT

    result: Dict[str, Any] = {
        "status":        DRIFT_ALERT_STATUS if drifted else DRIFT_OK_STATUS,
        "feature":       "Annual_Miles",
        "training_mean": round(train_mean, 2),
        "incoming_mean": round(batch_mean, 2),
        "pct_shift_pct": round(pct_shift * 100, 2),
        "n_samples":     len(incoming_batch),
    }
    if drifted:
        log.warning(
            "  ⚠  DATA DRIFT on Annual_Miles  "
            "│  training_mean=%.0f  incoming_mean=%.0f  shift=%+.1f %%",
            train_mean, batch_mean, pct_shift * 100,
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4b – SAMPLE WEIGHTS  (cost-sensitive: High 3× / Medium 2× penalty)
# ─────────────────────────────────────────────────────────────────────────────
def compute_sample_weights(
    y:                 pd.Series,
    le:                LabelEncoder,
    high_multiplier:   float = HIGH_RISK_WEIGHT_MULTIPLIER,
    medium_multiplier: float = MEDIUM_RISK_WEIGHT_MULTIPLIER,
) -> np.ndarray:
    """
    Cost-sensitive inverse-frequency weights with economic multipliers
    for the High-Risk and Medium-Risk classes.

    Base formula:   w_c  = N / (K × n_c)        [standard inverse-frequency]
    High-Risk:      w_Hi = base_w_Hi × 3.0      [economic cost adjustment]
    Medium-Risk:    w_Me = base_w_Me × 2.0      [recall-collapse prevention]
    Low-Risk:       w_Lo = base_w_Lo × 1.0      [no extra penalty]

    WHY the 3× / 2× multipliers?
    ─────────────────────────────
    High  (3×):  Missing a high-risk driver leads to an underpriced policy
                 and eventual claims losses — the most expensive error.
    Medium (2×): Without a boost, XGBoost collapses nearly all Medium
                 predictions into Low (recall ≈ 0.005).  A 2× multiplier
                 forces meaningful gradient signal on Medium misclassifications
                 while keeping the dominant Low class at base weight.
    Low   (1×):  Largest class — no boost needed.

    Class indices are resolved dynamically from le.classes_ so this
    function remains correct even if label encoding order changes.
    """
    counts     = Counter(y.tolist())
    total      = len(y)
    n_cls      = len(counts)
    high_idx   = int(np.where(le.classes_ == "High")[0][0])
    medium_idx = int(np.where(le.classes_ == "Medium")[0][0])

    w_map: Dict[int, float] = {}
    for cls, cnt in counts.items():
        base_w = total / (n_cls * cnt)
        if cls == high_idx:
            w_map[cls] = base_w * high_multiplier
        elif cls == medium_idx:
            w_map[cls] = base_w * medium_multiplier
        else:
            w_map[cls] = base_w          # Low — no extra penalty

    weights = np.array([w_map[int(yi)] for yi in y])
    log.info(
        "── Step 4b: Sample weights  range [%.4f – %.4f]  "
        "(High=%.1f× │ Medium=%.1f× │ Low=1.0×)",
        weights.min(), weights.max(), high_multiplier, medium_multiplier,
    )
    return weights


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

    WHY neg_log_loss (not accuracy)?
    ─────────────────────────────────
    log(p) → −∞ as p → 0, so a confidently wrong prediction receives an
    enormous penalty.  This forces the search to select hyperparameters
    that produce well-calibrated probability distributions — not just the
    highest label-accuracy score.

    5-fold StratifiedKFold ensures every fold preserves the ~6/76/18 %
    High/Low/Medium class ratio.  250 total fits (50 iter × 5 folds).
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

    method='isotonic' — non-parametric monotone mapping from raw softmax
                        scores to true posterior probabilities.  More
                        flexible than Platt scaling; recommended for
                        n_calib > ~1 000 samples.

    cv='prefit'       — base estimator is already fitted on X_subtrain;
                        only the isotonic layer is fitted here on the
                        separate 16 % calibration holdout.  This prevents
                        any leakage between the training and calibration steps.
    """
    log.info("── Step 6: Isotonic probability calibration ─────────────────")
    calibrated = CalibratedClassifierCV(base_xgb, method="isotonic", cv="prefit")
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

    WHY the base estimator (not the calibrated wrapper)?
    ─────────────────────────────────────────────────────
    CalibratedClassifierCV wraps XGBoost inside isotonic regressors; SHAP's
    TreeExplainer cannot traverse the isotonic layer.  The base XGBoost
    preserves identical feature-importance rankings — calibration only
    adjusts probability magnitudes, not the rank order of decisions.

    feature_perturbation='interventional' produces causally interpretable
    SHAP values (suitable for regulatory audit), using a 500-row background.
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
#  SHARED ENCODING HELPER  (training & inference use same path)
# ─────────────────────────────────────────────────────────────────────────────
def _build_inference_row(
    quote_data:    Dict[str, Any],
    ohe:           OneHotEncoder,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Encode a raw quote dict into the 11-column feature vector expected by
    both the OOD detector and the XGBoost model.

    Computes all three interaction features from the raw inputs so that
    training-time and inference-time feature construction share one code path.
    """
    row = dict(quote_data)

    # ── Interaction features ──────────────────────────────────────────────
    miles = float(row["Annual_Miles"])
    exp   = float(row["Driving_Exp"])
    age   = float(row["Driver_Age"])
    row["Miles_Per_Exp"]   = miles / (exp + 1)
    row["Total_Incidents"] = float(row["Prev_Accidents"]) + float(row["Prev_Citations"])
    row["Age_Exp_Gap"]     = age - exp - 16

    # ── OHE Veh_Usage ─────────────────────────────────────────────────────
    veh_df      = pd.DataFrame([[row["Veh_Usage"]]], columns=["Veh_Usage"])
    veh_enc_arr = np.array(ohe.transform(veh_df))
    veh_cols    = list(ohe.get_feature_names_out(["Veh_Usage"]))
    for col, val in zip(veh_cols, veh_enc_arr[0]):
        row[col] = val

    return pd.DataFrame([row])[feature_names].astype(float)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 9 – PREDICT & EXPLAIN  (OOD safety gate → XGBoost → SHAP)
# ─────────────────────────────────────────────────────────────────────────────
def explain_risk_prediction(
    quote_data:   Dict[str, Any],
    *,
    model:        Any,                   # CalibratedClassifierCV at runtime
    explainer:    shap.TreeExplainer,
    ohe:          OneHotEncoder,
    le:           LabelEncoder,
    feature_names: List[str],
    ood_detector: IsolationForest,
    ood_threshold: float,
) -> Dict[str, Any]:
    """
    Predict Risk Tier for ONE customer quote with a three-stage pipeline:

    ┌──────────────────────────────────────────────────────────────────┐
    │ Stage 1 │ Build 11-feature vector from raw quote dict            │
    │ Stage 2 │ OOD Gate — IsolationForest.predict()                   │
    │         │   -1 (outlier) → return {"status": OOD_FLAG}          │
    │         │    1 (normal)  → continue to Stage 3                  │
    │ Stage 3 │ Calibrated XGBoost prediction + SHAP explanation       │
    └──────────────────────────────────────────────────────────────────┘

    Input  (quote_data keys):
        Prev_Accidents, Prev_Citations, Driving_Exp,
        Driver_Age, Annual_Miles, Veh_Usage

    Normal output keys:
        status, predicted_tier, predicted_class_id, confidence,
        class_probabilities, top_3_features, all_shap_values

    Anomaly output keys:
        status (= OOD_FLAG), input_data, message
    """
    # ── Stage 1a: Deterministic Physics Check ─────────────────────────────
    # Hard-coded domain rules that catch logically impossible inputs before
    # the statistical OOD detector even runs.  This is the *last* line of
    # defence when the predictor is called directly (without the Pydantic API
    # layer).  Each rule mirrors a physical or legal constraint that no valid
    # insurance quote can violate.
    age = float(quote_data.get("Driver_Age", 0))
    exp = float(quote_data.get("Driving_Exp", 0))
    miles = float(quote_data.get("Annual_Miles", 0))
    acc = float(quote_data.get("Prev_Accidents", 0))
    cit = float(quote_data.get("Prev_Citations", 0))

    physics_violations: List[str] = []
    if age < 16:
        physics_violations.append(f"Driver_Age={age} < 16 (below legal driving age)")
    if miles < 0:
        physics_violations.append(f"Annual_Miles={miles} < 0 (negative mileage impossible)")
    if acc < 0:
        physics_violations.append(f"Prev_Accidents={acc} < 0 (negative count impossible)")
    if cit < 0:
        physics_violations.append(f"Prev_Citations={cit} < 0 (negative count impossible)")
    if exp > (age - 15):
        physics_violations.append(
            f"Driving_Exp={exp} > Driver_Age−15={age - 15} "
            f"(more experience than years since legal driving age)"
        )

    if physics_violations:
        log.warning(
            "  🛑  Deterministic Physics Check FAILED for input: %s │ %s",
            quote_data, physics_violations,
        )
        return {
            "status":     OOD_FLAG,
            "input_data": quote_data,
            "message": (
                "Deterministic Physics Check Failed: "
                "Logically impossible driver inputs detected. "
                + "; ".join(physics_violations)
            ),
        }

    # ── Stage 1b: Encode ──────────────────────────────────────────────────
    X_input = _build_inference_row(quote_data, ohe, feature_names)

    # ── Stage 2: OOD Safety Gate (IsolationForest) ────────────────────────
    # score_samples returns a continuous anomaly score; lower = more anomalous.
    # We flag if the score falls below the pre-computed training percentile
    # threshold rather than using predict() which depends on contamination %.
    # IMPORTANT: slice to RAW_FEATURES_FOR_OOD only — the detector was trained
    # on the 8 raw features (5 numeric + 3 OHE), NOT the 3 interaction features.
    # Passing interaction features causes a feature-name mismatch error.
    anomaly_score = float(ood_detector.score_samples(X_input[RAW_FEATURES_FOR_OOD])[0])
    if anomaly_score < ood_threshold:
        log.warning("  ⚠  OOD anomaly detected for input: %s", quote_data)
        return {
            "status":     OOD_FLAG,
            "input_data": quote_data,
            "message": (
                "This quote falls outside the distribution of known training data. "
                "XGBoost prediction skipped to prevent a confident-but-wrong result. "
                "Please route to a human underwriter for manual review."
            ),
        }

    # ── Stage 3a: Calibrated prediction ───────────────────────────────────
    pred_class = int(model.predict(X_input)[0])
    pred_proba = model.predict_proba(X_input)[0]
    pred_tier  = str(le.inverse_transform([pred_class])[0])

    # ── Stage 3b: SHAP from base XGBoost TreeExplainer ────────────────────
    shap_exp  = explainer(X_input, check_additivity=False)
    shap_vals = np.array(shap_exp.values)           # cast → (1, n_features, n_classes)
    sv        = shap_vals[0, :, pred_class]          # (n_features,) for predicted class
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
        "status":              "OK",
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
        # result["status"] == "OK" → normal prediction
        # result["status"] == OOD_FLAG → route to human underwriter
    """

    def __init__(
        self,
        model:           CalibratedClassifierCV,
        explainer:       shap.TreeExplainer,
        ohe:             OneHotEncoder,
        le:              LabelEncoder,
        feature_names:   List[str],
        ood_detector:    IsolationForest,
        ood_threshold:   float,
        training_stats:  Dict[str, float],
    ) -> None:
        self.model           = model
        self.explainer       = explainer
        self.ohe             = ohe
        self.le              = le
        self.feature_names   = feature_names
        self.ood_detector    = ood_detector
        self.ood_threshold   = ood_threshold
        self.training_stats  = training_stats

    @classmethod
    def from_artifacts(
        cls, model_dir: str = "../models/"
    ) -> "RiskProfilerPredictor":
        """Load all seven serialised artifacts from disk and return a ready predictor."""
        p = Path(model_dir)
        log.info("Loading artifacts from %s", p.resolve())
        return cls(
            model           = joblib.load(p / "calibrated_risk_profiler.pkl"),
            explainer       = joblib.load(p / "shap_explainer.pkl"),
            ohe             = joblib.load(p / "ohe_encoder.pkl"),
            le              = joblib.load(p / "label_encoder.pkl"),
            feature_names   = joblib.load(p / "feature_names.pkl"),
            ood_detector    = joblib.load(p / "ood_detector.pkl"),
            ood_threshold   = joblib.load(p / "ood_threshold.pkl"),
            training_stats  = joblib.load(p / "training_stats.pkl"),
        )

    def generate_counterfactual_advice(
        self,
        quote_data:   Dict[str, Any],
        current_tier: str,
    ) -> Optional[str]:
        """
        Counterfactual 'What-If' Analyzer for High / Medium risk customers.

        Strategy
        ────────
        Two levers are probed in isolation:

        Lever A – Annual Mileage Reduction
          Iteratively lowers Annual_Miles by CF_MILES_STEP (1 000 mi) until the
          predicted tier improves to 'Low' or a floor of 1 000 mi is reached.

        Lever B – Incident Record Clearing
          Sets Prev_Accidents and Prev_Citations to zero and checks whether
          that single change flips the tier to 'Low'.

        The first lever that achieves a tier improvement is reported in plain
        English.  If neither lever is sufficient, a generic safe-driver guidance
        string is returned so the dashboard always has actionable advice.

        Returns None if current_tier is already 'Low' (no improvement needed).
        """
        if current_tier == "Low":
            return None  # already lowest risk tier — no action needed

        # ── Lever A: iterative mileage reduction ──────────────────────────────
        probe_a       = dict(quote_data)
        original_miles = int(float(probe_a.get("Annual_Miles", 0)))
        target         = max(original_miles - CF_MILES_STEP, 1_000)
        iterations     = 0
        while target >= 1_000 and iterations < CF_MAX_ITER:
            probe_a["Annual_Miles"] = target
            trial = explain_risk_prediction(
                probe_a,
                model=self.model,
                explainer=self.explainer,
                ohe=self.ohe,
                le=self.le,
                feature_names=self.feature_names,
                ood_detector=self.ood_detector,
                ood_threshold=self.ood_threshold,
            )
            if trial.get("status") == "OK" and trial.get("predicted_tier") == "Low":
                return (
                    f"Reducing annual mileage from {original_miles:,} to "
                    f"{target:,} miles would likely transition this profile "
                    f"to a Low Risk tier."
                )
            target     -= CF_MILES_STEP
            iterations += 1

        # ── Lever B: clear incident record ────────────────────────────────────
        probe_b = dict(quote_data)
        probe_b["Prev_Accidents"] = 0
        probe_b["Prev_Citations"] = 0
        trial_b = explain_risk_prediction(
            probe_b,
            model=self.model,
            explainer=self.explainer,
            ohe=self.ohe,
            le=self.le,
            feature_names=self.feature_names,
            ood_detector=self.ood_detector,
            ood_threshold=self.ood_threshold,
        )
        if trial_b.get("status") == "OK" and trial_b.get("predicted_tier") == "Low":
            total_incidents = (
                int(float(quote_data.get("Prev_Accidents", 0)))
                + int(float(quote_data.get("Prev_Citations", 0)))
            )
            return (
                f"Clearing the {total_incidents} incident record(s) "
                f"(accidents + citations) would likely transition this profile "
                f"to a Low Risk tier."
            )

        # ── Fallback: generic guidance ─────────────────────────────────────────
        return (
            f"This profile is currently classified as {current_tier} Risk. "
            "Sustained incident-free driving and a gradual reduction in annual "
            "mileage are the strongest levers for improving the risk tier over time."
        )

    def predict_and_explain(self, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full inference pipeline entry point for CrewAI tool's _run() method.

        Pipeline
        ────────
        1. explain_risk_prediction() → OOD gate → XGBoost → SHAP
        2. calculate_feature_drift() → compares quote against training mean
        3. generate_counterfactual_advice() → What-If mileage / incident levers
        4. Assembles dashboard_metrics dict for the frontend What-If tab
        """
        result = explain_risk_prediction(
            quote_data,
            model=self.model,
            explainer=self.explainer,
            ohe=self.ohe,
            le=self.le,
            feature_names=self.feature_names,
            ood_detector=self.ood_detector,
            ood_threshold=self.ood_threshold,
        )

        # OOD gate fired — skip drift and counterfactual (input is corrupt)
        if result.get("status") == OOD_FLAG:
            return result

        # ── Drift Monitor ─────────────────────────────────────────────────────
        drift_status = calculate_feature_drift([quote_data], self.training_stats)

        # ── Counterfactual Advice ─────────────────────────────────────────────
        current_tier        = result.get("predicted_tier", "Low")
        counterfactual_advice = self.generate_counterfactual_advice(
            quote_data, current_tier
        )

        result["dashboard_metrics"] = {
            "drift_status":          drift_status,
            "counterfactual_advice": counterfactual_advice,
        }
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 11 – ARTIFACT EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export_artifacts(
    calibrated:     CalibratedClassifierCV,
    explainer:      shap.TreeExplainer,
    ohe:            OneHotEncoder,
    le:             LabelEncoder,
    feature_names:  List[str],
    ood_detector:   IsolationForest,
    ood_threshold:  float,
    training_stats: Dict[str, float],
    df_processed:   pd.DataFrame,
) -> None:
    """
    Serialise all eight pipeline components with joblib (compress=3).

    Artifacts saved to models/
    ──────────────────────────
    calibrated_risk_profiler.pkl  — CalibratedClassifierCV (the live model)
    shap_explainer.pkl            — TreeExplainer bound to base XGBoost
    ohe_encoder.pkl               — OneHotEncoder for Veh_Usage
    label_encoder.pkl             — LabelEncoder (High/Low/Medium ↔ 0/1/2)
    feature_names.pkl             — Ordered list of 11 feature names
    ood_detector.pkl              — IsolationForest anomaly detector
    ood_threshold.pkl             — Pre-computed score threshold (0.1th percentile)
    training_stats.pkl            — Feature means for drift monitoring
    manifest.json                 — Human-readable pipeline record for audit

    compress=3: ~50 % size reduction vs uncompressed at minimal I/O cost.
    """
    log.info("── Step 11: Exporting artifacts ─────────────────────────────")

    pkl_artifacts = {
        "calibrated_risk_profiler.pkl": calibrated,
        "shap_explainer.pkl":            explainer,
        "ohe_encoder.pkl":               ohe,
        "label_encoder.pkl":             le,
        "feature_names.pkl":             feature_names,
        "ood_detector.pkl":              ood_detector,
        "ood_threshold.pkl":             ood_threshold,
        "training_stats.pkl":            training_stats,
    }

    for fname, obj in pkl_artifacts.items():
        out = MODEL_DIR / fname
        joblib.dump(obj, out, compress=3)
        log.info("  💾  Saved  models/%-38s %d KB", fname, out.stat().st_size // 1024)

    manifest = {
        "agent":                    "Agent 1 – Risk Profiler v3",
        "model_type":               "CalibratedClassifierCV(isotonic, cv=prefit) → XGBClassifier",
        "objective":                "multi:softprob (3-class)",
        "tuning_metric":            "neg_log_loss",
        "noise_scale":              NOISE_SCALE,
        "high_risk_weight_multiplier": HIGH_RISK_WEIGHT_MULTIPLIER,
        "classes":                  list(le.classes_),
        "n_features":               len(feature_names),
        "feature_names":            feature_names,
        "interaction_features":     INTERACTION_FEATURES,
        "ood_detector":             (
            f"IsolationForest(n_estimators={OOD_N_ESTIMATORS})"
        ),
        "ood_threshold_percentile":  OOD_SCORE_PERCENTILE,
        "ood_threshold_value":       round(ood_threshold, 6),
        "shap_method":              "TreeExplorer (interventional) on base XGBClassifier",
        "drift_monitor":            {
            "monitored_feature":  "Annual_Miles",
            "threshold_pct":      DRIFT_THRESHOLD_PCT * 100,
            "training_stats":     training_stats,
        },
        "artifacts":                list(pkl_artifacts.keys()),
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
    print("  AGENT 1 – RISK PROFILER v3  │  Training Pipeline")
    print("═" * 64 + "\n")

    # ── Step 0: Wipe output directories ───────────────────────────────────
    cleanup_previous_artifacts()

    # ── Step 1: Load & clean ──────────────────────────────────────────────
    df = load_and_prepare_data()

    # ── Step 2: Generate noisy Risk_Tier labels ───────────────────────────
    df = generate_risk_labels(df, noise_scale=NOISE_SCALE)

    # ── Step 3a: Add interaction / reasoning features ─────────────────────
    df = add_interaction_features(df)

    # ── Step 3b: Encode (11 features: 5 numeric + 3 interaction + 3 OHE) ─
    X, y, ohe, le, feature_names = encode_features(df)
    n_classes = len(le.classes_)

    # ── Step 4: Three-way stratified split  64 % │ 16 % │ 20 % ──────────
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

    # ── Step 4a: Train OOD detector on subtrain feature space ────────────
    ood_detector, ood_threshold = train_ood_detector(X_subtrain)

    # ── Step 4a+: Capture training distribution for drift monitoring ─────
    training_stats: Dict[str, float] = {
        f"{col}_mean": float(X_subtrain[col].mean())
        for col in NUMERIC_FEATURES
    }
    log.info(
        "  Training stats captured for drift monitor: %s",
        {k: round(v, 2) for k, v in training_stats.items()},
    )

    # ── Step 4b: Cost-sensitive sample weights (High = 3× base weight) ───
    sw_subtrain = compute_sample_weights(y_subtrain, le)

    # ── Step 5: Hyperparameter search ────────────────────────────────────
    best_xgb, best_params = run_hyperparameter_search(
        X_subtrain, y_subtrain, n_classes, sw_subtrain, n_iter=50
    )

    # ── Step 6: Calibrate on held-out calibration set ────────────────────
    calibrated = calibrate_model(best_xgb, X_calib, y_calib)

    # ── Step 7: Evaluate calibrated model on test set ────────────────────
    evaluate_model(calibrated, X_test, y_test, le)

    # ── Step 8: Build SHAP on base XGBoost ───────────────────────────────
    explainer = build_shap_explainer(calibrated.estimator, X_subtrain)

    # ── Step 9: Demo predictions (includes OOD gate test) ────────────────
    print("─" * 64)
    print("  DEMO: explain_risk_prediction()  (incl. OOD safety gate)")
    print("─" * 64)

    demo_customers = [
        {
            "label": "High-Risk   │ accident + citation, young novice, business",
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
        {
            "label": "OOD Test    │ impossible values (age=−5, miles=9,999,999)",
            "data":  {
                "Prev_Accidents": 0, "Prev_Citations": 0,
                "Driving_Exp":    0, "Driver_Age":    -5,
                "Annual_Miles": 9_999_999, "Veh_Usage": "Pleasure",
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
            ood_detector=ood_detector,
            ood_threshold=ood_threshold,
        )
        print(f"\n  🧑  {demo['label']}")
        if result.get("status") == OOD_FLAG:
            print(f"  🚨  STATUS : {OOD_FLAG}")
            print(f"      {result['message']}")
        else:
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

    # ── Step 9b: Adversarial OOD Gate Test ──────────────────────────────
    print("\n" + "─" * 64)
    print("  ADVERSARIAL SAFETY TEST  (Red-Team OOD Gate)")
    print("─" * 64)
    generate_adversarial_examples(ood_detector, ood_threshold, ohe, feature_names)

    # ── Step 11: Export all artifacts ─────────────────────────────────────
    print("\n" + "─" * 64)
    print("  EXPORTING ARTIFACTS")
    print("─" * 64)

    keep_cols = (
        NUMERIC_FEATURES
        + INTERACTION_FEATURES
        + CAT_FEATURES
        + ["Risk_Tier"]
        + (["Annual_Miles_Range"] if "Annual_Miles_Range" in df.columns else [])
    )
    df_processed = df[[c for c in keep_cols if c in df.columns]].copy()

    export_artifacts(
        calibrated, explainer, ohe, le, feature_names,
        ood_detector, ood_threshold, training_stats, df_processed,
    )

    print("\n" + "═" * 64)
    print("  ✅  Agent 1 – Risk Profiler v3 : Training Complete!")
    print(f"  📁  Models  → {MODEL_DIR}/")
    print(f"  📄  Data    → {PROC_DIR / AGENT1_PROCESSED_FILE}")
    print("═" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
