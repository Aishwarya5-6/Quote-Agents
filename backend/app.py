"""
app.py — Agent 1 Risk Profiler  │  Production FastAPI Service
═══════════════════════════════════════════════════════════════
A safety-first, production-ready inference service that wraps the
Agent 1 XGBoost Risk Profiler behind six defensive layers before a
probability estimate ever leaves the system.

Routes
──────
  POST /predict/risk  →  full inference pipeline (see docstring)
  GET  /health        →  manifest.json version & artifact status

Safety Layers (in order)
────────────────────────
  Layer 1 │ Pydantic QuoteRequest      — type & range validation at the door
  Layer 2 │ IsolationForest OOD gate   — blocks impossible / corrupt inputs
  Layer 3 │ Interaction features       — computed inside predictor (actuarial)
  Layer 4 │ Cost-sensitive weights     — baked into the model at training time
  Layer 5 │ Confidence gate (0.60)     — escalates borderline predictions

Usage
─────
  # Start the server (artifacts must exist in ./models/ first):
  python app.py

  # Or with auto-reload for development:
  uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal

# ── Make the agents package importable when running from the project root ─────
sys.path.insert(0, str(Path(__file__).parent))
from agents.agent1_risk_profiler import OOD_FLAG, RiskProfilerPredictor  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = Path(__file__).parent / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

# ── Confidence gate threshold ─────────────────────────────────────────────────
CONFIDENCE_GATE      = 0.60
LOW_CONF_STATUS      = "LOW_CONFIDENCE_ESCALATE"
OOD_ESCALATE_STATUS  = "ACTION_REQUIRED: DATA_ANOMALY_ESCALATE"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent1.api")


# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 1 — Pydantic Input Schema (The Gateway)
# ─────────────────────────────────────────────────────────────────────────────
class QuoteRequest(BaseModel):
    """
    Validates every field of an incoming insurance quote **before** it touches
    the model.  FastAPI automatically returns HTTP 422 with a structured error
    message if any field fails validation — no model inference is ever attempted
    on malformed input.

    Field constraints mirror the real-world domain of auto insurance:
      • Driver_Age         must be between 16 (minimum legal age) and 100.
      • Driving_Exp        cannot exceed Driver_Age − 16 (you can't have more
                           experience than years since you could legally drive).
      • Annual_Miles       capped at 200,000 (realistic upper bound).
      • Veh_Usage          strictly one of the three categories the model knows.
    """

    Driver_Age:     int     = Field(..., ge=16, le=100,
                                description="Driver age in years (16–100)")
    Driving_Exp:    int     = Field(..., ge=0,  le=84,
                                description="Years of driving experience (0–84)")
    Prev_Accidents: int     = Field(..., ge=0,  le=20,
                                description="Number of prior at-fault accidents")
    Prev_Citations: int     = Field(..., ge=0,  le=20,
                                description="Number of prior traffic citations")
    Annual_Miles:   int     = Field(..., ge=0,  le=200_000,
                                description="Estimated annual mileage (0–200,000)")
    Veh_Usage: Literal["Business", "Commute", "Pleasure"] = Field(
                                ...,
                                description="Primary vehicle use: Business | Commute | Pleasure")

    @model_validator(mode="after")
    def driving_exp_cannot_exceed_age_minus_16(self) -> "QuoteRequest":
        """
        A driver cannot have more years of experience than (Driver_Age − 16).
        This cross-field check catches logically impossible inputs that individual
        field validators cannot catch on their own.
        """
        max_exp = self.Driver_Age - 16
        if self.Driving_Exp > max_exp:
            raise ValueError(
                f"Driving_Exp ({self.Driving_Exp}) cannot exceed "
                f"Driver_Age − 16 = {max_exp}.  "
                f"A {self.Driver_Age}-year-old cannot have more than "
                f"{max_exp} year(s) of driving experience."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "Driver_Age":     30,
                "Driving_Exp":     8,
                "Prev_Accidents":  0,
                "Prev_Citations":  1,
                "Annual_Miles":  28_000,
                "Veh_Usage":    "Commute",
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Response models (for /docs auto-schema)
# ─────────────────────────────────────────────────────────────────────────────
class ShapFeature(BaseModel):
    feature:    str
    shap_value: float
    direction:  str
    magnitude:  str


class PredictResponse(BaseModel):
    status:              str
    predicted_tier:      Optional[str]     = None
    confidence:          Optional[float]   = None
    class_probabilities: Optional[Dict[str, float]] = None
    top_3_features:      Optional[List[ShapFeature]] = None
    escalation_reason:   Optional[str]     = None


class HealthResponse(BaseModel):
    status:       str
    agent:        Optional[str] = None
    model_type:   Optional[str] = None
    n_features:   Optional[int] = None
    classes:      Optional[List[str]] = None
    ood_detector: Optional[str] = None
    artifacts:    Optional[List[str]] = None


# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 0 — Singleton Predictor  (loaded exactly once on startup)
# ─────────────────────────────────────────────────────────────────────────────
predictor: Optional[RiskProfilerPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    On startup  → load all six serialised artifacts from ./models/ into a
                  single RiskProfilerPredictor instance shared across all
                  requests.  Expensive IO + deserialization happens exactly once.
    On shutdown → log a clean teardown message.
    """
    global predictor
    log.info("🚀  Loading Agent 1 artifacts from  %s …", MODELS_DIR.resolve())
    if not MODELS_DIR.exists():
        log.error(
            "❌  models/ directory not found at %s.  "
            "Run agents/agent1_risk_profiler.py first to generate artifacts.",
            MODELS_DIR.resolve(),
        )
    else:
        predictor = RiskProfilerPredictor.from_artifacts(str(MODELS_DIR))
        log.info("✅  Predictor ready — all 6 artifacts loaded.")
    yield
    log.info("🛑  Agent 1 API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI application
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agent 1 – Risk Profiler API",
    description=(
        "**Safety-first, production-ready** insurance risk scoring service.\n\n"
        "**Inference pipeline (per request)**\n\n"
        "1. **Layer 1 – Pydantic validation** — rejects malformed or out-of-range "
        "   fields before the model is ever invoked.\n"
        "2. **Layer 2 – IsolationForest OOD gate** — blocks inputs that fall outside "
        "   the training distribution (e.g., `Driver_Age = −5`, `Annual_Miles = 9 999 999`). "
        "   Returns `ACTION_REQUIRED: DATA_ANOMALY_ESCALATE`.\n"
        "3. **Layer 3 – Actuarial interaction features** — computed inside the "
        "   predictor (`Miles_Per_Exp`, `Risk_Score_Raw`, `Age_Experience_Gap`).\n"
        "4. **Layer 4 – Cost-sensitive XGBoost** — High-risk class carries 3× weight "
        "   to minimise false negatives on dangerous drivers.\n"
        "5. **Layer 5 – Confidence gate** — predictions below 60 % confidence are "
        "   returned as `LOW_CONFIDENCE_ESCALATE` for human underwriter review."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 5 (part 2) — Global Exception Handler
#  Never expose a raw Python traceback to the caller.
# ─────────────────────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler: converts any unhandled exception into a structured
    JSON error response so the API never returns an opaque HTTP 500 body.
    The full traceback is logged server-side for debugging.
    """
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "ERROR",
            "detail": (
                "An internal server error occurred. "
                "Please try again or contact the platform team."
            ),
            "error_type": type(exc).__name__,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /health — Model version & artifact status
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check and model version",
    tags=["Operations"],
)
async def health_check() -> Any:
    """
    Returns the model version and key metadata from `manifest.json`.

    Use this endpoint to:
    - Confirm the correct model version is deployed.
    - Verify that all expected artifact files are present.
    - Check that the predictor singleton loaded successfully on startup.

    Returns **503 Service Unavailable** if `manifest.json` is missing
    (which means the model training script hasn't been run yet).
    """
    if not MANIFEST_PATH.exists():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "DEGRADED",
                "detail": (
                    "manifest.json not found in ./models/. "
                    "Run agents/agent1_risk_profiler.py first to train and export artifacts."
                ),
            },
        )

    manifest: Dict[str, Any] = json.loads(MANIFEST_PATH.read_text())

    return {
        "status":       "OK" if predictor is not None else "DEGRADED",
        "agent":        manifest.get("agent"),
        "model_type":   manifest.get("model_type"),
        "n_features":   manifest.get("n_features"),
        "classes":      manifest.get("classes"),
        "ood_detector": manifest.get("ood_detector"),
        "artifacts":    manifest.get("artifacts"),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  POST /predict/risk — Full inference pipeline
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/predict/risk",
    response_model=PredictResponse,
    summary="Predict risk tier for an insurance quote",
    tags=["Inference"],
    responses={
        200: {"description": "Successful prediction (OK or LOW_CONFIDENCE_ESCALATE)"},
        422: {"description": "Input validation failed (Pydantic) or OOD data anomaly detected"},
        503: {"description": "Predictor not loaded — artifacts missing"},
    },
)
async def predict_risk(quote: QuoteRequest) -> Any:
    """
    Full **five-layer** inference pipeline for a single insurance quote.

    ---
    **Layer 1 – Input validation** (automatic via Pydantic)
    Fields are validated for type, range, and logical consistency
    (`Driving_Exp ≤ Driver_Age − 16`) before this function body is entered.
    A failed validation returns HTTP 422 immediately.

    **Layer 2 – OOD Safety Gate**
    The validated quote is scored by an `IsolationForest` trained on the
    5 raw numeric features of all 146 K training quotes.  If the anomaly
    score falls below the 0.1th-percentile threshold, the request is rejected
    with status `ACTION_REQUIRED: DATA_ANOMALY_ESCALATE`.  The XGBoost model
    is never invoked on suspicious data.

    **Layer 3 & 4 – Actuarial features + Cost-sensitive XGBoost**
    Interaction features are computed inside the predictor and fed to a
    `CalibratedClassifierCV` (isotonic regression) wrapping an `XGBClassifier`
    trained with 3× sample weight on the High-risk class.

    **Layer 5 – Confidence Gate**
    If `max(class_probability) < 0.60` the response status is upgraded to
    `LOW_CONFIDENCE_ESCALATE`.  The prediction and all SHAP values are still
    returned so the human reviewer has full context.
    """
    # ── Predictor availability guard ──────────────────────────────────────
    if predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "ERROR",
                "detail": (
                    "The predictor could not be loaded at startup. "
                    "Ensure agents/agent1_risk_profiler.py has been run and "
                    "./models/ contains all required artifacts."
                ),
            },
        )

    # ── Layer 1 complete — convert validated Pydantic model to plain dict ─
    quote_dict: Dict[str, Any] = quote.model_dump()

    # ── Layers 2–4: OOD gate → feature engineering → calibrated prediction ─
    result: Dict[str, Any] = predictor.predict_and_explain(quote_dict)

    # ── Layer 2: Handle OOD flag ───────────────────────────────────────────
    #  The predictor sets status = OOD_FLAG when IsolationForest rejects the
    #  input.  We surface this as a distinct _ESCALATE status to the API caller
    #  so downstream systems can route it to a human underwriter without any
    #  further processing.
    if result.get("status") == OOD_FLAG:
        log.warning(
            "OOD anomaly escalated for input: %s", quote_dict
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status":  OOD_ESCALATE_STATUS,
                "message": result.get("message"),
                "input":   result.get("input_data"),
            },
        )

    # ── Layer 5: Confidence gate ───────────────────────────────────────────
    #  Low-confidence predictions are not suppressed — the caller still gets
    #  the tier and SHAP values — but the status is escalated so the response
    #  contract clearly signals that a human should review the case.
    confidence: float = result.get("confidence", 1.0)
    escalation_reason: Optional[str] = None

    if confidence < CONFIDENCE_GATE:
        result["status"] = LOW_CONF_STATUS
        escalation_reason = (
            f"Model confidence {confidence:.1%} is below the {CONFIDENCE_GATE:.0%} threshold. "
            "This is a borderline case — route to a human underwriter for manual review."
        )
        log.info(
            "LOW_CONFIDENCE_ESCALATE │ tier=%s │ confidence=%.4f │ input=%s",
            result.get("predicted_tier"), confidence, quote_dict,
        )

    return {
        "status":              result.get("status"),
        "predicted_tier":      result.get("predicted_tier"),
        "confidence":          confidence,
        "class_probabilities": result.get("class_probabilities"),
        "top_3_features":      result.get("top_3_features"),
        "escalation_reason":   escalation_reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Entry-point  (python app.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
