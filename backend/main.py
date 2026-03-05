"""
main.py — Multi-Agent Quote Pipeline  │  Production FastAPI Orchestrator
═════════════════════════════════════════════════════════════════════════
Single entry-point for all four agents.  On startup, pre-trained artifacts
are loaded into app.state exactly once so every request pays zero reload cost.

Routes
──────
  POST /api/process_quote  →  full 4-agent pipeline (Agent 1 live; 2–4 ready to wire)
  GET  /api/health         →  liveness check + artifact inventory
  GET  /docs               →  Swagger UI (auto-generated)

Agent slots
───────────
  app.state.risk_engine    ← Agent 1  RiskProfilerPredictor  (✅ live)
  app.state.conv_engine    ← Agent 2  (plug in when ready)
  app.state.advisor        ← Agent 3  (plug in when ready)
  app.state.router         ← Agent 4  (plug in when ready)

Usage
─────
  cd backend
  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

# ── Make the agents package importable when running from backend/ ──────────────
sys.path.insert(0, str(Path(__file__).parent))
from agents.agent1_risk_profiler import OOD_FLAG, RiskProfilerPredictor  # noqa: E402
from agents.agent2_conversion_predictor import ConversionPredictor  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
#  Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent                  # …/backend/
MODELS_DIR    = BASE_DIR / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

CONFIDENCE_GATE     = 0.60
LOW_CONF_STATUS     = "LOW_CONFIDENCE_ESCALATE"
OOD_ESCALATE_STATUS = "ACTION_REQUIRED: DATA_ANOMALY_ESCALATE"

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline.api")

# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="InsurTech AI Pipeline API",
    description=(
        "**4-Agent** auto insurance quote engine.\n\n"
        "| Agent | Role | Status |\n"
        "|---|---|---|\n"
        "| Agent 1 | Risk Profiler (XGBoost + SHAP) | ✅ Live |\n"
        "| Agent 2 | Conversion Engine | 🔜 Wiring |\n"
        "| Agent 3 | AI Advisor (LLM) | 🔜 Wiring |\n"
        "| Agent 4 | Underwriting Router | 🔜 Wiring |\n"
    ),
    version="1.0.0",
)

# CORS — allows the Next.js frontend on localhost:3000 to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP — load all agent artifacts into app.state exactly once
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def load_agents() -> None:
    """
    Runs once when Uvicorn boots.  Deserializes all pre-trained artifacts from
    backend/models/ and stores them on app.state so every route handler can
    access them without triggering a re-load.

    app.state.risk_engine  ← RiskProfilerPredictor (Agent 1)
    app.state.conv_engine  ← None until Agent 2 is wired
    app.state.advisor      ← None until Agent 3 is wired
    app.state.router       ← None until Agent 4 is wired
    """
    # ── Agent 1 ───────────────────────────────────────────────────────────────
    if MODELS_DIR.exists():
        log.info("🚀  Agent 1 │ loading artifacts from %s …", MODELS_DIR.resolve())
        app.state.risk_engine = RiskProfilerPredictor.from_artifacts(MODELS_DIR)
        log.info("✅  Agent 1 │ RiskProfilerPredictor ready.")
    else:
        app.state.risk_engine = None
        log.error(
            "❌  Agent 1 │ models/ not found at %s — "
            "run backend/agents/agent1_risk_profiler.py first.",
            MODELS_DIR.resolve(),
        )

    # ── Agent 2 ───────────────────────────────────────────────────────────────
    try:
        log.info("🚀  Agent 2 │ loading artifacts from %s …", MODELS_DIR.resolve())
        app.state.conv_engine = ConversionPredictor.from_artifacts(MODELS_DIR)
        log.info("✅  Agent 2 │ ConversionPredictor ready.")
    except FileNotFoundError as exc:
        app.state.conv_engine = None
        log.warning("⚠️  Agent 2 │ artifacts not found — %s", exc)

    # ── Agent 3 slot (wire when ready) ────────────────────────────────────────
    app.state.advisor = None       # TODO: AdvisorAgent(api_key=os.getenv("OPENAI_API_KEY"))

    # ── Agent 4 slot (wire when ready) ────────────────────────────────────────
    app.state.router = None        # TODO: UnderwritingRouter.from_config(...)


# ─────────────────────────────────────────────────────────────────────────────
#  Global exception handler — never expose raw tracebacks to callers
# ─────────────────────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status":     "ERROR",
            "detail":     "An internal server error occurred. Please contact the platform team.",
            "error_type": type(exc).__name__,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  INPUT SCHEMA — QuoteRequest
# ─────────────────────────────────────────────────────────────────────────────
class QuoteRequest(BaseModel):
    """
    Validated input contract for a single insurance quote.
    FastAPI returns HTTP 422 automatically if any field fails.
    """

    Driver_Age:     int = Field(..., ge=16, le=100,    description="Driver age in years")
    Driving_Exp:    int = Field(..., ge=0,  le=84,     description="Years of driving experience")
    Prev_Accidents: int = Field(..., ge=0,  le=20,     description="Prior at-fault accidents")
    Prev_Citations: int = Field(..., ge=0,  le=20,     description="Prior traffic citations")
    Annual_Miles:   int = Field(..., ge=0,  le=200_000, description="Estimated annual mileage")
    Veh_Usage: Literal["Business", "Commute", "Pleasure"] = Field(
        ..., description="Primary vehicle use"
    )

    @model_validator(mode="after")
    def exp_cannot_exceed_age(self) -> "QuoteRequest":
        max_exp = self.Driver_Age - 16
        if self.Driving_Exp > max_exp:
            raise ValueError(
                f"Driving_Exp ({self.Driving_Exp}) cannot exceed "
                f"Driver_Age − 16 = {max_exp}."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "Driver_Age": 34, "Driving_Exp": 12,
                "Prev_Accidents": 0, "Prev_Citations": 1,
                "Annual_Miles": 22_000, "Veh_Usage": "Pleasure",
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class ShapDriver(BaseModel):
    feature:    str
    shap_value: float
    direction:  str
    magnitude:  str


class RiskAssessmentOut(BaseModel):
    predicted_tier:      str
    confidence_score:    float
    ood_flag:            str
    class_probabilities: Dict[str, float]
    top_shap_drivers:    List[ShapDriver]


class ConversionMetricsOut(BaseModel):
    bind_probability:       Optional[float] = None
    sales_status:           Optional[str]   = None
    distance_to_conversion: Optional[float] = None


class AdvisorStrategyOut(BaseModel):
    suggested_discount_pct:  Optional[float] = None
    customer_facing_message: Optional[str]   = None
    internal_reasoning:      Optional[str]   = None


class FinalRoutingOut(BaseModel):
    routing_status:            Optional[str] = None
    underwriter_justification: Optional[str] = None


class PipelineResponse(BaseModel):
    transaction_id:     str
    status:             str
    escalation_reason:  Optional[str]              = None
    risk_assessment:    Optional[RiskAssessmentOut] = None
    conversion_metrics: Optional[ConversionMetricsOut] = None
    advisor_strategy:   Optional[AdvisorStrategyOut]   = None
    final_routing:      Optional[FinalRoutingOut]      = None


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["Operations"], summary="Liveness check + artifact inventory")
async def health() -> Any:
    """Returns model version from manifest.json and the load status of each agent."""
    agent_status = {
        "agent_1_risk_profiler": "✅ loaded" if app.state.risk_engine else "❌ not loaded",
        "agent_2_conversion":    "✅ loaded" if app.state.conv_engine is not None else "❌ not loaded",
        "agent_3_advisor":       "🔜 not wired" if app.state.advisor    is None else "✅ loaded",
        "agent_4_router":        "🔜 not wired" if app.state.router     is None else "✅ loaded",
    }

    if not MANIFEST_PATH.exists():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "DEGRADED", "agents": agent_status,
                     "detail": "manifest.json missing — run agent1_risk_profiler.py first."},
        )

    manifest = json.loads(MANIFEST_PATH.read_text())
    return {
        "status":  "OK" if app.state.risk_engine else "DEGRADED",
        "version": manifest.get("agent"),
        "agents":  agent_status,
        "model":   manifest.get("model_type"),
        "classes": manifest.get("classes"),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/process_quote  — Full 4-agent pipeline
# ─────────────────────────────────────────────────────────────────────────────
import uuid   # noqa: E402  (stdlib, late import for readability)

@app.post(
    "/api/process_quote",
    response_model=PipelineResponse,
    tags=["Pipeline"],
    summary="Run all 4 agents on a single insurance quote",
    responses={
        200: {"description": "Pipeline complete — all agent results returned"},
        422: {"description": "Pydantic validation failed or OOD anomaly detected"},
        503: {"description": "Agent 1 artifacts not loaded"},
    },
)
async def process_quote(quote: QuoteRequest, request: Request) -> Any:
    """
    Chains all four agents on a single validated quote.

    **Current state**
    - Agent 1 (Risk Profiler) is live — returns `predicted_tier`, `confidence_score`,
      `top_shap_drivers`, and `dashboard_metrics`.
    - Agents 2–4 slots are stubbed with placeholder values so the frontend
      contract is stable while teammates build them.

    **To go live with agents 2–4:**
    Replace each `# TODO` stub below with a real call to `app.state.<engine>`.
    """
    transaction_id = str(uuid.uuid4())
    risk_engine: Optional[RiskProfilerPredictor] = request.app.state.risk_engine

    # ── Guard: Agent 1 must be loaded ─────────────────────────────────────────
    if risk_engine is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "ERROR",
                "detail": "Agent 1 not loaded. Run agent1_risk_profiler.py first.",
            },
        )

    # ── Agent 1 ── Risk assessment ────────────────────────────────────────────
    quote_dict: Dict[str, Any] = quote.model_dump()
    agent1_result: Dict[str, Any] = risk_engine.predict_and_explain(quote_dict)

    # OOD / Physics Check blocked the input → 422 immediately
    if agent1_result.get("status") == OOD_FLAG:
        log.warning("OOD anomaly blocked for tx=%s input=%s", transaction_id, quote_dict)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "transaction_id": transaction_id,
                "status":         OOD_ESCALATE_STATUS,
                "message":        agent1_result.get("message"),
                "input":          agent1_result.get("input_data"),
            },
        )

    # Confidence gate check
    confidence: float = agent1_result.get("confidence", 1.0)
    pipeline_status   = agent1_result.get("status", "OK")
    escalation_reason: Optional[str] = None

    if confidence < CONFIDENCE_GATE:
        pipeline_status   = LOW_CONF_STATUS
        escalation_reason = (
            f"Model confidence {confidence:.1%} is below the "
            f"{CONFIDENCE_GATE:.0%} threshold. Route to human underwriter."
        )

    risk_assessment = RiskAssessmentOut(
        predicted_tier      = agent1_result["predicted_tier"],
        confidence_score    = confidence,
        ood_flag            = "OK",
        class_probabilities = agent1_result.get("class_probabilities", {}),
        top_shap_drivers    = [
            ShapDriver(**f) for f in agent1_result.get("top_3_features", [])
        ],
    )

    # ── Agent 2 ── Conversion metrics ────────────────────────────────────────
    conv_engine: Optional[ConversionPredictor] = request.app.state.conv_engine
    if conv_engine is not None:
        conv_result = conv_engine.predict_conversion(
            input_data=quote_dict,
            risk_tier=risk_assessment.predicted_tier,   # Agent 1 → Agent 2 context
        )
        conversion_metrics = ConversionMetricsOut(
            bind_probability       = round(conv_result.bind_probability, 4),
            sales_status           = conv_result.sales_status,
            distance_to_conversion = round(conv_result.distance_to_conversion, 4),
        )
    else:
        conversion_metrics = ConversionMetricsOut(
            bind_probability       = None,
            sales_status           = None,
            distance_to_conversion = None,
        )

    # ── Agent 3 ── Advisor strategy (stub — wire when ready) ─────────────────
    # TODO: advisor_strategy = await request.app.state.advisor.generate(quote_dict, risk_assessment)
    advisor_strategy = AdvisorStrategyOut(
        suggested_discount_pct  = None,
        customer_facing_message = None,
        internal_reasoning      = None,
    )

    # ── Agent 4 ── Final routing (stub — wire when ready) ────────────────────
    # TODO: final_routing = request.app.state.router.decide(risk_assessment, conversion_metrics)
    final_routing = FinalRoutingOut(
        routing_status            = None,
        underwriter_justification = None,
    )

    # ── Assemble master response ──────────────────────────────────────────────
    return PipelineResponse(
        transaction_id     = transaction_id,
        status             = pipeline_status,
        escalation_reason  = escalation_reason,
        risk_assessment    = risk_assessment,
        conversion_metrics = conversion_metrics,
        advisor_strategy   = advisor_strategy,
        final_routing      = final_routing,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Entry-point  (python main.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
