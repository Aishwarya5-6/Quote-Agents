"""
main.py — LangGraph Multi-Agent Quote Pipeline  │  FastAPI Orchestrator
════════════════════════════════════════════════════════════════════════
LangGraph StateGraph wires all four agents into a typed, inspectable DAG.
ML models are loaded into module-level singletons at startup (zero reload
cost per request).  The graph is compiled once and reused for every call.

Routes
──────
  POST /api/v1/quote       →  full 4-agent LangGraph pipeline
  POST /api/process_quote  →  legacy alias (kept for backwards compat)
  GET  /api/health         →  liveness check + artifact inventory
  GET  /docs               →  Swagger UI (auto-generated)

LangGraph DAG
─────────────
  [START] → node_risk → node_conversion → node_advisor → node_router → [END]

AgentState fields
─────────────────
  input_data             raw quote dict from the HTTP request
  risk_results           Agent 1 output (predicted_tier, shap drivers …)
  conversion_results     Agent 2 output (bind_probability, sales_status …)
  advisor_pitch          Agent 3 output (recommended_premium, reason …)
  final_decision         Agent 4 full route_decision() output dict
  final_routing_decision Agent 4 canonical: AUTO_APPROVE | MANUAL_REVIEW | REJECT

Safety constraints
──────────────────
  • No node may write to backend/models/ — enforced in each node wrapper.
  • GROQ_API_KEY loaded from backend/.env via python-dotenv at startup.
  • Agents 3 & 4 LLM calls have silent rule-based fallbacks; graph never stalls.

Usage
─────
  cd backend
  uvicorn main:app --reload --port 8001
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

# LangGraph
from langgraph.graph import END, START, StateGraph

from dotenv import load_dotenv

# Load GROQ_API_KEY (and any other secrets) from backend/.env
# This is a no-op when the variable is already set in the environment.
load_dotenv(Path(__file__).parent / ".env")

# ── Make the agents package importable when running from backend/ ──────────────
sys.path.insert(0, str(Path(__file__).parent))
from agents.agent1_risk_profiler import OOD_FLAG, RiskProfilerPredictor   # noqa: E402
from agents.agent2_conversion_predictor import ConversionPredictor         # noqa: E402
from agents.agent3 import advise_premium                                   # noqa: E402
from agents.agent4 import (                                                # noqa: E402
    DEC_APPROVE, DEC_ESCALATE, DEC_FOLLOWUP, DEC_REJECT,
    RISK_HIGH, RISK_LOW, RISK_MEDIUM,
    route_decision,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent                  # …/backend/
MODELS_DIR    = BASE_DIR / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

CONFIDENCE_GATE     = 0.60
LOW_CONF_STATUS     = "LOW_CONFIDENCE_ESCALATE"
OOD_ESCALATE_STATUS = "ACTION_REQUIRED: DATA_ANOMALY_ESCALATE"

# High-risk tiers that should trigger human review
HIGH_RISK_TIERS = {"High"}

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
#  Module-level ML singletons  (loaded once at startup, never replaced)
# ─────────────────────────────────────────────────────────────────────────────
_risk_engine:  Optional[RiskProfilerPredictor] = None
_conv_engine:  Optional[ConversionPredictor]   = None

# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="InsurTech AI Pipeline API",
    description=(
        "**4-Agent** auto insurance quote engine powered by **LangGraph**.\n\n"
        "| Agent | Role | Status |\n"
        "|---|---|---|\n"
        "| Agent 1 | Risk Profiler (XGBoost + SHAP) | ✅ Live |\n"
        "| Agent 2 | Conversion Predictor (SMOTE + Calibration) | ✅ Live |\n"
        "| Agent 3 | Premium Advisor (Rules + Groq LLM) | ✅ Live |\n"
        "| Agent 4 | Underwriting Router | ✅ Live |\n"
    ),
    version="2.0.0",
)

# CORS: always allow localhost for dev; add production domains via FRONTEND_ORIGINS env var.
# Example:  FRONTEND_ORIGINS=https://quote-agents.vercel.app,https://my-app.vercel.app
_extra_origins = [
    o.strip() for o in os.environ.get("FRONTEND_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        *_extra_origins,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  LANGGRAPH  ── AgentState
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    """
    Shared mutable state threaded through the LangGraph DAG.

    Populated sequentially: each node reads upstream fields and writes its
    own.  input_data is READ-ONLY — no node may mutate it.
    """
    input_data:              Dict[str, Any]   # raw validated quote dict — READ ONLY
    risk_results:            Dict[str, Any]   # Agent 1: predicted_tier, shap, confidence …
    conversion_results:      Dict[str, Any]   # Agent 2: bind_probability, sales_status …
    advisor_pitch:           Dict[str, Any]   # Agent 3: recommended_premium, reason …
    final_decision:          Dict[str, Any]   # Agent 4: full route_decision() output dict
    final_routing_decision:  str              # Agent 4 canonical label: AUTO_APPROVE | MANUAL_REVIEW | REJECT


# ─────────────────────────────────────────────────────────────────────────────
#  LANGGRAPH  ── Node 1: Risk Profiler (Agent 1)
# ─────────────────────────────────────────────────────────────────────────────
def node_risk(state: AgentState) -> AgentState:
    """
    Calls RiskProfilerPredictor.predict_and_explain().

    Reads  : state["input_data"]
    Writes : state["risk_results"]
    Safety : never touches models/ directory.
    """
    if _risk_engine is None:
        return {**state, "risk_results": {"status": "ERROR", "message": "Agent 1 not loaded"}}

    result: Dict[str, Any] = _risk_engine.predict_and_explain(state["input_data"])
    log.info(
        "node_risk  │ tier=%s  conf=%.3f  ood=%s",
        result.get("predicted_tier", "?"),
        result.get("confidence", 0.0),
        result.get("status", "?"),
    )
    return {**state, "risk_results": result}


# ─────────────────────────────────────────────────────────────────────────────
#  LANGGRAPH  ── Node 2: Conversion Predictor (Agent 2)
# ─────────────────────────────────────────────────────────────────────────────
def node_conversion(state: AgentState) -> AgentState:
    """
    Calls ConversionPredictor.predict_conversion(), injecting predicted_tier
    from node_risk so Agent 2 benefits from Agent 1 cross-agent context.

    Reads  : state["input_data"], state["risk_results"]["predicted_tier"]
    Writes : state["conversion_results"]
    Safety : never touches models/ directory.
    """
    if _conv_engine is None:
        return {**state, "conversion_results": {
            "bind_probability": None,
            "sales_status":     None,
            "distance_to_conversion": None,
            "error": "Agent 2 not loaded",
        }}

    risk_tier: Optional[str] = state.get("risk_results", {}).get("predicted_tier")
    conv = _conv_engine.predict_conversion(
        input_data=state["input_data"],
        risk_tier=risk_tier,
    )
    result = {
        "bind_probability":       round(conv.bind_probability, 4),
        "sales_status":           conv.sales_status,
        "distance_to_conversion": round(conv.distance_to_conversion, 4),
        "conversion_score":       round(conv.bind_probability * 100, 2),
    }
    log.info(
        "node_conversion │ bind_prob=%.3f  status=%s",
        conv.bind_probability, conv.sales_status,
    )
    return {**state, "conversion_results": result}


# ─────────────────────────────────────────────────────────────────────────────
#  LANGGRAPH  ── Node 3: Premium Advisor (Agent 3)
# ─────────────────────────────────────────────────────────────────────────────
def node_advisor(state: AgentState) -> AgentState:
    """
    Calls agent3.advise_premium().  The 40-point activation gate is enforced
    inside advise_premium(); this node always runs safely.

    Reads  : state["input_data"], state["conversion_results"]["conversion_score"]
    Writes : state["advisor_pitch"]

    LLM safety: any timeout / missing key / exception is silently caught
    inside advise_premium() — the rule-based reason string is returned as
    fallback.  This node never raises or stalls the graph.
    Safety : never touches models/ directory.
    """
    conv  = state.get("conversion_results") or {}
    score = float(conv.get("conversion_score", 50.0))

    result = advise_premium(quote_dict=state["input_data"], conversion_score=score)
    log.info(
        "node_advisor │ flag=%s  adjustment=%s  recommended=%.2f",
        result.get("premium_flag"),
        result.get("adjustment"),
        result.get("recommended_premium", 0.0),
    )
    return {**state, "advisor_pitch": result}


# ─────────────────────────────────────────────────────────────────────────────
#  LANGGRAPH  ── Node 4: Underwriting Router (Agent 4)
# ─────────────────────────────────────────────────────────────────────────────

# Map Agent 1 predicted_tier string → int encoding agent4 expects
_TIER_TO_INT: Dict[str, int] = {"Low": RISK_LOW, "Medium": RISK_MEDIUM, "High": RISK_HIGH}

# Map agent4 DEC_* decision label → canonical final_routing_decision value
_DEC_TO_CANONICAL: Dict[str, str] = {
    DEC_APPROVE:  "AUTO_APPROVE",
    DEC_REJECT:   "REJECT",
    DEC_ESCALATE: "MANUAL_REVIEW",
    DEC_FOLLOWUP: "MANUAL_REVIEW",
}


def node_router(state: AgentState) -> AgentState:
    """
    Calls agent4.route_decision() with the full upstream state.

    Translates AgentState fields into the three dicts agent4.route_decision()
    expects, then writes both the raw output dict and the canonical
    final_routing_decision string back into state.

    agent4 input translation
    ────────────────────────
    agent1_output : risk_tier (int 0-2), risk_label (str)
    agent2_output : conversion_score (float 0-100), will_buy (bool)
    agent3_output : premium_flag, recommended_premium, adjustment, reason
    quote_dict    : Prev_Accidents, Prev_Citations, Driver_Age (from input_data)

    Reads  : state["risk_results"], state["conversion_results"],
             state["advisor_pitch"], state["input_data"]
    Writes : state["final_decision"], state["final_routing_decision"]
    Safety : never touches models/ directory.
    """
    risk    = state.get("risk_results")       or {}
    conv    = state.get("conversion_results") or {}
    advisor = state.get("advisor_pitch")      or {}
    inp     = state.get("input_data")         or {}

    # ── Build agent1_output dict ──────────────────────────────────────────────
    tier_str  = risk.get("predicted_tier", "Low")
    agent1_out = {
        "risk_tier":  _TIER_TO_INT.get(tier_str, RISK_LOW),
        "risk_label": tier_str,
    }

    # ── Build agent2_output dict ──────────────────────────────────────────────
    bind_prob       = float(conv.get("bind_probability") or 0.0)
    conversion_score = float(conv.get("conversion_score") or bind_prob * 100)
    agent2_out = {
        "conversion_score": conversion_score,
        "will_buy":         bind_prob >= 0.5,
        "confidence":       conv.get("sales_status", "Medium"),
    }

    # ── Build agent3_output dict ──────────────────────────────────────────────
    agent3_out = {
        "premium_flag":        bool(advisor.get("premium_flag", False)),
        "recommended_premium": float(advisor.get("recommended_premium") or 0.0),
        "adjustment":          advisor.get("adjustment", "none"),
        "reason":              advisor.get("reason", ""),
    }

    # ── Call agent4.route_decision() ─────────────────────────────────────────
    result: Dict[str, Any] = route_decision(
        agent1_output=agent1_out,
        agent2_output=agent2_out,
        agent3_output=agent3_out,
        quote_dict=inp,
    )

    canonical = _DEC_TO_CANONICAL.get(result.get("decision", ""), "MANUAL_REVIEW")
    log.info(
        "node_router  │ decision=%s  canonical=%s  priority=%s",
        result.get("decision"), canonical, result.get("priority"),
    )
    return {
        **state,
        "final_decision":         result,
        "final_routing_decision": canonical,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  LANGGRAPH  ── Compile the StateGraph (once at import time)
# ─────────────────────────────────────────────────────────────────────────────
def _build_graph() -> Any:
    builder = StateGraph(AgentState)
    builder.add_node("node_risk",       node_risk)
    builder.add_node("node_conversion", node_conversion)
    builder.add_node("node_advisor",    node_advisor)
    builder.add_node("node_router",     node_router)
    builder.add_edge(START,             "node_risk")
    builder.add_edge("node_risk",       "node_conversion")
    builder.add_edge("node_conversion", "node_advisor")
    builder.add_edge("node_advisor",    "node_router")
    builder.add_edge("node_router",     END)
    return builder.compile()


_pipeline = _build_graph()   # compiled once; reused for every request


# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP — load ML artifacts into module-level singletons exactly once
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def load_agents() -> None:
    """
    Deserializes pre-trained artifacts from backend/models/ into the
    module-level singletons (_risk_engine, _conv_engine).

    Agents 3 and 4 are stateless — no artifact loading required.
    """
    global _risk_engine, _conv_engine

    if MODELS_DIR.exists():
        log.info("🚀  Agent 1 │ loading artifacts from %s …", MODELS_DIR.resolve())
        _risk_engine = RiskProfilerPredictor.from_artifacts(MODELS_DIR)
        log.info("✅  Agent 1 │ RiskProfilerPredictor ready.")
    else:
        log.error(
            "❌  Agent 1 │ models/ not found at %s — "
            "run backend/agents/agent1_risk_profiler.py first.",
            MODELS_DIR.resolve(),
        )

    try:
        log.info("🚀  Agent 2 │ loading artifacts from %s …", MODELS_DIR.resolve())
        _conv_engine = ConversionPredictor.from_artifacts(MODELS_DIR)
        log.info("✅  Agent 2 │ ConversionPredictor ready.")
    except FileNotFoundError as exc:
        log.warning("⚠️  Agent 2 │ artifacts not found — %s", exc)

    log.info("✅  Agent 3 │ advise_premium() ready (stateless rule engine + Groq LLM)."
    )
    log.info("✅  Agent 4 │ route_decision() ready (agent4.py rule engine + Groq LLM).")
    log.info("✅  LangGraph pipeline compiled and ready.")

    # Expose on app.state for /api/health
    app.state.risk_engine = _risk_engine
    app.state.conv_engine = _conv_engine


# ─────────────────────────────────────────────────────────────────────────────
#  Global exception handler
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
    """Validated input contract for a single insurance quote."""

    Driver_Age:     int   = Field(..., ge=16,  le=100,     description="Driver age in years")
    Driving_Exp:    int   = Field(..., ge=0,   le=84,      description="Years of driving experience")
    Prev_Accidents: int   = Field(..., ge=0,   le=20,      description="Prior at-fault accidents")
    Prev_Citations: int   = Field(..., ge=0,   le=20,      description="Prior traffic citations")
    Annual_Miles:   int   = Field(..., ge=0,   le=200_000, description="Estimated annual mileage")
    Veh_Usage: Literal["Business", "Commute", "Pleasure"] = Field(
        ..., description="Primary vehicle use"
    )
    # Agent 3 optional fields — provided by CRM / quoting engine
    Quoted_Premium:    Optional[float] = Field(None, ge=0,  description="Current quoted premium ($)")
    Sal_Range:         Optional[int]   = Field(None, ge=0, le=4, description="Salary band 0–4")
    Coverage:          Optional[int]   = Field(None, ge=0, le=2, description="Coverage level 0–2")
    Vehicl_Cost_Range: Optional[int]   = Field(None, ge=0, le=4, description="Vehicle cost band 0–4")
    Re_Quote:          Optional[int]   = Field(None, ge=0, le=1, description="1 if re-quoting")

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
                "Quoted_Premium": 750.0, "Sal_Range": 1,
                "Coverage": 1, "Vehicl_Cost_Range": 2, "Re_Quote": 0,
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
    premium_flag:            bool            = False
    suggested_discount_pct:  Optional[str]   = None
    recommended_premium:     Optional[float] = None
    original_premium:        Optional[float] = None
    customer_facing_message: Optional[str]   = None


class FinalRoutingOut(BaseModel):
    # Agent 4 native fields
    decision:       Optional[str]       = None   # "Auto Approve" | "Escalate to Underwriter" | "Agent Follow-Up"
    reason:         Optional[str]       = None   # LLM-enriched 2-sentence justification
    human_required: bool                = False
    priority:       Optional[str]       = None   # "High" | "Medium" | "Low"
    action_items:   List[str]           = []
    # Canonical routing label surfaced at top-level state
    final_routing_decision: Optional[str] = None  # AUTO_APPROVE | MANUAL_REVIEW | REJECT


class PipelineResponse(BaseModel):
    transaction_id:          str
    status:                  str
    final_routing_decision:  Optional[str]               = None  # AUTO_APPROVE | MANUAL_REVIEW | REJECT
    escalation_reason:       Optional[str]               = None
    risk_assessment:         Optional[RiskAssessmentOut]    = None
    conversion_metrics:      Optional[ConversionMetricsOut] = None
    advisor_strategy:        Optional[AdvisorStrategyOut]   = None
    final_routing:           Optional[FinalRoutingOut]      = None


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/warmup", tags=["Operations"], summary="Zero-cost keep-alive ping — call every 14 min to prevent Render free-tier cold starts")
async def warmup() -> Any:
    """
    Lightweight endpoint designed to be pinged by UptimeRobot (or any cron)
    every 14 minutes.  Render free tier spins down after 15 min of inactivity;
    this keeps the container hot so the first real request never hits a cold start.

    Returns immediately with no ML inference — safe to call frequently.
    """
    return {"status": "warm", "agents_loaded": _risk_engine is not None}


@app.get("/api/health", tags=["Operations"], summary="Liveness check + artifact inventory")
async def health() -> Any:
    agent_status = {
        "agent_1_risk_profiler": "✅ loaded"          if app.state.risk_engine is not None else "❌ not loaded",
        "agent_2_conversion":    "✅ loaded"          if app.state.conv_engine is not None else "❌ not loaded",
        "agent_3_advisor":       "✅ loaded (stateless)",
        "agent_4_router":        "✅ loaded (stateless)",
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
#  SHARED PIPELINE HANDLER
# ─────────────────────────────────────────────────────────────────────────────
async def _run_pipeline(quote: QuoteRequest) -> Any:
    """
    Runs the LangGraph DAG and assembles a PipelineResponse.
    Called by both /api/v1/quote and /api/process_quote.
    """
    transaction_id = str(uuid.uuid4())

    if _risk_engine is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "ERROR",
                     "detail": "Agent 1 not loaded. Run agent1_risk_profiler.py first."},
        )

    quote_dict: Dict[str, Any] = quote.model_dump()

    initial_state: AgentState = {"input_data": quote_dict}

    # Wrap in a 90-second timeout to accommodate Render's free-tier CPU cold
    # starts and slow Groq LLM calls.  recursion_limit=50 is well above our
    # 4-node linear DAG but safely below LangGraph's compiled-graph ceiling.
    try:
        final_state: AgentState = await asyncio.wait_for(
            asyncio.to_thread(
                _pipeline.invoke,
                initial_state,
                {"recursion_limit": 50},
            ),
            timeout=90.0,
        )
    except asyncio.TimeoutError:
        log.error("Pipeline timed out after 90s  tx=%s", transaction_id)
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "transaction_id": transaction_id,
                "status": "TIMEOUT",
                "detail": "Pipeline took longer than 90 seconds. Please try again.",
            },
        )

    risk_res  = final_state.get("risk_results",       {})
    conv_res  = final_state.get("conversion_results", {})
    adv_res   = final_state.get("advisor_pitch",      {})
    route_res = final_state.get("final_decision",     {})

    # OOD gate — block immediately
    if risk_res.get("status") == OOD_FLAG:
        log.warning("OOD anomaly blocked tx=%s", transaction_id)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "transaction_id": transaction_id,
                "status":         OOD_ESCALATE_STATUS,
                "message":        risk_res.get("message"),
                "input":          risk_res.get("input_data"),
            },
        )

    confidence: float    = float(risk_res.get("confidence", 1.0))
    pipeline_status      = risk_res.get("status", "OK")
    escalation_reason: Optional[str] = None

    if confidence < CONFIDENCE_GATE:
        pipeline_status   = LOW_CONF_STATUS
        escalation_reason = (
            f"Model confidence {confidence:.1%} is below the "
            f"{CONFIDENCE_GATE:.0%} threshold. Route to human underwriter."
        )

    risk_assessment = RiskAssessmentOut(
        predicted_tier      = risk_res.get("predicted_tier", "Unknown"),
        confidence_score    = confidence,
        ood_flag            = "OK",
        class_probabilities = risk_res.get("class_probabilities", {}),
        top_shap_drivers    = [ShapDriver(**f) for f in risk_res.get("top_3_features", [])],
    )

    conversion_metrics = ConversionMetricsOut(
        bind_probability       = conv_res.get("bind_probability"),
        sales_status           = conv_res.get("sales_status"),
        distance_to_conversion = conv_res.get("distance_to_conversion"),
    )

    advisor_strategy = AdvisorStrategyOut(
        premium_flag            = bool(adv_res.get("premium_flag", False)),
        suggested_discount_pct  = adv_res.get("adjustment"),
        recommended_premium     = adv_res.get("recommended_premium"),
        original_premium        = adv_res.get("original_premium"),
        customer_facing_message = adv_res.get("reason"),
    )

    final_routing = FinalRoutingOut(
        decision               = route_res.get("decision"),
        # Always populate reason so the Sequential UI terminal card has a story
        # to tell — fall back to the decision label if both LLM and rules return
        # an empty string (should never happen, but this is the last safety net).
        reason                 = (
            route_res.get("reason")
            or "Referred for underwriter review based on standard risk-routing rules."
        ),
        human_required         = bool(route_res.get("human_required", False)),
        priority               = route_res.get("priority"),
        action_items           = route_res.get("action_items", []),
        final_routing_decision = final_state.get("final_routing_decision"),
    )

    return PipelineResponse(
        transaction_id          = transaction_id,
        status                  = pipeline_status,
        final_routing_decision  = final_state.get("final_routing_decision"),
        escalation_reason       = escalation_reason,
        risk_assessment         = risk_assessment,
        conversion_metrics      = conversion_metrics,
        advisor_strategy        = advisor_strategy,
        final_routing           = final_routing,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/v1/quote  — Primary LangGraph endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/api/v1/quote",
    response_model=PipelineResponse,
    tags=["Pipeline v1"],
    summary="Run all 4 agents via LangGraph on a single insurance quote",
    responses={
        200: {"description": "Pipeline complete — full AgentState returned"},
        422: {"description": "Pydantic validation failure or OOD anomaly"},
        503: {"description": "Agent 1 artifacts not loaded"},
    },
)
async def quote_v1(quote: QuoteRequest) -> Any:
    """
    Primary endpoint.  Chains all four agents through the LangGraph DAG:

    **[START] → node_risk → node_conversion → node_advisor → node_router → [END]**

    Returns the full AgentState serialised as a `PipelineResponse`.
    """
    return await _run_pipeline(quote)


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/v1/full-analysis  — Full stateful analysis (named alias)
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/api/v1/full-analysis",
    response_model=PipelineResponse,
    tags=["Pipeline v1"],
    summary="Complete 4-agent analysis — returns full AgentState including final_routing_decision",
    responses={
        200: {"description": "Complete pipeline result with final_routing_decision"},
        422: {"description": "Pydantic validation failure or OOD anomaly"},
        503: {"description": "Agent 1 artifacts not loaded"},
    },
)
async def full_analysis(quote: QuoteRequest) -> Any:
    """
    Identical to `/api/v1/quote` — runs the full LangGraph DAG and returns
    the complete `AgentState` including `final_routing_decision`
    (`AUTO_APPROVE` | `MANUAL_REVIEW` | `REJECT`).
    """
    return await _run_pipeline(quote)


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/process_quote  — Legacy alias
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/api/process_quote",
    response_model=PipelineResponse,
    tags=["Pipeline (legacy)"],
    summary="Alias for /api/v1/quote — kept for backwards compatibility",
)
async def process_quote(quote: QuoteRequest, request: Request) -> Any:
    return await _run_pipeline(quote)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False, log_level="info")
