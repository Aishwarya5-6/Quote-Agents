// ─────────────────────────────────────────────────────────────────────────────
//  CONTRACT-FIRST API TYPES
//  This file is the single source of truth for the shape returned by
//  POST http://localhost:8000/api/process_quote
//
//  Each type maps directly to one agent's output:
//    RiskAssessment    ← Agent 1  (agent1_risk_profiler.py)
//    ConversionMetrics ← Agent 2
//    AdvisorStrategy   ← Agent 3
//    FinalRouting      ← Agent 4
// ─────────────────────────────────────────────────────────────────────────────

export type RiskTier      = "Low" | "Medium" | "High";
export type OodFlag       = "OK" | "ACTION_REQUIRED: DATA_ANOMALY";
export type SalesStatus   = "NEAR_MISS" | "HOT_LEAD" | "COLD" | "CONVERTED";
export type RoutingStatus = "AUTO_APPROVE" | "ESCALATE" | "MANUAL_REVIEW";

// ── Agent 1 ──────────────────────────────────────────────────────────────────
export interface ShapDriver {
  /** Raw feature name from the XGBoost model */
  feature: string;
  /** Raw SHAP value for the predicted class. Sign is relative to predicted class — use fixShapLabel() before displaying */
  value: number;
  /** Raw label from agent — use fixShapLabel() to correct for Low-tier predictions */
  impact: string;
}

export interface RiskAssessment {
  predicted_tier:    RiskTier;
  /** Calibrated probability for the predicted class: 0.0 – 1.0 */
  confidence_score:  number;
  /** IsolationForest + Physics Check gate result */
  ood_flag:          OodFlag;
  /** Top SHAP contributors sorted by |value| descending */
  top_shap_drivers:  ShapDriver[];
}

// ── Agent 2 ──────────────────────────────────────────────────────────────────
export interface ConversionMetrics {
  /** Predicted probability that the quote converts to a bound policy: 0.0 – 1.0 */
  bind_probability:          number;
  sales_status:              SalesStatus;
  /** How far (as a decimal fraction) the quote is below the conversion threshold */
  distance_to_conversion:    number;
}

// ── Agent 3 ──────────────────────────────────────────────────────────────────
export interface AdvisorStrategy {
  suggested_discount_pct:   number;
  /** LLM-generated message shown directly to the customer */
  customer_facing_message:  string;
  /** LLM chain-of-thought reasoning visible to agents/underwriters only */
  internal_reasoning:       string;
}

// ── Agent 4 ──────────────────────────────────────────────────────────────────
export interface FinalRouting {
  routing_status:              RoutingStatus;
  underwriter_justification:   string;
}

// ── Master pipeline response ──────────────────────────────────────────────────
export interface PipelineResponse {
  /** UUID v4 — unique per quote request */
  transaction_id:      string;
  risk_assessment:     RiskAssessment;
  conversion_metrics:  ConversionMetrics;
  advisor_strategy:    AdvisorStrategy;
  final_routing:       FinalRouting;
}

// ─────────────────────────────────────────────────────────────────────────────
//  MOCK PIPELINE RESPONSE
//  Copy-paste ready for frontend development.
//  Replace the fetch() call in app/page.tsx to go live.
// ─────────────────────────────────────────────────────────────────────────────
export const MOCK_PIPELINE_RESPONSE: PipelineResponse = {
  transaction_id: "a3f7c9d2-81b4-4e2a-9c6f-1d8e2b4a7f3c",

  risk_assessment: {
    predicted_tier:   "Low",
    confidence_score: 0.89,
    ood_flag:         "OK",
    top_shap_drivers: [
      { feature: "Prev_Accidents",   value: -0.45, impact: "decreases risk" },
      { feature: "Annual_Miles",     value: -0.31, impact: "decreases risk" },
      { feature: "Driving_Exp",      value:  0.28, impact: "increases risk" },
      { feature: "Age_Exp_Gap",      value: -0.19, impact: "decreases risk" },
      { feature: "Total_Incidents",  value: -0.14, impact: "decreases risk" },
    ],
  },

  conversion_metrics: {
    bind_probability:       0.35,
    sales_status:           "NEAR_MISS",
    distance_to_conversion: 0.03,
  },

  advisor_strategy: {
    suggested_discount_pct:  5.0,
    customer_facing_message: "Great news! Your clean driving record qualifies you for a 5% loyalty discount. Lock in this rate today.",
    internal_reasoning:      "Driver is Low Risk (89% confidence) but a NEAR_MISS for conversion. A 5% discount bridges the remaining 3% gap without eroding underwriting margin.",
  },

  final_routing: {
    routing_status:            "AUTO_APPROVE",
    underwriter_justification: "Low Risk tier · 89% model confidence · OOD gate clear · No adverse SHAP drivers. Auto-approval criteria met.",
  },
};
