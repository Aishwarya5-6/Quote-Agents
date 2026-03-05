// ─────────────────────────────────────────────────────────────────────────────
//  Agent Pipeline · Mock Data Contract
//  Mirrors the exact shape returned by predict_and_explain() in
//  agents/agent1_risk_profiler.py (+ downstream agents 2–4).
// ─────────────────────────────────────────────────────────────────────────────

export type RiskTier = "Low" | "Medium" | "High";
export type SalesStatus = "NEAR_MISS" | "HOT_LEAD" | "COLD" | "CONVERTED";
export type RoutingStatus = "AUTO_APPROVE" | "ESCALATE" | "MANUAL_REVIEW";

export interface ShapDriver {
  feature: string;
  /** Raw SHAP value for the predicted class. Sign is relative to predicted class. */
  value: number;
  /** Raw label from the model — use fixShapLabel() in RiskGauge for display */
  impact: string;
}

export interface RiskAssessment {
  predicted_tier: RiskTier;
  confidence_score: number;
  top_shap_drivers: ShapDriver[];
}

export interface ConversionMetrics {
  bind_probability: number;
  /** How far (as a fraction) the quote is from the conversion threshold */
  distance_to_conversion: number;
  sales_status: SalesStatus;
}

export interface AdvisorStrategy {
  suggested_discount_pct: number;
  customer_facing_message: string;
  internal_reasoning: string;
}

export interface FinalRouting {
  routing_status: RoutingStatus;
  underwriter_justification: string;
}

export interface MasterResponse {
  transaction_id: string;
  risk_assessment: RiskAssessment;
  conversion_metrics: ConversionMetrics;
  advisor_strategy: AdvisorStrategy;
  final_routing: FinalRouting;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mock payload — copy-paste ready for frontend development.
//  Replace the fetch call in app/page.tsx with a real POST when the
//  backend is live at http://localhost:8001/api/process_quote.
// ─────────────────────────────────────────────────────────────────────────────
export const mockMasterResponse: MasterResponse = {
  transaction_id: "123",
  risk_assessment: {
    predicted_tier: "Low",
    confidence_score: 0.89,
    top_shap_drivers: [
      { feature: "Prev_Accidents", value: -0.45, impact: "decreases risk" },
    ],
  },
  conversion_metrics: {
    bind_probability: 0.35,
    distance_to_conversion: 0.03,
    sales_status: "NEAR_MISS",
  },
  advisor_strategy: {
    suggested_discount_pct: 5.0,
    customer_facing_message: "Clean driving record! Here is a 5% discount.",
    internal_reasoning: "Driver is Low Risk but a near miss for conversion.",
  },
  final_routing: {
    routing_status: "AUTO_APPROVE",
    underwriter_justification: "Approved.",
  },
};
