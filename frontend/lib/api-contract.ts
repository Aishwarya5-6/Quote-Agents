// ─────────────────────────────────────────────────────────────────────────────
//  CONTRACT-FIRST API TYPES
//  Single source of truth matching the Pydantic schemas in backend/main.py
//
//  Endpoint: POST http://localhost:8001/api/v1/full-analysis
//
//  Each type maps to one agent's output:
//    RiskAssessment    ← Agent 1  (XGBoost + SHAP + OOD Gate)
//    ConversionMetrics ← Agent 2  (SMOTE + CalibratedClassifierCV)
//    AdvisorStrategy   ← Agent 3  (Rules + Groq LLM)
//    FinalRouting      ← Agent 4  (Rules + Groq LLM)
// ─────────────────────────────────────────────────────────────────────────────

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

// ── Enums ────────────────────────────────────────────────────────────────────
export type RiskTier = "Low" | "Medium" | "High";

export type SalesStatus =
  | "HIGH_PROPENSITY"
  | "NEAR_MISS_FOR_ADVISOR"
  | "LOW_PROB"
  | "UNCERTAIN";

export type RoutingDecision = "AUTO_APPROVE" | "MANUAL_REVIEW" | "REJECT";

export type AgentDecision =
  | "Auto Approve"
  | "Escalate to Underwriter"
  | "Agent Follow-Up"
  | "Reject Application";

export type ShapMagnitude = "HIGH" | "MEDIUM" | "LOW";

// ── Agent 1 — Risk Profiler ─────────────────────────────────────────────────
export interface ShapDriver {
  feature:    string;
  shap_value: number;
  direction:  string;   // "↑ increases risk" | "↓ decreases risk"
  magnitude:  ShapMagnitude;
}

export interface RiskAssessment {
  predicted_tier:      RiskTier;
  confidence_score:    number;          // 0.0 – 1.0
  ood_flag:            string;          // "OK"
  class_probabilities: Record<string, number>;
  top_shap_drivers:    ShapDriver[];
}

// ── Agent 2 — Conversion Predictor ──────────────────────────────────────────
export interface ConversionMetrics {
  bind_probability:       number | null;  // 0.0 – 1.0
  sales_status:           SalesStatus | null;
  distance_to_conversion: number | null;  // ≥ 0
}

// ── Agent 3 — Premium Advisor ───────────────────────────────────────────────
export interface AdvisorStrategy {
  premium_flag:            boolean;
  suggested_discount_pct:  string | null;  // e.g. "-15%" or "none"
  recommended_premium:     number | null;
  original_premium:        number | null;
  customer_facing_message: string | null;  // LLM-enriched or rule-based reason
}

// ── Agent 4 — Underwriting Router ───────────────────────────────────────────
export interface FinalRouting {
  decision:               AgentDecision | null;
  reason:                 string | null;   // LLM-enriched 2-sentence justification
  human_required:         boolean;
  priority:               string | null;   // "High" | "Medium" | "Low"
  action_items:           string[];
  final_routing_decision: RoutingDecision | null;
}

// ── Master pipeline response ────────────────────────────────────────────────
export interface PipelineResponse {
  transaction_id:         string;
  status:                 string;  // "OK" | "LOW_CONFIDENCE_ESCALATE"
  final_routing_decision: RoutingDecision | null;
  escalation_reason:      string | null;
  risk_assessment:        RiskAssessment  | null;
  conversion_metrics:     ConversionMetrics | null;
  advisor_strategy:       AdvisorStrategy | null;
  final_routing:          FinalRouting    | null;
}

// ── OOD Error response (HTTP 422) ───────────────────────────────────────────
export interface OodErrorResponse {
  transaction_id: string;
  status:         string;   // "ACTION_REQUIRED: DATA_ANOMALY_ESCALATE"
  message:        string;
  input:          Record<string, unknown>;
}

// ── Quote input ─────────────────────────────────────────────────────────────
export interface QuoteInput {
  Driver_Age:        number;
  Driving_Exp:       number;
  Prev_Accidents:    number;
  Prev_Citations:    number;
  Annual_Miles:      number;
  Veh_Usage:         "Business" | "Commute" | "Pleasure";
  Quoted_Premium?:   number;
  Sal_Range?:        number;
  Coverage?:         number;
  Vehicl_Cost_Range?: number;
  Re_Quote?:         number;
}

// ── Conversion threshold (from Agent 2 v2 training) ─────────────────────────
export const CONVERSION_THRESHOLD = 0.3370;
