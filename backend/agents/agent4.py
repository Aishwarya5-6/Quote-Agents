# pip install groq python-dotenv
"""
=============================================================================
  Agent 4 – Decision Router  |  Auto Insurance Multi-Agent Pipeline
  Rule-based routing + Groq LLM reason generation.

  Role
  ────
  Final stage of the pipeline.  Consumes outputs from Agents 1, 2, and 3
  and makes ONE of three routing decisions:

    ESCALATE TO UNDERWRITER  — high-risk or very-low-conversion cases that
                               require immediate human expert review.
    AUTO APPROVE             — clean, low-risk, high-conversion cases that
                               can be processed without human involvement.
    AGENT FOLLOW-UP          — everything else; a sales/service agent
                               should engage the customer.

  Rule priority
  ─────────────
  Escalation rules are evaluated FIRST (strict priority).  A case that
  meets ANY escalation condition is escalated regardless of whether it
  would otherwise qualify for auto-approval.  Auto-approval is only
  reached when NONE of the escalation conditions fire.

  LLM layer
  ─────────
  After the rules determine the decision the reason field is enriched by a
  Groq LLM call (llama3-8b-8192, max_tokens=150, temperature=0.7).
  If the call fails for ANY reason the original rule-generated reason
  string is used as a silent fallback — the function never raises.

  API key
  ───────
  Loaded from the GROQ_API_KEY environment variable via python-dotenv.
  Create backend/.env with:  GROQ_API_KEY=your_key_here
  Never commit .env — ensure it is listed in .gitignore.

  Public API
  ──────────
  route_decision(agent1_output, agent2_output, agent3_output, quote_dict)
      Importable from pipeline.py with no side-effects.
=============================================================================
"""

import os
from dotenv import load_dotenv

# Load GROQ_API_KEY from backend/.env (or any parent .env on the dotenv search
# path).  This is a no-op when the variable is already in the environment.
load_dotenv()

try:
    from groq import Groq as _Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


# ---------------------------------------------------------------------------
#  GROQ LLM HELPER
# ---------------------------------------------------------------------------

_GROQ_MODEL       = "llama3-8b-8192"
_GROQ_MAX_TOKENS  = 150
_GROQ_TEMPERATURE = 0.7

# Prompt template filled at call time with rule-derived values
_PROMPT_TEMPLATE = """\
You are a senior insurance underwriter explaining a routing decision.
Risk tier: {risk_label} ({risk_tier}/2)
Conversion score: {conversion_score}/100
Premium adjustment recommended: {premium_flag}
Decision made: {decision}
Internal reason: {internal_reason}

Write exactly 2 sentences explaining this routing decision to a junior agent.
Be professional, factual, and never mention AI or algorithms."""


def _llm_reason(
    risk_label:      str,
    risk_tier:       int,
    conversion_score: float,
    premium_flag:    bool,
    decision:        str,
    internal_reason: str,
    fallback:        str,
) -> str:
    """
    Call Groq to generate a 2-sentence underwriter-facing reason string.

    Returns the LLM text on success, or `fallback` on any failure
    (missing key, wrong key, network error, timeout, API error,
    import error).  The caller is guaranteed a non-empty string.
    """
    if not _GROQ_AVAILABLE:
        return fallback

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return fallback

    try:
        client = _Groq(api_key=api_key, timeout=8.0)
        prompt = _PROMPT_TEMPLATE.format(
            risk_label=risk_label,
            risk_tier=risk_tier,
            conversion_score=conversion_score,
            premium_flag=premium_flag,
            decision=decision,
            internal_reason=internal_reason,
        )
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_GROQ_MAX_TOKENS,
            temperature=_GROQ_TEMPERATURE,
        )
        text = response.choices[0].message.content.strip()
        return text if text else fallback
    except Exception:  # noqa: BLE001 — intentional broad catch for silent fallback
        return fallback


# ---------------------------------------------------------------------------
#  CONSTANTS
# ---------------------------------------------------------------------------

# Risk tier encoded values
RISK_LOW    = 0
RISK_MEDIUM = 1
RISK_HIGH   = 2

# Decision labels
DEC_ESCALATE  = "Escalate to Underwriter"
DEC_REJECT    = "Reject Application"
DEC_APPROVE   = "Auto Approve"
DEC_FOLLOWUP  = "Agent Follow-Up"

# Priority levels
PRI_HIGH   = "High"
PRI_MEDIUM = "Medium"
PRI_LOW    = "Low"

# Conversion score thresholds
ESCALATE_HARD_THRESHOLD    = 30   # below this → always escalate
ESCALATE_ACCIDENT_THRESHOLD = 50  # accident present AND score below this → escalate
REJECT_CONVERSION_CEILING  = 20   # High risk + score below this → reject outright
AUTO_APPROVE_MIN_SCORE     = 50   # minimum score for auto-approval (calibrated to model's realistic 50-54% ceiling)


# ---------------------------------------------------------------------------
#  CORE FUNCTION
# ---------------------------------------------------------------------------

def route_decision(
    agent1_output: dict,
    agent2_output: dict,
    agent3_output: dict,
    quote_dict:    dict,
) -> dict:
    """
    Make the final routing decision by combining outputs from all three
    upstream agents.

    Parameters
    ----------
    agent1_output : dict
        Keys: risk_tier (int 0-2), risk_label (str "Low"/"Medium"/"High")

    agent2_output : dict
        Keys: conversion_score (float 0-100), will_buy (bool),
              confidence (str)

    agent3_output : dict
        Keys: premium_flag (bool), recommended_premium (float),
              adjustment (str), reason (str)

    quote_dict : dict
        Raw customer quote features. Keys used here:
          Prev_Accidents (int), Prev_Citations (int), Driver_Age (int)

    Returns
    -------
    dict with keys:
      decision       str   — "Escalate to Underwriter" | "Auto Approve" |
                             "Agent Follow-Up"
      reason         str   — 2-sentence LLM explanation (or rule fallback)
      human_required bool  — True for Escalate and Follow-Up
      priority       str   — "High" | "Medium" | "Low"
      action_items   list  — ordered list of strings for the handling agent
    """

    # Unpack inputs with safe defaults
    risk_tier        = int(agent1_output.get("risk_tier", RISK_LOW))
    risk_label       = str(agent1_output.get("risk_label", "Low"))
    conversion_score = float(agent2_output.get("conversion_score", 0.0))
    premium_flag     = bool(agent3_output.get("premium_flag", False))
    adjustment       = str(agent3_output.get("adjustment", "none"))
    prev_accidents   = int(quote_dict.get("Prev_Accidents", 0))
    prev_citations   = int(quote_dict.get("Prev_Citations", 0))

    # ------------------------------------------------------------------
    # Stage 0 — REJECTION  (hard deny for the absolute worst-case profiles)
    # High risk + near-zero conversion + multiple accidents → the
    # risk-reward is so unfavourable that escalation to an underwriter
    # would waste their time.  Reject outright.
    # Evaluated BEFORE escalation so the worst cases never reach Stage 1.
    # ------------------------------------------------------------------
    reject = (
        risk_tier == RISK_HIGH
        and conversion_score < REJECT_CONVERSION_CEILING
        and prev_accidents >= 2
    )

    if reject:
        internal_reason = (
            f"High risk tier with critically low conversion score "
            f"({conversion_score:.0f}) and {prev_accidents} prior accidents — "
            f"application does not meet minimum underwriting criteria"
        )
        decision  = DEC_REJECT
        priority  = PRI_HIGH
        human_req = False
        action_items = [
            "Application automatically rejected",
            "Notify applicant of denial",
            "Retain record for audit trail",
        ]
        reason = _llm_reason(
            risk_label=risk_label,
            risk_tier=risk_tier,
            conversion_score=conversion_score,
            premium_flag=premium_flag,
            decision=decision,
            internal_reason=internal_reason,
            fallback=internal_reason,
        )
        return {
            "decision":       decision,
            "reason":         reason,
            "human_required": human_req,
            "priority":       priority,
            "action_items":   action_items,
        }

    # ------------------------------------------------------------------
    # Stage 1 — ESCALATION RULES  (strict priority)
    # Any single match routes immediately to underwriter review.
    # ------------------------------------------------------------------
    escalation_reasons = []

    # Rule E1 — High risk tier is a hard escalation trigger.
    # The risk profiler assigned this customer to the highest loss-
    # probability class; underwriter sign-off is mandatory.
    if risk_tier == RISK_HIGH:
        escalation_reasons.append(
            "Customer is classified as High Risk by the risk profiler"
        )

    # Rule E2 — Very low conversion score signals an extremely unlikely
    # purchase; escalating prevents wasted advisor time on non-viable leads
    # and flags them for potential pricing or coverage review.
    if conversion_score < ESCALATE_HARD_THRESHOLD:
        escalation_reasons.append(
            f"Conversion score {conversion_score:.0f} is critically low (threshold: {ESCALATE_HARD_THRESHOLD})"
        )

    # Rule E3 — Prior accident combined with a low conversion score.
    # An accident history raises claim probability; a borderline conversion
    # score means the risk-reward is unfavourable without expert review.
    if prev_accidents > 0 and conversion_score < ESCALATE_ACCIDENT_THRESHOLD:
        escalation_reasons.append(
            f"Prior accident on record with borderline conversion score ({conversion_score:.0f})"
        )

    # Rule E4 — Any citation, regardless of risk tier.
    # Citations indicate a pattern of traffic violations that compound the
    # risk profile and always require underwriter judgement on whether to
    # bind at the quoted terms — even for otherwise low-risk customers.
    if prev_citations > 0:
        escalation_reasons.append(
            f"Prior citation on record ({prev_citations} citation(s)) requires underwriter review regardless of risk tier"
        )

    if escalation_reasons:
        internal_reason = "; ".join(escalation_reasons)
        decision  = DEC_ESCALATE
        priority  = PRI_HIGH
        human_req = True
        action_items = [
            "Immediate underwriter review required",
            "Do not process automatically",
            "Check risk factors",
        ]
        reason = _llm_reason(
            risk_label=risk_label,
            risk_tier=risk_tier,
            conversion_score=conversion_score,
            premium_flag=premium_flag,
            decision=decision,
            internal_reason=internal_reason,
            fallback=internal_reason,
        )
        return {
            "decision":       decision,
            "reason":         reason,
            "human_required": human_req,
            "priority":       priority,
            "action_items":   action_items,
        }

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Stage 2 — AUTO APPROVAL  (ALL conditions must be true)
    # Only reached when no rejection or escalation rule fired.
    # ------------------------------------------------------------------

    # Rules A1-A4: clean slate — low risk, high buying intent, no price
    # intervention needed, and no accident history.
    auto_approve = (
        risk_tier == RISK_LOW           # A1: lowest risk classification
        and conversion_score >= AUTO_APPROVE_MIN_SCORE  # A2: strong buying signal
        and not premium_flag            # A3: no premium adjustment required
        and prev_accidents == 0         # A4: clean accident record
    )

    if auto_approve:
        internal_reason = (
            "Low risk tier, strong conversion score, no premium adjustment "
            "required, and clean accident record"
        )
        decision  = DEC_APPROVE
        priority  = PRI_LOW
        human_req = False
        action_items = [
            "Process policy automatically",
            "No human review needed",
        ]
        reason = _llm_reason(
            risk_label=risk_label,
            risk_tier=risk_tier,
            conversion_score=conversion_score,
            premium_flag=premium_flag,
            decision=decision,
            internal_reason=internal_reason,
            fallback=internal_reason,
        )
        return {
            "decision":       decision,
            "reason":         reason,
            "human_required": human_req,
            "priority":       priority,
            "action_items":   action_items,
        }

    # ------------------------------------------------------------------
    # Stage 3 — AGENT FOLLOW-UP  (catch-all)
    # Reached when the case is not bad enough to escalate but not clean
    # enough to auto-approve.
    # ------------------------------------------------------------------

    # Priority and action items depend on whether a premium adjustment
    # was recommended by Agent 3.
    if premium_flag:
        # Medium priority: there is a concrete lever (premium reduction)
        # that the agent should discuss with the customer.
        internal_reason = (
            f"Moderate case — premium adjustment of {adjustment} recommended "
            "to improve conversion likelihood"
        )
        priority    = PRI_MEDIUM
        action_items = [
            f"Review recommended premium adjustment of {adjustment}",
            "Follow up with customer within 24 hours",
        ]
    else:
        # Low priority: no obvious pricing lever; standard follow-up.
        internal_reason = (
            "Moderate case — no premium adjustment triggered; "
            "relationship-driven follow-up recommended"
        )
        priority    = PRI_LOW
        action_items = [
            "Follow up with customer within 24 hours",
            "Review conversion score",
        ]

    decision  = DEC_FOLLOWUP
    human_req = True
    reason = _llm_reason(
        risk_label=risk_label,
        risk_tier=risk_tier,
        conversion_score=conversion_score,
        premium_flag=premium_flag,
        decision=decision,
        internal_reason=internal_reason,
        fallback=internal_reason,
    )
    return {
        "decision":       decision,
        "reason":         reason,
        "human_required": human_req,
        "priority":       priority,
        "action_items":   action_items,
    }


# ---------------------------------------------------------------------------
#  SELF-TEST  (runs only when executing the file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 62)
    print("  AGENT 4 – DECISION ROUTER  |  Self-test")
    print("=" * 62)

    # ── Test 1: Auto Approve ───────────────────────────────────────────
    # Low risk, high conversion, no premium flag, clean record.
    t1_a1 = {"risk_tier": 0, "risk_label": "Low"}
    t1_a2 = {"conversion_score": 72, "will_buy": True,  "confidence": "High"}
    t1_a3 = {"premium_flag": False, "recommended_premium": 580.0,
              "adjustment": "none", "reason": "No adjustment needed"}
    t1_q  = {"Prev_Accidents": 0, "Prev_Citations": 0, "Driver_Age": 42}

    result_1 = route_decision(t1_a1, t1_a2, t1_a3, t1_q)
    print("\n  Test 1 — Auto Approve")
    print(f"  Input  : risk=Low  score=72  premium_flag=False  accidents=0")
    for k, v in result_1.items():
        print(f"    {k:<18}: {v}")

    # ── Test 2: Agent Follow-Up (with premium flag) ───────────────────
    # Medium risk, borderline conversion, premium adjustment triggered.
    t2_a1 = {"risk_tier": 1, "risk_label": "Medium"}
    t2_a2 = {"conversion_score": 45, "will_buy": False, "confidence": "Medium"}
    t2_a3 = {"premium_flag": True,  "recommended_premium": 650.0,
              "adjustment": "-10%", "reason": "Price sensitive customer"}
    t2_q  = {"Prev_Accidents": 0, "Prev_Citations": 0, "Driver_Age": 35}

    result_2 = route_decision(t2_a1, t2_a2, t2_a3, t2_q)
    print("\n  Test 2 — Agent Follow-Up (premium flag active)")
    print(f"  Input  : risk=Medium  score=45  premium_flag=True  accidents=0")
    for k, v in result_2.items():
        print(f"    {k:<18}: {v}")

    # ── Test 3: Escalate to Underwriter ───────────────────────────────
    # High risk tier — hard escalation regardless of anything else.
    t3_a1 = {"risk_tier": 2, "risk_label": "High"}
    t3_a2 = {"conversion_score": 38, "will_buy": False, "confidence": "Low"}
    t3_a3 = {"premium_flag": True,  "recommended_premium": 900.0,
              "adjustment": "-15%", "reason": "High premium for income level"}
    t3_q  = {"Prev_Accidents": 1, "Prev_Citations": 1, "Driver_Age": 22}

    result_3 = route_decision(t3_a1, t3_a2, t3_a3, t3_q)
    print("\n  Test 3 — Escalate to Underwriter")
    print(f"  Input  : risk=High  score=38  accidents=1  citations=1")
    for k, v in result_3.items():
        print(f"    {k:<18}: {v}")

    print("\n" + "=" * 62 + "\n")
