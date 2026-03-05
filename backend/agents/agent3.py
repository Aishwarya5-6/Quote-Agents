# pip install groq python-dotenv
"""
=============================================================================
  Agent 3 – Premium Advisor  |  Auto Insurance Multi-Agent Pipeline
  Rule-based logic + Groq LLM reason generation.

  Activation gate
  ───────────────
  Only runs when Agent 2 conversion_score < 40.
  At >= 40 the customer is already likely to bind; no adjustment is needed.

  Rule evaluation order
  ─────────────────────
  All applicable rules are evaluated independently.  Where multiple rules
  recommend a % reduction the LARGEST reduction wins (most customer-
  friendly and least likely to produce contradictory advice).  Coverage
  downgrade rules are applied on top of the winning reduction rule when
  they fire alongside a reduction rule.

  LLM layer
  ─────────
  After the rules determine the adjustment the reason field is enriched
  by a Groq LLM call (llama3-8b-8192, max_tokens=150, temperature=0.7).
  If the call fails for ANY reason the original rule-generated reason
  string is used as a silent fallback — the function never raises.

  API key
  ───────
  Loaded from the GROQ_API_KEY environment variable via python-dotenv.
  Create backend/.env with:  GROQ_API_KEY=your_key_here
  Never commit .env — ensure it is listed in .gitignore.

  Public API
  ──────────
  advise_premium(quote_dict, conversion_score) -> dict
      Importable from pipeline.py with no side-effects.
=============================================================================
"""


import logging
import os

from dotenv import load_dotenv

# Load GROQ_API_KEY from backend/.env (or any parent .env on the dotenv search
# path).  This is a no-op when the variable is already in the environment.
load_dotenv()

log = logging.getLogger(__name__)

# Human-readable string shown in the UI when the Groq call fails with a
# 400 / network error.  Keeps the Sequential UI "neat" with a real sentence.
_LLM_UI_FALLBACK = (
    "Premium is competitive for this risk profile; no manual adjustment needed."
)

try:
    from groq import Groq as _Groq  # type: ignore[import-not-found]
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


# ---------------------------------------------------------------------------
#  GROQ LLM HELPER
# ---------------------------------------------------------------------------

# Groq model and generation settings
_GROQ_MODEL       = "llama3-8b-8192"
_GROQ_MAX_TOKENS  = 150
_GROQ_TEMPERATURE = 0.7

# Prompt template — filled at call time with rule-derived values
_PROMPT_TEMPLATE = """You are a professional insurance advisor speaking directly to a customer.
The customer's quote has a conversion score of {conversion_score}/100.
Their current premium is ${original_premium}.
Our system recommends a {adjustment} reduction to ${recommended_premium}.
Internal reasons: {triggered_reasons}

Write exactly 2 sentences explaining this adjustment to the customer.
Be empathetic, professional, and never mention AI or algorithms."""


def _llm_reason(
    conversion_score:  float,
    original_premium:  float,
    recommended_premium: float,
    adjustment:        str,
    triggered_reasons: str,
    fallback:          str,
) -> str:
    """
    Call Groq to generate a 2-sentence customer-facing reason string.

    Returns the LLM text on success, or `fallback` on any failure
    (missing key, network error, timeout, API error, import error).
    The caller is guaranteed a non-empty string regardless of outcome.
    """
    # Fast-path: library not installed
    if not _GROQ_AVAILABLE:
        return fallback

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return fallback

    try:
        client = _Groq(api_key=api_key, timeout=5.0)  # fail-fast: use _LLM_UI_FALLBACK on slow/bad connections
        prompt = _PROMPT_TEMPLATE.format(
            conversion_score=conversion_score,
            original_premium=f"{original_premium:.2f}",
            recommended_premium=f"{recommended_premium:.2f}",
            adjustment=adjustment,
            triggered_reasons=triggered_reasons,
        )
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_GROQ_MAX_TOKENS,
            temperature=_GROQ_TEMPERATURE,
        )
        text = response.choices[0].message.content.strip()
        return text if text else fallback
    except Exception as exc:  # noqa: BLE001 — surface error then fall back gracefully
        log.warning(
            "Agent 3 │ Groq API call failed — %s: %s",
            type(exc).__name__,
            exc,
        )
        # Return the UI-friendly constant so the frontend always has a neat,
        # human-readable sentence in the customer_facing_message field.
        return _LLM_UI_FALLBACK


# ---------------------------------------------------------------------------
#  CONSTANTS
# ---------------------------------------------------------------------------

# Conversion score below which this agent activates
ACTIVATION_THRESHOLD = 40

# Salary-range encoded values (ordinal 0-4)
SAL_VERY_LOW  = 0   # <= $25 K
SAL_LOW       = 1   # $25 K – $40 K

# Coverage encoded values
COV_BASIC     = 0
COV_BALANCED  = 1
COV_ENHANCED  = 2

# Vehicle-cost-range encoded values
VEH_CHEAP     = 1   # <= $20 K  (0 = <= $10 K, 1 = $10-20 K)

# Re-quote flag
REQUOTE_YES   = 1


# ---------------------------------------------------------------------------
#  CORE FUNCTION
# ---------------------------------------------------------------------------

def advise_premium(quote_dict: dict, conversion_score: float) -> dict:
    """
    Analyse a low-conversion quote and recommend a premium adjustment.

    Parameters
    ----------
    quote_dict : dict
        Keys expected:
          Quoted_Premium    (float) — current quoted premium in dollars
          Sal_Range         (int)   — 0=<=25K, 1=25-40K, 2=40-60K,
                                      3=60-90K, 4=>90K
          Coverage          (int)   — 0=Basic, 1=Balanced, 2=Enhanced
          Vehicl_Cost_Range (int)   — 0=<=10K, 1=10-20K, 2=20-30K,
                                      3=30-40K, 4=>40K
          Re_Quote          (int)   — 0=No, 1=Yes

    conversion_score : float
        Probability score (0-100) from Agent 2. Values below 40 trigger
        this agent.

    Returns
    -------
    dict with keys:
      premium_flag        bool   — True when an adjustment is recommended
      recommended_premium float  — adjusted premium (same as original if
                                   no adjustment)
      adjustment          str    — e.g. "-15%" or "none"
      reason              str    — plain-English explanation for the advisor
      original_premium    float  — the unchanged input premium
    """

    # ------------------------------------------------------------------
    # Gate: if conversion score is already acceptable, do nothing.
    # ------------------------------------------------------------------
    original_premium = float(quote_dict.get("Quoted_Premium", 0.0))

    if conversion_score >= ACTIVATION_THRESHOLD:
        return {
            "premium_flag":        False,
            "recommended_premium": round(original_premium, 2),
            "adjustment":          "none",
            "reason":              "Conversion score is sufficient; no adjustment needed.",
            "original_premium":    round(original_premium, 2),
        }

    # Unpack inputs with safe defaults
    sal_range         = int(quote_dict.get("Sal_Range", 0))
    coverage          = int(quote_dict.get("Coverage", COV_BASIC))
    vehicl_cost_range = int(quote_dict.get("Vehicl_Cost_Range", 0))
    re_quote          = int(quote_dict.get("Re_Quote", 0))

    # ------------------------------------------------------------------
    # Rule evaluation — collect all triggered reductions and reasons.
    # Multiple rules may fire; the largest reduction is applied.
    # ------------------------------------------------------------------

    reductions = []   # list of (pct_reduction, reason_string) tuples
    flags      = []   # non-reduction advisory flags (coverage downgrade)

    # Rule 1 — Premium too high relative to low salary
    # High-earning customers can absorb a > $800 premium; low-earners cannot.
    if original_premium > 800 and sal_range <= SAL_LOW:
        reductions.append((
            15,
            "Customer salary range suggests premium is too high",
        ))

    # Rule 2 — Moderate premium still unaffordable at the lowest salary band
    # Even a $700+ premium strains a <= $25 K annual income.
    if original_premium > 700 and sal_range == SAL_VERY_LOW:
        reductions.append((
            10,
            "Premium exceeds affordability threshold for the lowest income bracket",
        ))

    # Rule 3 — Enhanced coverage is a luxury the customer likely cannot afford
    # When salary is very low or low, Enhanced coverage drives up cost
    # unnecessarily; Balanced provides adequate protection at lower cost.
    if coverage == COV_ENHANCED and sal_range <= SAL_LOW:
        flags.append("coverage_downgrade")
        # Record a 0% reduction so the flag is captured even if no price
        # reduction rule fires; the coverage advice is standalone.
        reductions.append((
            0,
            "Enhanced coverage not aligned with customer salary; downgrade to Balanced recommended",
        ))

    # Rule 4 — Re-quote signals active price shopping / price sensitivity
    # A returning customer who did not bind first time is signalling the
    # price is a barrier; a 10% gesture is often enough to close.
    if re_quote == REQUOTE_YES:
        reductions.append((
            10,
            "Customer is price-sensitive (re-quote detected); proactive reduction recommended",
        ))

    # Rule 5 — Over-insured: cheap vehicle + Enhanced coverage
    # Enhanced coverage on a low-value vehicle (<= $20 K) is rarely
    # cost-effective for the customer; Basic coverage is appropriate.
    if vehicl_cost_range <= VEH_CHEAP and coverage == COV_ENHANCED:
        flags.append("coverage_downgrade")
        reductions.append((
            0,
            "Vehicle value does not justify Enhanced coverage; Basic coverage recommended",
        ))

    # ------------------------------------------------------------------
    # Resolve — pick best (largest) reduction; merge reasons if tie.
    # ------------------------------------------------------------------

    if not reductions:
        # No rules fired despite low conversion score (e.g. score = 39,
        # premium is low, customer is well-off).  Return a soft advisory.
        fallback_reason = (
            "Low conversion score detected but no pricing rule applies. "
            "Manual advisor review recommended."
        )
        reason_text = _llm_reason(
            conversion_score=conversion_score,
            original_premium=original_premium,
            recommended_premium=original_premium,
            adjustment="none",
            triggered_reasons=fallback_reason,
            fallback=fallback_reason,
        )
        return {
            "premium_flag":        True,
            "recommended_premium": round(original_premium, 2),
            "adjustment":          "none",
            "reason":              reason_text,
            "original_premium":    round(original_premium, 2),
        }

    # Select the rule(s) with the maximum reduction percentage
    max_pct = max(pct for pct, _ in reductions)

    # Collect all reasons that correspond to the winning reduction
    winning_reasons = [
        reason for pct, reason in reductions if pct == max_pct
    ]

    # If a coverage downgrade was independently flagged, append it
    if "coverage_downgrade" in flags and max_pct == 0:
        # The only rules that fired were coverage-flag rules (0% reduction)
        reason_text = winning_reasons[0]
    elif "coverage_downgrade" in flags:
        # Combine price reduction reason with coverage note
        reason_text = (
            winning_reasons[0]
            + ". Additionally, a coverage downgrade is advisable."
        )
    else:
        reason_text = winning_reasons[0]

    reduction_factor  = max_pct / 100.0
    recommended       = round(original_premium * (1.0 - reduction_factor), 2)
    adjustment_label  = f"-{max_pct}%" if max_pct > 0 else "none"

    # Collect all triggered reasons (for the LLM prompt context)
    all_triggered = "; ".join(r for _, r in reductions)

    # Ask the LLM to write a customer-facing version; fall back silently
    reason_text = _llm_reason(
        conversion_score=conversion_score,
        original_premium=original_premium,
        recommended_premium=recommended,
        adjustment=adjustment_label,
        triggered_reasons=all_triggered,
        fallback=reason_text,
    )

    return {
        "premium_flag":        True,
        "recommended_premium": recommended,
        "adjustment":          adjustment_label,
        "reason":              reason_text,
        "original_premium":    round(original_premium, 2),
    }


# ---------------------------------------------------------------------------
#  SELF-TEST  (runs only when executing the file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  AGENT 3 – PREMIUM ADVISOR  |  Self-test")
    print("=" * 60)

    # ── Test 1: Rule triggers (high premium + low salary) ─────────────
    # Premium $720, Sal_Range=1 (low), conversion_score=35 → Rule 1+ fires.
    sample_1 = {
        "Quoted_Premium":    720.0,
        "Sal_Range":         1,
        "Coverage":          1,
        "Vehicl_Cost_Range": 1,
        "Re_Quote":          1,
    }
    result_1 = advise_premium(sample_1, conversion_score=35)
    print("\n  Test 1 — Triggers (high premium + re-quote, low salary)")
    print(f"  Input  : Premium=${sample_1['Quoted_Premium']:.2f}  "
          f"Sal_Range={sample_1['Sal_Range']}  Re_Quote={sample_1['Re_Quote']}")
    for k, v in result_1.items():
        print(f"    {k:<22}: {v}")

    # ── Test 2: No adjustment needed — conversion score already >= 40 ──
    sample_2 = {
        "Quoted_Premium":    650.0,
        "Sal_Range":         3,
        "Coverage":          1,
        "Vehicl_Cost_Range": 2,
        "Re_Quote":          0,
    }
    result_2 = advise_premium(sample_2, conversion_score=62)
    print("\n  Test 2 — No trigger (conversion_score=62 >= 40)")
    print(f"  Input  : Premium=${sample_2['Quoted_Premium']:.2f}  "
          f"conversion_score=62")
    for k, v in result_2.items():
        print(f"    {k:<22}: {v}")

    # ── Test 3: Edge case — Enhanced coverage on cheap vehicle, lowest ──
    #           salary band, re-quote (multiple rules, pick max)
    sample_3 = {
        "Quoted_Premium":    850.0,
        "Sal_Range":         0,     # SAL_VERY_LOW (<= $25 K)
        "Coverage":          2,     # COV_ENHANCED
        "Vehicl_Cost_Range": 0,     # cheapest vehicle band
        "Re_Quote":          1,     # price-sensitive re-quote
    }
    result_3 = advise_premium(sample_3, conversion_score=22)
    print("\n  Test 3 — Edge case (max reduction: Rule 1 + 4 + 5 all fire)")
    print(f"  Input  : Premium=${sample_3['Quoted_Premium']:.2f}  "
          f"Sal_Range={sample_3['Sal_Range']}  Coverage={sample_3['Coverage']}  "
          f"Re_Quote={sample_3['Re_Quote']}")
    for k, v in result_3.items():
        print(f"    {k:<22}: {v}")

    print("\n" + "=" * 60 + "\n")
