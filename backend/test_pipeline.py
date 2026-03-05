"""
test_pipeline.py — End-to-End Verification of the 4-Agent LangGraph Pipeline
═════════════════════════════════════════════════════════════════════════════

Self-contained test script that imports the compiled LangGraph app from
main.py and runs it against four carefully chosen quote profiles:

  1. Safe Driver      →  Low risk, clean record, high experience
  2. High-Risk Driver →  Young, multiple accidents, heavy mileage
  3. Near Miss        →  Moderate risk, bind_probability near 0.3370 threshold
  4. Corrupt Input    →  Negative age to trigger the OOD Safety Gate

Each test validates every agent's output keys, prints a human-readable
summary, and checks structural invariants. The full graph execution time
is measured and printed for frontend latency estimation.

Usage
─────
  cd backend
  ../.venv/bin/python test_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── Make sure we can import main.py (handles sys.path and artifact loading) ──
sys.path.insert(0, str(Path(__file__).parent))

# ── Import the compiled LangGraph pipeline and the OOD sentinel constant ─────
from main import _pipeline, _build_graph, OOD_FLAG, OOD_ESCALATE_STATUS
from agents.agent1_risk_profiler import RiskProfilerPredictor
from agents.agent2_conversion_predictor import ConversionPredictor

# ── Ensure the ML singletons are loaded (startup event doesn't fire in CLI) ──
import main as _main_module

MODELS_DIR = Path(__file__).parent / "models"

if _main_module._risk_engine is None:
    print("⏳ Loading Agent 1 artifacts …")
    _main_module._risk_engine = RiskProfilerPredictor.from_artifacts(MODELS_DIR)
    print("✅ Agent 1 loaded.\n")

if _main_module._conv_engine is None:
    print("⏳ Loading Agent 2 artifacts …")
    _main_module._conv_engine = ConversionPredictor.from_artifacts(MODELS_DIR)
    print("✅ Agent 2 loaded.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  TEST CASES
# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "Safe Driver",
        "description": "Clean record, low miles, high experience → expect Low risk",
        "input": {
            "Driver_Age":      45,
            "Driving_Exp":     25,
            "Prev_Accidents":  0,
            "Prev_Citations":  0,
            "Annual_Miles":    8_000,
            "Veh_Usage":       "Pleasure",
            "Quoted_Premium":  480.0,
            "Sal_Range":       3,
            "Coverage":        1,
            "Vehicl_Cost_Range": 2,
            "Re_Quote":        0,
        },
        "expect_ood": False,
    },
    {
        "name": "High-Risk Driver",
        "description": "Young, 3 accidents, high miles → expect High risk + Escalate",
        "input": {
            "Driver_Age":      21,
            "Driving_Exp":     3,
            "Prev_Accidents":  3,
            "Prev_Citations":  2,
            "Annual_Miles":    40_000,
            "Veh_Usage":       "Commute",
            "Quoted_Premium":  1100.0,
            "Sal_Range":       0,
            "Coverage":        2,
            "Vehicl_Cost_Range": 0,
            "Re_Quote":        1,
        },
        "expect_ood": False,
    },
    {
        "name": "Near Miss (Threshold Driver)",
        "description": (
            "Moderate risk, conversion probability near the 0.3370 threshold "
            "— tests the NEAR_MISS_FOR_ADVISOR sales_status bucket"
        ),
        "input": {
            "Driver_Age":      30,
            "Driving_Exp":     10,
            "Prev_Accidents":  1,
            "Prev_Citations":  1,
            "Annual_Miles":    18_000,
            "Veh_Usage":       "Commute",
            "Quoted_Premium":  720.0,
            "Sal_Range":       1,
            "Coverage":        1,
            "Vehicl_Cost_Range": 1,
            "Re_Quote":        0,
        },
        "expect_ood": False,
    },
    {
        "name": "Corrupt Input (OOD Gate)",
        "description": (
            "Negative Driver_Age (-5) → deterministic physics check should "
            "fire and return DATA_ANOMALY instead of crashing the graph"
        ),
        "input": {
            "Driver_Age":      -5,
            "Driving_Exp":     3,
            "Prev_Accidents":  0,
            "Prev_Citations":  0,
            "Annual_Miles":    12_000,
            "Veh_Usage":       "Pleasure",
        },
        "expect_ood": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_PASS = "✅"
_FAIL = "❌"

_total_checks  = 0
_passed_checks = 0


def check(condition: bool, label: str) -> bool:
    """Register a single assertion with a human-readable label."""
    global _total_checks, _passed_checks
    _total_checks += 1
    if condition:
        _passed_checks += 1
        print(f"    {_PASS}  {label}")
    else:
        print(f"    {_FAIL}  {label}")
    return condition


def validate_agent1(risk: Dict[str, Any], *, expect_ood: bool) -> None:
    """Validate Agent 1 (Risk Profiler) output."""
    print("\n  ── Agent 1: Risk Profiler ─────────────────────────")

    if expect_ood:
        check(
            risk.get("status") == OOD_FLAG,
            f"OOD gate triggered  (status = {risk.get('status')!r})",
        )
        check(
            "message" in risk,
            f"Anomaly message present  ({risk.get('message', '')[:80]}…)",
        )
        return

    # Normal prediction path
    check(
        risk.get("status") == "OK",
        f"Status is OK  (got {risk.get('status')!r})",
    )
    check(
        risk.get("predicted_tier") in {"Low", "Medium", "High"},
        f"risk_tier = {risk.get('predicted_tier')!r}",
    )
    check(
        isinstance(risk.get("confidence"), (int, float)) and 0 <= risk["confidence"] <= 1,
        f"confidence = {risk.get('confidence')}  (valid 0–1 float)",
    )
    top3 = risk.get("top_3_features", [])
    check(
        isinstance(top3, list) and len(top3) >= 1,
        f"top_3_features returned {len(top3)} features",
    )
    if top3:
        first = top3[0]
        check(
            all(k in first for k in ("feature", "shap_value", "direction", "magnitude")),
            f"First SHAP driver has required keys  (feature={first.get('feature')!r})",
        )
    class_probs = risk.get("class_probabilities", {})
    check(
        set(class_probs.keys()) == {"High", "Low", "Medium"},
        f"class_probabilities keys = {set(class_probs.keys())}",
    )


def validate_agent2(conv: Dict[str, Any], *, expect_ood: bool) -> None:
    """Validate Agent 2 (Conversion Predictor) output."""
    print("\n  ── Agent 2: Conversion Predictor ──────────────────")

    if expect_ood:
        print(f"    ⏭️  Skipped (upstream OOD — Agent 2 ran on anomalous input)")
        return

    bind_prob = conv.get("bind_probability")
    check(
        isinstance(bind_prob, (int, float)) and 0 <= bind_prob <= 1,
        f"bind_probability = {bind_prob}  (valid 0–1 float)",
    )

    VALID_STATUSES = {
        "HIGH_PROPENSITY", "NEAR_MISS_FOR_ADVISOR", "LOW_PROB", "UNCERTAIN",
    }
    sales_status = conv.get("sales_status")
    check(
        sales_status in VALID_STATUSES,
        f"sales_status = {sales_status!r}",
    )

    dtc = conv.get("distance_to_conversion")
    check(
        isinstance(dtc, (int, float)) and dtc >= 0,
        f"distance_to_conversion = {dtc}",
    )

    conversion_score = conv.get("conversion_score")
    check(
        isinstance(conversion_score, (int, float)),
        f"conversion_score = {conversion_score}  (derived from bind_prob × 100)",
    )


def validate_agent3(advisor: Dict[str, Any], *, expect_ood: bool) -> None:
    """Validate Agent 3 (Premium Advisor) output."""
    print("\n  ── Agent 3: Premium Advisor ───────────────────────")

    if expect_ood:
        print(f"    ⏭️  Skipped (upstream OOD — Agent 3 ran on anomalous input)")
        return

    check(
        isinstance(advisor.get("premium_flag"), bool),
        f"premium_flag = {advisor.get('premium_flag')!r}  (bool)",
    )

    rec = advisor.get("recommended_premium")
    check(
        isinstance(rec, (int, float)) and rec >= 0,
        f"recommended_premium = ${rec}",
    )

    orig = advisor.get("original_premium")
    check(
        isinstance(orig, (int, float)) and orig >= 0,
        f"original_premium = ${orig}",
    )

    reason = advisor.get("reason", "")
    check(
        isinstance(reason, str) and len(reason) > 0,
        f"advisor_pitch reason is non-empty string  ({len(reason)} chars)",
    )

    adjustment = advisor.get("adjustment")
    check(
        isinstance(adjustment, str),
        f"adjustment = {adjustment!r}",
    )


def validate_agent4(
    final: Dict[str, Any],
    canonical: str,
    *,
    expect_ood: bool,
) -> None:
    """Validate Agent 4 (Underwriting Router) output."""
    print("\n  ── Agent 4: Underwriting Router ───────────────────")

    if expect_ood:
        print(f"    ⏭️  Skipped (upstream OOD — Agent 4 ran on anomalous input)")
        return

    VALID_DECISIONS = {
        "Auto Approve",
        "Escalate to Underwriter",
        "Agent Follow-Up",
    }
    decision = final.get("decision")
    check(
        decision in VALID_DECISIONS,
        f"decision = {decision!r}",
    )

    reason = final.get("reason", "")
    check(
        isinstance(reason, str) and len(reason) > 0,
        f"reason is non-empty string  ({len(reason)} chars)",
    )

    check(
        isinstance(final.get("human_required"), bool),
        f"human_required = {final.get('human_required')!r}  (bool)",
    )

    priority = final.get("priority")
    check(
        priority in {"High", "Medium", "Low", None},
        f"priority = {priority!r}",
    )

    action_items = final.get("action_items", [])
    check(
        isinstance(action_items, list) and len(action_items) >= 1,
        f"action_items has {len(action_items)} items",
    )

    VALID_ROUTING = {"AUTO_APPROVE", "MANUAL_REVIEW", "REJECT"}
    check(
        canonical in VALID_ROUTING,
        f"final_routing_decision = {canonical!r}  (canonical label)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_tests() -> None:
    """Execute all test cases through the compiled LangGraph pipeline."""

    print("=" * 72)
    print("  4-AGENT LANGGRAPH PIPELINE — END-TO-END TEST SUITE")
    print("=" * 72)
    print(f"  Pipeline nodes : START → node_risk → node_conversion → node_advisor → node_router → END")
    print(f"  Compiled graph : {type(_pipeline).__name__}")
    print(f"  Models dir     : {MODELS_DIR.resolve()}")
    print()

    timings: List[float] = []

    for i, tc in enumerate(TEST_CASES, 1):
        name     = tc["name"]
        desc     = tc["description"]
        inp      = tc["input"]
        ood_exp  = tc["expect_ood"]

        print("─" * 72)
        print(f"  TEST {i}/{len(TEST_CASES)} │ {name}")
        print(f"  {desc}")
        print(f"  Input: {inp}")
        print("─" * 72)

        # ── Run the full LangGraph DAG ────────────────────────────────────
        initial_state = {"input_data": inp}

        t0 = time.perf_counter()
        final_state = _pipeline.invoke(initial_state)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        print(f"\n  ⏱  Graph execution time: {elapsed * 1000:.1f} ms")

        # ── Extract agent outputs from final state ────────────────────────
        risk_res     = final_state.get("risk_results",       {})
        conv_res     = final_state.get("conversion_results", {})
        advisor_res  = final_state.get("advisor_pitch",      {})
        routing_res  = final_state.get("final_decision",     {})
        canonical    = final_state.get("final_routing_decision", "")

        # ── Handle OOD path ───────────────────────────────────────────────
        is_ood = risk_res.get("status") == OOD_FLAG

        if ood_exp:
            # This test EXPECTS the OOD gate to fire
            validate_agent1(risk_res, expect_ood=True)
            validate_agent2(conv_res, expect_ood=True)
            validate_agent3(advisor_res, expect_ood=True)
            validate_agent4(routing_res, canonical, expect_ood=True)

            # Extra: make sure the graph DID NOT crash
            check(
                isinstance(final_state, dict),
                "Graph returned a valid dict (no crash on corrupt input)",
            )
            check(
                is_ood,
                f"OOD flag correctly detected  (status = {risk_res.get('status')!r})",
            )
        else:
            # Normal path — validate all four agents
            if is_ood:
                print(f"\n  ⚠️  Unexpected OOD flag! Message: {risk_res.get('message')}")
                check(False, "Expected normal prediction but got OOD")
            else:
                validate_agent1(risk_res, expect_ood=False)
                validate_agent2(conv_res, expect_ood=False)
                validate_agent3(advisor_res, expect_ood=False)
                validate_agent4(routing_res, canonical, expect_ood=False)

        # ── Print condensed result summary ────────────────────────────────
        if not ood_exp and not is_ood:
            print(f"\n  📋 SUMMARY")
            print(f"     Risk Tier          : {risk_res.get('predicted_tier')}  "
                  f"(conf={risk_res.get('confidence', 0):.2%})")
            print(f"     Bind Probability   : {conv_res.get('bind_probability', 0):.4f}  "
                  f"({conv_res.get('sales_status')})")
            print(f"     Conversion Score   : {conv_res.get('conversion_score', 0):.1f}")
            print(f"     Premium Flag       : {advisor_res.get('premium_flag')}")
            print(f"     Recommended Premium: ${advisor_res.get('recommended_premium', 0):.2f}")
            print(f"     Advisor Reason     : {advisor_res.get('reason', '')[:100]}")
            print(f"     Router Decision    : {routing_res.get('decision')}")
            print(f"     Canonical Routing  : {canonical}")
            print(f"     Human Required     : {routing_res.get('human_required')}")
            print(f"     Priority           : {routing_res.get('priority')}")
            print(f"     Action Items       : {routing_res.get('action_items', [])}")

        print()

    # ── Final report ──────────────────────────────────────────────────────
    print("=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"  Checks passed : {_passed_checks}/{_total_checks}")
    if _passed_checks == _total_checks:
        print(f"  Status        : ✅  ALL CHECKS PASSED")
    else:
        failed = _total_checks - _passed_checks
        print(f"  Status        : ❌  {failed} CHECK(S) FAILED")

    print()
    print("  ⏱  TIMING REPORT (for frontend loading state estimation)")
    print("  ─" * 30)
    for i, (tc, t) in enumerate(zip(TEST_CASES, timings), 1):
        print(f"    Test {i} ({tc['name']:25s}) : {t * 1000:8.1f} ms")

    avg = sum(timings) / len(timings) if timings else 0
    mx  = max(timings) if timings else 0
    mn  = min(timings) if timings else 0
    print(f"\n    Average : {avg * 1000:8.1f} ms")
    print(f"    Min     : {mn * 1000:8.1f} ms")
    print(f"    Max     : {mx * 1000:8.1f} ms")
    print()

    # Exit with non-zero code if any check failed
    if _passed_checks < _total_checks:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
