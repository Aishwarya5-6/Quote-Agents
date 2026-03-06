import type { QuoteInput } from "./api-contract";

// ─────────────────────────────────────────────────────────────────────────────
//  Weighted random helper
//  Returns the index of the chosen bucket; weights need not sum to 1.
// ─────────────────────────────────────────────────────────────────────────────
function weightedPick(weights: number[]): number {
  const total = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

// ─────────────────────────────────────────────────────────────────────────────
//  generateRandomQuote
//
//  Distributions are calibrated against the Agent 1 training-data statistics
//  recorded in backend/models/manifest.json:
//    mean Prev_Accidents ≈ 0.11   → most drivers have 0
//    mean Prev_Citations ≈ 0.11   → most drivers have 0
//    mean Driver_Age     ≈ 41.5   → working-age skew
//    mean Annual_Miles   ≈ 23,269 → normal commuter range
//
//  Roughly 6 % High-risk, 18 % Medium-risk, 76 % Low-risk outcomes expected
//  across a large batch, matching the original dataset's label distribution.
// ─────────────────────────────────────────────────────────────────────────────
const VEH_USAGES: Array<"Business" | "Commute" | "Pleasure"> = [
  "Business",
  "Commute",
  "Pleasure",
];

export function generateRandomQuote(): QuoteInput {
  // ── Driver age: skew toward working-age adults ────────────────────────────
  const ageBands  = [16, 21, 31, 41, 51, 61, 71] as const;
  const ageSpans  = [ 4,  9, 10, 10, 10, 10,  9] as const;
  const ageWts    = [ 3,  8, 15, 18, 15,  8,  3];
  const aBand     = weightedPick(ageWts);
  const driverAge = ageBands[aBand] + Math.floor(Math.random() * ageSpans[aBand]);

  // ── Driving experience: 0 → (age − 16) ───────────────────────────────────
  const maxExp    = driverAge - 16;
  const drivingExp = Math.floor(Math.random() * (maxExp + 1));

  // ── Accidents & citations: heavy 0-weight (mean ≈ 0.11) ──────────────────
  const prevAccidents = weightedPick([0.70, 0.18, 0.08, 0.03, 0.01]);
  const prevCitations = weightedPick([0.70, 0.18, 0.08, 0.04]);

  // ── Annual miles: mode around 18K–25K ────────────────────────────────────
  const milesBases  = [5_000,  8_000, 12_000, 18_000, 25_000, 35_000, 48_000];
  const milesRanges = [3_000,  3_000,  5_000,  6_000,  8_000, 10_000, 14_000];
  const milesWts    = [0.06,   0.10,   0.18,   0.28,   0.20,   0.12,   0.06];
  const mBand       = weightedPick(milesWts);
  const annualMiles = milesBases[mBand] + Math.floor(Math.random() * milesRanges[mBand]);

  // ── Vehicle usage: mostly Commute ─────────────────────────────────────────
  const vehUsage = VEH_USAGES[weightedPick([0.15, 0.55, 0.30])];

  // ── Premium: correlated with accidents and youth ─────────────────────────
  const basePremium   = 350 + prevAccidents * 180 + (driverAge < 25 ? 200 : 0);
  const quotedPremium = basePremium + Math.floor(Math.random() * 700);

  // ── Optional CRM fields ───────────────────────────────────────────────────
  const salRange        = weightedPick([0.12, 0.22, 0.30, 0.24, 0.12]); // 0-4
  const coverage        = weightedPick([0.25, 0.50, 0.25]);             // 0-2
  const vehiclCostRange = weightedPick([0.15, 0.25, 0.30, 0.20, 0.10]); // 0-4
  const reQuote         = Math.random() < 0.15 ? 1 : 0;

  return {
    Driver_Age:        driverAge,
    Driving_Exp:       drivingExp,
    Prev_Accidents:    prevAccidents,
    Prev_Citations:    prevCitations,
    Annual_Miles:      annualMiles,
    Veh_Usage:         vehUsage,
    Quoted_Premium:    quotedPremium,
    Sal_Range:         salRange,
    Coverage:          coverage,
    Vehicl_Cost_Range: vehiclCostRange,
    Re_Quote:          reQuote,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
//  quoteLabel  — compact one-liner for dashboard table rows
// ─────────────────────────────────────────────────────────────────────────────
export function quoteLabel(q: QuoteInput): string {
  const miles =
    q.Annual_Miles >= 1_000
      ? `${(q.Annual_Miles / 1_000).toFixed(0)}K mi`
      : `${q.Annual_Miles} mi`;
  return `Age ${q.Driver_Age} · ${q.Driving_Exp}yr · ${q.Prev_Accidents}acc/${q.Prev_Citations}cit · ${miles}`;
}
