"use client";

import type { RiskAssessment, RiskTier, ShapDriver } from "@/lib/api-contract";

// ─────────────────────────────────────────────────────────────────────────────
//  RiskDetails — the detailed SHAP waterfall, gauge, and magnitude badges.
//  This is the accordion body content for the Risk Profiler StorytellingCard.
//  The original RiskPanel is preserved but now lives inside a card wrapper.
// ─────────────────────────────────────────────────────────────────────────────

// ── fixShapLabel ─────────────────────────────────────────────────────────────
function fixShapLabel(
  tier: RiskTier,
  value: number
): { label: string; isGood: boolean } {
  if (tier === "Low") {
    return value > 0
      ? { label: "Reduces Risk", isGood: true }
      : { label: "Increases Risk", isGood: false };
  }
  return value > 0
    ? { label: "Increases Risk", isGood: false }
    : { label: "Reduces Risk", isGood: true };
}

// ── Tier visual config ───────────────────────────────────────────────────────
const TIER_CONFIG = {
  Low: {
    label: "LOW RISK",
    color: "text-emerald-400",
    hex: "#10b981",
  },
  Medium: {
    label: "MEDIUM RISK",
    color: "text-amber-400",
    hex: "#f59e0b",
  },
  High: {
    label: "HIGH RISK",
    color: "text-rose-400",
    hex: "#f43f5e",
  },
} as const;

// ── SVG Semicircle Confidence Gauge ──────────────────────────────────────────
function ConfidenceGauge({ value, color }: { value: number; color: string }) {
  const arcPath = "M 28 90 A 72 72 0 0 1 172 90";
  return (
    <svg viewBox="0 0 200 100" className="w-full max-w-[200px]">
      <path d={arcPath} fill="none" stroke="#1e293b" strokeWidth="14" strokeLinecap="round" />
      <path
        d={arcPath} fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
        pathLength="1" strokeDasharray="1" strokeDashoffset={1 - value}
        style={{ transition: "stroke-dashoffset 0.9s cubic-bezier(.4,0,.2,1)" }}
      />
      <path
        d={arcPath} fill="none" stroke={color} strokeWidth="3" strokeLinecap="round"
        pathLength="1" strokeDasharray="1" strokeDashoffset={1 - value}
        opacity="0.35" style={{ filter: "blur(3px)", transition: "stroke-dashoffset 0.9s cubic-bezier(.4,0,.2,1)" }}
      />
      <text x="100" y="76" textAnchor="middle" fill="white" fontSize="26" fontWeight="bold" fontFamily="ui-monospace,monospace">
        {Math.round(value * 100)}%
      </text>
      <text x="100" y="91" textAnchor="middle" fill="#475569" fontSize="8.5" fontFamily="ui-monospace,monospace" letterSpacing="2.5">
        CONFIDENCE
      </text>
    </svg>
  );
}

// ── SHAP Waterfall ───────────────────────────────────────────────────────────
function ShapWaterfall({ drivers, tier }: { drivers: ShapDriver[]; tier: RiskTier }) {
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.shap_value)), 0.01);

  return (
    <div>
      <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-4">
        SHAP Feature Impact
      </p>
      <div className="flex flex-col gap-3">
        {drivers.map((driver, i) => {
          const { label, isGood } = fixShapLabel(tier, driver.shap_value);
          const barPct = (Math.abs(driver.shap_value) / maxAbs) * 45;
          const barColor = isGood ? "#10b981" : "#f43f5e";
          const isNeg = driver.shap_value < 0;

          return (
            <div key={i} className="flex items-center gap-3">
              <span className="text-xs font-mono text-slate-400 w-[8.5rem] shrink-0 truncate">
                {driver.feature}
              </span>
              <div className="relative flex-1 h-5 flex items-center">
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700/70" />
                <div
                  className="absolute h-3.5 rounded-sm"
                  style={{
                    ...(isNeg
                      ? { right: "50%", width: `${barPct}%` }
                      : { left: "50%", width: `${barPct}%` }),
                    backgroundColor: barColor,
                    opacity: 0.75,
                  }}
                />
              </div>
              <div className="flex flex-col items-end shrink-0 w-[6.5rem]">
                <span className={`text-[10px] font-mono ${isGood ? "text-emerald-400" : "text-rose-400"}`}>
                  {label}
                </span>
                <span className="text-[10px] font-mono text-slate-500 tabular-nums">
                  {driver.shap_value > 0 ? "+" : ""}
                  {driver.shap_value.toFixed(3)}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex items-center justify-center gap-8 mt-4 pt-4 border-t border-slate-800/80">
        <div className="flex items-center gap-1.5">
          <div className="w-3.5 h-2 rounded-sm bg-rose-400/70" />
          <span className="text-[10px] font-mono text-slate-500">Increases Risk</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3.5 h-2 rounded-sm bg-emerald-400/70" />
          <span className="text-[10px] font-mono text-slate-500">Reduces Risk</span>
        </div>
      </div>
    </div>
  );
}

// ── Magnitude badge config ───────────────────────────────────────────────────
const MAG_COLOR = {
  HIGH:   "border-rose-500/40 text-rose-400 bg-rose-500/10",
  MEDIUM: "border-amber-500/40 text-amber-400 bg-amber-500/10",
  LOW:    "border-slate-600 text-slate-400 bg-slate-800/50",
} as const;

// ─────────────────────────────────────────────────────────────────────────────
//  RiskDetails — full detail view (used inside StorytellingCard accordion)
// ─────────────────────────────────────────────────────────────────────────────
export default function RiskDetails({ data }: { data: RiskAssessment }) {
  const t = TIER_CONFIG[data.predicted_tier];

  return (
    <div className="flex flex-col gap-6">
      {/* Gauge + tier label */}
      <div className="flex items-center gap-5">
        <div className="w-36 shrink-0">
          <ConfidenceGauge value={data.confidence_score} color={t.hex} />
        </div>
        <div>
          <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-1.5">
            Predicted Tier
          </p>
          <p className={`text-3xl font-black tracking-tight ${t.color}`}>{t.label}</p>
          <p className="text-xs font-mono text-slate-500 mt-1">
            {Math.round(data.confidence_score * 100)}% calibrated probability
          </p>
        </div>
      </div>

      {/* SHAP waterfall */}
      <ShapWaterfall drivers={data.top_shap_drivers} tier={data.predicted_tier} />

      {/* Magnitude badges */}
      <div>
        <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-3">
          Driver Magnitude
        </p>
        <div className="flex flex-wrap gap-2">
          {data.top_shap_drivers.map((d, i) => (
            <span
              key={i}
              className={`text-[10px] font-mono px-2.5 py-1 rounded-full border ${MAG_COLOR[d.magnitude]}`}
            >
              {d.feature}: {d.magnitude}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: derive verdict + summary from risk data
// ─────────────────────────────────────────────────────────────────────────────
export function getRiskVerdict(data: RiskAssessment) {
  const tier = data.predicted_tier;
  const conf = Math.round(data.confidence_score * 100);
  const topDriver = data.top_shap_drivers[0]?.feature ?? "profile";

  const variant = tier === "Low" ? "positive" : tier === "Medium" ? "caution" : "negative";
  const label = `${tier} Risk`;
  const summary =
    tier === "Low"
      ? `This driver's profile is ${conf}% likely to be low risk. The strongest factor is "${topDriver}".`
      : tier === "Medium"
      ? `Moderate risk detected with ${conf}% confidence. "${topDriver}" is the primary concern.`
      : `High risk profile with ${conf}% confidence. "${topDriver}" is the dominant driver.`;

  return { verdict: { label, variant: variant as "positive" | "caution" | "negative" }, summary };
}
