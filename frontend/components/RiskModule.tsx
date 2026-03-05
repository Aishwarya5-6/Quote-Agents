"use client";

import type { RiskAssessment, RiskTier, ShapDriver } from "@/lib/api-contract";

// ─────────────────────────────────────────────────────────────────────────────
//  fixShapLabel
//
//  SHAP values are signed relative to the PREDICTED class, not the High class.
//  For a Low-tier prediction, positive SHAP pushes TOWARD Low (reduces risk),
//  but the raw model label reads "increases risk" — which is wrong for UX.
//
//  Rule:
//    Low  tier:  value > 0 → "Reduces Risk" (✓ green)  |  value < 0 → "Increases Risk" (✗ red)
//    High/Med:   value > 0 → "Increases Risk" (✗ red)  |  value < 0 → "Reduces Risk"  (✓ green)
// ─────────────────────────────────────────────────────────────────────────────
function fixShapLabel(
  tier: RiskTier,
  value: number
): { label: string; isGood: boolean } {
  if (tier === "Low") {
    return value > 0
      ? { label: "Reduces Risk",   isGood: true  }
      : { label: "Increases Risk", isGood: false };
  }
  return value > 0
    ? { label: "Increases Risk", isGood: false }
    : { label: "Reduces Risk",   isGood: true  };
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tier visual config
// ─────────────────────────────────────────────────────────────────────────────
const TIER_CONFIG = {
  Low: {
    label:  "LOW RISK",
    color:  "text-emerald-400",
    border: "border-emerald-500/40",
    bg:     "bg-emerald-500/8",
    hex:    "#10b981",
  },
  Medium: {
    label:  "MEDIUM RISK",
    color:  "text-amber-400",
    border: "border-amber-500/40",
    bg:     "bg-amber-500/8",
    hex:    "#f59e0b",
  },
  High: {
    label:  "HIGH RISK",
    color:  "text-rose-400",
    border: "border-rose-500/40",
    bg:     "bg-rose-500/8",
    hex:    "#f43f5e",
  },
} satisfies Record<RiskTier, object>;

// ─────────────────────────────────────────────────────────────────────────────
//  SVG Semicircle Confidence Gauge
//  Uses pathLength="1" trick — no manual circumference math needed.
// ─────────────────────────────────────────────────────────────────────────────
function ConfidenceGauge({ value, color }: { value: number; color: string }) {
  const arcPath = "M 28 90 A 72 72 0 0 1 172 90";
  return (
    <svg viewBox="0 0 200 100" className="w-full max-w-[200px]">
      {/* Track */}
      <path
        d={arcPath}
        fill="none"
        stroke="#1e293b"
        strokeWidth="14"
        strokeLinecap="round"
      />
      {/* Fill — pathLength normalises to 1 so dashoffset = 1 - value */}
      <path
        d={arcPath}
        fill="none"
        stroke={color}
        strokeWidth="14"
        strokeLinecap="round"
        pathLength="1"
        strokeDasharray="1"
        strokeDashoffset={1 - value}
        style={{ transition: "stroke-dashoffset 0.9s cubic-bezier(.4,0,.2,1)" }}
      />
      {/* Glow */}
      <path
        d={arcPath}
        fill="none"
        stroke={color}
        strokeWidth="3"
        strokeLinecap="round"
        pathLength="1"
        strokeDasharray="1"
        strokeDashoffset={1 - value}
        opacity="0.35"
        style={{ filter: "blur(3px)", transition: "stroke-dashoffset 0.9s cubic-bezier(.4,0,.2,1)" }}
      />
      {/* Value */}
      <text
        x="100"
        y="76"
        textAnchor="middle"
        fill="white"
        fontSize="26"
        fontWeight="bold"
        fontFamily="ui-monospace,monospace"
      >
        {Math.round(value * 100)}%
      </text>
      <text
        x="100"
        y="91"
        textAnchor="middle"
        fill="#475569"
        fontSize="8.5"
        fontFamily="ui-monospace,monospace"
        letterSpacing="2.5"
      >
        CONFIDENCE
      </text>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  SHAP Waterfall (force-plot style)
//  Bars extend left/right from a center baseline.
//  Direction = sign of value. Color = fixShapLabel result.
// ─────────────────────────────────────────────────────────────────────────────
function ShapWaterfall({
  drivers,
  tier,
}: {
  drivers: ShapDriver[];
  tier: RiskTier;
}) {
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.value)));

  return (
    <div>
      <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-4">
        SHAP Feature Impact
      </p>

      <div className="flex flex-col gap-3">
        {drivers.map((driver, i) => {
          const { label, isGood } = fixShapLabel(tier, driver.value);
          // Normalise to 45% max so bars never overflow the container half
          const barPct  = (Math.abs(driver.value) / maxAbs) * 45;
          const barColor = isGood ? "#10b981" : "#f43f5e";
          const isNeg    = driver.value < 0;

          return (
            <div key={i} className="flex items-center gap-3">
              {/* Feature name */}
              <span className="text-xs font-mono text-slate-400 w-[8.5rem] shrink-0 truncate">
                {driver.feature}
              </span>

              {/* Bar zone — center-baseline */}
              <div className="relative flex-1 h-5 flex items-center">
                {/* Center baseline */}
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700/70" />

                {isNeg ? (
                  // ← extends LEFT from center
                  <div
                    className="absolute h-3.5 rounded-sm"
                    style={{
                      right:           "50%",
                      width:           `${barPct}%`,
                      backgroundColor: barColor,
                      opacity:         0.75,
                    }}
                  />
                ) : (
                  // extends RIGHT from center →
                  <div
                    className="absolute h-3.5 rounded-sm"
                    style={{
                      left:            "50%",
                      width:           `${barPct}%`,
                      backgroundColor: barColor,
                      opacity:         0.75,
                    }}
                  />
                )}
              </div>

              {/* Label + value */}
              <div className="flex flex-col items-end shrink-0 w-[6.5rem]">
                <span
                  className={`text-[10px] font-mono ${
                    isGood ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {label}
                </span>
                <span className="text-[10px] font-mono text-slate-500 tabular-nums">
                  {driver.value > 0 ? "+" : ""}
                  {driver.value.toFixed(3)}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
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

// ─────────────────────────────────────────────────────────────────────────────
//  RiskModule — assembled card
// ─────────────────────────────────────────────────────────────────────────────
export default function RiskModule({ data }: { data: RiskAssessment }) {
  const t = TIER_CONFIG[data.predicted_tier];
  const isOodClean = data.ood_flag === "OK";

  return (
    <div className={`rounded-xl border ${t.border} p-6 flex flex-col gap-6`}
         style={{ background: "rgba(15,23,42,0.5)" }}>

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-mono tracking-[0.2em] text-slate-500 uppercase">
          Agent 1 · Risk Profiler
        </span>
        <span
          className={`text-[10px] font-mono px-2.5 py-0.5 rounded-full border ${
            isOodClean
              ? "border-emerald-500/40 text-emerald-400"
              : "border-rose-500/40 text-rose-400"
          }`}
        >
          {isOodClean ? "OOD ✓ Clear" : "⚠ OOD Flag"}
        </span>
      </div>

      {/* ── Gauge + tier label ── */}
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

      {/* ── SHAP waterfall ── */}
      <ShapWaterfall drivers={data.top_shap_drivers} tier={data.predicted_tier} />
    </div>
  );
}
