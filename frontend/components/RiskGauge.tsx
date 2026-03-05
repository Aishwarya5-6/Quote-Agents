"use client";

import { ShieldCheck, ShieldAlert, Shield, Zap } from "lucide-react";
import type { RiskAssessment, RiskTier } from "@/lib/mockData";

// ─────────────────────────────────────────────────────────────────────────────
//  fixShapLabel
//
//  SHAP values are computed against the *predicted class*.
//  For a Low-tier prediction, a positive SHAP pushes *toward Low* (good),
//  but the raw model label reads "increases risk" — which is confusing.
//  This helper translates to user-friendly language based on tier + sign.
// ─────────────────────────────────────────────────────────────────────────────
function fixShapLabel(
  tier: RiskTier,
  value: number
): { label: string; positive: boolean } {
  if (tier === "Low") {
    return value > 0
      ? { label: "Reduces Risk", positive: true }
      : { label: "Increases Risk", positive: false };
  }
  // High / Medium: positive SHAP pushes toward that (bad) class
  return value > 0
    ? { label: "Increases Risk", positive: false }
    : { label: "Reduces Risk", positive: true };
}

// ─────────────────────────────────────────────────────────────────────────────
//  Per-tier visual config
// ─────────────────────────────────────────────────────────────────────────────
const TIER = {
  Low: {
    label: "LOW RISK",
    color: "text-emerald-400",
    border: "border-emerald-500/40",
    bg: "bg-emerald-500/10",
    hex: "#10b981",
    Icon: ShieldCheck,
  },
  Medium: {
    label: "MEDIUM RISK",
    color: "text-amber-400",
    border: "border-amber-500/40",
    bg: "bg-amber-500/10",
    hex: "#f59e0b",
    Icon: Shield,
  },
  High: {
    label: "HIGH RISK",
    color: "text-rose-400",
    border: "border-rose-500/40",
    bg: "bg-rose-500/10",
    hex: "#f43f5e",
    Icon: ShieldAlert,
  },
} as const;

export default function RiskGauge({ data }: { data: RiskAssessment }) {
  const t = TIER[data.predicted_tier];
  const { Icon } = t;
  const confidencePct = Math.round(data.confidence_score * 100);

  return (
    <div className={`rounded-xl border ${t.border} ${t.bg} p-6 flex flex-col gap-6`}>
      {/* ── Header ──────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-[10px] font-mono tracking-[0.2em] text-slate-500 uppercase">
            Agent 1 · Risk Profiler
          </span>
        </div>
        <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${t.border} ${t.color}`}>
          v3
        </span>
      </div>

      {/* ── Tier badge ──────────────────────────────────────────────── */}
      <div className="flex items-center gap-4">
        <div className={`p-3 rounded-lg ${t.bg} border ${t.border}`}>
          <Icon className={`w-7 h-7 ${t.color}`} />
        </div>
        <div>
          <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-0.5">
            Predicted Tier
          </p>
          <p className={`text-2xl font-bold tracking-tight ${t.color}`}>{t.label}</p>
        </div>
      </div>

      {/* ── Confidence bar ──────────────────────────────────────────── */}
      <div>
        <div className="flex justify-between mb-2">
          <span className="text-xs font-mono text-slate-500">Model Confidence</span>
          <span className={`text-xs font-mono font-bold ${t.color}`}>{confidencePct}%</span>
        </div>
        <div className="w-full bg-slate-800 rounded-full h-2">
          <div
            className="h-2 rounded-full transition-all duration-700"
            style={{ width: `${confidencePct}%`, backgroundColor: t.hex }}
          />
        </div>
      </div>

      {/* ── SHAP Drivers ────────────────────────────────────────────── */}
      <div>
        <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-3">
          Top SHAP Drivers
        </p>
        <div className="flex flex-col gap-3">
          {data.top_shap_drivers.map((driver, i) => {
            const { label, positive } = fixShapLabel(data.predicted_tier, driver.value);
            // Scale |SHAP| → bar width (0.45 → 90%, cap at 100%)
            const barPct = Math.min(Math.abs(driver.value) * 200, 100);
            const barColor = positive ? "#10b981" : "#f43f5e";
            const labelColor = positive ? "text-emerald-400" : "text-rose-400";

            return (
              <div key={i} className="flex flex-col gap-1.5">
                <div className="flex justify-between items-baseline">
                  <span className="text-xs font-mono text-slate-300">{driver.feature}</span>
                  <span className={`text-xs font-mono ${labelColor}`}>{label}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 bg-slate-800 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full transition-all duration-700"
                      style={{ width: `${barPct}%`, backgroundColor: barColor }}
                    />
                  </div>
                  <span className="text-[10px] font-mono text-slate-500 w-14 text-right tabular-nums">
                    {driver.value > 0 ? "+" : ""}
                    {driver.value.toFixed(3)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
