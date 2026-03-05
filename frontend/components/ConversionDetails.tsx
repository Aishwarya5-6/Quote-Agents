"use client";

import { TrendingUp, Target, Activity } from "lucide-react";
import type { ConversionMetrics, SalesStatus } from "@/lib/api-contract";
import { CONVERSION_THRESHOLD } from "@/lib/api-contract";

// ─────────────────────────────────────────────────────────────────────────────
//  ConversionDetails — detailed conversion view for StorytellingCard accordion
// ─────────────────────────────────────────────────────────────────────────────

const STATUS_CONFIG: Record<
  SalesStatus,
  { label: string; desc: string; color: string; border: string; bg: string; hex: string }
> = {
  HIGH_PROPENSITY: {
    label:  "High Propensity",
    desc:   "Strong buying signal",
    color:  "text-emerald-400",
    border: "border-emerald-500/40",
    bg:     "bg-emerald-500/10",
    hex:    "#10b981",
  },
  NEAR_MISS_FOR_ADVISOR: {
    label:  "Near Miss",
    desc:   "Just below conversion threshold",
    color:  "text-amber-400",
    border: "border-amber-500/40",
    bg:     "bg-amber-500/10",
    hex:    "#f59e0b",
  },
  LOW_PROB: {
    label:  "Low Probability",
    desc:   "Unlikely to convert",
    color:  "text-slate-400",
    border: "border-slate-600",
    bg:     "bg-slate-800/40",
    hex:    "#64748b",
  },
  UNCERTAIN: {
    label:  "Uncertain",
    desc:   "Borderline conversion signal",
    color:  "text-sky-400",
    border: "border-sky-500/40",
    bg:     "bg-sky-500/10",
    hex:    "#38bdf8",
  },
};

const FALLBACK = STATUS_CONFIG.LOW_PROB;

export default function ConversionDetails({ data }: { data: ConversionMetrics }) {
  const s = (data.sales_status && STATUS_CONFIG[data.sales_status]) || FALLBACK;
  const bindProb = data.bind_probability ?? 0;
  const bindPct  = Math.round(bindProb * 100);
  const thresholdPct = Math.round(CONVERSION_THRESHOLD * 100);
  const gapPct   = Math.round((data.distance_to_conversion ?? 0) * 100);

  return (
    <div className="flex flex-col gap-5">
      {/* Stat tiles */}
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-slate-700/60 bg-slate-900/60 p-4">
          <div className="flex items-center gap-1.5 mb-2">
            <TrendingUp className={`w-3.5 h-3.5 ${s.color}`} />
            <span className="text-[10px] font-mono text-slate-500">Bind Probability</span>
          </div>
          <p className={`text-3xl font-bold tabular-nums ${s.color}`}>{bindPct}%</p>
        </div>

        <div className="rounded-lg border border-slate-700/60 bg-slate-900/60 p-4">
          <div className="flex items-center gap-1.5 mb-2">
            <Target className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-[10px] font-mono text-slate-500">Gap to Convert</span>
          </div>
          <p className="text-3xl font-bold tabular-nums text-slate-300">{gapPct}%</p>
          <p className="text-[10px] text-slate-500 font-mono mt-1.5">{s.desc}</p>
        </div>
      </div>

      {/* Horizontal progress bar with threshold marker */}
      <div>
        <div className="flex justify-between mb-2">
          <span className="text-xs font-mono text-slate-500">Conversion Proximity</span>
          <span className={`text-xs font-mono ${s.color}`}>{bindPct}%</span>
        </div>

        <div className="relative w-full bg-slate-800 rounded-full h-3">
          <div
            className="h-3 rounded-full transition-all duration-700 relative z-10"
            style={{
              width: `${Math.min(bindPct, 100)}%`,
              background: `linear-gradient(90deg, #1e293b, ${s.hex})`,
            }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 z-20 flex flex-col items-center"
            style={{ left: `${thresholdPct}%` }}
          >
            <div className="w-0.5 h-5 bg-white/40 rounded-full" />
          </div>
        </div>

        <div className="flex justify-between mt-1.5">
          <span className="text-[10px] font-mono text-slate-600">0%</span>
          <span className="text-[10px] font-mono text-white/40">
            ▲ {CONVERSION_THRESHOLD} threshold
          </span>
          <span className="text-[10px] font-mono text-slate-600">100%</span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: derive verdict + summary from conversion data
// ─────────────────────────────────────────────────────────────────────────────
export function getConversionVerdict(data: ConversionMetrics) {
  const bindPct = Math.round((data.bind_probability ?? 0) * 100);
  const status = data.sales_status ?? "LOW_PROB";

  const variantMap: Record<SalesStatus, "positive" | "caution" | "negative" | "info"> = {
    HIGH_PROPENSITY:       "positive",
    NEAR_MISS_FOR_ADVISOR: "caution",
    LOW_PROB:              "negative",
    UNCERTAIN:             "info",
  };

  const labelMap: Record<SalesStatus, string> = {
    HIGH_PROPENSITY:       "High Propensity",
    NEAR_MISS_FOR_ADVISOR: "Near Miss",
    LOW_PROB:              "Low Probability",
    UNCERTAIN:             "Uncertain",
  };

  const variant = variantMap[status] ?? "neutral";
  const label = labelMap[status] ?? status;

  const summary =
    status === "HIGH_PROPENSITY"
      ? `Strong buying signal — ${bindPct}% likelihood this customer will bind. Pursue confidently.`
      : status === "NEAR_MISS_FOR_ADVISOR"
      ? `This customer is close to converting at ${bindPct}%. A small nudge (discount or follow-up) could close the deal.`
      : status === "UNCERTAIN"
      ? `Borderline conversion signal at ${bindPct}%. More data or advisor judgment is recommended.`
      : `Low conversion probability at ${bindPct}%. This lead is unlikely to convert without significant intervention.`;

  return { verdict: { label, variant }, summary };
}
