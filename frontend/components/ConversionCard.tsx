"use client";

import { TrendingUp, Target, Activity } from "lucide-react";
import type { ConversionMetrics, SalesStatus } from "@/lib/mockData";

// ─────────────────────────────────────────────────────────────────────────────
//  Per-status visual config
// ─────────────────────────────────────────────────────────────────────────────
const STATUS = {
  NEAR_MISS: {
    label: "Near Miss",
    desc: "Just below conversion threshold",
    color: "text-amber-400",
    border: "border-amber-500/40",
    bg: "bg-amber-500/10",
    hex: "#f59e0b",
  },
  HOT_LEAD: {
    label: "Hot Lead",
    desc: "High conversion probability",
    color: "text-emerald-400",
    border: "border-emerald-500/40",
    bg: "bg-emerald-500/10",
    hex: "#10b981",
  },
  COLD: {
    label: "Cold Lead",
    desc: "Low conversion probability",
    color: "text-slate-400",
    border: "border-slate-600",
    bg: "bg-slate-800/40",
    hex: "#64748b",
  },
  CONVERTED: {
    label: "Converted",
    desc: "Policy bound successfully",
    color: "text-sky-400",
    border: "border-sky-500/40",
    bg: "bg-sky-500/10",
    hex: "#38bdf8",
  },
} satisfies Record<SalesStatus, object>;

export default function ConversionCard({ data }: { data: ConversionMetrics }) {
  const s = STATUS[data.sales_status] ?? STATUS.COLD;
  const bindPct = Math.round(data.bind_probability * 100);
  const gapPct = Math.round(data.distance_to_conversion * 100);
  // The conversion threshold lives at bindPct + gapPct along the bar
  const thresholdLeft = Math.min(bindPct + gapPct, 97);

  return (
    <div className={`rounded-xl border ${s.border} ${s.bg} p-6 flex flex-col gap-6`}>
      {/* ── Header ──────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-[10px] font-mono tracking-[0.2em] text-slate-500 uppercase">
            Agent 2 · Conversion Engine
          </span>
        </div>
        <span className={`text-xs font-mono font-semibold px-2.5 py-0.5 rounded-full border ${s.border} ${s.color}`}>
          {s.label}
        </span>
      </div>

      {/* ── Stat tiles ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3">
        {/* Bind probability */}
        <div className="rounded-lg border border-slate-700/60 bg-slate-900/60 p-4">
          <div className="flex items-center gap-1.5 mb-2">
            <TrendingUp className={`w-3.5 h-3.5 ${s.color}`} />
            <span className="text-[10px] font-mono text-slate-500">Bind Probability</span>
          </div>
          <p className={`text-3xl font-bold tabular-nums ${s.color}`}>{bindPct}%</p>
        </div>

        {/* Gap to convert */}
        <div className="rounded-lg border border-slate-700/60 bg-slate-900/60 p-4">
          <div className="flex items-center gap-1.5 mb-2">
            <Target className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-[10px] font-mono text-slate-500">Gap to Convert</span>
          </div>
          <p className="text-3xl font-bold tabular-nums text-slate-300">{gapPct}%</p>
          <p className="text-[10px] text-slate-500 font-mono mt-1.5">{s.desc}</p>
        </div>
      </div>

      {/* ── Proximity bar ───────────────────────────────────────────── */}
      <div>
        <div className="flex justify-between mb-2">
          <span className="text-xs font-mono text-slate-500">Conversion Proximity</span>
          <span className={`text-xs font-mono ${s.color}`}>{100 - gapPct}% of threshold</span>
        </div>

        {/* Bar + threshold marker */}
        <div className="relative w-full bg-slate-800 rounded-full h-2.5">
          <div
            className="h-2.5 rounded-full transition-all duration-700"
            style={{
              width: `${bindPct}%`,
              background: `linear-gradient(90deg, #1e293b, ${s.hex})`,
            }}
          />
          {/* Threshold tick */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-0.5 h-4 bg-white/25 rounded-full"
            style={{ left: `${thresholdLeft}%` }}
          />
        </div>

        <div className="flex justify-end mt-1.5">
          <span className="text-[10px] font-mono text-slate-600">▲ conv. threshold</span>
        </div>
      </div>
    </div>
  );
}
