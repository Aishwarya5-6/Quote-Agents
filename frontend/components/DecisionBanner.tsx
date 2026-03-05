"use client";

import { motion } from "framer-motion";
import {
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Terminal,
  AlertTriangle,
  CheckCircle2,
  Clock,
} from "lucide-react";
import type { FinalRouting, RoutingDecision, AgentDecision } from "@/lib/api-contract";

// ─────────────────────────────────────────────────────────────────────────────
//  Visual config for the three canonical routing decisions
// ─────────────────────────────────────────────────────────────────────────────
const ROUTING_CONFIG: Record<
  RoutingDecision,
  {
    label: string;
    color: string;
    bg: string;
    border: string;
    Icon: typeof ShieldCheck;
    dot: string;
  }
> = {
  AUTO_APPROVE: {
    label:  "Auto-Approved",
    color:  "text-emerald-400",
    bg:     "bg-emerald-500/10",
    border: "border-emerald-500/50",
    Icon:   ShieldCheck,
    dot:    "bg-emerald-400",
  },
  MANUAL_REVIEW: {
    label:  "Manual Review Required",
    color:  "text-amber-400",
    bg:     "bg-amber-500/10",
    border: "border-amber-500/50",
    Icon:   ShieldAlert,
    dot:    "bg-amber-400",
  },
  REJECT: {
    label:  "Rejected",
    color:  "text-rose-400",
    bg:     "bg-rose-500/10",
    border: "border-rose-500/50",
    Icon:   ShieldX,
    dot:    "bg-rose-400",
  },
};

const FALLBACK_CONFIG = ROUTING_CONFIG.MANUAL_REVIEW;

// ─────────────────────────────────────────────────────────────────────────────
//  Priority badge config
// ─────────────────────────────────────────────────────────────────────────────
const PRIORITY_CONFIG: Record<string, { color: string; Icon: typeof AlertTriangle }> = {
  High:   { color: "text-rose-400",    Icon: AlertTriangle },
  Medium: { color: "text-amber-400",   Icon: Clock },
  Low:    { color: "text-emerald-400", Icon: CheckCircle2 },
};

export default function DecisionBanner({ data }: { data: FinalRouting }) {
  const routing = data.final_routing_decision ?? "MANUAL_REVIEW";
  const rc = ROUTING_CONFIG[routing] ?? FALLBACK_CONFIG;
  const { Icon } = rc;
  const pc = data.priority ? PRIORITY_CONFIG[data.priority] : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={`rounded-xl border-2 ${rc.border} ${rc.bg} p-6 flex flex-col gap-5`}
    >
      {/* ── High-contrast Final Action banner ──────────────────────── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono tracking-[0.2em] text-slate-500 uppercase">
            Agent 4 · Final Decision
          </span>
        </div>
        {pc && (
          <div className={`flex items-center gap-1.5 text-[10px] font-mono ${pc.color}`}>
            <pc.Icon className="w-3 h-3" />
            Priority: {data.priority}
          </div>
        )}
      </div>

      <div className="flex items-center gap-4">
        <div className={`p-3 rounded-xl ${rc.bg} border ${rc.border}`}>
          <Icon className={`w-8 h-8 ${rc.color}`} />
        </div>
        <div>
          <p className={`text-2xl font-black tracking-tight ${rc.color}`}>
            {rc.label}
          </p>
          <p className="text-xs font-mono text-slate-500 mt-1">
            {data.decision || "No decision"}
            {data.human_required && (
              <span className="ml-2 text-amber-400">· Human review required</span>
            )}
          </p>
        </div>
      </div>

      {/* ── Reason — terminal-style log ───────────────────────────── */}
      {data.reason && (
        <div>
          <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-2 flex items-center gap-1.5">
            <Terminal className="w-3 h-3" />
            Routing Justification
          </p>
          <div className="rounded-lg border border-slate-700/60 bg-slate-950 p-4 font-mono">
            {/* Window chrome */}
            <div className="flex items-center gap-2 mb-3 pb-2.5 border-b border-slate-800/80">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500/60" />
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500/60" />
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/60" />
              <span className="ml-2 text-[10px] text-slate-600 tracking-wider">
                agent4_router.log
              </span>
            </div>

            <div className="flex gap-2 text-xs mb-2">
              <span className="text-emerald-500 shrink-0 select-none">›</span>
              <p className="text-slate-400 leading-relaxed font-typewriter">
                {data.reason}
              </p>
            </div>

            <div className="flex gap-2 text-xs mb-2">
              <span className="text-emerald-500 shrink-0 select-none">›</span>
              <p className="text-slate-600">
                routing_decision:{" "}
                <span className={rc.color}>{routing}</span>
              </p>
            </div>

            {/* Blinking cursor */}
            <div className="flex gap-2 text-xs items-center">
              <span className="text-emerald-500 shrink-0 select-none">›</span>
              <span className="inline-block w-1.5 h-3.5 bg-slate-400 animate-pulse rounded-sm" />
            </div>
          </div>
        </div>
      )}

      {/* ── Action items ──────────────────────────────────────────── */}
      {data.action_items.length > 0 && (
        <div>
          <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-2">
            Action Items
          </p>
          <ul className="flex flex-col gap-1.5">
            {data.action_items.map((item, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-xs text-slate-300"
              >
                <span className={`mt-0.5 w-1.5 h-1.5 rounded-full shrink-0 ${rc.dot}`} />
                {item}
              </li>
            ))}
          </ul>
        </div>
      )}
    </motion.div>
  );
}
