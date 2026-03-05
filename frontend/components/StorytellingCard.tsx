"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Terminal, Sparkles } from "lucide-react";
import VerdictBadge, { type VerdictVariant } from "./VerdictBadge";
import SkeletonLoader from "./SkeletonLoader";
import type { RiskAssessment, ConversionMetrics, AdvisorStrategy, FinalRouting } from "@/lib/api-contract";

// ─────────────────────────────────────────────────────────────────────────────
//  StorytellingCard — wraps each agent's output in a progressive-disclosure
//  card with:
//    1. Verdict badge (one-liner summary)
//    2. Plain-English summary sentence
//    3. "View Details" accordion (closed by default)
//    4. Skeleton loader while the agent is running
//    5. AnimatePresence enter/exit transitions
// ─────────────────────────────────────────────────────────────────────────────

interface StorytellingCardProps {
  /** Agent index 0-3 for accent colors */
  agentIndex: number;
  /** Header label, e.g. "Agent 1 · Risk Profiler" */
  agentLabel: string;
  /** Header icon */
  icon: React.ReactNode;
  /** Current state of this card */
  state: "hidden" | "loading" | "revealed";
  /** One-liner verdict badge */
  verdict?: { label: string; variant: VerdictVariant; icon?: React.ReactNode };
  /** Plain English summary sentence shown below verdict */
  summary?: string;
  /** Children = the detailed panel content (accordion body) */
  children?: React.ReactNode;
  /** Start with details open? (default: false) */
  defaultOpen?: boolean;
  /** Agent-specific data for custom reasoning display */
  agentData?: {
    riskAssessment?: RiskAssessment | null;
    conversionMetrics?: ConversionMetrics | null;
    advisorStrategy?: AdvisorStrategy | null;
    finalRouting?: FinalRouting | null;
  };
}

const AGENT_ACCENT: Record<number, { border: string; header: string; glow: string }> = {
  0: {
    border: "border-emerald-500/30",
    header: "text-emerald-400",
    glow:   "shadow-emerald-500/5",
  },
  1: {
    border: "border-sky-500/30",
    header: "text-sky-400",
    glow:   "shadow-sky-500/5",
  },
  2: {
    border: "border-violet-500/30",
    header: "text-violet-400",
    glow:   "shadow-violet-500/5",
  },
  3: {
    border: "border-amber-500/30",
    header: "text-amber-400",
    glow:   "shadow-amber-500/5",
  },
};

export default function StorytellingCard({
  agentIndex,
  agentLabel,
  icon,
  state,
  verdict,
  summary,
  children,
  defaultOpen = false,
  agentData,
}: StorytellingCardProps) {
  const [detailsOpen, setDetailsOpen] = useState(defaultOpen);
  const accent = AGENT_ACCENT[agentIndex] ?? AGENT_ACCENT[0];

  // Generate agent-specific data-driven reasoning
  const getDataDrivenReason = (): React.ReactNode => {
    if (!agentData) return null;

    switch (agentIndex) {
      case 0: // Agent 1: Risk Profiler
        if (agentData.riskAssessment?.top_shap_drivers?.[0]) {
          const topDriver = agentData.riskAssessment.top_shap_drivers[0];
          return (
            <div className="mt-3 p-3 rounded-lg border border-emerald-500/20 bg-emerald-500/5">
              <p className="text-xs text-emerald-300 leading-relaxed">
                Risk level determined by <span className="font-mono text-emerald-200">{topDriver.feature}</span>{" "}
                <span className="text-emerald-400">({topDriver.direction})</span>
              </p>
            </div>
          );
        }
        return (
          <div className="mt-3 p-3 rounded-lg border border-slate-600/30 bg-slate-800/20">
            <p className="text-xs text-slate-400 leading-relaxed italic">Finalizing risk assessment criteria...</p>
          </div>
        );

      case 1: // Agent 2: Conversion Engine
        if (agentData.conversionMetrics?.distance_to_conversion !== null && agentData.conversionMetrics?.distance_to_conversion !== undefined) {
          const distance = agentData.conversionMetrics.distance_to_conversion;
          const distanceText = distance === 0 
            ? "Above threshold" 
            : `${Math.round(distance * 100)}% from the buying threshold`;
          return (
            <div className="mt-3 p-3 rounded-lg border border-sky-500/20 bg-sky-500/5">
              <p className="text-xs text-sky-300 leading-relaxed">
                Customer is <span className="font-mono text-sky-200">{distanceText}</span>
              </p>
            </div>
          );
        }
        return (
          <div className="mt-3 p-3 rounded-lg border border-slate-600/30 bg-slate-800/20">
            <p className="text-xs text-slate-400 leading-relaxed italic">Finalizing conversion analysis...</p>
          </div>
        );

      case 2: // Agent 3: AI Advisor
        if (agentData.advisorStrategy?.customer_facing_message) {
          return (
            <div className="mt-3 p-4 rounded-lg border border-violet-500/30 bg-violet-500/5 relative">
              <div className="flex items-start gap-3">
                <Sparkles className="w-4 h-4 text-violet-400 mt-0.5 shrink-0" />
                <div className="min-w-0">
                  <p className="text-xs text-violet-300 leading-relaxed">
                    {agentData.advisorStrategy.customer_facing_message}
                  </p>
                  <p className="text-[10px] text-violet-500/80 mt-2 font-mono">AI-Generated Pricing Logic</p>
                </div>
              </div>
            </div>
          );
        }
        return (
          <div className="mt-3 p-3 rounded-lg border border-slate-600/30 bg-slate-800/20">
            <p className="text-xs text-slate-400 leading-relaxed italic">Finalizing premium strategy...</p>
          </div>
        );

      case 3: // Agent 4: Underwriting Router
        if (agentData.finalRouting?.reason) {
          const routing = agentData.finalRouting.final_routing_decision ?? "MANUAL_REVIEW";
          const routingColor = routing === "AUTO_APPROVE" ? "text-emerald-400" : routing === "REJECT" ? "text-rose-400" : "text-amber-400";
          return (
            <div className="mt-3 rounded-lg border border-slate-700/60 bg-slate-950 overflow-hidden">
              <div className="flex items-center gap-2 px-3 py-2 bg-slate-900/50 border-b border-slate-800/80">
                <span className="w-2 h-2 rounded-full bg-rose-500/60" />
                <span className="w-2 h-2 rounded-full bg-amber-500/60" />
                <span className="w-2 h-2 rounded-full bg-emerald-500/60" />
                <span className="ml-2 text-[10px] text-slate-600 tracking-wider font-mono">decision_engine.log</span>
              </div>
              <div className="p-3">
                <div className="flex gap-2 text-xs mb-2">
                  <span className="text-emerald-500 shrink-0 select-none font-mono">›</span>
                  <p className="text-slate-400 leading-relaxed font-mono text-[11px]">
                    {agentData.finalRouting.reason}
                  </p>
                </div>
                <div className="flex gap-2 text-xs">
                  <span className="text-emerald-500 shrink-0 select-none font-mono">›</span>
                  <p className="text-slate-600 font-mono text-[11px]">
                    routing_decision: <span className={routingColor}>{routing}</span>
                  </p>
                </div>
              </div>
            </div>
          );
        }
        return (
          <div className="mt-3 p-3 rounded-lg border border-slate-600/30 bg-slate-800/20">
            <p className="text-xs text-slate-400 leading-relaxed italic">Finalizing decision criteria...</p>
          </div>
        );

      default:
        return null;
    }
  };

  if (state === "hidden") return null;

  return (
    <AnimatePresence mode="wait">
      {state === "loading" && (
        <motion.div key="skeleton">
          <SkeletonLoader agentIndex={agentIndex} variant="medium" />
        </motion.div>
      )}

      {state === "revealed" && (
        <motion.div
          key="revealed"
          initial={{ opacity: 0, y: 20, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
          className={`rounded-xl border ${accent.border} bg-slate-900/60 shadow-lg ${accent.glow} overflow-hidden`}
        >
          {/* ── Card Header ─────────────────────────────────────────── */}
          <div className="px-6 pt-5 pb-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                {icon}
                <span
                  className={`text-[10px] font-mono tracking-[0.2em] uppercase ${accent.header}`}
                >
                  {agentLabel}
                </span>
              </div>
              {verdict && (
                <VerdictBadge
                  label={verdict.label}
                  variant={verdict.variant}
                  icon={verdict.icon}
                />
              )}
            </div>

            {/* Plain-English summary */}
            {summary && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.15 }}
                className="text-sm text-slate-300 leading-relaxed"
              >
                {summary}
              </motion.p>
            )}

            {/* Data-driven reasoning (only appears after reveal) */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              {getDataDrivenReason()}
            </motion.div>
          </div>

          {/* ── Accordion Toggle ────────────────────────────────────── */}
          {children && (
            <>
              <button
                onClick={() => setDetailsOpen((prev) => !prev)}
                className="w-full flex items-center justify-between px-6 py-3 
                           border-t border-slate-800/80 
                           hover:bg-slate-800/30 transition-colors group"
              >
                <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest group-hover:text-slate-400 transition-colors">
                  {detailsOpen ? "Hide Details" : "View Details"}
                </span>
                <motion.div
                  animate={{ rotate: detailsOpen ? 180 : 0 }}
                  transition={{ duration: 0.25 }}
                >
                  <ChevronDown className="w-4 h-4 text-slate-500 group-hover:text-slate-400" />
                </motion.div>
              </button>

              {/* ── Accordion Body ───────────────────────────────────── */}
              <AnimatePresence initial={false}>
                {detailsOpen && (
                  <motion.div
                    key="details"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
                    className="overflow-hidden"
                  >
                    <div className="px-6 pb-6 pt-2">{children}</div>
                  </motion.div>
                )}
              </AnimatePresence>
            </>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
