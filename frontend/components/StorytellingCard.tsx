"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown } from "lucide-react";
import VerdictBadge, { type VerdictVariant } from "./VerdictBadge";
import SkeletonLoader from "./SkeletonLoader";

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
}: StorytellingCardProps) {
  const [detailsOpen, setDetailsOpen] = useState(defaultOpen);
  const accent = AGENT_ACCENT[agentIndex] ?? AGENT_ACCENT[0];

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
