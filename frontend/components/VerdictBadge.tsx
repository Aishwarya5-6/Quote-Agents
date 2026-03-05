"use client";

import { motion } from "framer-motion";

// ─────────────────────────────────────────────────────────────────────────────
//  VerdictBadge — plain-English, color-coded verdict pill
//  Shows the agent's bottom-line result at a glance.
// ─────────────────────────────────────────────────────────────────────────────

export type VerdictVariant =
  | "positive"   // emerald — Low Risk, Auto-Approve, High Propensity
  | "caution"    // amber  — Medium Risk, Near-Miss, Manual Review
  | "negative"   // rose   — High Risk, Reject, Low Prob
  | "info"       // violet — No Adjustment, Uncertain
  | "neutral";   // slate  — fallback

interface VerdictBadgeProps {
  label: string;
  variant: VerdictVariant;
  /** Optional icon shown before the label */
  icon?: React.ReactNode;
}

const VARIANT_CLASSES: Record<VerdictVariant, string> = {
  positive: "border-emerald-500/50 bg-emerald-500/10 text-emerald-400",
  caution:  "border-amber-500/50 bg-amber-500/10 text-amber-400",
  negative: "border-rose-500/50 bg-rose-500/10 text-rose-400",
  info:     "border-violet-500/50 bg-violet-500/10 text-violet-400",
  neutral:  "border-slate-600 bg-slate-800/50 text-slate-400",
};

export default function VerdictBadge({ label, variant, icon }: VerdictBadgeProps) {
  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
      className={`
        inline-flex items-center gap-1.5
        text-xs font-mono font-semibold
        px-3 py-1 rounded-full border
        ${VARIANT_CLASSES[variant]}
      `}
    >
      {icon}
      {label}
    </motion.span>
  );
}
