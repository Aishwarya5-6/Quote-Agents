"use client";

import { motion } from "framer-motion";

// ─────────────────────────────────────────────────────────────────────────────
//  SkeletonLoader — pulsing placeholder while an agent is running
//  Three variants for different card heights (compact, medium, tall)
// ─────────────────────────────────────────────────────────────────────────────

interface SkeletonLoaderProps {
  /** Which agent index (0-3) — controls accent color */
  agentIndex: number;
  /** Variant: compact = 2 lines, medium = 4 lines, tall = 6 lines */
  variant?: "compact" | "medium" | "tall";
}

const AGENT_GLOW: Record<number, string> = {
  0: "from-emerald-500/5 to-transparent",
  1: "from-sky-500/5 to-transparent",
  2: "from-violet-500/5 to-transparent",
  3: "from-amber-500/5 to-transparent",
};

const AGENT_BORDER: Record<number, string> = {
  0: "border-emerald-500/20",
  1: "border-sky-500/20",
  2: "border-violet-500/20",
  3: "border-amber-500/20",
};

const AGENT_SHIMMER: Record<number, string> = {
  0: "bg-emerald-500/10",
  1: "bg-sky-500/10",
  2: "bg-violet-500/10",
  3: "bg-amber-500/10",
};

const LINE_COUNTS = { compact: 2, medium: 4, tall: 6 };

export default function SkeletonLoader({
  agentIndex,
  variant = "medium",
}: SkeletonLoaderProps) {
  const lines = LINE_COUNTS[variant];
  const glow = AGENT_GLOW[agentIndex] ?? AGENT_GLOW[0];
  const border = AGENT_BORDER[agentIndex] ?? AGENT_BORDER[0];
  const shimmer = AGENT_SHIMMER[agentIndex] ?? AGENT_SHIMMER[0];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.35 }}
      className={`rounded-xl border ${border} bg-gradient-to-b ${glow} p-6`}
    >
      {/* Header skeleton */}
      <div className="flex items-center justify-between mb-5">
        <div className={`h-3 w-36 rounded ${shimmer} animate-pulse`} />
        <div className={`h-5 w-20 rounded-full ${shimmer} animate-pulse`} />
      </div>

      {/* Content skeleton lines */}
      <div className="flex flex-col gap-3">
        {Array.from({ length: lines }).map((_, i) => (
          <motion.div
            key={i}
            className={`rounded ${shimmer} animate-pulse`}
            style={{
              height: i === 0 ? "2rem" : "0.75rem",
              width: i === 0 ? "60%" : `${85 - i * 8}%`,
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: i * 0.08 }}
          />
        ))}
      </div>

      {/* Bottom row skeleton */}
      <div className="flex gap-2 mt-5">
        {[1, 2, 3].map((_, i) => (
          <div
            key={i}
            className={`h-6 rounded-full ${shimmer} animate-pulse`}
            style={{ width: `${28 - i * 4}%` }}
          />
        ))}
      </div>
    </motion.div>
  );
}
