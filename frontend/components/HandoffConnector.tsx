"use client";

import { motion } from "framer-motion";

// ─────────────────────────────────────────────────────────────────────────────
//  HandoffConnector — growing vertical line + pulse dot between agent cards
//  Animates when the previous agent completes, bridging to the next.
// ─────────────────────────────────────────────────────────────────────────────

interface HandoffConnectorProps {
  /** Whether to animate (previous agent completed) */
  active: boolean;
  /** Tailwind color class for the line (e.g. "bg-emerald-500") */
  color?: string;
}

export default function HandoffConnector({
  active,
  color = "bg-violet-500",
}: HandoffConnectorProps) {
  return (
    <div className="flex flex-col items-center py-1">
      {/* Growing vertical line */}
      <div className="relative w-0.5 h-10 bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          className={`absolute inset-x-0 top-0 rounded-full ${color}`}
          initial={{ height: "0%" }}
          animate={{ height: active ? "100%" : "0%" }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>

      {/* Pulse dot at the midpoint */}
      {active && (
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: [0, 1.3, 1], opacity: 1 }}
          transition={{ duration: 0.4, delay: 0.3 }}
          className={`w-2 h-2 rounded-full ${color} shadow-lg`}
          style={{ boxShadow: `0 0 8px currentColor` }}
        />
      )}

      {/* Second segment of the line */}
      <div className="relative w-0.5 h-10 bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          className={`absolute inset-x-0 top-0 rounded-full ${color}`}
          initial={{ height: "0%" }}
          animate={{ height: active ? "100%" : "0%" }}
          transition={{ duration: 0.5, ease: "easeOut", delay: 0.35 }}
        />
      </div>
    </div>
  );
}
