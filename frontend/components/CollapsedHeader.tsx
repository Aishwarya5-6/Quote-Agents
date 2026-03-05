"use client";

import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, User, Car, MapPin, DollarSign, Shield } from "lucide-react";
import type { QuoteInput } from "@/lib/api-contract";

// ─────────────────────────────────────────────────────────────────────────────
//  CollapsedHeader — slim summary strip replacing the full QuoteForm
//  After submission, the form collapses into a horizontal summary showing
//  the key input values. Click "Edit" to re-expand the full form.
// ─────────────────────────────────────────────────────────────────────────────

interface CollapsedHeaderProps {
  input: QuoteInput;
  onEdit: () => void;
}

const USAGE_LABELS: Record<string, string> = {
  Pleasure: "Pleasure",
  Commute:  "Commute",
  Business: "Business",
};

export default function CollapsedHeader({ input, onEdit }: CollapsedHeaderProps) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
      className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden"
    >
      <div className="px-5 py-3 flex items-center gap-6 flex-wrap">
        {/* Age + Experience */}
        <div className="flex items-center gap-1.5">
          <User className="w-3.5 h-3.5 text-violet-400" />
          <span className="text-xs font-mono text-slate-400">
            Age <span className="text-slate-200 font-semibold">{input.Driver_Age}</span>
          </span>
          <span className="text-slate-700 mx-1">·</span>
          <span className="text-xs font-mono text-slate-400">
            <span className="text-slate-200 font-semibold">{input.Driving_Exp}</span> yr exp
          </span>
        </div>

        {/* Driving record */}
        <div className="flex items-center gap-1.5">
          <Shield className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-xs font-mono text-slate-400">
            <span className="text-slate-200 font-semibold">{input.Prev_Accidents}</span> acc ·{" "}
            <span className="text-slate-200 font-semibold">{input.Prev_Citations}</span> cit
          </span>
        </div>

        {/* Miles + Usage */}
        <div className="flex items-center gap-1.5">
          <MapPin className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-xs font-mono text-slate-400">
            <span className="text-slate-200 font-semibold">
              {(input.Annual_Miles / 1000).toFixed(0)}K
            </span>{" "}
            mi · {USAGE_LABELS[input.Veh_Usage] ?? input.Veh_Usage}
          </span>
        </div>

        {/* Premium */}
        <div className="flex items-center gap-1.5">
          <DollarSign className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-xs font-mono text-slate-400">
            Premium{" "}
            <span className="text-emerald-400 font-semibold">
              ${input.Quoted_Premium ?? "—"}
            </span>
          </span>
        </div>

        {/* Spacer + Edit button */}
        <div className="ml-auto">
          <button
            onClick={onEdit}
            className="text-[10px] font-mono text-violet-400 hover:text-violet-300 
                       border border-violet-500/30 hover:border-violet-500/60
                       rounded-lg px-3 py-1.5 transition-colors
                       flex items-center gap-1"
          >
            Edit Quote
            <ChevronDown className="w-3 h-3" />
          </button>
        </div>
      </div>
    </motion.div>
  );
}
