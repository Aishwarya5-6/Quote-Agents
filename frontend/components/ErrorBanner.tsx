"use client";

import { motion } from "framer-motion";
import { AlertTriangle, X } from "lucide-react";

// ─────────────────────────────────────────────────────────────────────────────
//  Error banner for OOD anomaly (HTTP 422) and general errors
// ─────────────────────────────────────────────────────────────────────────────
interface ErrorBannerProps {
  title: string;
  message: string;
  details?: Record<string, unknown>;
  onDismiss?: () => void;
  onRetry?: () => void;
  isOod?: boolean;
}

export default function ErrorBanner({ title, message, details, onDismiss, onRetry, isOod = false }: ErrorBannerProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="rounded-xl border-2 border-rose-500/50 bg-rose-500/10 p-6"
    >
      <div className="flex items-start gap-4">
        <div className="p-2 rounded-lg bg-rose-500/20 border border-rose-500/30 shrink-0">
          <AlertTriangle className="w-6 h-6 text-rose-400" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-bold text-rose-400">{title}</h3>
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="p-1 rounded-lg hover:bg-slate-800 transition-colors"
              >
                <X className="w-4 h-4 text-slate-500" />
              </button>
            )}
          </div>

          <p className="text-sm text-slate-300 leading-relaxed mb-3">{message}</p>

          {onRetry && (
            <button
              onClick={onRetry}
              className="mb-3 flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-rose-300 text-sm font-mono transition-colors"
            >
              ↺ Retry
            </button>
          )}

          {details && Object.keys(details).length > 0 && (
            <div className="rounded-lg border border-slate-700/60 bg-slate-950 p-3 font-mono">
              <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">
                Input Data That Triggered the Anomaly
              </p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(details).map(([k, v]) => (
                  <span
                    key={k}
                    className="text-[10px] font-mono px-2 py-1 rounded border border-slate-700 bg-slate-900 text-slate-400"
                  >
                    {k}: <span className="text-rose-400">{String(v)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {isOod && (
            <p className="text-xs font-mono text-slate-500 mt-3">
              This quote has been flagged for human underwriter review.
              The AI pipeline was halted to prevent a confident-but-wrong prediction.
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
}
