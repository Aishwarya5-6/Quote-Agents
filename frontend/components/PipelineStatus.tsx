"use client";

import { motion } from "framer-motion";
import { Shield, BarChart2, MessageSquare, Gavel, Check } from "lucide-react";

// ─────────────────────────────────────────────────────────────────────────────
//  Step definitions — one entry per agent
// ─────────────────────────────────────────────────────────────────────────────
const STEPS = [
  {
    Icon:       Shield,
    agent:      "Agent 1",
    name:       "Risk Profiler",
    desc:       "XGBoost · SHAP · OOD Gate",
    textColor:  "text-emerald-400",
    borderIdle: "border-slate-700",
    borderOn:   "border-emerald-500/60",
    bgOn:       "bg-emerald-500/15",
    lineColor:  "bg-emerald-500",
  },
  {
    Icon:       BarChart2,
    agent:      "Agent 2",
    name:       "Conversion Engine",
    desc:       "Bind probability scoring",
    textColor:  "text-sky-400",
    borderIdle: "border-slate-700",
    borderOn:   "border-sky-500/60",
    bgOn:       "bg-sky-500/15",
    lineColor:  "bg-sky-500",
  },
  {
    Icon:       MessageSquare,
    agent:      "Agent 3",
    name:       "AI Advisor",
    desc:       "LLM discount strategy",
    textColor:  "text-violet-400",
    borderIdle: "border-slate-700",
    borderOn:   "border-violet-500/60",
    bgOn:       "bg-violet-500/15",
    lineColor:  "bg-violet-500",
  },
  {
    Icon:       Gavel,
    agent:      "Agent 4",
    name:       "Underwriting Router",
    desc:       "Final routing decision",
    textColor:  "text-amber-400",
    borderIdle: "border-slate-700",
    borderOn:   "border-amber-500/60",
    bgOn:       "bg-amber-500/15",
    lineColor:  "bg-amber-500",
  },
] as const;

// ─────────────────────────────────────────────────────────────────────────────
//  Props
//  currentStep: -1 = idle | 0–3 = that agent is running | 4 = all complete
// ─────────────────────────────────────────────────────────────────────────────
interface PipelineStatusProps {
  currentStep: number;
}

type StepState = "idle" | "active" | "complete";

export default function PipelineStatus({ currentStep }: PipelineStatusProps) {
  const getState = (i: number): StepState => {
    if (currentStep < 0)  return "idle";
    if (i < currentStep)  return "complete";
    if (i === currentStep) return "active";
    return "idle";
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-7">
        <span className="text-[10px] font-mono tracking-[0.2em] text-slate-500 uppercase">
          Pipeline · 4 Agents
        </span>
        {currentStep >= 4 && (
          <motion.span
            initial={{ opacity: 0, scale: 0.7 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
            className="text-[10px] font-mono text-emerald-400 border border-emerald-500/30 rounded-full px-2.5 py-0.5"
          >
            ✓ Complete
          </motion.span>
        )}
      </div>

      {/* Steps */}
      <div className="flex flex-col">
        {STEPS.map((step, i) => {
          const state = getState(i);
          const { Icon } = step;
          const isLast = i === STEPS.length - 1;

          return (
            <div key={i} className="flex gap-4">
              {/* ── Left column: circle + connector line ── */}
              <div className="flex flex-col items-center">
                {/* Step circle */}
                <motion.div
                  animate={
                    state === "active"
                      ? { scale: [1, 1.18, 1], opacity: 1 }
                      : state === "complete"
                      ? { scale: 1, opacity: 1 }
                      : { scale: 1, opacity: 0.3 }
                  }
                  transition={
                    state === "active"
                      ? { duration: 0.9, repeat: Infinity, ease: "easeInOut" }
                      : { duration: 0.25 }
                  }
                  className={[
                    "w-10 h-10 rounded-full border flex items-center justify-center shrink-0",
                    state === "idle"     && `${step.borderIdle} bg-slate-800/50`,
                    state === "active"   && `${step.borderOn} ${step.bgOn}`,
                    state === "complete" && "border-emerald-500/50 bg-emerald-500/10",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                >
                  {state === "complete" ? (
                    <motion.div
                      initial={{ scale: 0, rotate: -45 }}
                      animate={{ scale: 1, rotate: 0 }}
                      transition={{ type: "spring", stiffness: 350, damping: 20 }}
                    >
                      <Check className="w-4 h-4 text-emerald-400" />
                    </motion.div>
                  ) : (
                    <Icon
                      className={`w-4 h-4 ${
                        state === "active" ? step.textColor : "text-slate-600"
                      }`}
                    />
                  )}
                </motion.div>

                {/* Connector line segment */}
                {!isLast && (
                  <div className="relative w-0.5 flex-1 min-h-[2.25rem] bg-slate-800 my-1 rounded-full overflow-hidden">
                    <motion.div
                      className={`absolute inset-x-0 top-0 rounded-full ${step.lineColor}`}
                      initial={{ height: "0%" }}
                      animate={{ height: state === "complete" ? "100%" : "0%" }}
                      transition={{ duration: 0.4, ease: "easeOut", delay: 0.1 }}
                    />
                  </div>
                )}
              </div>

              {/* ── Right column: text ── */}
              <motion.div
                className={`pt-1.5 ${isLast ? "" : "pb-6"}`}
                animate={{ opacity: state === "idle" ? 0.3 : 1 }}
                transition={{ duration: 0.25 }}
              >
                <p
                  className={`text-[10px] font-mono tracking-widest uppercase mb-0.5 ${
                    state === "active" ? step.textColor : "text-slate-500"
                  }`}
                >
                  {step.agent}
                </p>
                <p
                  className={`text-sm font-semibold leading-snug ${
                    state === "idle" ? "text-slate-500" : "text-slate-100"
                  }`}
                >
                  {step.name}
                </p>
                <p className="text-xs text-slate-600 font-mono mt-0.5">{step.desc}</p>

                {/* Active pulse label */}
                {state === "active" && (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 1.2, repeat: Infinity }}
                    className={`text-[10px] font-mono mt-1 ${step.textColor}`}
                  >
                    ● running…
                  </motion.p>
                )}
              </motion.div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
