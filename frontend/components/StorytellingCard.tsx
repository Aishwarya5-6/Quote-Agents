"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Terminal, Sparkles } from "lucide-react";
import VerdictBadge, { type VerdictVariant } from "./VerdictBadge";
import SkeletonLoader from "./SkeletonLoader";
import type { RiskAssessment, ConversionMetrics, AdvisorStrategy, FinalRouting, QuoteInput } from "@/lib/api-contract";

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
    quoteInput?: QuoteInput | null;
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
      case 0: { // Agent 1: Risk Profiler
        const risk = agentData.riskAssessment;
        const inp  = agentData.quoteInput;
        if (!risk) return (
          <div className="mt-3 p-3 rounded-lg border border-slate-600/30 bg-slate-800/20">
            <p className="text-xs text-slate-400 italic">Finalizing risk assessment criteria...</p>
          </div>
        );
        const tier   = risk.predicted_tier;
        const conf   = Math.round(risk.confidence_score * 100);
        const d0     = risk.top_shap_drivers[0];
        const d1     = risk.top_shap_drivers[1];
        const incStr = inp ? `${inp.Prev_Accidents} accident${inp.Prev_Accidents !== 1 ? "s" : ""} and ${inp.Prev_Citations} citation${inp.Prev_Citations !== 1 ? "s" : ""}` : "incident history";
        const expStr = inp ? `${inp.Driving_Exp} years of driving experience` : "driving experience";
        const miles  = inp ? `${(inp.Annual_Miles / 1000).toFixed(0)}K miles/year` : null;
        const line1  = `Age ${inp?.Driver_Age ?? "–"} · ${expStr} · ${incStr}${miles ? ` · ${miles}` : ""}.`;
        const line2  = d0 ? `Primary risk factor: ${d0.feature} (${d0.direction})${d1 ? `, followed by ${d1.feature} (${d1.direction})` : ""}.` : "No dominant risk factor identified.";
        const line3  = `Model is ${conf}% confident in the ${tier} Risk classification.`;
        return (
          <div className="mt-3 p-3 rounded-lg border border-emerald-500/20 bg-emerald-500/5">
            <p className="text-xs text-emerald-200 leading-relaxed mb-1">{line1}</p>
            <p className="text-xs text-emerald-300 leading-relaxed mb-1">{line2}</p>
            <p className="text-xs text-emerald-400/80 leading-relaxed font-mono">{line3}</p>
          </div>
        );
      }

      case 1: { // Agent 2: Conversion Engine
        const conv = agentData.conversionMetrics;
        if (!conv) return (
          <div className="mt-3 p-3 rounded-lg border border-slate-600/30 bg-slate-800/20">
            <p className="text-xs text-slate-400 italic">Finalizing conversion analysis...</p>
          </div>
        );
        const bindPct  = Math.round((conv.bind_probability ?? 0) * 100);
        const dist     = conv.distance_to_conversion ?? null;
        const status   = conv.sales_status;
        const distText = dist === null ? null : dist === 0 ? "above the conversion threshold" : `${Math.round(dist * 100)}% from the buying threshold`;
        const statusMessages: Record<string, string> = {
          HIGH_PROPENSITY:       `A ${bindPct}% bind probability is a strong buying signal — pursue confidently with no incentives needed.`,
          NEAR_MISS_FOR_ADVISOR: `At ${bindPct}%, this customer is close to converting. A small premium nudge or follow-up call could close the deal.`,
          LOW_PROB:              `At ${bindPct}%, this lead is unlikely to bind without significant intervention. Consider reassignment or deep discount.`,
          UNCERTAIN:             `${bindPct}% bind probability is borderline. Advisor judgment or additional data is recommended before proceeding.`,
        };
        const line1 = `Calibrated bind probability: ${bindPct}%${distText ? ` — customer is ${distText}` : ""}.`;
        const line2 = status ? (statusMessages[status] ?? `Classified as ${status}.`) : "Conversion signal computed.";
        return (
          <div className="mt-3 p-3 rounded-lg border border-sky-500/20 bg-sky-500/5">
            <p className="text-xs text-sky-200 leading-relaxed mb-1 font-mono">{line1}</p>
            <p className="text-xs text-sky-300 leading-relaxed">{line2}</p>
          </div>
        );
      }

      case 2: { // Agent 3: AI Advisor
        const adv = agentData.advisorStrategy;
        if (!adv) return (
          <div className="mt-3 p-3 rounded-lg border border-slate-600/30 bg-slate-800/20">
            <p className="text-xs text-slate-400 italic">Finalizing premium strategy...</p>
          </div>
        );
        const orig     = adv.original_premium ?? agentData.quoteInput?.Quoted_Premium ?? null;
        const rec      = adv.recommended_premium;
        const discount = adv.suggested_discount_pct;
        const msg      = adv.customer_facing_message;
        const hasAdj   = adv.premium_flag && discount && discount !== "none";
        const savings  = orig && rec ? Math.round(orig - rec) : 0;
        const line1    = orig
          ? (hasAdj
            ? `Quoted premium of $${Math.round(orig)} → recommended $${Math.round(rec ?? orig)} (${discount}, saving $${savings}).`
            : `Quoted premium of $${Math.round(orig)} requires no adjustment.`)
          : (hasAdj ? `A discount of ${discount} is recommended.` : "Premium holds at current rate.");
        const line2 = msg ?? (hasAdj
          ? `Lowering the premium improves conversion chances. The adjustment targets the activation threshold.`
          : `Conversion score is strong enough — a price reduction is unnecessary at this stage.`);
        return (
          <div className="mt-3 p-4 rounded-lg border border-violet-500/30 bg-violet-500/5">
            <div className="flex items-start gap-3">
              <Sparkles className="w-4 h-4 text-violet-400 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs text-violet-200 leading-relaxed mb-1 font-mono">{line1}</p>
                <p className="text-xs text-violet-300 leading-relaxed">{line2}</p>
                <p className="text-[10px] text-violet-500/70 mt-2 font-mono">AI-Generated Pricing Logic</p>
              </div>
            </div>
          </div>
        );
      }

      case 3: { // Agent 4: Underwriting Router
        const routing = agentData.finalRouting;
        if (!routing) return (
          <div className="mt-3 p-3 rounded-lg border border-slate-600/30 bg-slate-800/20">
            <p className="text-xs text-slate-400 italic">Finalizing decision criteria...</p>
          </div>
        );
        const decision   = routing.final_routing_decision ?? "MANUAL_REVIEW";
        const reason     = routing.reason ?? "Routing criteria evaluated.";
        const priority   = routing.priority;
        const humanReq   = routing.human_required;
        const firstAction = routing.action_items?.[0];
        const decisionColor = decision === "AUTO_APPROVE" ? "text-emerald-400" : decision === "REJECT" ? "text-rose-400" : "text-amber-400";
        const priorityColor = priority === "High" ? "text-rose-400" : priority === "Low" ? "text-emerald-400" : "text-amber-400";
        return (
          <div className="mt-3 rounded-lg border border-slate-700/60 bg-slate-950 overflow-hidden">
            <div className="flex items-center gap-2 px-3 py-2 bg-slate-900/50 border-b border-slate-800/80">
              <span className="w-2 h-2 rounded-full bg-rose-500/60" />
              <span className="w-2 h-2 rounded-full bg-amber-500/60" />
              <span className="w-2 h-2 rounded-full bg-emerald-500/60" />
              <span className="ml-2 text-[10px] text-slate-600 tracking-wider font-mono">decision_engine.log</span>
            </div>
            <div className="p-3 space-y-2">
              <div className="flex gap-2 text-xs">
                <span className="text-emerald-500 shrink-0 select-none font-mono">›</span>
                <p className="text-slate-300 leading-relaxed font-mono text-[11px]">{reason}</p>
              </div>
              <div className="flex gap-2 text-xs">
                <span className="text-emerald-500 shrink-0 select-none font-mono">›</span>
                <p className="text-slate-500 font-mono text-[11px]">
                  routing_decision: <span className={decisionColor}>{decision}</span>
                  {priority && <span className="ml-2">· priority: <span className={priorityColor}>{priority}</span></span>}
                  {humanReq && <span className="ml-2 text-amber-400">· human_review: required</span>}
                </p>
              </div>
              {firstAction && (
                <div className="flex gap-2 text-xs">
                  <span className="text-emerald-500 shrink-0 select-none font-mono">›</span>
                  <p className="text-slate-400 font-mono text-[11px]">next_action: <span className="text-slate-300">{firstAction}</span></p>
                </div>
              )}
              <div className="flex gap-2 text-xs items-center">
                <span className="text-emerald-500 shrink-0 select-none font-mono">›</span>
                <span className="inline-block w-1.5 h-3 bg-slate-500 animate-pulse rounded-sm" />
              </div>
            </div>
          </div>
        );
      }

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
