"use client";

import { useState, useCallback, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Cpu, Zap, Clock, Shield, BarChart2, MessageSquare, Gavel } from "lucide-react";

import {
  API_BASE,
  type PipelineResponse,
  type OodErrorResponse,
  type QuoteInput,
} from "@/lib/api-contract";

import QuoteForm       from "@/components/QuoteForm";
import ErrorBanner     from "@/components/ErrorBanner";

// Storytelling components
import CollapsedHeader from "@/components/CollapsedHeader";
import StorytellingCard from "@/components/StorytellingCard";
import HandoffConnector from "@/components/HandoffConnector";

// Detail panels (accordion bodies)
import RiskDetails,       { getRiskVerdict }       from "@/components/RiskDetails";
import ConversionDetails, { getConversionVerdict } from "@/components/ConversionDetails";
import AdvisorDetails,    { getAdvisorVerdict }    from "@/components/AdvisorDetails";
import DecisionDetails,   { getDecisionVerdict }   from "@/components/DecisionDetails";

// ─────────────────────────────────────────────────────────────────────────────
//  Agent processing timing: each agent "completes" on a timer while the fetch
//  runs. When the real data arrives, all remaining agents complete instantly.
// ─────────────────────────────────────────────────────────────────────────────
const AGENT_REVEAL_DELAY_MS = 1200;

// Card state per agent
type CardState = "hidden" | "loading" | "revealed";

// App-level state machine
type AppState =
  | { kind: "idle" }
  | {
      kind: "running";
      input: QuoteInput;
      cardStates: [CardState, CardState, CardState, CardState];
      data: PipelineResponse | null;
    }
  | {
      kind: "complete";
      input: QuoteInput;
      data: PipelineResponse;
      elapsed: number;
    }
  | { kind: "ood_error"; error: OodErrorResponse }
  | { kind: "error"; title: string; message: string };

// ─────────────────────────────────────────────────────────────────────────────
//  Handoff connector colors for each agent → next agent transition
// ─────────────────────────────────────────────────────────────────────────────
const CONNECTOR_COLORS = [
  "bg-emerald-500",
  "bg-sky-500",
  "bg-violet-500",
];

export default function DashboardPage() {
  const [state, setState] = useState<AppState>({ kind: "idle" });
  const [formVisible, setFormVisible] = useState(true);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // ── handleAnalyze — progressive agent reveal ───────────────────────────
  const handleAnalyze = useCallback(async (input: QuoteInput) => {
    // Collapse form
    setFormVisible(false);

    // Initialize: Agent 0 starts loading
    setState({
      kind: "running",
      input,
      cardStates: ["loading", "hidden", "hidden", "hidden"],
      data: null,
    });

    // Progressive reveal: advance each agent from hidden → loading → revealed
    let step = 0;
    timerRef.current = setInterval(() => {
      step++;
      setState((prev) => {
        if (prev.kind !== "running") return prev;
        const next = [...prev.cardStates] as [CardState, CardState, CardState, CardState];

        // Previous agent becomes "revealed" (with placeholder until real data arrives)
        if (step - 1 >= 0 && step - 1 < 4) {
          next[step - 1] = "revealed";
        }
        // Current agent starts loading
        if (step < 4) {
          next[step] = "loading";
        }

        return { ...prev, cardStates: next };
      });

      if (step >= 4) {
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }, AGENT_REVEAL_DELAY_MS);

    const t0 = performance.now();
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90_000); // 90s for cold starts

    try {
      const res = await fetch(`${API_BASE}/api/v1/full-analysis`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(input),
        signal:  controller.signal,
      });

      clearTimeout(timeoutId);
      if (timerRef.current) clearInterval(timerRef.current);
      const elapsed = performance.now() - t0;

      if (res.ok) {
        const json: PipelineResponse = await res.json();
        setState({
          kind: "complete",
          input,
          data: json,
          elapsed,
        });
        return;
      }

      // OOD anomaly (HTTP 422)
      if (res.status === 422) {
        const json = await res.json();
        if (json.status && String(json.status).includes("DATA_ANOMALY")) {
          setState({ kind: "ood_error", error: json as OodErrorResponse });
          return;
        }

        const detail = json.detail;
        const msg = Array.isArray(detail)
          ? detail.map((d: { msg: string; loc: string[] }) => `${d.loc.join(".")}: ${d.msg}`).join("\n")
          : typeof detail === "string"
          ? detail
          : JSON.stringify(json);

        setState({ kind: "error", title: "Validation Error", message: msg });
        return;
      }

      if (res.status === 503) {
        const json = await res.json();
        setState({
          kind: "error",
          title: "Service Unavailable",
          message: json.detail || "Agent artifacts not loaded.",
        });
        return;
      }

      if (res.status === 504 || res.status === 502) {
        const body = await res.json().catch(() => ({}));
        if (body.status === "TIMEOUT") {
          setState({
            kind: "error",
            title: "Pipeline Timed Out",
            message: "The analysis took too long to complete. This can happen with complex risk profiles — please try again.",
          });
        } else {
          setState({
            kind: "error",
            title: "Backend Waking Up",
            message: "The server is starting up after a period of inactivity. This takes up to 60 seconds on the free tier — please wait a moment and try again.",
          });
        }
        return;
      }

      const text = await res.text();
      setState({ kind: "error", title: `HTTP ${res.status}`, message: text || "Unexpected error." });
    } catch (err) {
      clearTimeout(timeoutId);
      if (timerRef.current) clearInterval(timerRef.current);
      const isTimeout = err instanceof DOMException && err.name === "AbortError";
      setState({
        kind: "error",
        title: isTimeout ? "Request Timed Out" : "Connection Failed",
        message: isTimeout
          ? "The backend took too long to respond. It may be waking up from sleep — please wait 30 seconds and try again."
          : "Could not reach the backend. Please check your connection and try again.",
      });
    }
  }, []);

  // ── handleEdit — re-expand the form ────────────────────────────────────
  const handleEdit = useCallback(() => {
    setFormVisible(true);
    setState({ kind: "idle" });
  }, []);

  // ── Convenience getters ────────────────────────────────────────────────
  const isRunning   = state.kind === "running";
  const isComplete  = state.kind === "complete";
  const data        = isComplete ? state.data : state.kind === "running" ? state.data : null;
  const input       = isComplete ? state.input : isRunning ? state.input : null;
  const elapsed     = isComplete ? state.elapsed : 0;

  // Card states — when complete, all are revealed
  const cardStates: [CardState, CardState, CardState, CardState] =
    isComplete
      ? ["revealed", "revealed", "revealed", "revealed"]
      : isRunning
      ? state.cardStates
      : ["hidden", "hidden", "hidden", "hidden"];

  // Verdicts (only available with real data)
  const riskVerdict       = data?.risk_assessment       ? getRiskVerdict(data.risk_assessment)             : null;
  const conversionVerdict = data?.conversion_metrics    ? getConversionVerdict(data.conversion_metrics)    : null;
  const advisorVerdict    = data?.advisor_strategy      ? getAdvisorVerdict(data.advisor_strategy)         : null;
  const decisionVerdict   = data?.final_routing         ? getDecisionVerdict(data.final_routing)           : null;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200">
      <div className="max-w-3xl mx-auto px-4 py-8 md:py-12">
        {/* ── Top bar ───────────────────────────────────────────────── */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Cpu className="w-5 h-5 text-violet-400" />
              <h1 className="text-xl font-bold tracking-tight">
                InsurTech AI Pipeline
              </h1>
            </div>
            <p className="text-sm text-slate-500 font-mono">
              4-Agent LangGraph Quote Engine · v3.0
            </p>
          </div>

          <div className="flex items-center gap-3">
            {isComplete && data && (
              <>
                <span className="text-xs font-mono text-slate-500 border border-slate-700 rounded-full px-3 py-1 flex items-center gap-1.5">
                  <Zap className="w-3 h-3 text-violet-500" />
                  tx: {data.transaction_id.slice(0, 8)}…
                </span>
                <span className="text-xs font-mono text-slate-500 border border-slate-700 rounded-full px-3 py-1 flex items-center gap-1.5">
                  <Clock className="w-3 h-3 text-slate-500" />
                  {(elapsed / 1000).toFixed(1)}s
                </span>
              </>
            )}
          </div>
        </div>

        {/* ── Quote Form / Collapsed Header ─────────────────────────── */}
        <AnimatePresence mode="wait">
          {formVisible ? (
            <motion.div
              key="form"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20, height: 0 }}
              transition={{ duration: 0.35 }}
              className="mb-8"
            >
              <QuoteForm
                onSubmit={handleAnalyze}
                disabled={isRunning}
              />
            </motion.div>
          ) : input ? (
            <motion.div
              key="collapsed"
              className="mb-6"
            >
              <CollapsedHeader input={input} onEdit={handleEdit} />
            </motion.div>
          ) : null}
        </AnimatePresence>

        {/* ── Idle state ────────────────────────────────────────────── */}
        {state.kind === "idle" && !input && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center justify-center py-24 text-center gap-3"
          >
            <div className="w-16 h-16 rounded-2xl border border-slate-800 bg-slate-900 flex items-center justify-center mb-2">
              <Cpu className="w-7 h-7 text-slate-700" />
            </div>
            <p className="text-slate-400 font-semibold">No pipeline run yet</p>
            <p className="text-slate-600 font-mono text-xs max-w-xs">
              Fill in the quote form and click &ldquo;Run Analysis&rdquo; to
              execute all 4 agents through the LangGraph DAG.
            </p>
          </motion.div>
        )}

        {/* ── OOD Error ─────────────────────────────────────────────── */}
        {state.kind === "ood_error" && (
          <ErrorBanner
            title="Data Anomaly Detected"
            message={state.error.message}
            details={state.error.input}
            onDismiss={handleEdit}
            isOod
          />
        )}

        {/* ── General error ──────────────────────────────────────────── */}
        {state.kind === "error" && (
          <ErrorBanner
            title={state.title}
            message={state.message}
            onDismiss={handleEdit}
          />
        )}

        {/* ── Sequential Agent Cards ─────────────────────────────────── */}
        {(isRunning || isComplete) && (
          <div className="flex flex-col items-center">
            {/* Cold-start notice — only shown while loading */}
            {isRunning && !data && (
              <motion.div
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full mb-4 flex items-center gap-2.5 px-4 py-2.5 rounded-lg border border-amber-500/20 bg-amber-500/5 text-xs text-amber-300/80 font-mono"
              >
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse shrink-0" />
                First request may take up to 60s — backend wakes from sleep on Render free tier. Please wait…
              </motion.div>
            )}
            {/* ─── Agent 1: Risk Profiler ──────────────────────────── */}
            <div className="w-full">
              <StorytellingCard
                agentIndex={0}
                agentLabel="Agent 1 · Risk Profiler"
                icon={<Shield className="w-3.5 h-3.5 text-emerald-400" />}
                state={cardStates[0]}
                verdict={riskVerdict?.verdict}
                summary={riskVerdict?.summary}
                agentData={{ riskAssessment: data?.risk_assessment, quoteInput: input }}
              >
                {data?.risk_assessment && (
                  <RiskDetails data={data.risk_assessment} />
                )}
              </StorytellingCard>
            </div>

            {/* Handoff 1 → 2 */}
            {cardStates[0] !== "hidden" && (
              <HandoffConnector
                active={cardStates[0] === "revealed"}
                color={CONNECTOR_COLORS[0]}
              />
            )}

            {/* ─── Agent 2: Conversion Engine ─────────────────────── */}
            <div className="w-full">
              <StorytellingCard
                agentIndex={1}
                agentLabel="Agent 2 · Conversion Engine"
                icon={<BarChart2 className="w-3.5 h-3.5 text-sky-400" />}
                state={cardStates[1]}
                verdict={conversionVerdict?.verdict}
                summary={conversionVerdict?.summary}
                agentData={{ conversionMetrics: data?.conversion_metrics, quoteInput: input }}
              >
                {data?.conversion_metrics && (
                  <ConversionDetails data={data.conversion_metrics} />
                )}
              </StorytellingCard>
            </div>

            {/* Handoff 2 → 3 */}
            {cardStates[1] !== "hidden" && (
              <HandoffConnector
                active={cardStates[1] === "revealed"}
                color={CONNECTOR_COLORS[1]}
              />
            )}

            {/* ─── Agent 3: AI Premium Advisor ────────────────────── */}
            <div className="w-full">
              <StorytellingCard
                agentIndex={2}
                agentLabel="Agent 3 · AI Premium Advisor"
                icon={<MessageSquare className="w-3.5 h-3.5 text-violet-400" />}
                state={cardStates[2]}
                verdict={advisorVerdict?.verdict}
                summary={advisorVerdict?.summary}
                agentData={{ advisorStrategy: data?.advisor_strategy, quoteInput: input }}
              >
                {data?.advisor_strategy && (
                  <AdvisorDetails data={data.advisor_strategy} />
                )}
              </StorytellingCard>
            </div>

            {/* Handoff 3 → 4 */}
            {cardStates[2] !== "hidden" && (
              <HandoffConnector
                active={cardStates[2] === "revealed"}
                color={CONNECTOR_COLORS[2]}
              />
            )}

            {/* ─── Agent 4: Underwriting Router ───────────────────── */}
            <div className="w-full">
              <StorytellingCard
                agentIndex={3}
                agentLabel="Agent 4 · Underwriting Router"
                icon={<Gavel className="w-3.5 h-3.5 text-amber-400" />}
                state={cardStates[3]}
                verdict={decisionVerdict?.verdict}
                summary={decisionVerdict?.summary}
                agentData={{ finalRouting: data?.final_routing, quoteInput: input }}
                defaultOpen={true}
              >
                {data?.final_routing && (
                  <DecisionDetails data={data.final_routing} />
                )}
              </StorytellingCard>
            </div>

            {/* ── Escalation notice ─────────────────────────────────── */}
            {isComplete && data?.escalation_reason && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="w-full mt-6 rounded-xl border border-amber-500/30 bg-amber-500/5 p-4"
              >
                <p className="text-[10px] font-mono text-amber-400 uppercase tracking-widest mb-1">
                  Escalation Notice
                </p>
                <p className="text-sm text-slate-300 font-typewriter">
                  {data.escalation_reason}
                </p>
              </motion.div>
            )}

            {/* ── New Analysis button ───────────────────────────────── */}
            {isComplete && (
              <motion.button
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                onClick={handleEdit}
                className="mt-8 mb-4 text-sm font-mono text-violet-400 hover:text-violet-300 
                           border border-violet-500/30 hover:border-violet-500/60
                           rounded-lg px-5 py-2.5 transition-colors"
              >
                ↻ Run New Analysis
              </motion.button>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
