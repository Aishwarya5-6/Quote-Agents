"use client";

import { useState, useCallback } from "react";
import { Cpu, Zap, Clock } from "lucide-react";

import {
  API_BASE,
  type PipelineResponse,
  type OodErrorResponse,
  type QuoteInput,
} from "@/lib/api-contract";

import PipelineStepper from "@/components/PipelineStepper";
import QuoteForm       from "@/components/QuoteForm";
import RiskPanel       from "@/components/RiskPanel";
import ConversionPanel from "@/components/ConversionPanel";
import AdvisorPanel    from "@/components/AdvisorPanel";
import DecisionBanner  from "@/components/DecisionBanner";
import ErrorBanner     from "@/components/ErrorBanner";

// ─────────────────────────────────────────────────────────────────────────────
//  Pipeline step timing: animate through 4 agents while the fetch runs.
//  Steps complete as real data arrives; the stepper auto-advances.
// ─────────────────────────────────────────────────────────────────────────────
const STEP_INTERVAL_MS = 800;

type AppState =
  | { kind: "idle" }
  | { kind: "loading"; step: number }
  | { kind: "success"; data: PipelineResponse; elapsed: number }
  | { kind: "ood_error"; error: OodErrorResponse }
  | { kind: "error"; title: string; message: string };

export default function DashboardPage() {
  const [state, setState] = useState<AppState>({ kind: "idle" });

  // ── handleAnalyze — fetch POST /api/v1/full-analysis ───────────────────
  const handleAnalyze = useCallback(async (input: QuoteInput) => {
    setState({ kind: "loading", step: 0 });

    // Animate the stepper through agents 0 → 3 on a timer
    let currentStep = 0;
    const ticker = setInterval(() => {
      currentStep = Math.min(currentStep + 1, 3);
      setState((prev) =>
        prev.kind === "loading" ? { ...prev, step: currentStep } : prev
      );
    }, STEP_INTERVAL_MS);

    const t0 = performance.now();

    try {
      const res = await fetch(`${API_BASE}/api/v1/full-analysis`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(input),
      });

      clearInterval(ticker);
      const elapsed = performance.now() - t0;

      if (res.ok) {
        const json: PipelineResponse = await res.json();
        setState({ kind: "success", data: json, elapsed });
        return;
      }

      // ── OOD anomaly (HTTP 422) ──────────────────────────────────────
      if (res.status === 422) {
        const json = await res.json();

        // Check if this is the OOD gate response (has `status` field with DATA_ANOMALY)
        if (json.status && String(json.status).includes("DATA_ANOMALY")) {
          setState({
            kind: "ood_error",
            error: json as OodErrorResponse,
          });
          return;
        }

        // Pydantic validation error (missing/invalid fields)
        const detail = json.detail;
        const msg = Array.isArray(detail)
          ? detail.map((d: { msg: string; loc: string[] }) => `${d.loc.join(".")}: ${d.msg}`).join("\n")
          : typeof detail === "string"
          ? detail
          : JSON.stringify(json);

        setState({
          kind: "error",
          title: "Validation Error",
          message: msg,
        });
        return;
      }

      // ── 503: agents not loaded ──────────────────────────────────────
      if (res.status === 503) {
        const json = await res.json();
        setState({
          kind: "error",
          title: "Service Unavailable",
          message: json.detail || "Agent artifacts not loaded. Run agent1_risk_profiler.py first.",
        });
        return;
      }

      // ── Other HTTP errors ───────────────────────────────────────────
      const text = await res.text();
      setState({
        kind: "error",
        title: `HTTP ${res.status}`,
        message: text || "An unexpected error occurred.",
      });
    } catch (err) {
      clearInterval(ticker);

      // Network error — backend not running
      setState({
        kind: "error",
        title: "Connection Failed",
        message:
          `Could not reach the backend at ${API_BASE}. ` +
          "Make sure uvicorn is running: cd backend && uvicorn main:app --port 8001",
      });
    }
  }, []);

  // ── Convenience getters ────────────────────────────────────────────────
  const isLoading = state.kind === "loading";
  const data      = state.kind === "success" ? state.data : null;
  const elapsed   = state.kind === "success" ? state.elapsed : 0;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200 p-6 md:p-10">
      {/* ── Top bar ─────────────────────────────────────────────────── */}
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
          {data && (
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

      {/* ── Main layout: sidebar form + results ────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* ── Left sidebar: Quote form + Pipeline stepper ──────────── */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          <QuoteForm onSubmit={handleAnalyze} disabled={isLoading} />
          <PipelineStepper
            currentStep={
              state.kind === "loading"
                ? state.step
                : state.kind === "success"
                ? 4
                : -1
            }
          />
        </div>

        {/* ── Right: Results area ──────────────────────────────────── */}
        <div className="lg:col-span-9">
          {/* Idle state */}
          {state.kind === "idle" && (
            <div className="flex flex-col items-center justify-center py-32 text-center gap-3">
              <div className="w-16 h-16 rounded-2xl border border-slate-800 bg-slate-900 flex items-center justify-center mb-2">
                <Cpu className="w-7 h-7 text-slate-700" />
              </div>
              <p className="text-slate-400 font-semibold">No pipeline run yet</p>
              <p className="text-slate-600 font-mono text-xs max-w-xs">
                Fill in the quote form and click &ldquo;Run Analysis&rdquo; to
                execute all 4 agents through the LangGraph DAG.
              </p>
            </div>
          )}

          {/* Loading state — horizontal stepper centered */}
          {isLoading && (
            <div className="flex flex-col items-center justify-center py-24 gap-6">
              <PipelineStepper
                currentStep={state.kind === "loading" ? state.step : 0}
                direction="horizontal"
              />
              <p className="text-sm text-slate-500 font-mono animate-pulse">
                Running 4-agent pipeline…
              </p>
            </div>
          )}

          {/* OOD Error */}
          {state.kind === "ood_error" && (
            <ErrorBanner
              title="Data Anomaly Detected"
              message={state.error.message}
              details={state.error.input}
              onDismiss={() => setState({ kind: "idle" })}
            />
          )}

          {/* General error */}
          {state.kind === "error" && (
            <ErrorBanner
              title={state.title}
              message={state.message}
              onDismiss={() => setState({ kind: "idle" })}
            />
          )}

          {/* ── Success: Results grid ──────────────────────────────── */}
          {data && (
            <div className="flex flex-col gap-6">
              {/* Decision banner — full width, high contrast */}
              {data.final_routing && (
                <DecisionBanner data={data.final_routing} />
              )}

              {/* Agent cards — 2-column grid */}
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                {/* Agent 1: Risk */}
                {data.risk_assessment && (
                  <RiskPanel data={data.risk_assessment} />
                )}

                {/* Agent 2: Conversion */}
                {data.conversion_metrics && (
                  <ConversionPanel data={data.conversion_metrics} />
                )}

                {/* Agent 3: Advisor */}
                {data.advisor_strategy && (
                  <AdvisorPanel data={data.advisor_strategy} />
                )}
              </div>

              {/* Escalation reason if present */}
              {data.escalation_reason && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
                  <p className="text-[10px] font-mono text-amber-400 uppercase tracking-widest mb-1">
                    Escalation Notice
                  </p>
                  <p className="text-sm text-slate-300 font-typewriter">
                    {data.escalation_reason}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
