"use client";

import { useState } from "react";
import { Loader2, Play, Cpu, Zap } from "lucide-react";

import { MOCK_PIPELINE_RESPONSE, type PipelineResponse } from "@/lib/api-contract";
import PipelineStatus from "@/components/PipelineStatus";
import RiskModule     from "@/components/RiskModule";
import ConversionCard from "@/components/ConversionCard";
import AdvisorPanel   from "@/components/AdvisorPanel";

export default function DashboardPage() {
  const [data,        setData]        = useState<PipelineResponse | null>(null);
  const [loading,     setLoading]     = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);

  // ── Simulation handler ─────────────────────────────────────────────────────
  // Steps agents 0 → 1 → 2 → 3 at 500 ms intervals (total: 2 s)
  // ── TO GO LIVE: uncomment the fetch() block and delete the mock line ────────
  const handleSimulate = async () => {
    setData(null);
    setLoading(true);
    setCurrentStep(0);

    // Animate through the 4 agents
    for (let step = 1; step <= 4; step++) {
      await new Promise<void>((r) => setTimeout(r, 500));
      setCurrentStep(step);
    }

    // ── MOCK (swap this block when backend is ready) ─────────────────────────
    const json = MOCK_PIPELINE_RESPONSE;
    // ── LIVE (uncomment to connect to the real pipeline) ─────────────────────
    // const res  = await fetch("http://localhost:8000/api/process_quote", {
    //   method:  "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body:    JSON.stringify({
    //     Driver_Age: 34, Driving_Exp: 12, Prev_Accidents: 0,
    //     Prev_Citations: 1, Annual_Miles: 22000, Veh_Usage: "Pleasure",
    //   }),
    // });
    // const json: PipelineResponse = await res.json();
    // ─────────────────────────────────────────────────────────────────────────

    setData(json);
    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200 p-6 md:p-10">

      {/* ── Top bar ─────────────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-10">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Cpu className="w-5 h-5 text-violet-400" />
            <h1 className="text-xl font-bold tracking-tight">InsurTech AI Pipeline</h1>
          </div>
          <p className="text-sm text-slate-500 font-mono">
            4-Agent Multi-Modal Quote Engine · v3.0
          </p>
        </div>

        <div className="flex items-center gap-3">
          {data && (
            <span className="text-xs font-mono text-slate-500 border border-slate-700 rounded-full px-3 py-1 flex items-center gap-1.5">
              <Zap className="w-3 h-3 text-violet-500" />
              tx: {data.transaction_id.slice(0, 8)}…
            </span>
          )}

          <button
            onClick={handleSimulate}
            disabled={loading}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-violet-600 hover:bg-violet-500 active:bg-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-semibold shadow-lg shadow-violet-500/20"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Running Pipeline…
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Simulate AI Pipeline
              </>
            )}
          </button>
        </div>
      </div>

      {/* ── Idle state ──────────────────────────────────────────────────────── */}
      {!loading && !data && (
        <div className="flex flex-col items-center justify-center py-32 text-center gap-3">
          <div className="w-16 h-16 rounded-2xl border border-slate-800 bg-slate-900 flex items-center justify-center mb-2">
            <Cpu className="w-7 h-7 text-slate-700" />
          </div>
          <p className="text-slate-400 font-semibold">No pipeline run yet</p>
          <p className="text-slate-600 font-mono text-xs max-w-xs">
            Click &ldquo;Simulate AI Pipeline&rdquo; to execute all 4 agents
            and render the full risk · conversion · advisor · routing dashboard.
          </p>
        </div>
      )}

      {/* ── Loading state — animated stepper centred ────────────────────────── */}
      {loading && (
        <div className="max-w-xs mx-auto">
          <PipelineStatus currentStep={currentStep} />
        </div>
      )}

      {/* ── Results layout ──────────────────────────────────────────────────── */}
      {data && !loading && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">

          {/* Pipeline stepper — left sidebar */}
          <div className="lg:col-span-1">
            <PipelineStatus currentStep={4} />
          </div>

          {/* Data cards — responsive 3-up grid */}
          <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {/* Agent 1 */}
            <RiskModule data={data.risk_assessment} />

            {/* Agent 2 */}
            <ConversionCard data={data.conversion_metrics} />

            {/* Agents 3 & 4 — spans 2 cols on md, 1 on xl */}
            <div className="md:col-span-2 xl:col-span-1">
              <AdvisorPanel
                advisor={data.advisor_strategy}
                routing={data.final_routing}
              />
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
