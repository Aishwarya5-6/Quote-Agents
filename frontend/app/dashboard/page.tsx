"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Cpu,
  ArrowLeft,
  Pause,
  Play,
  Trash2,
  Shield,
  Zap,
  TrendingUp,
} from "lucide-react";
import Link from "next/link";

import {
  API_BASE,
  type PipelineResponse,
  type QuoteInput,
  type RoutingDecision,
} from "@/lib/api-contract";
import { generateRandomQuote, quoteLabel } from "@/lib/quote-simulator";

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────
const MAX_RECORDS = 60;

const SPEED_OPTIONS = [
  { label: "Slow (5s)",   ms: 5000 },
  { label: "Normal (3s)", ms: 3000 },
  { label: "Fast (1.5s)", ms: 1500 },
] as const;

// ─────────────────────────────────────────────────────────────────────────────
//  Types
// ─────────────────────────────────────────────────────────────────────────────
type RecordStatus = "processing" | "done" | "error" | "ood";

interface QuoteRecord {
  id:        string;
  seq:       number;
  input:     QuoteInput;
  status:    RecordStatus;
  result:    PipelineResponse | null;
  elapsed:   number;
  timestamp: number;
  errorMsg?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Colour helpers
// ─────────────────────────────────────────────────────────────────────────────
function routingTextColor(d: RoutingDecision | null | undefined): string {
  if (d === "AUTO_APPROVE") return "text-emerald-400";
  if (d === "REJECT")       return "text-rose-400";
  return "text-amber-400";
}

function routingBadgeClass(d: RoutingDecision | null | undefined): string {
  if (d === "AUTO_APPROVE") return "border-emerald-500/30 bg-emerald-500/10 text-emerald-400";
  if (d === "REJECT")       return "border-rose-500/30 bg-rose-500/10 text-rose-400";
  return "border-amber-500/30 bg-amber-500/5 text-amber-400";
}

function routingLabel(d: RoutingDecision | null | undefined): string {
  if (d === "AUTO_APPROVE") return "✓ APPROVE";
  if (d === "REJECT")       return "✗ REJECT";
  return "⚑ REVIEW";
}

function tierTextColor(t: string | null | undefined): string {
  if (t === "Low")  return "text-emerald-400";
  if (t === "High") return "text-rose-400";
  return "text-amber-400";
}

function priorityBadgeClass(p: string | null | undefined): string {
  if (p === "High") return "border-rose-500/40 bg-rose-500/10 text-rose-400";
  if (p === "Low")  return "border-emerald-500/30 bg-emerald-500/5 text-emerald-400";
  return "border-amber-500/30 bg-amber-500/5 text-amber-400";
}

// ─────────────────────────────────────────────────────────────────────────────
//  StatCard sub-component
// ─────────────────────────────────────────────────────────────────────────────
interface StatCardProps {
  label:       string;
  value:       number;
  icon:        React.ReactNode;
  color:       string;
  borderColor: string;
  subLabel?:   string;
}

function StatCard({ label, value, icon, color, borderColor, subLabel }: StatCardProps) {
  return (
    <div className={`rounded-xl border ${borderColor} bg-slate-900 p-4 flex flex-col gap-1.5`}>
      <div className="flex items-center justify-between">
        <span className={`text-[10px] font-mono uppercase tracking-widest ${color} opacity-60`}>
          {label}
        </span>
        <span className={`${color} opacity-70`}>{icon}</span>
      </div>
      <motion.span
        key={value}
        initial={{ scale: 1.25, opacity: 0.5 }}
        animate={{ scale: 1,    opacity: 1 }}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
        className={`text-4xl font-bold font-mono ${color}`}
      >
        {value}
      </motion.span>
      {subLabel && (
        <span className="text-[10px] font-mono text-slate-600">{subLabel}</span>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  MAIN DASHBOARD PAGE
// ─────────────────────────────────────────────────────────────────────────────
export default function OperationsDashboardPage() {
  const [running,    setRunning]    = useState(false);
  const [speedIdx,   setSpeedIdx]   = useState(1);        // default "Normal (3s)"
  const [records,    setRecords]    = useState<QuoteRecord[]>([]);
  const [warmingUp,  setWarmingUp]  = useState(false);   // cold-start banner

  const intervalRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const seqRef         = useRef(0);
  const inFlightRef    = useRef(false);   // prevent concurrent requests
  const timeoutStreak  = useRef(0);       // consecutive timeout counter

  // ── Fire a single quote through the pipeline ────────────────────────────
  const fireQuote = useCallback(async () => {
    // Skip if a request is already in-flight (backend may be cold-starting)
    if (inFlightRef.current) return;

    inFlightRef.current = true;
    const id    = crypto.randomUUID();
    const seq   = ++seqRef.current;
    const input = generateRandomQuote();
    const t0    = performance.now();

    // Append as "processing" at the top of the list
    setRecords(prev => [
      { id, seq, input, status: "processing", result: null, elapsed: 0, timestamp: Date.now() },
      ...prev.slice(0, MAX_RECORDS - 1),
    ]);

    try {
      const res = await fetch(`${API_BASE}/api/v1/full-analysis`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(input),
        signal:  AbortSignal.timeout(90_000),  // 90s — matches Render backend timeout
      });

      const elapsed = performance.now() - t0;
      timeoutStreak.current = 0;
      setWarmingUp(false);

      if (res.ok) {
        const json: PipelineResponse = await res.json();
        setRecords(prev =>
          prev.map(r => r.id === id ? { ...r, status: "done", result: json, elapsed } : r)
        );
        inFlightRef.current = false;
        return;
      }

      // OOD anomaly (HTTP 422 with DATA_ANOMALY body)
      if (res.status === 422) {
        const json = await res.json();
        const isOod = json.status && String(json.status).includes("DATA_ANOMALY");
        setRecords(prev =>
          prev.map(r =>
            r.id === id
              ? { ...r, status: isOod ? "ood" : "error", elapsed, errorMsg: json.message ?? "Validation error" }
              : r
          )
        );
        inFlightRef.current = false;
        return;
      }

      // Any other HTTP error
      setRecords(prev =>
        prev.map(r =>
          r.id === id
            ? { ...r, status: "error", elapsed, errorMsg: `HTTP ${res.status}` }
            : r
        )
      );
    } catch (err: unknown) {
      const elapsed = performance.now() - t0;
      const msg = err instanceof Error ? err.message : "Connection failed";

      // Backend is asleep (connection refused) — silently drop this record
      // and show the warming banner; the interval will auto-retry
      if (msg === "Failed to fetch") {
        setRecords(prev => prev.filter(r => r.id !== id));
        setWarmingUp(true);
        // inFlightRef released in finally; interval fires again automatically
        return;
      }

      // Detect cold-start timeout (>= 2 consecutive timeouts)
      if (msg.toLowerCase().includes("timeout") || msg.toLowerCase().includes("timed out")) {
        timeoutStreak.current += 1;
        if (timeoutStreak.current >= 2) setWarmingUp(true);
      } else {
        timeoutStreak.current = 0;
        setWarmingUp(false);
      }
      setRecords(prev =>
        prev.map(r =>
          r.id === id
            ? { ...r, status: "error", elapsed, errorMsg: msg }
            : r
        )
      );
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  // ── Start / stop the interval ────────────────────────────────────────────
  useEffect(() => {
    if (running) {
      fireQuote(); // fire once immediately on start
      intervalRef.current = setInterval(fireQuote, SPEED_OPTIONS[speedIdx].ms);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [running, speedIdx, fireQuote]);

  // ── Derived stats ────────────────────────────────────────────────────────
  const doneRecords  = records.filter(r => r.status === "done");
  const approved     = doneRecords.filter(r => r.result?.final_routing_decision === "AUTO_APPROVE").length;
  const manualReview = doneRecords.filter(r => r.result?.final_routing_decision === "MANUAL_REVIEW").length;
  const rejected     = doneRecords.filter(r => r.result?.final_routing_decision === "REJECT").length;
  const processing   = records.filter(r => r.status === "processing").length;

  // Approval rate %
  const approvalRate = doneRecords.length > 0
    ? Math.round((approved / doneRecords.length) * 100)
    : 0;

  // Escalation queue: MANUAL_REVIEW rows sorted High → Medium → Low, then newest first
  const PRIORITY_ORDER: Record<string, number> = { High: 0, Medium: 1, Low: 2 };
  const escalationQueue = [...doneRecords]
    .filter(r => r.result?.final_routing_decision === "MANUAL_REVIEW")
    .sort((a, b) => {
      const pa = PRIORITY_ORDER[a.result?.final_routing?.priority ?? ""] ?? 3;
      const pb = PRIORITY_ORDER[b.result?.final_routing?.priority ?? ""] ?? 3;
      return pa !== pb ? pa - pb : b.timestamp - a.timestamp;
    })
    .slice(0, 20);

  const errorCount = records.filter(r => r.status === "error").length;

  const handleClear = () => {
    setRecords([]);
    seqRef.current = 0;
  };

  const handleDismissErrors = () => {
    setRecords(prev => prev.filter(r => r.status !== "error"));
  };

  // ─────────────────────────────────────────────────────────────────────────
  //  Render
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <main className="min-h-screen bg-slate-950 text-slate-200">
      <div className="max-w-screen-xl mx-auto px-4 py-6">

        {/* ── Top Bar ───────────────────────────────────────────────────── */}
        <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="flex items-center gap-1.5 text-xs font-mono text-slate-500 hover:text-slate-300 border border-slate-800 hover:border-slate-600 rounded-lg px-3 py-1.5 transition-colors"
            >
              <ArrowLeft className="w-3 h-3" />
              Single Quote
            </Link>
            <div className="w-px h-5 bg-slate-800" />
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-violet-400" />
              <h1 className="text-lg font-bold tracking-tight">
                Agent Operations Dashboard
              </h1>
            </div>
            {running && (
              <span className="flex items-center gap-1.5 text-[10px] font-mono text-emerald-400 border border-emerald-500/30 bg-emerald-500/5 rounded-full px-2 py-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                LIVE
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 text-xs font-mono text-slate-500">
            {processing > 0 && (
              <span className="flex items-center gap-1.5 text-violet-400">
                <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
                {processing} in pipeline
              </span>
            )}
            <span className="text-slate-700">{records.length}/{MAX_RECORDS} records</span>
          </div>
        </div>

        {/* ── KPI Stats Row ─────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-5">
          <StatCard
            label="Processed"
            value={doneRecords.length}
            icon={<Cpu className="w-4 h-4" />}
            color="text-violet-400"
            borderColor="border-violet-500/20"
            subLabel="total completed"
          />
          <StatCard
            label="Auto-Approved"
            value={approved}
            icon={<CheckCircle2 className="w-4 h-4" />}
            color="text-emerald-400"
            borderColor="border-emerald-500/20"
            subLabel={doneRecords.length > 0 ? `${approvalRate}% rate` : "—"}
          />
          <StatCard
            label="Manual Review"
            value={manualReview}
            icon={<AlertTriangle className="w-4 h-4" />}
            color="text-amber-400"
            borderColor="border-amber-500/20"
            subLabel="in escalation queue"
          />
          <StatCard
            label="Rejected"
            value={rejected}
            icon={<XCircle className="w-4 h-4" />}
            color="text-rose-400"
            borderColor="border-rose-500/20"
            subLabel="hard denials"
          />
          <StatCard
            label="Throughput"
            value={approved + manualReview + rejected}
            icon={<TrendingUp className="w-4 h-4" />}
            color="text-sky-400"
            borderColor="border-sky-500/20"
            subLabel={`avg ${doneRecords.length > 0 ? (doneRecords.reduce((s, r) => s + r.elapsed, 0) / doneRecords.length / 1000).toFixed(1) : "–"}s / quote`}
          />
        </div>

        {/* ── Control Bar ───────────────────────────────────────────────── */}
        <div className="flex flex-wrap items-center gap-3 mb-5 px-4 py-3 rounded-xl border border-slate-800 bg-slate-900">
          {/* Start / Pause button */}
          <button
            onClick={() => setRunning(r => !r)}
            className={`flex items-center gap-2 text-xs font-mono font-semibold px-4 py-2 rounded-lg border transition-all ${
              running
                ? "border-amber-500/40 bg-amber-500/10 text-amber-400 hover:bg-amber-500/20"
                : "border-emerald-500/40 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20"
            }`}
          >
            {running
              ? <><Pause className="w-3.5 h-3.5" /> Pause Feed</>
              : <><Play  className="w-3.5 h-3.5" /> Start Feed</>
            }
          </button>

          {/* Speed selector */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-slate-600 uppercase tracking-wider">Speed</span>
            <div className="flex gap-1">
              {SPEED_OPTIONS.map((opt, i) => (
                <button
                  key={i}
                  onClick={() => setSpeedIdx(i)}
                  className={`text-xs font-mono px-3 py-1.5 rounded-md border transition-colors ${
                    speedIdx === i
                      ? "border-violet-500/50 bg-violet-500/15 text-violet-300"
                      : "border-slate-700 text-slate-500 hover:text-slate-300 hover:border-slate-600"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Clear / Dismiss errors */}
          <div className="ml-auto flex items-center gap-2">
            {errorCount > 0 && (
              <button
                onClick={handleDismissErrors}
                className="flex items-center gap-1.5 text-xs font-mono text-amber-500/70 hover:text-amber-400 border border-amber-500/20 hover:border-amber-500/40 rounded-lg px-3 py-1.5 transition-colors"
              >
                <XCircle className="w-3 h-3" />
                Dismiss {errorCount} error{errorCount !== 1 ? "s" : ""}
              </button>
            )}
            <button
              onClick={handleClear}
              disabled={records.length === 0}
              className="flex items-center gap-1.5 text-xs font-mono text-slate-500 hover:text-rose-400 border border-slate-700 hover:border-rose-500/40 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
            >
              <Trash2 className="w-3 h-3" />
              Clear All
            </button>
          </div>
        </div>

        {/* ── Warming-up banner (Render free tier cold start) ──────────── */}
        {warmingUp && (
          <div className="mb-4 flex items-center gap-3 px-4 py-3 rounded-xl border border-amber-500/30 bg-amber-500/5 text-amber-400">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse shrink-0" />
            <span className="text-xs font-mono">
              Backend is waking up (Render free tier cold start — up to 60s). Requests are queued and will resume automatically.
            </span>
          </div>
        )}

        {/* ── Main Content: Live Feed (left) + Escalation Queue (right) ── */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">

          {/* ── LIVE FEED TABLE — 2/3 width ─────────────────────────────── */}
          <div className="xl:col-span-2">
            <div className="rounded-xl border border-slate-800 bg-slate-900 overflow-hidden flex flex-col">

              {/* Table header bar */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 shrink-0">
                <div className="flex items-center gap-2">
                  <Activity className="w-3.5 h-3.5 text-violet-400" />
                  <span className="text-[10px] font-mono font-semibold text-slate-300 uppercase tracking-widest">
                    Live Feed
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  {running && (
                    <span className="flex items-center gap-1.5 text-[10px] font-mono text-emerald-400">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                      streaming
                    </span>
                  )}
                  <span className="text-[10px] font-mono text-slate-600">
                    {records.length} rows
                  </span>
                </div>
              </div>

              {/* Column header row */}
              <div className="grid items-center gap-2 px-4 py-2 border-b border-slate-800/60 bg-slate-900/80 shrink-0"
                style={{ gridTemplateColumns: "2.5rem 1fr 5.5rem 4.5rem 8rem 5rem 3.5rem" }}>
                {["#", "Driver Profile", "Risk Tier", "Bind %", "Decision", "Priority", "Time"].map(h => (
                  <span key={h} className="text-[9px] font-mono text-slate-600 uppercase tracking-widest">{h}</span>
                ))}
              </div>

              {/* Rows */}
              <div className="overflow-y-auto" style={{ maxHeight: "540px" }}>
                {records.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-24 gap-3">
                    <Zap className="w-8 h-8 text-slate-800" />
                    <p className="text-sm text-slate-600 font-mono text-center">
                      Press &ldquo;Start Feed&rdquo; to begin<br />processing quotes through the pipeline
                    </p>
                  </div>
                ) : (
                  <AnimatePresence initial={false}>
                    {records.map(record => {
                      const risk     = record.result?.risk_assessment;
                      const conv     = record.result?.conversion_metrics;
                      const routing  = record.result?.final_routing;
                      const decision = record.result?.final_routing_decision;
                      const bindPct  = conv?.bind_probability != null
                        ? Math.round(conv.bind_probability * 100)
                        : null;

                      return (
                        <motion.div
                          key={record.id}
                          layout
                          initial={{ opacity: 0, backgroundColor: "rgba(139,92,246,0.12)" }}
                          animate={{ opacity: 1, backgroundColor: "rgba(0,0,0,0)" }}
                          transition={{ duration: 0.45 }}
                          className="grid items-center gap-2 px-4 py-2.5 border-b border-slate-800/30 hover:bg-slate-800/25 transition-colors"
                          style={{ gridTemplateColumns: "2.5rem 1fr 5.5rem 4.5rem 8rem 5rem 3.5rem" }}
                        >
                          {/* Seq */}
                          <span className="text-[10px] font-mono text-slate-600">
                            #{record.seq}
                          </span>

                          {/* Driver profile */}
                          <span
                            className="text-xs font-mono text-slate-400 truncate"
                            title={quoteLabel(record.input)}
                          >
                            {quoteLabel(record.input)}
                          </span>

                          {/* Risk Tier */}
                          <span className={`text-xs font-mono font-semibold ${
                            record.status === "processing"
                              ? "text-slate-700"
                              : risk
                              ? tierTextColor(risk.predicted_tier)
                              : record.status === "ood"
                              ? "text-orange-400"
                              : "text-slate-600"
                          }`}>
                            {record.status === "processing"
                              ? <span className="animate-pulse">–</span>
                              : risk
                              ? risk.predicted_tier
                              : record.status === "ood"
                              ? "OOD"
                              : "–"}
                          </span>

                          {/* Bind % */}
                          <span className={`text-xs font-mono ${
                            bindPct != null ? "text-sky-400" : "text-slate-600"
                          }`}>
                            {record.status === "processing"
                              ? <span className="animate-pulse text-slate-700">…</span>
                              : bindPct != null
                              ? `${bindPct}%`
                              : "—"}
                          </span>

                          {/* Decision badge */}
                          {record.status === "processing" ? (
                            <span className="flex items-center gap-1 text-[10px] font-mono text-violet-400">
                              <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse shrink-0" />
                              running…
                            </span>
                          ) : decision ? (
                            <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border truncate ${routingBadgeClass(decision)}`}>
                              {routingLabel(decision)}
                            </span>
                          ) : record.status === "ood" ? (
                            <span className="text-[10px] font-mono text-orange-400 border border-orange-500/30 bg-orange-500/5 px-2 py-0.5 rounded">
                              ⚠ ANOMALY
                            </span>
                          ) : (
                            <span
                              className="text-[10px] font-mono text-rose-500/60 truncate"
                              title={record.errorMsg ?? "Unknown error"}
                            >
                              ✕ {record.errorMsg
                                ? record.errorMsg.length > 18
                                  ? record.errorMsg.slice(0, 18) + "…"
                                  : record.errorMsg
                                : "error"}
                            </span>
                          )}

                          {/* Priority */}
                          <span className={`text-[10px] font-mono font-semibold ${
                            record.status === "processing" ? "text-slate-700" : priorityBadgeClass(routing?.priority).includes("rose") ? "text-rose-400" : priorityBadgeClass(routing?.priority).includes("emerald") ? "text-emerald-400" : "text-amber-400"
                          }`}>
                            {record.status === "processing" ? "—" : routing?.priority ?? "—"}
                          </span>

                          {/* Elapsed time */}
                          <span className="text-[10px] font-mono text-slate-600">
                            {record.status === "processing"
                              ? <span className="animate-pulse">…</span>
                              : record.elapsed > 0
                              ? `${(record.elapsed / 1000).toFixed(1)}s`
                              : "—"}
                          </span>
                        </motion.div>
                      );
                    })}
                  </AnimatePresence>
                )}
              </div>
            </div>
          </div>

          {/* ── ESCALATION QUEUE — 1/3 width ───────────────────────────── */}
          <div className="xl:col-span-1">
            <div className="rounded-xl border border-amber-500/20 bg-slate-900 overflow-hidden flex flex-col h-full">

              {/* Queue header */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-amber-500/20 bg-amber-500/5 shrink-0">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                  <span className="text-[10px] font-mono font-semibold text-amber-400 uppercase tracking-widest">
                    Escalation Queue
                  </span>
                </div>
                <motion.span
                  key={manualReview}
                  initial={{ scale: 1.4, opacity: 0.6 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="text-xs font-mono font-bold text-amber-400 border border-amber-500/30 bg-amber-500/10 rounded-full w-6 h-6 flex items-center justify-center"
                >
                  {manualReview}
                </motion.span>
              </div>

              {/* Priority legend */}
              <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-800/60 shrink-0">
                {[
                  { label: "High",   cls: "text-rose-400" },
                  { label: "Medium", cls: "text-amber-400" },
                  { label: "Low",    cls: "text-emerald-400" },
                ].map(({ label, cls }) => (
                  <span key={label} className={`flex items-center gap-1 text-[9px] font-mono ${cls} opacity-70`}>
                    <span className="w-1.5 h-1.5 rounded-full bg-current" />
                    {label}
                  </span>
                ))}
                <span className="ml-auto text-[9px] font-mono text-slate-700">Sorted by priority ↓</span>
              </div>

              {/* Queue rows */}
              <div className="overflow-y-auto flex-1" style={{ maxHeight: "490px" }}>
                {escalationQueue.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-16 gap-3 px-4">
                    <Shield className="w-8 h-8 text-slate-800" />
                    <p className="text-xs text-slate-600 font-mono text-center leading-relaxed">
                      No escalations yet.<br />
                      MANUAL_REVIEW cases appear<br />here, sorted by priority.
                    </p>
                  </div>
                ) : (
                  <AnimatePresence initial={false}>
                    {escalationQueue.map(record => {
                      const risk    = record.result?.risk_assessment;
                      const routing = record.result?.final_routing;
                      const conv    = record.result?.conversion_metrics;
                      const bindPct = conv?.bind_probability != null
                        ? Math.round(conv.bind_probability * 100)
                        : null;
                      const pri = routing?.priority;

                      return (
                        <motion.div
                          key={record.id}
                          initial={{ opacity: 0, x: 16 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -8 }}
                          transition={{ duration: 0.3 }}
                          className="px-4 py-3 border-b border-slate-800/40 hover:bg-amber-500/5 transition-colors"
                        >
                          {/* Row header: seq + priority badge */}
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="text-[10px] font-mono text-slate-600">
                              #{record.seq}
                            </span>
                            <span className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded border ${priorityBadgeClass(pri)}`}>
                              {pri ?? "Med"} Priority
                            </span>
                          </div>

                          {/* Driver profile */}
                          <p className="text-xs font-mono text-slate-300 truncate mb-1.5">
                            {quoteLabel(record.input)}
                          </p>

                          {/* Risk + Bind inline */}
                          <div className="flex items-center gap-3 mb-2">
                            <span className={`text-[10px] font-mono font-semibold ${tierTextColor(risk?.predicted_tier)}`}>
                              {risk?.predicted_tier ?? "?"} Risk
                            </span>
                            {risk?.confidence_score != null && (
                              <span className="text-[10px] font-mono text-slate-600">
                                {Math.round(risk.confidence_score * 100)}% conf
                              </span>
                            )}
                            {bindPct != null && (
                              <span className="text-[10px] font-mono text-sky-400 ml-auto">
                                {bindPct}% bind
                              </span>
                            )}
                          </div>

                          {/* LLM reason (truncated) */}
                          {routing?.reason && (
                            <p className="text-[10px] text-slate-500 leading-relaxed line-clamp-2 italic">
                              &ldquo;{routing.reason}&rdquo;
                            </p>
                          )}

                          {/* Action items */}
                          {routing?.action_items && routing.action_items.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-1">
                              {routing.action_items.slice(0, 2).map((item, i) => (
                                <span
                                  key={i}
                                  className="text-[9px] font-mono text-amber-500/70 border border-amber-500/20 rounded px-1.5 py-0.5"
                                >
                                  {item}
                                </span>
                              ))}
                            </div>
                          )}

                          <div className="mt-1.5 text-[9px] font-mono text-slate-700 text-right">
                            {(record.elapsed / 1000).toFixed(1)}s
                          </div>
                        </motion.div>
                      );
                    })}
                  </AnimatePresence>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* ── Footer ──────────────────────────────────────────────────────── */}
        <div className="mt-6 flex items-center justify-between text-[10px] font-mono text-slate-800">
          <span>InsurTech AI Pipeline · Agent Operations Dashboard · v2.0.0</span>
          <span>API: {API_BASE}</span>
        </div>

      </div>
    </main>
  );
}
