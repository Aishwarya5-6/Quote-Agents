"use client";

import { useState } from "react";
import type { QuoteInput } from "@/lib/api-contract";

// ─────────────────────────────────────────────────────────────────────────────
//  Field definitions for the quote form
// ─────────────────────────────────────────────────────────────────────────────
const FIELDS: {
  key: keyof QuoteInput;
  label: string;
  min?: number;
  max?: number;
  step?: number;
  type: "number" | "select";
  options?: { label: string; value: string | number }[];
  required?: boolean;
  defaultValue: number | string;
}[] = [
  { key: "Driver_Age",       label: "Driver Age",        type: "number", min: 16,  max: 100, required: true, defaultValue: 34 },
  { key: "Driving_Exp",      label: "Driving Experience", type: "number", min: 0,   max: 84,  required: true, defaultValue: 12 },
  { key: "Prev_Accidents",   label: "Prior Accidents",   type: "number", min: 0,   max: 20,  required: true, defaultValue: 0 },
  { key: "Prev_Citations",   label: "Prior Citations",   type: "number", min: 0,   max: 20,  required: true, defaultValue: 1 },
  { key: "Annual_Miles",     label: "Annual Miles",      type: "number", min: 0,   max: 200000, step: 1000, required: true, defaultValue: 22000 },
  {
    key: "Veh_Usage", label: "Vehicle Usage", type: "select", required: true, defaultValue: "Pleasure",
    options: [
      { label: "Pleasure",  value: "Pleasure" },
      { label: "Commute",   value: "Commute" },
      { label: "Business",  value: "Business" },
    ],
  },
  { key: "Quoted_Premium",    label: "Quoted Premium ($)", type: "number", min: 0, step: 10, defaultValue: 750 },
  {
    key: "Sal_Range", label: "Salary Range", type: "select", defaultValue: 2,
    options: [
      { label: "≤$25K",     value: 0 },
      { label: "$25K–$40K", value: 1 },
      { label: "$40K–$60K", value: 2 },
      { label: "$60K–$90K", value: 3 },
      { label: ">$90K",     value: 4 },
    ],
  },
  {
    key: "Coverage", label: "Coverage Level", type: "select", defaultValue: 1,
    options: [
      { label: "Basic",    value: 0 },
      { label: "Balanced", value: 1 },
      { label: "Enhanced", value: 2 },
    ],
  },
  {
    key: "Vehicl_Cost_Range", label: "Vehicle Cost", type: "select", defaultValue: 2,
    options: [
      { label: "≤$10K",     value: 0 },
      { label: "$10K–$20K", value: 1 },
      { label: "$20K–$30K", value: 2 },
      { label: "$30K–$40K", value: 3 },
      { label: ">$40K",     value: 4 },
    ],
  },
  {
    key: "Re_Quote", label: "Re-Quote?", type: "select", defaultValue: 0,
    options: [
      { label: "No",  value: 0 },
      { label: "Yes", value: 1 },
    ],
  },
];

// ─────────────────────────────────────────────────────────────────────────────
//  Build default values from field definitions
// ─────────────────────────────────────────────────────────────────────────────
function buildDefaults(): Record<string, string> {
  const defaults: Record<string, string> = {};
  for (const f of FIELDS) {
    defaults[f.key] = String(f.defaultValue);
  }
  return defaults;
}

// ─────────────────────────────────────────────────────────────────────────────
//  QuoteForm component
// ─────────────────────────────────────────────────────────────────────────────
interface QuoteFormProps {
  onSubmit: (data: QuoteInput) => void;
  disabled?: boolean;
}

export default function QuoteForm({ onSubmit, disabled }: QuoteFormProps) {
  const [values, setValues] = useState<Record<string, string>>(buildDefaults);

  const handleChange = (key: string, val: string) => {
    setValues((prev) => ({ ...prev, [key]: val }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const result: Record<string, unknown> = {};
    for (const f of FIELDS) {
      const raw = values[f.key];
      if (f.type === "select" && f.key === "Veh_Usage") {
        result[f.key] = raw;
      } else {
        const n = Number(raw);
        if (!isNaN(n) && raw !== "") {
          result[f.key] = n;
        }
      }
    }

    onSubmit(result as unknown as QuoteInput);
  };

  const inputCls =
    "w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 " +
    "font-mono tabular-nums focus:outline-none focus:border-violet-500/60 focus:ring-1 focus:ring-violet-500/30 " +
    "placeholder:text-slate-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors";

  return (
    <form onSubmit={handleSubmit} className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <p className="text-[10px] font-mono tracking-[0.2em] text-slate-500 uppercase mb-4">
        Quote Input
      </p>

      <div className="grid grid-cols-2 gap-x-3 gap-y-3">
        {FIELDS.map((f) => (
          <div key={f.key} className={f.key === "Annual_Miles" ? "col-span-2" : ""}>
            <label className="block text-[10px] font-mono text-slate-500 mb-1 uppercase tracking-wider">
              {f.label} {f.required && <span className="text-violet-400">*</span>}
            </label>
            {f.type === "select" ? (
              <select
                value={values[f.key]}
                onChange={(e) => handleChange(f.key, e.target.value)}
                disabled={disabled}
                className={inputCls}
              >
                {f.options!.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                value={values[f.key]}
                onChange={(e) => handleChange(f.key, e.target.value)}
                min={f.min}
                max={f.max}
                step={f.step ?? 1}
                disabled={disabled}
                className={inputCls}
              />
            )}
          </div>
        ))}
      </div>

      <button
        type="submit"
        disabled={disabled}
        className="w-full mt-4 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-violet-600 hover:bg-violet-500 active:bg-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-semibold shadow-lg shadow-violet-500/20"
      >
        {disabled ? "Processing…" : "Run Analysis"}
      </button>
    </form>
  );
}
