"use client";

import React from "react";
import { AdjudicationVerdict } from "../lib/types";

interface PredictionBadgeProps {
  verdict: AdjudicationVerdict["verdict"];
  size?: "sm" | "md" | "lg";
}

export function PredictionBadge({ verdict, size = "md" }: PredictionBadgeProps) {
  const configs = {
    TRUE: {
      bg: "bg-emerald-500/15 border-emerald-500/30 text-emerald-300",
      iconBg: "bg-emerald-500/20 text-emerald-400",
      label: "VERIFIED TRUE",
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
        </svg>
      ),
      desc: "Corroborated by authoritative Tier-1/Tier-2 primary sources.",
    },
    FALSE: {
      bg: "bg-rose-500/15 border-rose-500/30 text-rose-300",
      iconBg: "bg-rose-500/20 text-rose-400",
      label: "FABRICATED / FALSE",
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
        </svg>
      ),
      desc: "Directly contradicted or debunked by authoritative investigations.",
    },
    MISLEADING: {
      bg: "bg-amber-500/15 border-amber-500/30 text-amber-300",
      iconBg: "bg-amber-500/20 text-amber-400",
      label: "MISLEADING / OUT OF CONTEXT",
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2.5}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
      ),
      desc: "Contains partial truths presented with deceptive context or altered framing.",
    },
    UNVERIFIABLE: {
      bg: "bg-slate-500/15 border-slate-500/30 text-slate-300",
      iconBg: "bg-slate-500/20 text-slate-400",
      label: "UNVERIFIABLE",
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2.5}
            d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
      desc: "Insufficient independent reporting or conflicting non-authoritative claims.",
    },
  };

  const cfg = configs[verdict] || configs.UNVERIFIABLE;

  if (size === "sm") {
    return (
      <span
        className={`inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-bold border shadow-sm ${cfg.bg}`}
      >
        <span className="shrink-0">{cfg.icon}</span>
        <span>{cfg.label}</span>
      </span>
    );
  }

  if (size === "lg") {
    return (
      <div className={`flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-5 rounded-2xl border-2 p-6 sm:p-7 shadow-2xl backdrop-blur-md ${cfg.bg}`}>
        <div className={`flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl shadow-lg ${cfg.iconBg}`}>
          <div className="scale-150 transform">
            {cfg.icon}
          </div>
        </div>
        <div className="space-y-1.5 flex-1">
          <div className="flex flex-wrap items-center gap-2.5">
            <span className="text-2xl sm:text-3xl font-black tracking-tight text-white">{cfg.label}</span>
            <span className="rounded-full bg-slate-900/60 px-3 py-1 text-xs uppercase font-mono font-bold tracking-wider border border-white/20 text-slate-200">
              Final Consensus Verdict
            </span>
          </div>
          <p className="text-base sm:text-lg text-slate-100 font-medium leading-relaxed">{cfg.desc}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex items-center gap-4 rounded-xl border-2 p-4 sm:p-5 ${cfg.bg}`}>
      <div className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl ${cfg.iconBg}`}>
        <div className="scale-125 transform">
          {cfg.icon}
        </div>
      </div>
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold tracking-tight text-white">{cfg.label}</span>
          <span className="rounded bg-slate-900/50 px-2 py-0.5 text-xs uppercase font-mono font-bold border border-white/10 text-slate-200">
            Adjudicated
          </span>
        </div>
        <p className="text-sm opacity-95 text-slate-200 font-medium">{cfg.desc}</p>
      </div>
    </div>
  );
}

export function MLVerdictPill({
  predictedClass,
  confidence,
}: {
  predictedClass: "real" | "fake" | "REAL_CLASSIFIED" | "FAKE_CLASSIFIED";
  confidence?: number;
}) {
  const isReal = predictedClass.toLowerCase().includes("real");
  return (
    <div
      className={`inline-flex items-center gap-2.5 rounded-xl border px-3.5 py-2 text-sm font-bold shadow-sm ${
        isReal
          ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
          : "border-rose-500/40 bg-rose-500/15 text-rose-300"
      }`}
    >
      <span className={`h-2.5 w-2.5 rounded-full ${isReal ? "bg-emerald-400" : "bg-rose-400"}`} />
      <span>
        ML Model: {isReal ? "REAL / AUTHENTIC" : "FAKE / UNRELIABLE"}
      </span>
      {confidence !== undefined && (
        <span className="text-slate-300 font-mono text-xs">
          ({(confidence * 100).toFixed(1)}% confidence)
        </span>
      )}
    </div>
  );
}

export function AgreementIndicator({
  agreement,
}: {
  agreement: "agrees" | "disagrees" | "unresolved";
}) {
  const styles = {
    agrees: {
      badge: "border-emerald-500/40 bg-emerald-500/15 text-emerald-300",
      text: "ML patterns & External Evidence Agree",
      icon: "✓",
    },
    disagrees: {
      badge: "border-amber-500/40 bg-amber-500/15 text-amber-300",
      text: "Conflict: ML features contradict retrieved evidence",
      icon: "⚠",
    },
    unresolved: {
      badge: "border-slate-500/40 bg-slate-500/15 text-slate-300",
      text: "Unresolved / Insufficient evidence cross-check",
      icon: "ℹ",
    },
  };

  const style = styles[agreement] || styles.unresolved;

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-xl border px-3.5 py-2 text-sm font-bold shadow-sm ${style.badge}`}
    >
      <span className="text-base font-extrabold">{style.icon}</span>
      <span>{style.text}</span>
    </span>
  );
}
