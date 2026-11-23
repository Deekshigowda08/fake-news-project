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
      bg: "bg-black border-zinc-500 text-white",
      iconBg: "bg-white text-black",
      label: "VERIFIED AUTHENTIC",
      icon: (
        <svg className="w-5 h-5 font-bold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
        </svg>
      ),
      desc: "Corroborated by authoritative real-world primary sources and verified news records.",
    },
    FALSE: {
      bg: "bg-black border-zinc-700 text-white",
      iconBg: "bg-zinc-800 text-white border border-zinc-600",
      label: "CONFIRMED FABRICATED / FAKE",
      icon: (
        <svg className="w-5 h-5 font-bold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
        </svg>
      ),
      desc: "Directly contradicted or debunked by authoritative investigative sources.",
    },
    MISLEADING: {
      bg: "bg-black border-zinc-700 text-zinc-200",
      iconBg: "bg-zinc-800 text-zinc-300 border border-zinc-600",
      label: "MISLEADING / OUT OF CONTEXT",
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2.5}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
      ),
      desc: "Contains factual elements presented with deceptive framing or inaccurate attribution.",
    },
    UNVERIFIABLE: {
      bg: "bg-black border-zinc-800 text-zinc-300",
      iconBg: "bg-zinc-900 text-zinc-400 border border-zinc-700",
      label: "INSUFFICIENT EVIDENCE",
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2.5}
            d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
      desc: "No conclusive primary or secondary records retrieved across news wires to adjudicate claims.",
    },
  };

  const config = configs[verdict] || configs.UNVERIFIABLE;

  return (
    <div className={`rounded-xl border p-5 sm:p-6 ${config.bg} space-y-3`}>
      <div className="flex items-center space-x-3.5">
        <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${config.iconBg}`}>
          {config.icon}
        </div>
        <div>
          <span className="text-[10px] font-mono tracking-wider text-zinc-400 uppercase font-medium">
            Evidence Verification Verdict
          </span>
          <h3 className="text-xl sm:text-2xl font-bold tracking-tight text-white">{config.label}</h3>
        </div>
      </div>
      <p className="text-xs sm:text-sm text-zinc-400 pl-1">{config.desc}</p>
    </div>
  );
}

export function MLVerdictPill({ predictedClass, confidence }: { predictedClass: string; confidence: number }) {
  const isFake = predictedClass.toLowerCase() === "fake";
  return (
    <div
      className={`inline-flex items-center space-x-2 rounded-lg px-3 py-1.5 border text-xs font-semibold ${
        isFake
          ? "bg-zinc-900 border-zinc-700 text-white"
          : "bg-white border-white text-black"
      }`}
    >
      <span className={`h-2 w-2 rounded-full ${isFake ? "bg-zinc-400" : "bg-black"}`} />
      <span>Model Prediction: {isFake ? "FAKE" : "REAL"}</span>
      <span className={isFake ? "font-mono text-zinc-400" : "font-mono text-zinc-600"}>
        ({Math.round(confidence * 100)}%)
      </span>
    </div>
  );
}

export function AgreementIndicator({ agreement }: { agreement: "agrees" | "disagrees" | "unresolved" }) {
  const configs = {
    agrees: {
      color: "text-zinc-200 border-zinc-700 bg-zinc-900",
      text: "Consensus Alignment: Neural ML Prior Corroborates Evidence",
    },
    disagrees: {
      color: "text-zinc-200 border-zinc-700 bg-zinc-900",
      text: "Divergence Detected: Neural Prior Conflicts With Real-World Evidence",
    },
    unresolved: {
      color: "text-zinc-400 border-zinc-800 bg-zinc-950",
      text: "Consensus Pending Further Investigation",
    },
  };

  const c = configs[agreement] || configs.unresolved;

  return (
    <div className={`rounded-lg border px-3 py-2 text-xs font-medium ${c.color} flex items-center gap-2`}>
      <span className="h-1.5 w-1.5 rounded-full bg-zinc-400" />
      <span>{c.text}</span>
    </div>
  );
}
