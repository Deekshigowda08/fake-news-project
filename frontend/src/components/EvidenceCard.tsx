"use client";

import React from "react";
import { EvidenceCitation } from "../lib/types";

interface EvidenceCardProps {
  evidence: EvidenceCitation;
}

export function EvidenceCard({ evidence }: EvidenceCardProps) {
  const stanceConfig = {
    SUPPORTS: {
      border: "border-emerald-500/30",
      bg: "bg-emerald-500/10 text-emerald-300",
      label: "SUPPORTS CLAIM",
      icon: "✓",
    },
    CONTRADICTS: {
      border: "border-rose-500/30",
      bg: "bg-rose-500/10 text-rose-300",
      label: "CONTRADICTS / DEBUNKS",
      icon: "✗",
    },
    MIXED: {
      border: "border-amber-500/30",
      bg: "bg-amber-500/10 text-amber-300",
      label: "MIXED STANCE",
      icon: "~",
    },
    INSUFFICIENT: {
      border: "border-slate-500/30",
      bg: "bg-slate-500/10 text-slate-300",
      label: "INSUFFICIENT DETAILS",
      icon: "?",
    },
  };

  const tierConfig = {
    tier_1: {
      bg: "bg-cyan-500/15 border-cyan-500/30 text-cyan-300",
      label: "Tier 1: High Authority (Official / Primary Wire)",
    },
    tier_2: {
      bg: "bg-indigo-500/15 border-indigo-500/30 text-indigo-300",
      label: "Tier 2: Reputable News & Fact Checkers",
    },
    tier_3: {
      bg: "bg-slate-500/15 border-slate-500/30 text-slate-400",
      label: "Tier 3: Standard Web Source",
    },
  };

  const stanceStyle = stanceConfig[evidence.stance] || stanceConfig.INSUFFICIENT;
  const tierStyle = tierConfig[evidence.credibility_tier] || tierConfig.tier_3;

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 sm:p-6 transition-all hover:border-slate-700 hover:bg-slate-900/90 shadow-md space-y-3">
      {/* Top badges */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 border-b border-slate-800/80 pb-3">
        <div className="flex flex-wrap items-center gap-2">
          {/* Stance Badge */}
          <span
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1 text-xs sm:text-sm font-extrabold border ${stanceStyle.bg} ${stanceStyle.border}`}
          >
            <span className="text-sm">{stanceStyle.icon}</span>
            <span>{stanceStyle.label}</span>
          </span>

          {/* Credibility Tier Badge */}
          <span
            className={`inline-flex items-center rounded-lg border px-3 py-1 text-xs font-bold ${tierStyle.bg}`}
          >
            {tierStyle.label}
          </span>
        </div>

        <span className="text-xs sm:text-sm font-bold text-slate-300 bg-slate-800/80 px-2.5 py-1 rounded-md">
          {evidence.publisher}
        </span>
      </div>

      {/* Article Title */}
      <h5 className="text-base sm:text-lg font-bold text-white leading-snug">
        {evidence.title || "Retrieved Source Document"}
      </h5>

      {/* Verbatim Excerpt */}
      {evidence.excerpt && (
        <blockquote className="rounded-xl border-l-4 border-indigo-500 bg-slate-950/70 p-3.5 text-sm text-slate-200 italic leading-relaxed">
          "{evidence.excerpt}"
        </blockquote>
      )}

      {/* Source Citation Link */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-800/80 pt-3">
        <span className="text-xs font-mono font-semibold text-slate-400">
          Source Reference #{evidence.source_id}
        </span>

        <div className="flex items-center gap-4">
          {evidence.url && (
            <a
              href={evidence.url.startsWith("http") ? evidence.url : `https://${evidence.url}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-sm font-bold text-indigo-400 hover:text-indigo-300 hover:underline transition-colors"
              title="Open direct authoritative publication link"
            >
              <span>Visit Publication</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                />
              </svg>
            </a>
          )}

          <a
            href={`https://www.google.com/search?q=${encodeURIComponent(
              `${evidence.publisher || ""} ${evidence.title || "fact check"}`.trim()
            )}`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm font-medium text-slate-300 hover:text-white transition-colors"
            title="Search Google for this exact publication & fact check"
          >
            <span>Search Verification</span>
            <svg className="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </a>
        </div>
      </div>
    </div>
  );
}
