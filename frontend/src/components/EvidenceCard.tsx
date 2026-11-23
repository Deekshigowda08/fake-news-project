"use client";

import React from "react";
import { EvidenceCitation } from "../lib/types";

interface EvidenceCardProps {
  evidence: EvidenceCitation;
}

export function EvidenceCard({ evidence }: EvidenceCardProps) {
  const stanceConfig = {
    SUPPORTS: {
      border: "border-white",
      bg: "bg-white text-black",
      label: "SUPPORTS CLAIM",
      icon: "✓",
    },
    CONTRADICTS: {
      border: "border-zinc-600",
      bg: "bg-zinc-800 text-white",
      label: "CONTRADICTS / DEBUNKS",
      icon: "✕",
    },
    MIXED: {
      border: "border-zinc-700",
      bg: "bg-zinc-900 text-zinc-300",
      label: "MIXED STANCE",
      icon: "~",
    },
    INSUFFICIENT: {
      border: "border-zinc-800",
      bg: "bg-zinc-950 text-zinc-400",
      label: "INSUFFICIENT DETAILS",
      icon: "?",
    },
  };

  const tierConfig = {
    tier_1: {
      bg: "bg-zinc-900 border-zinc-700 text-zinc-200",
      label: "Tier 1: Primary Wire / Official",
    },
    tier_2: {
      bg: "bg-zinc-900 border-zinc-800 text-zinc-300",
      label: "Tier 2: Major News & Fact Checker",
    },
    tier_3: {
      bg: "bg-zinc-950 border-zinc-800 text-zinc-400",
      label: "Tier 3: Web Source",
    },
  };

  const stanceStyle = stanceConfig[evidence.stance] || stanceConfig.INSUFFICIENT;
  const tierStyle = tierConfig[evidence.credibility_tier] || tierConfig.tier_3;

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-5 transition-colors hover:border-zinc-700 space-y-3">
      {/* Top badges */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 border-b border-zinc-800 pb-3">
        <div className="flex flex-wrap items-center gap-2">
          {/* Stance Badge */}
          <span
            className={`inline-flex items-center gap-1.5 rounded px-2.5 py-0.5 text-xs font-bold border ${stanceStyle.bg} ${stanceStyle.border}`}
          >
            <span>{stanceStyle.icon}</span>
            <span>{stanceStyle.label}</span>
          </span>

          {/* Credibility Tier Badge */}
          <span
            className={`inline-flex items-center rounded border px-2.5 py-0.5 text-xs font-medium ${tierStyle.bg}`}
          >
            {tierStyle.label}
          </span>
        </div>

        <span className="text-xs font-semibold text-zinc-300 bg-zinc-900 border border-zinc-800 px-2.5 py-0.5 rounded">
          {evidence.publisher}
        </span>
      </div>

      {/* Article Title */}
      <h5 className="text-sm sm:text-base font-bold text-white leading-snug">
        {evidence.title || "Retrieved Source Document"}
      </h5>

      {/* Verbatim Excerpt */}
      {evidence.excerpt && (
        <blockquote className="rounded-lg border-l-2 border-white bg-zinc-900/60 p-3 text-xs sm:text-sm text-zinc-300 leading-relaxed">
          "{evidence.excerpt}"
        </blockquote>
      )}

      {/* Source Citation Link */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-zinc-800 pt-3 text-xs">
        <span className="font-mono text-zinc-500">
          Source #{evidence.source_id}
        </span>

        <div className="flex items-center gap-3">
          {evidence.url && (
            <a
              href={evidence.url.startsWith("http") ? evidence.url : `https://${evidence.url}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 font-semibold text-white hover:underline transition-colors"
              title="Open publication"
            >
              <span>Visit Article</span>
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
            className="inline-flex items-center gap-1 text-zinc-400 hover:text-white transition-colors"
          >
            <span>Search Web</span>
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </a>
        </div>
      </div>
    </div>
  );
}
