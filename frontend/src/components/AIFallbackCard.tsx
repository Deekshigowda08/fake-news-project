"use client";

import React, { useState } from "react";
import { AIAnswer } from "../lib/types";

interface AIFallbackCardProps {
  aiAnswer?: AIAnswer;
  ensembleConfidence?: number;
}

export function AIFallbackCard({ aiAnswer, ensembleConfidence }: AIFallbackCardProps) {
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  if (!aiAnswer) {
    return null;
  }

  const isTriggered = aiAnswer.is_triggered;
  const verdict = aiAnswer.verdict || "UNVERIFIABLE";
  const confidence = aiAnswer.confidence !== undefined && aiAnswer.confidence !== null
    ? Math.round(aiAnswer.confidence * 100)
    : 85;

  const getVerdictBadge = (v: string) => {
    const uv = v.toUpperCase();
    if (uv.includes("REAL") || uv.includes("TRUE")) {
      return {
        bg: "bg-white",
        border: "border-white",
        text: "text-black",
        icon: "✓",
        label: "AI VERDICT: VERIFIED REAL",
      };
    }
    if (uv.includes("FAKE") || uv.includes("FALSE")) {
      return {
        bg: "bg-zinc-900",
        border: "border-zinc-500",
        text: "text-white",
        icon: "✕",
        label: "AI VERDICT: DEBUNKED / FAKE",
      };
    }
    return {
      bg: "bg-zinc-800",
      border: "border-zinc-600",
      text: "text-zinc-200",
      icon: "—",
      label: `AI VERDICT: ${uv}`,
    };
  };

  const badge = getVerdictBadge(verdict);

  return (
    <div
      className={`rounded-2xl border transition-all ${
        isTriggered
          ? "border-zinc-600 bg-zinc-950 text-white"
          : "border-zinc-800 bg-zinc-950/60"
      }`}
    >
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-5 border-b border-zinc-800">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-900 border border-zinc-700 text-white font-bold text-sm">
            AI
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm sm:text-base font-bold text-white tracking-tight">
                AI Fact-Check & Reasoning Summary
              </h3>
              {isTriggered ? (
                <span className="rounded border border-zinc-600 bg-zinc-800 px-2 py-0.5 text-[10px] font-mono font-medium text-white">
                  Active
                </span>
              ) : (
                <span className="rounded bg-zinc-900 border border-zinc-800 px-2 py-0.5 text-[10px] font-medium text-zinc-400">
                  Standby
                </span>
              )}
            </div>
            <p className="text-[11px] text-zinc-400 mt-0.5">
              {isTriggered
                ? "Triggered automatically when neural models exhibit uncertainty or conflict"
                : "Neural models achieved high confidence with consensus"}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {aiAnswer.engine && (
            <span className="rounded bg-zinc-900 border border-zinc-800 px-2.5 py-1 text-[11px] font-mono text-zinc-400">
              {aiAnswer.engine}
            </span>
          )}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="rounded border border-zinc-800 bg-zinc-900 px-3 py-1 text-xs text-zinc-300 hover:text-white hover:bg-zinc-800 transition-colors cursor-pointer"
          >
            {isExpanded ? "Collapse ▲" : "Expand ▼"}
          </button>
        </div>
      </div>

      {/* Trigger alert notification */}
      {isTriggered && isExpanded && aiAnswer.trigger_reasons && aiAnswer.trigger_reasons.length > 0 && (
        <div className="bg-zinc-900 border-b border-zinc-800 px-5 py-3">
          <div className="space-y-1">
            <span className="text-xs font-semibold text-zinc-300">
              Intervention Trigger:
            </span>
            <ul className="text-xs text-zinc-400 space-y-0.5 list-disc list-inside">
              {aiAnswer.trigger_reasons.map((reason, idx) => (
                <li key={idx}>{reason}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Main Content Body */}
      {isExpanded && (
        <div className="p-5 sm:p-6 space-y-5">
          {isTriggered ? (
            <>
              {/* Verdict Row */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-xl bg-zinc-900 border border-zinc-800">
                <div className="flex items-center gap-3">
                  <div className={`px-3 py-1.5 rounded-lg border font-bold text-xs tracking-wide flex items-center gap-1.5 ${badge.bg} ${badge.border} ${badge.text}`}>
                    <span>{badge.icon}</span>
                    <span>{badge.label}</span>
                  </div>
                  <div className="text-xs text-zinc-400">
                    Confidence: <span className="text-white font-mono font-bold">{confidence}%</span>
                  </div>
                </div>

                {/* Confidence Bar */}
                <div className="w-full sm:w-48 space-y-1">
                  <div className="flex justify-between text-[10px] text-zinc-400">
                    <span>Certainty</span>
                    <span className="font-mono">{confidence}%</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-zinc-800 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-white transition-all duration-300"
                      style={{ width: `${confidence}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Executive Summary */}
              {aiAnswer.summary && (
                <div className="space-y-1">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-400">
                    Executive Finding
                  </h4>
                  <p className="text-sm font-medium text-white leading-relaxed bg-zinc-900/60 p-3.5 rounded-xl border border-zinc-800">
                    "{aiAnswer.summary}"
                  </p>
                </div>
              )}

              {/* Detailed Explanation */}
              {aiAnswer.detailed_explanation && (
                <div className="space-y-1.5">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-400">
                    Fact-Check Analysis
                  </h4>
                  <div className="text-xs sm:text-sm text-zinc-300 leading-relaxed bg-zinc-900/60 p-4 rounded-xl border border-zinc-800 space-y-2">
                    {aiAnswer.detailed_explanation.split("\n\n").map((para, i) => (
                      <p key={i}>{para}</p>
                    ))}
                  </div>
                </div>
              )}

              {/* Key Points */}
              {aiAnswer.key_points && aiAnswer.key_points.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-400">
                    Key Fact Anchors
                  </h4>
                  <div className="grid grid-cols-1 gap-2">
                    {aiAnswer.key_points.map((pt, idx) => (
                      <div
                        key={idx}
                        className="flex items-start gap-2.5 rounded-lg border border-zinc-800 bg-zinc-900/50 p-2.5 text-xs text-zinc-300"
                      >
                        <span className="text-white font-bold shrink-0">•</span>
                        <span>{pt}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Model Critique */}
              {aiAnswer.model_critique && (
                <div className="rounded-lg border border-zinc-800 bg-zinc-900/30 p-3 space-y-1">
                  <span className="text-[11px] font-bold text-zinc-400 uppercase tracking-wider">
                    Model Assessment (BERT / ViT / CNN+LSTM)
                  </span>
                  <p className="text-xs text-zinc-400 italic">
                    "{aiAnswer.model_critique}"
                  </p>
                </div>
              )}
            </>
          ) : (
            <div className="py-2 text-xs text-zinc-400">
              <p>
                {aiAnswer.message ||
                  "The neural models achieved high confidence and consistent consensus. AI fallback was not required."}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
