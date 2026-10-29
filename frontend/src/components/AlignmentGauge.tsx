"use client";

import React from "react";
import { SemanticAlignment } from "../lib/types";

interface AlignmentGaugeProps {
  alignment?: SemanticAlignment | null;
  hasImage: boolean;
}

export function AlignmentGauge({ alignment, hasImage }: AlignmentGaugeProps) {
  if (!hasImage || !alignment || alignment.alignment_score === null) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 sm:p-6 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-xl">🖼️</span>
            <h4 className="text-base sm:text-lg font-bold text-white">CLIP Cross-Modal Semantic Alignment</h4>
          </div>
          <span className="rounded-lg bg-slate-800 px-3 py-1 text-xs font-bold text-slate-300">
            Image Not Provided
          </span>
        </div>
        <p className="text-sm text-slate-300 leading-relaxed">
          Cross-modal vision-language alignment evaluates semantic consistency between the headline text vector and the image vector. When an image is uploaded, OpenAI CLIP computes cosine similarity between both latent embeddings.
        </p>
      </div>
    );
  }

  const score = alignment.alignment_score;
  const percentage = Math.min(100, Math.max(0, Math.round(((score + 0.2) / 0.8) * 100)));

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 sm:p-6 backdrop-blur-md space-y-4 shadow-md">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🔬</span>
          <div>
            <h4 className="text-base sm:text-lg font-bold text-white">CLIP Text-to-Image Vector Cosine Alignment</h4>
            <p className="text-xs sm:text-sm text-indigo-300 font-mono">Backbone Model: {alignment.model_name}</p>
          </div>
        </div>

        <div className="text-right">
          <span className="text-xl sm:text-2xl font-black font-mono text-cyan-400">
            {score.toFixed(3)}
          </span>
          <span className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">Cosine Similarity</span>
        </div>
      </div>

      {/* Progress Gauge */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs sm:text-sm text-slate-300 font-medium">
          <span>Unrelated Media (-0.20)</span>
          <span className="text-cyan-300 font-bold bg-cyan-500/10 px-2.5 py-0.5 rounded-md border border-cyan-500/20">
            {alignment.interpretation}
          </span>
          <span>Strong Semantic Match (+0.80)</span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-slate-800/90 border border-slate-700/80">
          <div
            className="h-full bg-gradient-to-r from-rose-500 via-indigo-500 to-cyan-400 transition-all duration-700"
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>

      {/* Critical Scientific Caveat Alert */}
      <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3.5 text-xs sm:text-sm text-amber-100 flex gap-3 items-start">
        <svg
          className="w-5 h-5 text-amber-400 shrink-0 mt-0.5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <div className="leading-relaxed">
          <span className="font-bold text-amber-300">Methodological Clarification: </span>
          <span>{alignment.caveat}</span>
        </div>
      </div>
    </div>
  );
}
