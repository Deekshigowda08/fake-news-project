"use client";

import React from "react";
import { ModelPrediction } from "../lib/types";

interface ProbabilityChartProps {
  title: string;
  subtitle: string;
  prediction?: ModelPrediction | null;
  modalityUsed?: string;
  isTextOnlyFallback?: boolean;
}

export function ProbabilityChart({
  title,
  subtitle,
  prediction,
  modalityUsed,
  isTextOnlyFallback,
}: ProbabilityChartProps) {
  if (!prediction || prediction.status === "failed") {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-slate-200">{title}</h4>
          <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400">
            Unavailable
          </span>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          {prediction?.error_detail || "Model inference was not executed for this input."}
        </p>
      </div>
    );
  }

  const realProb = prediction.class_probabilities?.real ?? 0.5;
  const fakeProb = prediction.class_probabilities?.fake ?? 0.5;
  const isReal = prediction.predicted_class === "real";
  const confidence = (prediction.confidence * 100).toFixed(1);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-sm">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-white">{title}</h4>
            {modalityUsed && (
              <span className="rounded bg-indigo-500/15 border border-indigo-500/20 px-1.5 py-0.5 text-[10px] font-medium text-indigo-300">
                {modalityUsed}
              </span>
            )}
            {isTextOnlyFallback && (
              <span className="rounded bg-amber-500/15 border border-amber-500/20 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">
                Text-Only Mode
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400">{subtitle}</p>
        </div>

        {/* Prediction Tag */}
        <div
          className={`shrink-0 rounded-md px-2.5 py-1 text-xs font-semibold border ${
            isReal
              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
              : "border-rose-500/30 bg-rose-500/10 text-rose-300"
          }`}
        >
          {isReal ? "REAL" : "FAKE"} ({confidence}%)
        </div>
      </div>

      {/* Probability Dual Bar */}
      <div className="mt-4 space-y-2">
        <div className="flex items-center justify-between text-xs font-medium">
          <span className="text-emerald-400">Real: {(realProb * 100).toFixed(1)}%</span>
          <span className="text-rose-400">Fake: {(fakeProb * 100).toFixed(1)}%</span>
        </div>

        <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800 flex">
          <div
            className="h-full bg-emerald-500 transition-all duration-500"
            style={{ width: `${Math.round(realProb * 100)}%` }}
            title={`Real: ${(realProb * 100).toFixed(1)}%`}
          />
          <div
            className="h-full bg-rose-500 transition-all duration-500"
            style={{ width: `${Math.round(fakeProb * 100)}%` }}
            title={`Fake: ${(fakeProb * 100).toFixed(1)}%`}
          />
        </div>
      </div>

      {/* Fine-Grained 6-Class Fakeddit Distribution (If present, e.g. for BERT) */}
      {(prediction as any).fine_grained_probabilities && (
        <div className="mt-3 pt-3 border-t border-slate-800/80 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
              6-Class Fine-Grained Distribution
            </span>
            {(prediction as any).fine_grained_class && (
              <span className="rounded bg-indigo-500/10 border border-indigo-500/20 px-1.5 py-0.2 text-[10px] font-mono text-indigo-300">
                {(prediction as any).fine_grained_class.toUpperCase()}
              </span>
            )}
          </div>
          <div className="grid grid-cols-3 gap-1.5 text-[10px]">
            {Object.entries((prediction as any).fine_grained_probabilities).map(([lbl, pVal]: [string, any]) => {
              const pct = (Number(pVal) * 100).toFixed(1);
              const isTop = (prediction as any).fine_grained_class === lbl;
              return (
                <div
                  key={lbl}
                  className={`rounded p-1 border text-center transition-all ${
                    isTop
                      ? "border-indigo-500/40 bg-indigo-500/15 text-indigo-200 font-semibold"
                      : "border-slate-800/60 bg-slate-950/40 text-slate-400"
                  }`}
                >
                  <span className="block truncate capitalize">{lbl.replace("_", " ")}</span>
                  <span className="font-mono text-[9px] text-slate-300">{pct}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Transparent Feature Statistics */}
      {((prediction as any).preprocessing_details || (prediction as any).visual_feature_details) && (
        <div className="mt-2.5 pt-2 border-t border-slate-800/60 flex flex-wrap items-center gap-2 text-[10px] text-slate-400">
          {(prediction as any).preprocessing_details && (
            <>
              <span className="rounded bg-slate-800/70 px-1.5 py-0.5 font-mono text-slate-300">
                {(prediction as any).preprocessing_details.token_count} Tokens
              </span>
              <span className="rounded bg-slate-800/70 px-1.5 py-0.5 font-mono text-slate-300">
                BERT 768-d CLS
              </span>
              <span className="rounded bg-slate-800/70 px-1.5 py-0.5 text-slate-300">
                Sensationalism: {Math.round((prediction as any).preprocessing_details.sensationalism_score * 100)}%
              </span>
            </>
          )}
          {(prediction as any).visual_feature_details && (
            <>
              <span className="rounded bg-slate-800/70 px-1.5 py-0.5 font-mono text-slate-300">
                {(prediction as any).visual_feature_details.vit_patch_grid}
              </span>
              <span className="rounded bg-slate-800/70 px-1.5 py-0.5 font-mono text-slate-300">
                ViT 768-d Vector
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
}

