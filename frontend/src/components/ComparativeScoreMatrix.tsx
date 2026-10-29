"use client";

import React from "react";
import { ComparativeScoreItem, ModelPrediction, EnsemblePrediction } from "../lib/types";

interface ComparativeScoreMatrixProps {
  scores?: ComparativeScoreItem[];
  ensemble?: EnsemblePrediction;
  bert?: ModelPrediction;
  vit?: ModelPrediction;
  fusion?: ModelPrediction;
  hasImage: boolean;
}

export function ComparativeScoreMatrix({
  scores,
  ensemble,
  bert,
  vit,
  fusion,
  hasImage,
}: ComparativeScoreMatrixProps) {
  // Build fallback comparative scores if not provided directly
  const items: ComparativeScoreItem[] = scores && scores.length > 0 ? scores : [
    {
      model_key: "bert",
      model_name: "BERT Text Classifier",
      modality: "Text (WordPiece)",
      predicted_class: bert?.predicted_class || "real",
      p_real: bert?.class_probabilities?.real ?? 0.5,
      p_fake: bert?.class_probabilities?.fake ?? 0.5,
      confidence: bert?.confidence ?? 0.5,
      fine_grained_class: (bert as any)?.fine_grained_class,
      ensemble_weight: hasImage ? 0.35 : 0.44,
      backbone: "bert-base-uncased (110M params)",
    },
    ...(hasImage && vit && vit.status === "success"
      ? [
          {
            model_key: "vit",
            model_name: "Vision Transformer (ViT)",
            modality: "Image (196 Patches)",
            predicted_class: vit?.predicted_class || "real",
            p_real: vit?.class_probabilities?.real ?? 0.5,
            p_fake: vit?.class_probabilities?.fake ?? 0.5,
            confidence: vit?.confidence ?? 0.5,
            fine_grained_class: null,
            ensemble_weight: 0.20,
            backbone: "google/vit-base-patch16-224 (86M params)",
          },
        ]
      : []),
    {
      model_key: "multimodal_fusion",
      model_name: "Multimodal Feature Fusion",
      modality: hasImage ? "Cross-Modal Concatenation (1536-d)" : "Text-Only Dynamic Routing",
      predicted_class: fusion?.predicted_class || "real",
      p_real: fusion?.class_probabilities?.real ?? 0.5,
      p_fake: fusion?.class_probabilities?.fake ?? 0.5,
      confidence: fusion?.confidence ?? 0.5,
      fine_grained_class: null,
      ensemble_weight: hasImage ? 0.45 : 0.56,
      backbone: "BERT [768-d] + ViT [768-d] -> MLP Head",
    },
    {
      model_key: "ensemble",
      model_name: "Soft-Voting Consensus Ensemble",
      modality: "Probability-Weighted Meta-Model",
      predicted_class: ensemble?.predicted_class || "real",
      p_real: ensemble?.class_probabilities?.real ?? 0.5,
      p_fake: ensemble?.class_probabilities?.fake ?? 0.5,
      confidence: ensemble?.confidence ?? 0.5,
      fine_grained_class: null,
      ensemble_weight: 1.0,
      backbone: "Soft Voting Consensus",
    },
  ];

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 sm:p-7 backdrop-blur-md space-y-6 shadow-xl">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-3">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-500/15 border border-indigo-500/30 text-sm font-mono font-bold text-indigo-400">
              VS
            </span>
            <h3 className="text-lg sm:text-xl font-bold text-white">
              Comparative Model Performance & Score Matrix
            </h3>
          </div>
          <p className="text-sm text-slate-300 mt-1">
            Side-by-side evaluation comparing base transformers, feature concatenation, and soft-voting ensemble
          </p>
        </div>

        <span className="rounded-xl bg-indigo-500/15 border border-indigo-500/30 px-3 py-1.5 text-sm font-mono font-bold text-indigo-300">
          Ensemble Strategy: {ensemble?.ensemble_method || "Soft Voting Consensus"}
        </span>
      </div>

      {/* Comparison Table for Academic Evaluation */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm border-collapse">
          <thead>
            <tr className="border-b border-slate-700 text-xs sm:text-sm font-extrabold uppercase tracking-wider text-slate-300 bg-slate-950/60">
              <th className="py-3.5 px-4">Model Architecture</th>
              <th className="py-3.5 px-4">Input Modality</th>
              <th className="py-3.5 px-4">Predicted Class</th>
              <th className="py-3.5 px-4">P(Real)</th>
              <th className="py-3.5 px-4">P(Fake)</th>
              <th className="py-3.5 px-4">Confidence</th>
              <th className="py-3.5 px-4">Ensemble Weight</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800 font-sans">
            {items.map((row) => {
              const isReal = row.predicted_class === "real";
              const isEnsemble = row.model_key === "ensemble";

              return (
                <tr
                  key={row.model_key}
                  className={`transition-colors ${
                    isEnsemble
                      ? "bg-indigo-950/30 font-semibold border-l-4 border-indigo-500"
                      : "hover:bg-slate-800/40"
                  }`}
                >
                  <td className="py-4 px-4">
                    <span className="text-white block font-bold text-base">
                      {row.model_name}
                    </span>
                    <span className="text-xs text-slate-400 font-mono block mt-0.5">
                      {row.backbone}
                    </span>
                  </td>

                  <td className="py-4 px-4">
                    <span className="rounded-md bg-slate-800 px-2.5 py-1 text-xs text-slate-200 font-mono font-medium">
                      {row.modality}
                    </span>
                  </td>

                  <td className="py-4 px-4">
                    <span
                      className={`inline-block rounded-lg px-3 py-1 text-xs font-black border ${
                        isReal
                          ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
                          : "border-rose-500/40 bg-rose-500/15 text-rose-300"
                      }`}
                    >
                      {isReal ? "REAL" : "FAKE"}
                    </span>
                    {row.fine_grained_class && (
                      <span className="block text-xs text-indigo-300 font-mono mt-1 font-bold uppercase">
                        ({row.fine_grained_class})
                      </span>
                    )}
                  </td>

                  <td className="py-4 px-4 font-mono font-bold text-base text-emerald-400">
                    {((row.p_real ?? 0.5) * 100).toFixed(1)}%
                  </td>

                  <td className="py-4 px-4 font-mono font-bold text-base text-rose-400">
                    {((row.p_fake ?? 0.5) * 100).toFixed(1)}%
                  </td>

                  <td className="py-4 px-4 font-mono text-white font-extrabold text-base">
                    {((row.confidence ?? 0.5) * 100).toFixed(1)}%
                  </td>

                  <td className="py-4 px-4 font-mono font-bold text-sm text-indigo-300">
                    {isEnsemble ? "Meta Consensus" : `${Math.round((row.ensemble_weight ?? 0.33) * 100)}%`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Comparative Analysis Takeaways */}
      <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-4 space-y-2 text-xs leading-relaxed text-slate-300">
        <h4 className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
          Academic Research Takeaways & Ablation Insights
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-slate-300">
          <div className="space-y-1">
            <span className="text-white font-semibold block">1. Modality Robustness:</span>
            <p className="text-[11px] text-slate-400">
              BERT excels at detecting semantic hyperbole and factual phrasing, while ViT evaluates visual texture anomalies. Combining them in feature concatenation yields stronger decision boundaries than either model independently.
            </p>
          </div>

          <div className="space-y-1">
            <span className="text-white font-semibold block">2. Soft-Voting Calibration:</span>
            <p className="text-[11px] text-slate-400">
              The ensemble dynamically re-weights base probabilities. When visual data is absent, the pipeline automatically re-allocates 100% of the voting weight between text networks to eliminate dummy visual bias.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
