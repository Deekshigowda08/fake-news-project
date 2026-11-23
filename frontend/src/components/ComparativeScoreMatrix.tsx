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
  const items: ComparativeScoreItem[] = scores && scores.length > 0 ? scores : [
    {
      model_key: "cnn_lstm",
      model_name: "CNN + BiLSTM Classifier",
      modality: "Text (Word Embeddings)",
      predicted_class: bert?.predicted_class || "real",
      p_real: bert?.class_probabilities?.real ?? 0.5,
      p_fake: bert?.class_probabilities?.fake ?? 0.5,
      confidence: bert?.confidence ?? 0.5,
      fine_grained_class: null,
      ensemble_weight: 0.25,
      backbone: "TextCNN (k=[2,3,4,5]) + BiLSTM (2-layer, 256-d)",
    },
    {
      model_key: "bert",
      model_name: "BERT Contextual Classifier",
      modality: "Text (WordPiece)",
      predicted_class: bert?.predicted_class || "real",
      p_real: bert?.class_probabilities?.real ?? 0.5,
      p_fake: bert?.class_probabilities?.fake ?? 0.5,
      confidence: bert?.confidence ?? 0.5,
      fine_grained_class: (bert as any)?.fine_grained_class,
      ensemble_weight: hasImage ? 0.25 : 0.40,
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
            ensemble_weight: 0.15,
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
      ensemble_weight: hasImage ? 0.35 : 0.35,
      backbone: "Text Embeddings + Image Patches -> Dense MLP",
    },
  ];

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-5 sm:p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 rounded-full bg-white" />
            <h3 className="text-base sm:text-lg font-bold text-white tracking-tight">
              Multi-Model Intelligence Matrix
            </h3>
          </div>
          <p className="text-xs text-zinc-400 mt-1">
            Comparative predictions across CNN+LSTM, BERT contextual transformer, and multimodal heads.
          </p>
        </div>
        <div className="rounded border border-zinc-700 bg-zinc-900 px-2.5 py-1 text-xs font-mono font-medium text-zinc-300">
          Ensemble: Soft Voting
        </div>
      </div>

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {items.map((item) => {
          const isFake = item.predicted_class === "fake";
          const isCNNLSTM = item.model_key === "cnn_lstm";

          return (
            <div
              key={item.model_key}
              className={`rounded-xl border p-4 flex flex-col justify-between transition-colors ${
                isCNNLSTM
                  ? "border-zinc-500 bg-zinc-900/90"
                  : "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
              }`}
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-2">
                  <span className={`text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded ${
                    isCNNLSTM
                      ? "bg-white text-black font-bold"
                      : "bg-zinc-800 text-zinc-300 border border-zinc-700"
                  }`}>
                    {isCNNLSTM ? "PyTorch Model" : item.modality}
                  </span>
                  <span className="text-[11px] font-mono text-zinc-400">
                    Wt: {Math.round(item.ensemble_weight * 100)}%
                  </span>
                </div>

                <h4 className="text-sm font-bold text-white mb-0.5">{item.model_name}</h4>
                <p className="text-[11px] text-zinc-400 font-mono mb-3 line-clamp-1" title={item.backbone}>
                  {item.backbone}
                </p>

                {/* Predicted Pill */}
                <div className="flex items-center justify-between rounded-lg bg-black border border-zinc-800 p-2.5 mb-3">
                  <span className="text-xs text-zinc-400 font-medium">Prediction:</span>
                  <span
                    className={`text-xs font-bold px-2 py-0.5 rounded ${
                      isFake
                        ? "bg-zinc-800 text-white border border-zinc-600"
                        : "bg-white text-black border border-white"
                    }`}
                  >
                    {isFake ? "FAKE" : "REAL"}
                  </span>
                </div>

                {/* Probability Bars */}
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-[11px] font-medium mb-1">
                      <span className="text-zinc-300">P(Real):</span>
                      <span className="font-mono text-zinc-200">{(item.p_real * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-white rounded-full transition-all duration-300"
                        style={{ width: `${item.p_real * 100}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] font-medium mb-1">
                      <span className="text-zinc-400">P(Fake):</span>
                      <span className="font-mono text-zinc-300">{(item.p_fake * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-zinc-500 rounded-full transition-all duration-300"
                        style={{ width: `${item.p_fake * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-zinc-800 flex items-center justify-between text-[11px]">
                <span className="text-zinc-400 font-medium">Confidence:</span>
                <span className="font-mono font-bold text-white">{(item.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Ensemble Summary Footer */}
      {ensemble && (
        <div className="rounded-xl border border-zinc-700 bg-zinc-900 p-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white text-black font-black text-sm">
              ∑
            </div>
            <div>
              <h5 className="text-xs font-bold text-white uppercase tracking-wider">
                Consensus Ensemble Verdict
              </h5>
              <p className="text-xs text-zinc-400">
                Combined neural probability across all active model architectures.
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <div className="text-right">
              <span className="text-[11px] text-zinc-400 block">Confidence</span>
              <span className="font-mono text-sm font-bold text-white">
                {(ensemble.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <span
              className={`text-xs font-bold uppercase px-3 py-1.5 rounded-lg border ${
                ensemble.predicted_class === "fake"
                  ? "bg-zinc-800 text-white border-zinc-600"
                  : "bg-white text-black border-white"
              }`}
            >
              {ensemble.predicted_class === "fake" ? "PREDICTED FAKE" : "PREDICTED REAL"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
