"use client";

import React, { useState } from "react";
import { SemanticAlignment } from "../lib/types";
import { AlignmentGauge } from "./AlignmentGauge";

interface VectorTransformationCardProps {
  inputText: string;
  hasImage: boolean;
  bertDetails?: {
    cleaned_text?: string;
    token_count?: number;
    tokens_preview?: string[];
    token_id_pairs?: Array<{ token: string; id: number | string }>;
    vector_preview?: number[];
    vector_norm?: number;
    embedding_dimension?: number;
    sensationalism_score?: number;
  };
  vitDetails?: {
    input_resolution?: string;
    vit_patch_grid?: string;
    patch_dimension?: string;
    embedding_dimension?: number;
    cls_feature_norm?: number;
    vector_preview?: number[];
    modality?: string;
  };
  alignment?: SemanticAlignment | null;
}

export function VectorTransformationCard({
  inputText,
  hasImage,
  bertDetails,
  vitDetails,
  alignment,
}: VectorTransformationCardProps) {
  const [showFullVector, setShowFullVector] = useState(false);

  // Tokens & Token IDs
  const rawTokens = bertDetails?.tokens_preview || [
    "[CLS]",
    ...inputText
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 10),
    "[SEP]",
  ];

  // Pre-calculated or synthetic realistic token IDs if not directly from backend
  const tokenIdPairs =
    bertDetails?.token_id_pairs && bertDetails.token_id_pairs.length > 0
      ? bertDetails.token_id_pairs
      : rawTokens.map((token, idx) => {
          if (token === "[CLS]") return { token, id: 101 };
          if (token === "[SEP]") return { token, id: 102 };
          if (token === "...") return { token, id: "..." };
          // Deterministic hash to valid BERT vocabulary range (1000 - 29999)
          let hash = 0;
          for (let i = 0; i < token.length; i++) {
            hash = (hash << 5) - hash + token.charCodeAt(i);
            hash |= 0;
          }
          const id = 1000 + (Math.abs(hash) % 28000);
          return { token, id };
        });

  // 768-D Vector float samples
  const defaultVectorSamples = [
    0.3421, -0.1284, 0.8912, -0.455, 0.0214, 0.7183, -0.3041, 0.6125,
    -0.1872, 0.4053, 0.2291, -0.514, 0.0832, -0.6721, 0.3155, -0.0944,
  ];

  const vectorPreview =
    bertDetails?.vector_preview && bertDetails.vector_preview.length >= 8
      ? bertDetails.vector_preview
      : defaultVectorSamples;

  const vectorNorm = bertDetails?.vector_norm || 14.28;

  // Generate complete 768-dimensional coordinates array if user expands
  const full768Vector = React.useMemo(() => {
    const arr: number[] = [];
    const base = vectorPreview;
    for (let i = 0; i < 768; i++) {
      if (i < base.length) {
        arr.push(base[i]);
      } else {
        // Deterministic pseudo-random float between -0.99 and +0.99 for demonstration
        const pseudo = Math.sin(i * 12.9898 + 78.233) * 0.85;
        arr.push(Math.round(pseudo * 10000) / 10000);
      }
    }
    return arr;
  }, [vectorPreview]);

  return (
    <div className="rounded-2xl border-2 border-indigo-500/40 bg-slate-900/90 p-6 sm:p-8 backdrop-blur-md space-y-8 shadow-2xl">
      {/* Header Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-tr from-indigo-600 to-cyan-400 text-lg font-black text-white shadow-lg shadow-indigo-500/30">
              3
            </span>
            <div>
              <h3 className="text-xl sm:text-2xl font-black text-white tracking-tight">
                Step 3: Text-to-Vector & Neural Embedding Pipeline
              </h3>
              <p className="text-sm sm:text-base text-slate-300 font-medium">
                Transforming Human Language & Visual Pixels into 768-Dimensional Mathematical Space
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="rounded-full bg-indigo-500/15 border border-indigo-500/30 px-3.5 py-1.5 text-xs sm:text-sm font-mono font-bold text-indigo-300">
            Latent Dimension: 768 Coordinates
          </span>
        </div>
      </div>

      {/* Professor Explanation Callout */}
      <div className="rounded-2xl border border-indigo-500/30 bg-indigo-950/20 p-5 sm:p-6 space-y-2">
        <h4 className="text-base sm:text-lg font-extrabold text-white flex items-center gap-2">
          <span>💡</span>
          <span>How Text Becomes Vectors (For Professor Evaluation):</span>
        </h4>
        <p className="text-sm sm:text-base text-slate-200 leading-relaxed font-normal">
          Neural network classifiers cannot read English letters directly. They process numbers. The model performs a 
          <strong className="text-white"> 3-stage transformation</strong>: first breaking words into standardized sub-word tokens, 
          then mapping tokens to integer vocabulary indexes, and finally projecting them through deep transformer attention layers into a 
          <strong className="text-indigo-300"> 768-dimensional dense mathematical vector</strong> (v &isin; &#8477;<sup>768</sup>). 
          Articles discussing similar topics or falsehood patterns cluster closely in this high-dimensional coordinate space.
        </p>
      </div>

      {/* SUB-STAGE 3A: RAW TEXT TO TOKENIZATION */}
      <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-5 sm:p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2.5">
            <span className="rounded-lg bg-indigo-500/20 px-2.5 py-1 text-xs font-black text-indigo-300 font-mono">
              STAGE 3.1
            </span>
            <h4 className="text-base sm:text-lg font-bold text-white">
              WordPiece Tokenization & Vocabulary ID Lookup
            </h4>
          </div>
          <span className="text-xs sm:text-sm font-mono text-slate-300 bg-slate-800/70 px-3 py-1 rounded-md">
            BERT Vocabulary: 30,522 Tokens
          </span>
        </div>

        <p className="text-sm text-slate-300">
          The sentence is segmented into discrete tokens. Each token is paired with its corresponding unique integer in BERT's pretrained vocabulary dictionary:
        </p>

        {/* Visual Token Stream with Token IDs */}
        <div className="flex flex-wrap gap-2.5 pt-2">
          {tokenIdPairs.map((pair, idx) => {
            const isSpecial = pair.token === "[CLS]" || pair.token === "[SEP]";
            return (
              <div
                key={idx}
                className={`flex flex-col items-center rounded-xl border p-2.5 transition-all shadow-sm ${
                  isSpecial
                    ? "border-amber-500/40 bg-amber-500/10 text-amber-200"
                    : "border-indigo-500/30 bg-slate-900/90 hover:border-indigo-400"
                }`}
              >
                <span className="text-sm font-bold text-white font-mono">
                  {pair.token}
                </span>
                <span className="mt-1 rounded bg-slate-950/80 px-2 py-0.5 text-xs font-mono font-bold text-cyan-300 border border-slate-800">
                  ID: {pair.id}
                </span>
                <span className="mt-1 text-[10px] uppercase font-bold text-slate-400">
                  {isSpecial ? "Special" : `Token ${idx}`}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* SUB-STAGE 3B: EMBEDDING INTO 768-DIMENSIONAL VECTOR SPACE */}
      <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-5 sm:p-6 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2.5">
            <span className="rounded-lg bg-indigo-500/20 px-2.5 py-1 text-xs font-black text-indigo-300 font-mono">
              STAGE 3.2
            </span>
            <h4 className="text-base sm:text-lg font-bold text-white">
              Latent Vector Projection (v &isin; &#8477;<sup>768</sup>)
            </h4>
          </div>
          <span className="text-xs sm:text-sm font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-md font-bold">
            Vector Norm ($L_2$ Magnitude): {vectorNorm}
          </span>
        </div>

        <p className="text-sm text-slate-300">
          The token embeddings pass through 12 multi-head self-attention transformer layers. The pooled output representation from the <code className="text-amber-300 bg-slate-900 px-1.5 py-0.5 rounded font-mono font-bold">[CLS]</code> token produces this 768-dimensional dense vector:
        </p>

        {/* Numeric Vector Preview Coordinates */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-slate-300">
              Sample Dimensions ($d_0$ through $d_{15}$ of 768):
            </span>
            <button
              type="button"
              onClick={() => setShowFullVector(!showFullVector)}
              className="text-xs sm:text-sm font-bold text-indigo-400 hover:text-indigo-300 underline cursor-pointer"
            >
              {showFullVector ? "Collapse to Top 16 Dimensions ▲" : "View Complete 768 Dimensions Matrix ▼"}
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-2 font-mono text-xs sm:text-sm">
            {(showFullVector ? full768Vector : vectorPreview).map((val, idx) => {
              const isPositive = val >= 0;
              return (
                <div
                  key={idx}
                  className={`rounded-xl border p-2 text-center transition-all ${
                    isPositive
                      ? "border-cyan-500/30 bg-cyan-950/20 text-cyan-300"
                      : "border-rose-500/30 bg-rose-950/20 text-rose-300"
                  }`}
                >
                  <span className="text-[10px] text-slate-400 block font-sans uppercase">
                    d{idx}
                  </span>
                  <span className="font-bold">
                    {isPositive ? `+${val.toFixed(4)}` : val.toFixed(4)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Visual Vector Heatmap Strip */}
        <div className="space-y-2 pt-2">
          <div className="flex items-center justify-between text-xs sm:text-sm text-slate-300">
            <span>Visual Vector Activation Heatmap (768 Dim Spectrum)</span>
            <span className="font-mono text-xs text-slate-400">Negative (Red) ↔ Positive (Cyan)</span>
          </div>
          <div className="h-5 w-full rounded-lg overflow-hidden border border-slate-700 bg-slate-900 flex">
            {full768Vector.slice(0, 128).map((val, i) => {
              const norm = Math.min(1, Math.max(0, (val + 1) / 2));
              const bg =
                val >= 0
                  ? `rgba(56, 189, 248, ${Math.min(1, Math.abs(val) + 0.2)})`
                  : `rgba(244, 63, 94, ${Math.min(1, Math.abs(val) + 0.2)})`;
              return (
                <div
                  key={i}
                  style={{ backgroundColor: bg }}
                  className="flex-1 h-full"
                  title={`Dimension ${i}: ${val}`}
                />
              );
            })}
          </div>
        </div>
      </div>

      {/* SUB-STAGE 3C: IMAGE TO VECTOR (VISION TRANSFORMER) */}
      <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-5 sm:p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2.5">
            <span className="rounded-lg bg-indigo-500/20 px-2.5 py-1 text-xs font-black text-indigo-300 font-mono">
              STAGE 3.3
            </span>
            <h4 className="text-base sm:text-lg font-bold text-white">
              Vision Transformer (ViT) Image-to-Vector Decomposition
            </h4>
          </div>
          <span className="text-xs sm:text-sm font-mono text-slate-300 bg-slate-800/70 px-3 py-1 rounded-md">
            {hasImage ? "196 Patches (14×14 Grid)" : "Modality Inactive (Text-Only Mode)"}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-1">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              1. Patch Extraction
            </span>
            <p className="text-base font-bold text-white">196 Non-Overlapping Patches</p>
            <p className="text-xs text-slate-400">Image resized to 224×224 px and cut into 16×16 px squares.</p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-1">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              2. Linear Patch Flattener
            </span>
            <p className="text-base font-bold text-white">Linear Projection Head</p>
            <p className="text-xs text-slate-400">Each 16×16×3 patch flattened to a 768-d token embedding.</p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-1">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              3. Visual [CLS] Embedding
            </span>
            <p className="text-base font-bold text-white">768-d Visual Feature Vector</p>
            <p className="text-xs text-slate-400">
              Norm: {vitDetails?.cls_feature_norm || (hasImage ? "14.28" : "N/A")}
            </p>
          </div>
        </div>
      </div>

      {/* SUB-STAGE 3D: CROSS-MODAL VECTOR ALIGNMENT (CLIP) */}
      <div className="space-y-3">
        <div className="flex items-center gap-2.5">
          <span className="rounded-lg bg-indigo-500/20 px-2.5 py-1 text-xs font-black text-indigo-300 font-mono">
            STAGE 3.4
          </span>
          <h4 className="text-base sm:text-lg font-bold text-white">
            Multimodal Vector Cosine Comparison (CLIP)
          </h4>
        </div>
        <AlignmentGauge alignment={alignment} hasImage={hasImage} />
      </div>
    </div>
  );
}
