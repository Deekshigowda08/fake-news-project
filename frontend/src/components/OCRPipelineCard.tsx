"use client";

import React, { useState } from "react";
import { OCRPipelineResult } from "../lib/types";

interface OCRPipelineCardProps {
  ocr?: OCRPipelineResult | null;
  imagePreviewUrl?: string | null;
  inputHeadline?: string;
}

export function OCRPipelineCard({
  ocr,
  imagePreviewUrl,
  inputHeadline,
}: OCRPipelineCardProps) {
  const [activeStepTab, setActiveStepTab] = useState<number>(1);

  if (!ocr) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-800 text-xs font-bold text-slate-400">
              OCR
            </span>
            <h3 className="text-sm font-bold text-white">
              Optical Character Recognition (OCR) Engine
            </h3>
          </div>
          <span className="rounded bg-slate-800/80 px-2 py-0.5 text-xs text-slate-400">
            No Image Provided
          </span>
        </div>
        <p className="mt-2 text-xs text-slate-400 leading-relaxed">
          Upload a news image, article screenshot, or infographic to initiate optical text extraction, visual bounding region detection, and cross-modal claim discrepancy verification.
        </p>
      </div>
    );
  }

  const isDiscrepant = ocr.is_discrepant || ocr.concordance_verdict === "CROSS_MODAL_DISCREPANCY";
  const isMatch = ocr.concordance_verdict === "CONCORDANT_MATCH";
  const isPureVisual = ocr.concordance_verdict === "PURE_VISUAL_MEDIA";

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 sm:p-6 backdrop-blur-md space-y-5 shadow-xl">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono font-bold text-xs">
            OCR
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-white">
                Image Text Reading & Cross-Modal OCR Pipeline
              </h3>
              <span className="rounded bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 text-[10px] font-semibold text-cyan-300">
                5-Step Neural Inspection
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Extracts printed typography, detects overlay banners, and compares visual text against input claim
            </p>
          </div>
        </div>

        {/* Status Pill */}
        <div className="flex items-center gap-2">
          {isDiscrepant ? (
            <span className="rounded-lg border border-rose-500/40 bg-rose-500/15 px-3 py-1 text-xs font-bold text-rose-300 animate-pulse flex items-center gap-1.5">
              <span>⚠️</span>
              <span>Discrepancy / Mismatched Media</span>
            </span>
          ) : isMatch ? (
            <span className="rounded-lg border border-emerald-500/40 bg-emerald-500/15 px-3 py-1 text-xs font-bold text-emerald-300 flex items-center gap-1.5">
              <span>✓</span>
              <span>Image Text Corroborates Claim</span>
            </span>
          ) : isPureVisual ? (
            <span className="rounded-lg border border-blue-500/40 bg-blue-500/15 px-3 py-1 text-xs font-medium text-blue-300">
              Pure Visual Media (No Typography)
            </span>
          ) : (
            <span className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1 text-xs font-medium text-slate-300">
              {ocr.concordance_verdict}
            </span>
          )}
        </div>
      </div>

      {/* Interactive Step Navigator for Professors */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Select Inspection Step to Present:
          </span>
          <span className="text-[11px] text-indigo-400 font-mono">
            Step {activeStepTab} of 5
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-1.5">
          {[
            { num: 1, label: "1. Optical Quality" },
            { num: 2, label: "2. Text Detection" },
            { num: 3, label: "3. Transcription" },
            { num: 4, label: "4. Cross-Modal Match" },
            { num: 5, label: "5. Forensics Verdict" },
          ].map((tab) => (
            <button
              key={tab.num}
              type="button"
              onClick={() => setActiveStepTab(tab.num)}
              className={`rounded-lg px-2.5 py-2 text-xs font-semibold transition-all border text-left truncate ${
                activeStepTab === tab.num
                  ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200 shadow-sm"
                  : "border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700 hover:text-slate-300"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* STEP CONTENT PANELS */}
      <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 sm:p-5 space-y-4">
        {/* STEP 1: Optical Quality */}
        {activeStepTab === 1 && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                Step 1: Ingestion Diagnostics & Optical Structural Integrity
              </h4>
              <span className="text-xs text-slate-400 font-mono">
                {ocr.quality_metrics?.dimensions || "Visual Media"}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="rounded-lg border border-slate-800/80 bg-slate-900/50 p-3 space-y-1">
                <span className="text-[10px] text-slate-400 block uppercase">Dimensions</span>
                <span className="text-sm font-bold text-white font-mono">
                  {ocr.quality_metrics?.dimensions || "N/A"}
                </span>
                <span className="text-[10px] text-slate-500 block">
                  Aspect Ratio: {ocr.quality_metrics?.aspect_ratio}:1
                </span>
              </div>

              <div className="rounded-lg border border-slate-800/80 bg-slate-900/50 p-3 space-y-1">
                <span className="text-[10px] text-slate-400 block uppercase">Optical Clarity</span>
                <span className="text-sm font-bold text-emerald-300">
                  {ocr.quality_metrics?.optical_clarity || "High"}
                </span>
                <span className="text-[10px] text-slate-500 block">
                  Sharpness Index: {ocr.quality_metrics?.sharpness_score}
                </span>
              </div>

              <div className="rounded-lg border border-slate-800/80 bg-slate-900/50 p-3 space-y-1">
                <span className="text-[10px] text-slate-400 block uppercase">Payload Size</span>
                <span className="text-sm font-bold text-white font-mono">
                  {ocr.quality_metrics?.payload_size_kb || 0} KB
                </span>
                <span className="text-[10px] text-slate-500 block">
                  Format: {ocr.quality_metrics?.color_format || "JPEG"}
                </span>
              </div>

              <div className="rounded-lg border border-slate-800/80 bg-slate-900/50 p-3 space-y-1">
                <span className="text-[10px] text-slate-400 block uppercase">Luminance Channel</span>
                <span className="text-sm font-bold text-white font-mono">
                  {ocr.quality_metrics?.mean_luminance || 128} / 255
                </span>
                <span className="text-[10px] text-slate-500 block">
                  Standardized Exposure
                </span>
              </div>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Optical validation verifies image decodability, filters zero-byte corruption or steganographic metadata payloads, and measures edge sharpness to guarantee legibility before invoking neural OCR models.
            </p>
          </div>
        )}

        {/* STEP 2: Text Detection & Bounding Blocks */}
        {activeStepTab === 2 && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                Step 2: Typography Detection & Visual Region Segmentation
              </h4>
              <span className="text-xs text-slate-400">
                {ocr.ocr_blocks?.length || 0} Distinct Region(s) Identified
              </span>
            </div>

            {ocr.ocr_blocks && ocr.ocr_blocks.length > 0 ? (
              <div className="space-y-2">
                {ocr.ocr_blocks.map((block) => (
                  <div
                    key={block.block_id}
                    className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 rounded-lg border border-slate-800/80 bg-slate-900/50 p-3 text-xs"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="rounded bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 text-[10px] font-mono text-cyan-300">
                          Region: {block.region.replace("_", " ")}
                        </span>
                        <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-[10px] font-mono text-emerald-400">
                          Confidence: {(block.estimated_confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="font-medium text-white font-mono">
                        "{block.text}"
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-lg border border-slate-800/80 bg-slate-900/40 p-4 text-center text-xs text-slate-400">
                No printed typographic regions were located. Visual scene evaluated as natural photography or diagrammatic imagery.
              </div>
            )}
          </div>
        )}

        {/* STEP 3: Full Transcription */}
        {activeStepTab === 3 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                Step 3: Complete Verbatim Text Transcription
              </h4>
              <span className="text-xs text-slate-400 font-mono">
                {ocr.word_count} Word(s) Extracted
              </span>
            </div>

            <div className="rounded-lg border border-slate-800 bg-slate-950 p-4 space-y-2 font-mono text-xs">
              <span className="text-[10px] text-slate-500 block uppercase font-sans font-semibold">
                Raw Extracted OCR Stream:
              </span>
              <p className="text-slate-200 whitespace-pre-wrap leading-relaxed">
                {ocr.raw_text || "[No readable text found in this image]"}
              </p>
            </div>
          </div>
        )}

        {/* STEP 4: Cross-Modal Concordance */}
        {activeStepTab === 4 && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                Step 4: Cross-Modal Concordance (Headline Claim vs Image Text)
              </h4>
              <span className="rounded bg-indigo-500/10 px-2 py-0.5 text-xs text-indigo-300 font-mono">
                Overlap: {Math.round((ocr.comparison?.jaccard_word_overlap_ratio || 0) * 100)}%
              </span>
            </div>

            {/* Comparison Details */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3 space-y-1">
                <span className="text-[10px] font-bold text-slate-400 uppercase">
                  User Headline Claim:
                </span>
                <p className="text-xs text-slate-200">
                  "{inputHeadline || "N/A"}"
                </p>
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3 space-y-1">
                <span className="text-[10px] font-bold text-slate-400 uppercase">
                  Text Extracted From Image:
                </span>
                <p className="text-xs text-slate-200 font-mono truncate">
                  "{ocr.raw_text || "No text detected"}"
                </p>
              </div>
            </div>

            {/* Overlap Metrics */}
            <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-3.5 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">Vocabulary Overlap (Jaccard Index):</span>
                <span className="text-cyan-300 font-mono font-bold">
                  {Math.round((ocr.comparison?.jaccard_word_overlap_ratio || 0) * 100)}%
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full bg-cyan-500 transition-all duration-500"
                  style={{ width: `${Math.round((ocr.comparison?.jaccard_word_overlap_ratio || 0) * 100)}%` }}
                />
              </div>

              {ocr.comparison?.shared_keywords && ocr.comparison.shared_keywords.length > 0 && (
                <div className="pt-2 flex flex-wrap items-center gap-1.5 text-xs">
                  <span className="text-[11px] text-slate-400">Shared Key Terms:</span>
                  {ocr.comparison.shared_keywords.map((kw, i) => (
                    <span
                      key={i}
                      className="rounded bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 text-[10px] font-mono text-cyan-300"
                    >
                      {kw}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Narrative Explanation */}
            <div
              className={`rounded-lg border p-3 text-xs leading-relaxed ${
                isDiscrepant
                  ? "border-rose-500/30 bg-rose-500/10 text-rose-300"
                  : isMatch
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                  : "border-slate-800 bg-slate-900 text-slate-300"
              }`}
            >
              {ocr.comparison?.concordance_explanation || "Concordance evaluated."}
            </div>
          </div>
        )}

        {/* STEP 5: Forensics Synthesis */}
        {activeStepTab === 5 && (
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-400">
              Step 5: Visual Forensics Summary for Academic Evaluation
            </h4>
            <div className="rounded-lg border border-slate-800/80 bg-slate-900/60 p-4 space-y-2 text-xs leading-relaxed text-slate-200">
              <p>
                <strong className="text-white">Academic Rationale:</strong> Modern multimodal disinformation rarely consists solely of altered photos. Instead, malicious actors frequently employ <span className="text-cyan-300 font-semibold">re-purposed authentic imagery with fabricated captions</span>.
              </p>
              <p>
                By executing optical character recognition, our pipeline independently reads the exact words printed within the image and verifies whether the graphic was authored to support the submitted claim or belongs to an unrelated reporting event.
              </p>
              <div className="mt-3 pt-3 border-t border-slate-800 flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Concordance Verdict:</span>
                <span className="text-cyan-400 font-bold">{ocr.concordance_verdict}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer Navigation Buttons */}
      <div className="flex items-center justify-between pt-1">
        <button
          type="button"
          disabled={activeStepTab === 1}
          onClick={() => setActiveStepTab((prev) => Math.max(1, prev - 1))}
          className="rounded-lg border border-slate-800 bg-slate-800/60 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
        >
          ← Previous Step
        </button>

        <span className="text-xs text-slate-500">
          Click any step above to present to professors
        </span>

        <button
          type="button"
          disabled={activeStepTab === 5}
          onClick={() => setActiveStepTab((prev) => Math.min(5, prev + 1))}
          className="rounded-lg border border-cyan-500/30 bg-cyan-500/10 hover:bg-cyan-500/20 px-3 py-1.5 text-xs font-medium text-cyan-300 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          Next Step →
        </button>
      </div>
    </div>
  );
}
