"use client";

import React, { useState } from "react";
import { FullAnalysisResponse } from "../lib/types";
import { OCRPipelineCard } from "./OCRPipelineCard";
import { ComparativeScoreMatrix } from "./ComparativeScoreMatrix";
import { VectorTransformationCard } from "./VectorTransformationCard";
import { EvidenceCard } from "./EvidenceCard";
import { TimelineDrawer } from "./TimelineDrawer";
import { PredictionBadge, MLVerdictPill, AgreementIndicator } from "./PredictionBadge";

interface PipelineStepperProps {
  analysis: FullAnalysisResponse;
  imagePreviewUrl?: string | null;
}

export function PipelineStepper({ analysis, imagePreviewUrl }: PipelineStepperProps) {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [viewMode, setViewMode] = useState<"all" | "stepper">("all");

  const hasImage = analysis.has_image;
  const bert = analysis.ml_prediction?.bert_prediction;
  const vit = analysis.ml_prediction?.vit_prediction;
  const fusion = analysis.ml_prediction?.multimodal_fusion_prediction;
  const ensemble = analysis.ml_prediction?.ensemble_prediction;
  const alignment = analysis.semantic_alignment || analysis.ml_prediction?.semantic_alignment;
  const adjudication = analysis.evidence_assessment?.adjudication;
  const isMLOnly = !analysis.evidence_assessment;

  const citations =
    adjudication?.recorded_citations && adjudication.recorded_citations.length > 0
      ? adjudication.recorded_citations
      : analysis.evidence_assessment?.claim_analyses?.flatMap((ca) => [
          ...(ca.supporting_sources || []).map((s: any) => ({ ...s, stance: "SUPPORTS" })),
          ...(ca.contradicting_sources || []).map((s: any) => ({ ...s, stance: "CONTRADICTS" })),
          ...(ca.neutral_or_insufficient_sources || []).map((s: any) => ({
            ...s,
            stance: "INSUFFICIENT",
          })),
        ]) || [];

  const prepDetails = (bert as any)?.preprocessing_details;
  const vitDetails = (vit as any)?.visual_feature_details;

  // If Model Only mode, only show stages 1 to 4
  const allPossibleSteps = [
    {
      id: 1,
      title: "1. Linguistic Forensics",
      desc: "Text Normalization & Sensationalism Scoring",
      badge: "Pre-Processing",
    },
    {
      id: 2,
      title: "2. Image OCR Extraction",
      desc: hasImage ? "Visual Text & Bounding Regions" : "Inactive (Text-Only Article)",
      badge: hasImage ? "5-Step Inspection" : "Skipped",
    },
    {
      id: 3,
      title: "3. Vector Representations",
      desc: "Dense Embeddings & Neural Representations",
      badge: "Feature Extraction",
    },
    {
      id: 4,
      title: "4. Multi-Model Intelligence Matrix",
      desc: "PyTorch CNN+LSTM, BERT & Soft Voting",
      badge: "Ensemble Consensus",
    },
    {
      id: 5,
      title: "5. Real-World Web Evidence",
      desc: "Web News Scrapers & Authoritative Citations",
      badge: "Live Fact-Check",
    },
    {
      id: 6,
      title: "6. Final Adjudication",
      desc: "Evidence Consensus & Explainable Verdict",
      badge: "Final Decision",
    },
  ];

  const stepsList = isMLOnly ? allPossibleSteps.slice(0, 4) : allPossibleSteps;

  const scrollToStep = (id: number) => {
    setCurrentStep(id);
    if (viewMode === "all") {
      const el = document.getElementById(`step-section-${id}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-6 flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="rounded bg-white text-black px-2 py-0.5 text-xs font-bold uppercase">
              {isMLOnly ? "4-Stage Model Pipeline" : "6-Stage Verification Pipeline"}
            </span>
            <span className="text-xs font-mono text-zinc-400">
              {isMLOnly ? "Mode: Model Only (Local Neural Models — No Gemini API / No AI Agents)" : "Mode: Full (Models + Autonomous Agents)"}
            </span>
          </div>
          <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
            {isMLOnly
              ? "Local Neural Model Classification & Feature Inspection"
              : "Comprehensive Verification & Evidence Pipeline"}
          </h2>
          <p className="text-xs text-zinc-400 max-w-2xl">
            {isMLOnly
              ? "Running 100% locally on the trained CNN+LSTM, BERT, and Soft-Voting ensemble. Zero Gemini API calls, web search requests, or AI agents are executed."
              : "Autonomous agents extract claims, scrape verified news wires, cross-examine stances, and synthesize an empirical adjudication."}
          </p>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center space-x-1 bg-zinc-900 p-1 rounded-lg border border-zinc-800">
          <button
            onClick={() => setViewMode("all")}
            className={`px-3 py-1.5 rounded text-xs font-semibold transition-colors ${
              viewMode === "all" ? "bg-white text-black" : "text-zinc-400 hover:text-white"
            }`}
          >
            Show All Stages
          </button>
          <button
            onClick={() => setViewMode("stepper")}
            className={`px-3 py-1.5 rounded text-xs font-semibold transition-colors ${
              viewMode === "stepper" ? "bg-white text-black" : "text-zinc-400 hover:text-white"
            }`}
          >
            Step-by-Step Focus
          </button>
        </div>
      </div>

      {/* Stage Selector Pills */}
      <div className={`grid grid-cols-2 sm:grid-cols-${stepsList.length} gap-2`}>
        {stepsList.map((step) => {
          const isActive = currentStep === step.id;
          return (
            <button
              key={step.id}
              onClick={() => scrollToStep(step.id)}
              className={`rounded-lg border p-3 text-left transition-colors cursor-pointer ${
                isActive
                  ? "border-white bg-zinc-900 text-white"
                  : "border-zinc-800 bg-zinc-950/60 text-zinc-400 hover:border-zinc-700 hover:text-white"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-bold text-white">{step.id}</span>
                <span className="text-[10px] font-mono font-medium rounded border border-zinc-800 bg-zinc-900 px-1.5 py-0.2 text-zinc-300">
                  {step.badge}
                </span>
              </div>
              <h4 className="text-xs font-bold truncate text-zinc-200">{step.title}</h4>
              <p className="text-[10px] text-zinc-500 truncate">{step.desc}</p>
            </button>
          );
        })}
      </div>

      {/* STAGE 1: Linguistic & Syntactic Forensics */}
      {(viewMode === "all" || currentStep === 1) && (
        <section id="step-section-1" className="space-y-4">
          <div className="flex items-center gap-2 border-b border-zinc-800 pb-2">
            <span className="flex h-5 w-5 items-center justify-center rounded bg-white text-black text-xs font-bold">
              1
            </span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Stage 1: Linguistic & Syntactic Forensics
            </h3>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-5 space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                <span className="text-[11px] text-zinc-500 block">Character Count</span>
                <span className="text-lg font-bold font-mono text-white">
                  {prepDetails?.normalized_text_length || analysis.input_text.length}
                </span>
              </div>
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                <span className="text-[11px] text-zinc-500 block">Token Length</span>
                <span className="text-lg font-bold font-mono text-white">
                  {prepDetails?.token_count || analysis.input_text.split(/\s+/).length}
                </span>
              </div>
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                <span className="text-[11px] text-zinc-500 block">Exclamation Frequency</span>
                <span className="text-lg font-bold font-mono text-white">
                  {prepDetails?.exclamation_count ?? (analysis.input_text.match(/!/g) || []).length}
                </span>
              </div>
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                <span className="text-[11px] text-zinc-500 block">Uppercase Ratio</span>
                <span className="text-lg font-bold font-mono text-white">
                  {prepDetails?.capitalization_ratio !== undefined
                    ? `${(prepDetails.capitalization_ratio * 100).toFixed(1)}%`
                    : "0.0%"}
                </span>
              </div>
            </div>

            <div className="rounded-lg border border-zinc-800 bg-zinc-900/30 p-3">
              <span className="text-[11px] font-mono text-zinc-400 block mb-1">Normalized Text:</span>
              <p className="text-xs sm:text-sm text-zinc-300 font-mono leading-relaxed">
                {prepDetails?.normalized_text || analysis.input_text}
              </p>
            </div>
          </div>
        </section>
      )}

      {/* STAGE 2: Image OCR Inspection */}
      {(viewMode === "all" || currentStep === 2) && (
        <section id="step-section-2" className="space-y-4">
          <div className="flex items-center gap-2 border-b border-zinc-800 pb-2">
            <span className="flex h-5 w-5 items-center justify-center rounded bg-white text-black text-xs font-bold">
              2
            </span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Stage 2: Image OCR Extraction & Bounding Inspection
            </h3>
          </div>
          <OCRPipelineCard
            imagePreviewUrl={imagePreviewUrl}
            ocr={analysis.ocr_analysis || (analysis.ml_prediction as any)?.ocr_analysis}
            inputHeadline={analysis.input_text}
          />
        </section>
      )}

      {/* STAGE 3: Vector Transformation */}
      {(viewMode === "all" || currentStep === 3) && (
        <section id="step-section-3" className="space-y-4">
          <div className="flex items-center gap-2 border-b border-zinc-800 pb-2">
            <span className="flex h-5 w-5 items-center justify-center rounded bg-white text-black text-xs font-bold">
              3
            </span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Stage 3: Vector Representations & Mathematical Embeddings
            </h3>
          </div>
          <VectorTransformationCard
            inputText={analysis.input_text}
            hasImage={hasImage}
            bertDetails={prepDetails}
            vitDetails={vitDetails}
            alignment={alignment}
          />
        </section>
      )}

      {/* STAGE 4: Multi-Model Intelligence Matrix */}
      {(viewMode === "all" || currentStep === 4) && (
        <section id="step-section-4" className="space-y-4">
          <div className="flex items-center gap-2 border-b border-zinc-800 pb-2">
            <span className="flex h-5 w-5 items-center justify-center rounded bg-white text-black text-xs font-bold">
              4
            </span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Stage 4: Multi-Model Intelligence Matrix
            </h3>
          </div>
          <ComparativeScoreMatrix
            scores={analysis.ml_prediction?.comparative_scores}
            ensemble={ensemble}
            bert={bert}
            vit={vit}
            fusion={fusion}
            hasImage={hasImage}
          />
        </section>
      )}

      {/* STAGE 5: Autonomous Evidence Retrieval (Hidden if ML Only) */}
      {!isMLOnly && (viewMode === "all" || currentStep === 5) && (
        <section id="step-section-5" className="space-y-4">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <div className="flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded bg-white text-black text-xs font-bold">
                5
              </span>
              <h3 className="text-base font-bold text-white tracking-tight">
                Stage 5: Autonomous Web Evidence Verification
              </h3>
            </div>
            <span className="text-xs text-zinc-400 font-mono">
              Agents 1-3: Claim Extraction & News Scraping
            </span>
          </div>

          <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-4 text-xs text-zinc-400 leading-relaxed">
            <p className="font-semibold text-zinc-300 mb-1">How Evidence Verification Works:</p>
            Autonomous agents extract the key claim, query live news databases (DuckDuckGo News, Reuters, AP, Bloomberg, CNBC), scrape the retrieved articles, and compare excerpts against the claim to verify authenticity.
          </div>

          {citations.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {citations.map((c: any, idx: number) => (
                <EvidenceCard key={c.source_id || idx} evidence={c} />
              ))}
            </div>
          ) : (
            <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-6 text-center text-xs text-zinc-400">
              No corroborating web sources were retrieved.
            </div>
          )}

          {analysis.evidence_assessment?.events && (
            <TimelineDrawer events={analysis.evidence_assessment.events} />
          )}
        </section>
      )}

      {/* STAGE 6: Final Adjudication (Hidden if ML Only) */}
      {!isMLOnly && (viewMode === "all" || currentStep === 6) && (
        <section id="step-section-6" className="space-y-4">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <div className="flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded bg-white text-black text-xs font-bold">
                6
              </span>
              <h3 className="text-base font-bold text-white tracking-tight">
                Stage 6: Final Adjudication & Consensus Verdict
              </h3>
            </div>
            <span className="text-xs text-zinc-400 font-mono">
              Agent 4: Adjudicator
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 space-y-4">
              <PredictionBadge verdict={adjudication?.verdict || (ensemble?.predicted_class === "fake" ? "FALSE" : "TRUE")} />
              {adjudication && (
                <AgreementIndicator agreement={adjudication.ml_agreement} />
              )}
            </div>

            <div className="lg:col-span-2 rounded-xl border border-zinc-800 bg-zinc-950 p-6 space-y-4">
              <div>
                <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-1">
                  Adjudication Rationale
                </h4>
                <p className="text-xs sm:text-sm text-zinc-300 leading-relaxed">
                  {adjudication?.verdict_explanation || "Evaluation synthesized from neural model priors and empirical evidence records."}
                </p>
              </div>

              {adjudication?.evidence_summary && (
                <div className="pt-4 border-t border-zinc-800">
                  <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-1">
                    Evidence Summary
                  </h4>
                  <p className="text-xs text-zinc-400 leading-relaxed">
                    {adjudication.evidence_summary}
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
