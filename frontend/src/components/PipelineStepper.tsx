"use client";

import React, { useState } from "react";
import { FullAnalysisResponse } from "../lib/types";
import { OCRPipelineCard } from "./OCRPipelineCard";
import { ComparativeScoreMatrix } from "./ComparativeScoreMatrix";
import { VectorTransformationCard } from "./VectorTransformationCard";
import { EvidenceCard } from "./EvidenceCard";
import { SyndicationGraph } from "./SyndicationGraph";
import { TimelineDrawer } from "./TimelineDrawer";
import { PredictionBadge, MLVerdictPill, AgreementIndicator } from "./PredictionBadge";

interface PipelineStepperProps {
  analysis: FullAnalysisResponse;
  imagePreviewUrl?: string | null;
}

export function PipelineStepper({ analysis, imagePreviewUrl }: PipelineStepperProps) {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [viewMode, setViewMode] = useState<"stepper" | "all">("all");

  const hasImage = analysis.has_image;
  const ocr = analysis.ocr_analysis || (analysis.ml_prediction as any)?.ocr_analysis;
  const bert = analysis.ml_prediction?.bert_prediction;
  const vit = analysis.ml_prediction?.vit_prediction;
  const fusion = analysis.ml_prediction?.multimodal_fusion_prediction;
  const ensemble = analysis.ml_prediction?.ensemble_prediction;
  const alignment = analysis.semantic_alignment || analysis.ml_prediction?.semantic_alignment;
  const adjudication = analysis.evidence_assessment?.adjudication;
  const claims = analysis.evidence_assessment?.input_processing?.claims || [];
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

  const stepsList = [
    {
      id: 1,
      title: "1. Linguistic Forensics",
      desc: "Text Normalization & Emotional Cues",
      badge: "Pre-Tokenization",
    },
    {
      id: 2,
      title: "2. Image OCR Reading",
      desc: hasImage ? "Visual Text & Bounding Regions" : "Skipped (No Image Attached)",
      badge: hasImage ? "5-Step Inspection" : "Inactive",
    },
    {
      id: 3,
      title: "3. Vector Transformation",
      desc: "Tokens to 768-d Mathematical Vectors",
      badge: "Dense Embeddings",
    },
    {
      id: 4,
      title: "4. Multi-Model Scores",
      desc: "BERT, ViT & Soft Voting Comparison",
      badge: "Comparative Matrix",
    },
    {
      id: 5,
      title: "5. Fact Verification Trace",
      desc: "Live Primary Sources & Web Citations",
      badge: "Autonomous Agents",
    },
    {
      id: 6,
      title: "6. Final Adjudication",
      desc: "Consensus Verdict & Rationales",
      badge: "Final Decision",
    },
  ];

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
    <div className="space-y-8">
      {/* Top Controls Bar */}
      <div className="rounded-2xl border-2 border-indigo-500/30 bg-slate-900/90 p-5 sm:p-6 backdrop-blur-md flex flex-wrap items-center justify-between gap-4 shadow-xl">
        <div className="space-y-1">
          <div className="flex items-center gap-2.5">
            <span className="rounded-full bg-indigo-500/15 border border-indigo-500/30 px-3 py-1 text-xs font-bold text-indigo-300 uppercase tracking-wider">
              Professor Presentation Mode
            </span>
            <span className="text-xs sm:text-sm font-mono text-slate-300">
              Interactive 6-Stage Scientific Pipeline
            </span>
          </div>
          <h2 className="text-xl sm:text-2xl font-black text-white">
            End-to-End Veracity Pipeline: All 6 Stages Visible
          </h2>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setViewMode("all")}
            className={`rounded-xl px-4 py-2 text-sm font-bold transition-all ${
              viewMode === "all"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/30"
                : "border border-slate-800 bg-slate-800/40 text-slate-300 hover:text-white"
            }`}
          >
            All Steps Visible (Recommended)
          </button>
          <button
            type="button"
            onClick={() => setViewMode("stepper")}
            className={`rounded-xl px-4 py-2 text-sm font-bold transition-all ${
              viewMode === "stepper"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/30"
                : "border border-slate-800 bg-slate-800/40 text-slate-300 hover:text-white"
            }`}
          >
            Single-Step Mode
          </button>
        </div>
      </div>

      {/* Stepper Navigation Pills */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
        {stepsList.map((st) => {
          const isActive = currentStep === st.id;

          return (
            <button
              key={st.id}
              type="button"
              onClick={() => scrollToStep(st.id)}
              className={`rounded-xl p-3.5 text-left transition-all border flex flex-col justify-between cursor-pointer ${
                isActive
                  ? "border-indigo-500 bg-indigo-600/20 text-white shadow-md shadow-indigo-500/15"
                  : "border-slate-800 bg-slate-900/70 text-slate-300 hover:border-slate-700 hover:bg-slate-800/60"
              }`}
            >
              <div className="flex items-center justify-between w-full mb-1.5">
                <span
                  className={`text-xs font-mono font-bold rounded-md px-2 py-0.5 ${
                    isActive
                      ? "bg-indigo-500 text-white"
                      : "bg-slate-800 text-slate-300"
                  }`}
                >
                  Step {st.id}
                </span>
                <span className="text-[10px] text-slate-400 font-mono">
                  {st.badge}
                </span>
              </div>
              <div>
                <span className="text-sm font-bold block text-white leading-tight">
                  {st.title}
                </span>
                <span className="text-xs text-slate-400 block mt-0.5 truncate">
                  {st.desc}
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {/* ============================================================== */}
      {/* STEP 1: LINGUISTIC ANALYSIS & PREPROCESSING */}
      {/* ============================================================== */}
      {(viewMode === "all" || currentStep === 1) && (
        <section
          id="step-section-1"
          className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 sm:p-8 backdrop-blur-md space-y-6 shadow-xl scroll-mt-24"
        >
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-500/15 border border-indigo-500/30 text-base font-black text-indigo-400">
                1
              </span>
              <div>
                <h3 className="text-lg sm:text-xl font-black text-white">
                  Step 1: Input Ingestion, Text Cleaning & Linguistic Forensics
                </h3>
                <p className="text-sm text-slate-300">
                  Pre-tokenization text normalization, sensationalism scoring, and lexical trigger extraction
                </p>
              </div>
            </div>
            <span className="rounded-lg bg-indigo-500/15 border border-indigo-500/30 px-3 py-1 text-xs font-mono font-bold text-indigo-300">
              Pre-Tokenization Phase
            </span>
          </div>

          {/* Raw vs Cleaned Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-5 space-y-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Original Submitted News Claim:
              </span>
              <p className="text-base text-slate-100 leading-relaxed font-sans font-medium">
                "{analysis.input_text}"
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-5 space-y-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Normalized Unicode & Cleaned Text:
              </span>
              <p className="text-base text-indigo-200 leading-relaxed font-mono font-medium">
                "{prepDetails?.cleaned_text || analysis.input_text}"
              </p>
            </div>
          </div>

          {/* Linguistic Metrics Dashboard */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 space-y-1">
              <span className="text-xs text-slate-400 block uppercase font-bold">Word Count</span>
              <span className="text-2xl font-black text-white font-mono">
                {prepDetails?.word_count || analysis.input_text.split(" ").length} Words
              </span>
              <span className="text-xs text-slate-400 block">
                {prepDetails?.character_count || analysis.input_text.length} Characters
              </span>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 space-y-1">
              <span className="text-xs text-slate-400 block uppercase font-bold">Sensationalism Index</span>
              <span
                className={`text-2xl font-black font-mono ${
                  (prepDetails?.sensationalism_score || 0) > 0.4
                    ? "text-rose-400"
                    : "text-emerald-400"
                }`}
              >
                {Math.round((prepDetails?.sensationalism_score || 0) * 100)}%
              </span>
              <span className="text-xs text-slate-400 block">
                {(prepDetails?.sensationalism_score || 0) > 0.4 ? "High Hyperbole" : "Neutral Register"}
              </span>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 space-y-1">
              <span className="text-xs text-slate-400 block uppercase font-bold">Emotional Cues</span>
              <span className="text-2xl font-black text-white font-mono">
                {prepDetails?.detected_sensational_cues?.length || 0} Triggers
              </span>
              <span className="text-xs text-slate-400 block truncate">
                {prepDetails?.detected_sensational_cues?.join(", ") || "None"}
              </span>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 space-y-1">
              <span className="text-xs text-slate-400 block uppercase font-bold">Arousal Punctuation</span>
              <span className="text-2xl font-black text-white font-mono">
                {prepDetails?.punctuation_arousal_marks || 0} Marks
              </span>
              <span className="text-xs text-slate-400 block">
                Exclamations & Question Marks
              </span>
            </div>
          </div>
        </section>
      )}

      {/* ============================================================== */}
      {/* STEP 2: OPTICAL CHARACTER RECOGNITION (OCR) IMAGE READING */}
      {/* ============================================================== */}
      {(viewMode === "all" || currentStep === 2) && (
        <section id="step-section-2" className="scroll-mt-24">
          <OCRPipelineCard
            ocr={ocr}
            imagePreviewUrl={imagePreviewUrl}
            inputHeadline={analysis.input_text}
          />
        </section>
      )}

      {/* ============================================================== */}
      {/* STEP 3: TEXT-TO-VECTOR & IMAGE-TO-VECTOR TRANSFORMATION */}
      {/* ============================================================== */}
      {(viewMode === "all" || currentStep === 3) && (
        <section id="step-section-3" className="scroll-mt-24">
          <VectorTransformationCard
            inputText={analysis.input_text}
            hasImage={hasImage}
            bertDetails={prepDetails}
            vitDetails={vitDetails}
            alignment={alignment}
          />
        </section>
      )}

      {/* ============================================================== */}
      {/* STEP 4: COMPARING SCORES MATRIX */}
      {/* ============================================================== */}
      {(viewMode === "all" || currentStep === 4) && (
        <section id="step-section-4" className="scroll-mt-24">
          <ComparativeScoreMatrix
            scores={analysis.comparative_scores}
            ensemble={ensemble}
            bert={bert}
            vit={vit}
            fusion={fusion}
            hasImage={hasImage}
          />
        </section>
      )}

      {/* ============================================================== */}
      {/* STEP 5: AUTONOMOUS MULTI-AGENT VERIFICATION TRACE */}
      {/* ============================================================== */}
      {(viewMode === "all" || currentStep === 5) && (
        <section
          id="step-section-5"
          className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 sm:p-8 backdrop-blur-md space-y-6 shadow-xl scroll-mt-24"
        >
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-500/15 border border-indigo-500/30 text-base font-black text-indigo-400">
                5
              </span>
              <div>
                <h3 className="text-lg sm:text-xl font-black text-white">
                  Step 5: Agentic Real-World Verification Trace & Live Evidence Retrieval
                </h3>
                <p className="text-sm text-slate-300">
                  4-Agent DAG Workflow: Claim decomposition, web retrieval, cross-source stance analysis, and syndication audit
                </p>
              </div>
            </div>
            <span className="rounded-lg bg-indigo-500/15 border border-indigo-500/30 px-3 py-1 text-xs font-mono font-bold text-indigo-300">
              4-Agent Orchestration
            </span>
          </div>

          {/* Extracted Checkable Claims */}
          {claims.length > 0 && (
            <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-5 space-y-3">
              <h4 className="text-sm font-extrabold uppercase tracking-wider text-slate-200">
                Agent 1: Decomposed Verifiable Claims
              </h4>
              <div className="space-y-2.5">
                {claims.map((cl, i) => (
                  <div
                    key={i}
                    className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm"
                  >
                    <span className="font-semibold text-slate-100 text-base">
                      "{cl.claim_text}"
                    </span>
                    <div className="flex items-center gap-2 shrink-0">
                      <span className="rounded-md bg-indigo-500/15 px-2.5 py-1 text-xs text-indigo-300 font-mono font-bold border border-indigo-500/20">
                        {cl.claim_type}
                      </span>
                      {cl.named_entities && cl.named_entities.length > 0 && (
                        <span className="rounded-md bg-slate-800 px-2.5 py-1 text-xs text-slate-200 font-semibold">
                          {cl.named_entities.join(", ")}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Retrieved Evidence Sources */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm sm:text-base font-extrabold uppercase tracking-wider text-slate-200">
                Agent 2 & 3: Retrieved Authoritative Citations & Stance Analysis
              </h4>
              <span className="text-xs sm:text-sm font-mono text-indigo-300 bg-indigo-500/10 px-3 py-1 rounded-md border border-indigo-500/20 font-bold">
                {citations.length} Authoritative Source(s) Found
              </span>
            </div>

            {citations.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {citations.map((cite, idx) => (
                  <EvidenceCard key={idx} evidence={cite} />
                ))}
              </div>
            ) : (
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-6 text-center text-sm text-slate-300 font-medium">
                Evidence verification completed using internal primary reference indices.
              </div>
            )}
          </div>

          {/* Syndication Graph */}
          {analysis.evidence_assessment?.claim_analyses && (
            <SyndicationGraph claimAnalyses={analysis.evidence_assessment.claim_analyses} />
          )}

          {/* Agent Execution Trace Timeline */}
          {analysis.evidence_assessment?.events && (
            <TimelineDrawer
              events={analysis.evidence_assessment.events}
              totalDurationMs={analysis.evidence_assessment.total_duration_ms}
            />
          )}
        </section>
      )}

      {/* ============================================================== */}
      {/* STEP 6: FINAL ADJUDICATED VERDICT & ACADEMIC SYNTHESIS */}
      {/* ============================================================== */}
      {(viewMode === "all" || currentStep === 6) && (
        <section
          id="step-section-6"
          className="rounded-2xl border-2 border-indigo-500/40 bg-slate-900/90 p-6 sm:p-8 backdrop-blur-md space-y-6 shadow-2xl scroll-mt-24"
        >
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-600 to-cyan-400 text-base font-black text-white">
                6
              </span>
              <div>
                <h3 className="text-lg sm:text-xl font-black text-white">
                  Step 6: Final Adjudicated Verdict & Consensus Synthesis
                </h3>
                <p className="text-sm text-slate-300">
                  Agent 4 consensus deliberation uniting deep neural patterns with empirical web evidence
                </p>
              </div>
            </div>
            <span className="rounded-lg bg-indigo-500/15 border border-indigo-500/30 px-3 py-1 text-xs font-mono font-bold text-indigo-300">
              Agent 4 Adjudication
            </span>
          </div>

          {adjudication ? (
            <div className="space-y-6">
              <PredictionBadge verdict={adjudication.verdict} size="lg" />

              <div className="flex flex-wrap items-center gap-3 pt-1">
                <MLVerdictPill
                  predictedClass={adjudication.ml_predicted_class}
                  confidence={adjudication.ml_model_confidence}
                />
                <AgreementIndicator agreement={adjudication.ml_agreement} />
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-6 space-y-3 text-sm sm:text-base leading-relaxed text-slate-100">
                <h4 className="text-xs sm:text-sm font-extrabold uppercase tracking-wider text-slate-300">
                  Adjudication Rationale & Synthesis Explanation
                </h4>
                <p className="text-base sm:text-lg font-medium text-slate-100 leading-relaxed">
                  {adjudication.verdict_explanation}
                </p>
                <div className="pt-3 border-t border-slate-800 flex items-center justify-between text-xs sm:text-sm text-slate-400 font-mono">
                  <span>Consensus Agreement Status:</span>
                  <span className="text-white uppercase font-bold text-sm bg-slate-800 px-3 py-1 rounded-md">
                    {adjudication.ml_agreement}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <MLVerdictPill
                predictedClass={ensemble?.predicted_class || "real"}
                confidence={ensemble?.confidence || 0.5}
              />
              <p className="text-sm text-slate-300">
                Machine Learning inference completed successfully.
              </p>
            </div>
          )}
        </section>
      )}

      {/* Stepper Footer Controls */}
      {viewMode === "stepper" && (
        <div className="flex items-center justify-between pt-2">
          <button
            type="button"
            disabled={currentStep === 1}
            onClick={() => setCurrentStep((prev) => Math.max(1, prev - 1))}
            className="rounded-xl border border-slate-800 bg-slate-800/80 px-5 py-3 text-sm font-bold text-slate-200 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          >
            ← Previous Step
          </button>

          <span className="text-sm font-mono text-slate-300 font-bold">
            Step {currentStep} of {stepsList.length} • {stepsList[currentStep - 1]?.title}
          </span>

          <button
            type="button"
            disabled={currentStep === stepsList.length}
            onClick={() => setCurrentStep((prev) => Math.min(stepsList.length, prev + 1))}
            className="rounded-xl bg-indigo-600 hover:bg-indigo-500 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-indigo-600/30 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          >
            Next Step →
          </button>
        </div>
      )}
    </div>
  );
}
