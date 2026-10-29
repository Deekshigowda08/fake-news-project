export interface ProbabilityDistribution {
  real: number;
  fake: number;
}

export interface ModelPrediction {
  model_name: string;
  predicted_class: "real" | "fake";
  class_probabilities: ProbabilityDistribution;
  confidence: number;
  raw_logits?: number[];
  status?: string;
  error_detail?: string;
}

export interface EnsemblePrediction {
  predicted_class: "real" | "fake";
  class_probabilities: ProbabilityDistribution;
  confidence: number;
  contributing_models: string[];
  ensemble_method: string;
  weights_applied?: Record<string, number>;
}

export interface SemanticAlignment {
  alignment_score: number | null;
  interpretation: string;
  model_name: string;
  caveat: string;
}

export interface EvidenceCitation {
  source_id: string;
  url: string;
  publisher: string;
  title: string;
  stance: "SUPPORTS" | "CONTRADICTS" | "MIXED" | "INSUFFICIENT";
  credibility_tier: "tier_1" | "tier_2" | "tier_3";
  excerpt: string;
}

export interface AdjudicationVerdict {
  verdict: "TRUE" | "FALSE" | "MISLEADING" | "UNVERIFIABLE";
  verdict_explanation: string;
  evidence_summary: string;
  ml_agreement: "agrees" | "disagrees" | "unresolved";
  ml_predicted_class: "REAL_CLASSIFIED" | "FAKE_CLASSIFIED";
  ml_model_confidence: number;
  recorded_citations: EvidenceCitation[];
  conflicting_evidence_present: boolean;
  uncertainty_reasons: string[];
  timestamp: string;
}

export interface AgentEvent {
  agent_name: string;
  event_type: string;
  status: string;
  duration_ms: number;
  structured_details: Record<string, any>;
  timestamp: number;
}

export interface ExtractedClaim {
  claim_id: string;
  claim_text: string;
  claim_type: string;
  named_entities: string[];
  temporal_anchor?: string;
  geographic_anchor?: string;
  is_verifiable: boolean;
}

export interface ComparativeScoreItem {
  model_key: string;
  model_name: string;
  modality: string;
  predicted_class: string;
  p_real: number;
  p_fake: number;
  confidence: number;
  fine_grained_class?: string | null;
  ensemble_weight: number;
  backbone: string;
}

export interface OCRTextBlock {
  block_id: number;
  text: string;
  region: string;
  estimated_confidence: number;
}

export interface OCRPipelineResult {
  has_image: boolean;
  has_extracted_text: boolean;
  raw_text: string;
  word_count: number;
  concordance_verdict: string;
  concordance_score: number;
  is_discrepant: boolean;
  steps: Array<{
    step_number: number;
    step_title: string;
    [key: string]: any;
  }>;
  quality_metrics?: {
    dimensions: string;
    aspect_ratio: number;
    color_format: string;
    sharpness_score: number;
    optical_clarity: string;
    mean_luminance: number;
    payload_size_kb: number;
  };
  ocr_blocks?: OCRTextBlock[];
  comparison?: {
    jaccard_word_overlap_ratio: number;
    sequence_similarity_ratio: number;
    shared_keywords: string[];
    concordance_verdict: string;
    concordance_explanation: string;
    is_manipulated_context_flag: boolean;
  };
}

export interface FullAnalysisResponse {
  analysis_id: string;
  status: string;
  input_text: string;
  has_image: boolean;
  submitted_url?: string;
  available_modalities: string[];
  ml_prediction: {
    headline_or_text: string;
    has_image: boolean;
    available_modalities: string[];
    bert_prediction: ModelPrediction;
    vit_prediction: ModelPrediction;
    multimodal_fusion_prediction: ModelPrediction & { modality_used: string };
    ensemble_prediction: EnsemblePrediction;
    semantic_alignment: SemanticAlignment;
    ocr_analysis?: OCRPipelineResult;
    comparative_scores?: ComparativeScoreItem[];
    calibrated_model_confidence: number;
    disclaimer: string;
  };
  evidence_assessment?: {
    input_processing: {
      normalized_text: string;
      claims: ExtractedClaim[];
      opinion_or_satire_detected: boolean;
    };
    claim_analyses: any[];
    adjudication: AdjudicationVerdict;
    events: AgentEvent[];
    total_duration_ms: number;
  };
  semantic_alignment?: SemanticAlignment;
  ocr_analysis?: OCRPipelineResult;
  comparative_scores?: ComparativeScoreItem[];
  created_at: string;
  completed_at: string;
}


export interface ExperimentMetric {
  name: string;
  description: string;
  modalities: string[];
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    brier_score?: number;
    ece?: number;
  };
}

export interface SystemHealth {
  status: string;
  database: string;
  gpu_available: boolean;
  device: string;
  version: string;
}

export interface ModelInfo {
  name: string;
  type: string;
  backbone: string;
  trainable_parameters?: number;
  status: string;
}
