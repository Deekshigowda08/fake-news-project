"""Pydantic schemas for Agentic AI Evidence-Verification layer."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StanceType(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    MIXED = "MIXED"
    INSUFFICIENT = "INSUFFICIENT"


class ClaimType(str, Enum):
    FACTUAL_ASSERTION = "factual_assertion"
    STATISTICAL_CLAIM = "statistical_claim"
    HISTORICAL_EVENT = "historical_event"
    SCIENTIFIC_CLAIM = "scientific_claim"
    OPINION_OR_COMMENTARY = "opinion_or_commentary"
    SATIRE_OR_PARODY = "satire_or_parody"


class VerdictType(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    MISLEADING = "MISLEADING"
    UNVERIFIABLE = "UNVERIFIABLE"


class ExtractedClaim(BaseModel):
    """Factual claim isolated from input article or headline."""
    claim_id: str = Field(..., description="Unique claim identifier e.g. claim_1")
    claim_text: str = Field(..., description="Verbatim or core extracted factual proposition")
    claim_type: ClaimType = Field(default=ClaimType.FACTUAL_ASSERTION)
    named_entities: List[str] = Field(default_factory=list, description="Entities mentioned in the claim")
    temporal_anchor: Optional[str] = Field(None, description="Date or time period referenced")
    geographic_anchor: Optional[str] = Field(None, description="Location referenced")
    is_verifiable: bool = Field(default=True, description="Whether this claim is empirically verifiable")


class InputProcessingResult(BaseModel):
    """Output of Agent 1 (Input Processing Agent)."""
    normalized_text: str
    submitted_url: Optional[str] = None
    url_domain: Optional[str] = None
    claims: List[ExtractedClaim]
    summary_of_context: str
    opinion_or_satire_detected: bool = False


class RetrievedEvidenceSource(BaseModel):
    """A distinct source retrieved by Agent 2 (Evidence Retrieval Agent)."""
    source_id: str
    claim_id: str
    url: str
    publisher: str
    title: str
    publication_date: Optional[str] = None
    relevant_excerpt: str
    retrieval_query: str
    query_intent: str = Field(..., description="'supporting_search' or 'contradicting_search'")
    is_primary_source: bool = False
    source_credibility_tier: str = Field(default="tier_2", description="tier_1 (gov/peer-reviewed/official wire), tier_2 (reputable press), tier_3 (general web)")


class EvidenceAnalysisResult(BaseModel):
    """Output of Agent 3 (Evidence Analysis Agent)."""
    claim_id: str
    stance: StanceType
    stance_confidence: float = Field(ge=0.0, le=1.0)
    supporting_sources: List[RetrievedEvidenceSource] = Field(default_factory=list)
    contradicting_sources: List[RetrievedEvidenceSource] = Field(default_factory=list)
    neutral_or_insufficient_sources: List[RetrievedEvidenceSource] = Field(default_factory=list)
    syndication_groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Groups sources that syndicate the exact same wire copy (not independent)",
    )
    temporal_match: bool = Field(default=True, description="Whether evidence matches the claimed timeframe")
    location_match: bool = Field(default=True, description="Whether evidence matches the claimed location")
    analysis_notes: str = ""


class AdjudicationVerdict(BaseModel):
    """Final output of Agent 4 (Adjudication Agent)."""
    verdict: VerdictType
    verdict_explanation: str
    evidence_summary: str
    ml_agreement: str = Field(
        ...,
        description="'agrees' | 'disagrees' | 'unresolved' between ML classification and real-world evidence",
    )
    ml_predicted_class: str
    ml_model_confidence: float
    recorded_citations: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Strictly valid citations matching retrieved sources",
    )
    conflicting_evidence_present: bool = False
    uncertainty_reasons: List[str] = Field(default_factory=list)
    timestamp: str
