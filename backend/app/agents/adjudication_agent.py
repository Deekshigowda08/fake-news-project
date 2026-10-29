"""Agent 4: Adjudication Agent.

Responsibilities:
- Synthesizes findings from ML classification and external evidence verification.
- Enforces strict deterministic rules for final verdicts:
  TRUE, FALSE, MISLEADING, UNVERIFIABLE.
- NEVER fabricates citations or overwrites ML model probabilities.
- Explains agreements and contradictions between ML patterns and empirical facts.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from backend.app.schemas.agent_schemas import (
    AdjudicationVerdict,
    EvidenceAnalysisResult,
    RetrievedEvidenceSource,
    StanceType,
    VerdictType,
)

logger = logging.getLogger(__name__)


class AdjudicationAgent:
    """Combines ML prediction and evidence stances to produce final explainable verdict."""

    def adjudicate(
        self,
        ml_prediction: Dict[str, Any],
        claim_analyses: List[EvidenceAnalysisResult],
    ) -> AdjudicationVerdict:
        """Determines final verdict, records valid citations, and compares with ML model."""
        ml_pred_class = ml_prediction.get("predicted_class", "unknown").lower()
        ml_confidence = float(ml_prediction.get("confidence", 0.5))

        # Collect all sources across all claims
        all_supporting: List[RetrievedEvidenceSource] = []
        all_contradicting: List[RetrievedEvidenceSource] = []
        all_insufficient: List[RetrievedEvidenceSource] = []
        uncertainties: List[str] = []

        for analysis in claim_analyses:
            all_supporting.extend(analysis.supporting_sources)
            all_contradicting.extend(analysis.contradicting_sources)
            all_insufficient.extend(analysis.neutral_or_insufficient_sources)

        has_supp = len(all_supporting) > 0
        has_contra = len(all_contradicting) > 0

        # Deterministic Verdict Rules
        if not has_supp and not has_contra:
            verdict = VerdictType.UNVERIFIABLE
            if "fake" in ml_pred_class and ml_confidence >= 0.60:
                verdict_exp = (
                    f"No authoritative empirical sources corroborate this claim. Neural linguistic models "
                    f"detected strong disinformation patterns ({round(ml_confidence * 100, 1)}% confidence). "
                    f"The assertion is unverified and unreliable."
                )
                evidence_sum = "Zero corroborating records located; neural models flag high likelihood of false or clickbait material."
            else:
                verdict_exp = (
                    "Insufficient external empirical evidence was retrieved to substantiate or refute the claim. "
                    "The system abstains from issuing a definitive factual verdict."
                )
                evidence_sum = "No corroborating or refuting authoritative records located."
            uncertainties.append("Zero authoritative corroborating citations located.")

        elif has_contra and not has_supp:
            verdict = VerdictType.FALSE
            tier1_sources = [s for s in all_contradicting if s.source_credibility_tier == "tier_1"]
            if tier1_sources:
                verdict_exp = (
                    f"Directly contradicted by {len(tier1_sources)} primary/authoritative record(s) "
                    f"including {tier1_sources[0].publisher}."
                )
            else:
                verdict_exp = f"Contradicted by {len(all_contradicting)} independent source(s)."
            evidence_sum = f"Identified {len(all_contradicting)} contradicting citation(s)."

        elif has_supp and not has_contra:
            verdict = VerdictType.TRUE
            tier1_sources = [s for s in all_supporting if s.source_credibility_tier == "tier_1"]
            if tier1_sources:
                verdict_exp = (
                    f"Corroborated by {len(tier1_sources)} primary/authoritative record(s) "
                    f"including {tier1_sources[0].publisher}."
                )
            else:
                verdict_exp = f"Supported by {len(all_supporting)} independent source(s)."
            evidence_sum = f"Identified {len(all_supporting)} supporting citation(s)."

        else:
            # Both supporting and contradicting sources exist
            verdict = VerdictType.MISLEADING
            verdict_exp = (
                "Retrieved sources present conflicting or mixed reports. The claim likely uses real elements "
                "out of original context or reports contested assertions."
            )
            evidence_sum = (
                f"Conflict detected: {len(all_supporting)} supporting vs {len(all_contradicting)} contradicting sources."
            )
            uncertainties.append("Conflicting evidence from multiple reporting bodies.")

        # ML vs Evidence Agreement Comparison
        # Map ML classes to boolean logic: 'real' -> True, 'fake' -> False
        if verdict in {VerdictType.TRUE, VerdictType.FALSE}:
            if (ml_pred_class == "real" and verdict == VerdictType.TRUE) or (
                ml_pred_class == "fake" and verdict == VerdictType.FALSE
            ):
                ml_agreement = "agrees"
            else:
                ml_agreement = "disagrees"
        else:
            ml_agreement = "unresolved"

        # Record strictly validated citations
        citations = []
        seen_urls = set()
        for s in all_supporting + all_contradicting + all_insufficient:
            if s.url in seen_urls:
                continue
            seen_urls.add(s.url)
            stance = "SUPPORTS" if s in all_supporting else ("CONTRADICTS" if s in all_contradicting else "INSUFFICIENT")
            citations.append({
                "source_id": s.source_id,
                "url": s.url,
                "publisher": s.publisher,
                "title": s.title,
                "stance": stance,
                "credibility_tier": s.source_credibility_tier,
                "excerpt": s.relevant_excerpt,
            })

        now_str = datetime.now(timezone.utc).isoformat()

        return AdjudicationVerdict(
            verdict=verdict,
            verdict_explanation=verdict_exp,
            evidence_summary=evidence_sum,
            ml_agreement=ml_agreement,
            ml_predicted_class=ml_pred_class.upper() + "_CLASSIFIED",
            ml_model_confidence=ml_confidence,
            recorded_citations=citations,
            conflicting_evidence_present=(has_supp and has_contra),
            uncertainty_reasons=uncertainties,
            timestamp=now_str,
        )
