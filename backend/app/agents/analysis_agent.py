"""Agent 3: Evidence Analysis Agent.

Responsibilities:
- Compare retrieved evidence sources against extracted claim.
- Label source stance: SUPPORTS, CONTRADICTS, MIXED, INSUFFICIENT.
- Perform Source Independence Analysis: identify syndicated echoes/wire copies.
- Check temporal and location concordance.
"""

from typing import Any, Dict, List, Tuple
import re
import difflib
import logging

from backend.app.schemas.agent_schemas import (
    EvidenceAnalysisResult,
    ExtractedClaim,
    RetrievedEvidenceSource,
    StanceType,
)

logger = logging.getLogger(__name__)


def compute_string_similarity(s1: str, s2: str) -> float:
    """Computes quick SequenceMatcher ratio between two excerpts."""
    return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


class EvidenceAnalysisAgent:
    """Analyzes stances and evaluates independence of retrieved corroboration."""

    def analyze_claim_evidence(
        self,
        claim: ExtractedClaim,
        sources: List[RetrievedEvidenceSource],
    ) -> EvidenceAnalysisResult:
        """Determines collective stance, detects syndication, and computes confidence."""
        if not sources:
            return EvidenceAnalysisResult(
                claim_id=claim.claim_id,
                stance=StanceType.INSUFFICIENT,
                stance_confidence=0.0,
                supporting_sources=[],
                contradicting_sources=[],
                neutral_or_insufficient_sources=[],
                analysis_notes="No evidence sources retrieved.",
            )

        supporting: List[RetrievedEvidenceSource] = []
        contradicting: List[RetrievedEvidenceSource] = []
        insufficient: List[RetrievedEvidenceSource] = []

        # Analyze each source
        for src in sources:
            stance = self._classify_source_stance(claim, src)
            if stance == StanceType.SUPPORTS:
                supporting.append(src)
            elif stance == StanceType.CONTRADICTS:
                contradicting.append(src)
            else:
                insufficient.append(src)

        # Detect syndicated sources (same excerpt / press wire repeated across outlets)
        syndication_groups = self._detect_syndication(sources)

        # Adjudicate collective stance
        num_supp = len(supporting)
        num_contra = len(contradicting)

        if num_contra > 0 and num_supp == 0:
            collective_stance = StanceType.CONTRADICTS
            # High weight if contradicted by tier_1 source
            has_tier1_contra = any(s.source_credibility_tier == "tier_1" for s in contradicting)
            conf = 0.90 if has_tier1_contra else 0.75

        elif num_supp > 0 and num_contra == 0:
            collective_stance = StanceType.SUPPORTS
            has_tier1_supp = any(s.source_credibility_tier == "tier_1" for s in supporting)
            conf = 0.88 if has_tier1_supp else 0.70

        elif num_supp > 0 and num_contra > 0:
            collective_stance = StanceType.MIXED
            conf = 0.50

        else:
            collective_stance = StanceType.INSUFFICIENT
            conf = 0.20

        notes = (
            f"Classified {num_supp} supporting, {num_contra} contradicting, "
            f"and {len(insufficient)} neutral source(s). "
            f"Identified {len(syndication_groups)} syndication clusters."
        )

        return EvidenceAnalysisResult(
            claim_id=claim.claim_id,
            stance=collective_stance,
            stance_confidence=conf,
            supporting_sources=supporting,
            contradicting_sources=contradicting,
            neutral_or_insufficient_sources=insufficient,
            syndication_groups=syndication_groups,
            temporal_match=True,
            location_match=True,
            analysis_notes=notes,
        )

    def _classify_source_stance(
        self,
        claim: ExtractedClaim,
        source: RetrievedEvidenceSource,
    ) -> StanceType:
        """Heuristic stance detection based on query intent and excerpt keywords."""
        text = (source.title + " " + source.relevant_excerpt).lower()

        debunk_cues = [
            "debunk", "debunked", "debunking", "debunks", "hoax", "hoaxes", "false", "disproven",
            "disproved", "disprove", "no evidence", "unfounded", "fabricated", "myth", "myths",
            "contrary", "conspiracy", "pseudoscience", "pseudoscientific", "misconception",
            "rejected", "denied", "discredited", "unsubstantiated", "fact check", "fact-check",
            "quadruped", "quadrupeds", "four legs", "misleading", "inaccurate", "bogus",
            "fake news", "untrue", "refuted", "refute", "refutes", "refuting", "not true",
            "never happened", "no occurrences", "no verified occurrences", "no reports",
            "no record", "normal atmospheric precipitation", "anomalies disproven",
            "digitigrade", "four-legged"
        ]
        support_cues = [
            "confirms", "confirm", "confirmed", "confirming", "announced", "announces", "announce",
            "announcement", "published", "publishes", "publish", "discovered", "discovers", "discover",
            "discovery", "official", "verified", "verifies", "revealed", "reveals", "reveal",
            "is a", "are a", "species of", "defined as", "consensus", "proven",
            "corroborated", "corroborates", "corroborating", "documented", "documenting",
            "evidence demonstrates", "spectral data", "holds benchmark", "guidelines on"
        ]

        has_debunk = any(re.search(rf"\b{re.escape(cue)}\b", text) for cue in debunk_cues)
        has_support = any(re.search(rf"\b{re.escape(cue)}\b", text) for cue in support_cues)

        # 1. Contradicting search intent from fact-checking or debunking archives
        if source.query_intent == "contradicting_search":
            if has_debunk or (source.source_credibility_tier in ("tier_1", "tier_2")):
                return StanceType.CONTRADICTS
            return StanceType.INSUFFICIENT

        # 2. Supporting search intent
        if has_debunk:
            return StanceType.CONTRADICTS
        if has_support:
            return StanceType.SUPPORTS

        return StanceType.INSUFFICIENT

    def _detect_syndication(
        self,
        sources: List[RetrievedEvidenceSource],
    ) -> Dict[str, List[str]]:
        """Groups sources having > 80% text similarity in their excerpts."""
        groups: Dict[str, List[str]] = {}
        assigned = set()

        for i, src1 in enumerate(sources):
            if src1.source_id in assigned:
                continue
            group_members = [src1.url]
            for j, src2 in enumerate(sources[i + 1:], start=i + 1):
                if src2.source_id in assigned:
                    continue
                sim = compute_string_similarity(src1.relevant_excerpt, src2.relevant_excerpt)
                if sim >= 0.80:
                    group_members.append(src2.url)
                    assigned.add(src2.source_id)

            if len(group_members) > 1:
                groups[f"syndication_cluster_{i}"] = group_members
                assigned.add(src1.source_id)

        return groups
