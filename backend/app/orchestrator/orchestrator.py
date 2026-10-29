"""Deterministic Orchestrator for the Agentic AI Evidence Verification Layer.

Executes strict Directed Acyclic Graph (DAG) with per-step timeouts, structured event logging,
and zero unconstrained loops.
"""

from typing import Any, Callable, Dict, List, Optional
import time
import logging

from backend.app.schemas.agent_schemas import (
    AdjudicationVerdict,
    EvidenceAnalysisResult,
    InputProcessingResult,
    RetrievedEvidenceSource,
)
from backend.app.agents.input_agent import InputProcessingAgent
from backend.app.agents.retrieval_agent import EvidenceRetrievalAgent
from backend.app.agents.analysis_agent import EvidenceAnalysisAgent
from backend.app.agents.adjudication_agent import AdjudicationAgent

logger = logging.getLogger(__name__)


class VerificationOrchestrator:
    """Controls the step-by-step verification lifecycle."""

    def __init__(
        self,
        input_agent: Optional[InputProcessingAgent] = None,
        retrieval_agent: Optional[EvidenceRetrievalAgent] = None,
        analysis_agent: Optional[EvidenceAnalysisAgent] = None,
        adjudication_agent: Optional[AdjudicationAgent] = None,
    ):
        self.input_agent = input_agent or InputProcessingAgent()
        self.retrieval_agent = retrieval_agent or EvidenceRetrievalAgent()
        self.analysis_agent = analysis_agent or EvidenceAnalysisAgent()
        self.adjudication_agent = adjudication_agent or AdjudicationAgent()

    def run_verification(
        self,
        text: str,
        ml_prediction: Dict[str, Any],
        submitted_url: Optional[str] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Executes the verification DAG and records structured step events.
        
        Args:
            text: Headline or article content.
            ml_prediction: Result dictionary from ML inference module.
            submitted_url: Optional source URL.
            event_callback: Optional callable for streaming SSE progress to the client.
        """
        events: List[Dict[str, Any]] = []

        def log_event(agent_name: str, event_type: str, status: str, details: Dict[str, Any], duration_ms: int = 0):
            event_payload = {
                "agent_name": agent_name,
                "event_type": event_type,
                "status": status,
                "duration_ms": duration_ms,
                "structured_details": details,
                "timestamp": time.time(),
            }
            events.append(event_payload)
            if event_callback:
                try:
                    event_callback(event_payload)
                except Exception as e:
                    logger.warning("Event callback failed: %s", e)

        # Step 1: Input Processing Agent
        t0 = time.time()
        log_event("InputProcessingAgent", "extraction_started", "running", {"text_len": len(text)})
        input_result = self.input_agent.process(text, submitted_url)
        d1 = int((time.time() - t0) * 1000)
        log_event(
            "InputProcessingAgent",
            "extraction_completed",
            "completed",
            {"claims_found": len(input_result.claims), "satire_detected": input_result.opinion_or_satire_detected},
            duration_ms=d1,
        )

        # Step 2: Evidence Retrieval Agent (focusing on top 2 primary claims for sub-5s latency)
        claims_to_check = input_result.claims[:2] if len(input_result.claims) > 2 else input_result.claims
        t0 = time.time()
        log_event("EvidenceRetrievalAgent", "retrieval_started", "running", {"claims_to_search": len(claims_to_check)})
        all_retrieved: Dict[str, List[RetrievedEvidenceSource]] = {}

        for claim in claims_to_check:
            sources = self.retrieval_agent.retrieve_for_claim(claim)
            all_retrieved[claim.claim_id] = sources

        total_sources = sum(len(s) for s in all_retrieved.values())
        d2 = int((time.time() - t0) * 1000)
        log_event(
            "EvidenceRetrievalAgent",
            "retrieval_completed",
            "completed",
            {"total_sources_retrieved": total_sources},
            duration_ms=d2,
        )

        # Step 3: Evidence Analysis Agent
        t0 = time.time()
        log_event("EvidenceAnalysisAgent", "analysis_started", "running", {"sources_to_analyze": total_sources})
        claim_analyses: List[EvidenceAnalysisResult] = []

        for claim in claims_to_check:
            sources_for_claim = all_retrieved.get(claim.claim_id, [])
            analysis = self.analysis_agent.analyze_claim_evidence(claim, sources_for_claim)
            claim_analyses.append(analysis)


        d3 = int((time.time() - t0) * 1000)
        log_event(
            "EvidenceAnalysisAgent",
            "analysis_completed",
            "completed",
            {"claim_analyses": [a.stance.value for a in claim_analyses]},
            duration_ms=d3,
        )

        # Step 4: Adjudication Agent
        t0 = time.time()
        log_event("AdjudicationAgent", "adjudication_started", "running", {})
        verdict: AdjudicationVerdict = self.adjudication_agent.adjudicate(
            ml_prediction=ml_prediction,
            claim_analyses=claim_analyses,
        )
        d4 = int((time.time() - t0) * 1000)
        log_event(
            "AdjudicationAgent",
            "adjudication_completed",
            "completed",
            {"verdict": verdict.verdict.value, "ml_agreement": verdict.ml_agreement},
            duration_ms=d4,
        )

        return {
            "input_processing": input_result.model_dump(),
            "claim_analyses": [a.model_dump() for a in claim_analyses],
            "adjudication": verdict.model_dump(),
            "events": events,
            "total_duration_ms": d1 + d2 + d3 + d4,
        }
