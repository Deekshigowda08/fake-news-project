"""Unit and integration tests for backend, agents, orchestrator, and security."""

import pytest
from backend.app.security.url_validator import validate_safe_url, SecurityValidationError
from backend.app.schemas.agent_schemas import (
    ExtractedClaim,
    ClaimType,
    RetrievedEvidenceSource,
    StanceType,
    VerdictType,
)
from backend.app.agents.input_agent import InputProcessingAgent
from backend.app.agents.retrieval_agent import EvidenceRetrievalAgent
from backend.app.agents.analysis_agent import EvidenceAnalysisAgent
from backend.app.agents.adjudication_agent import AdjudicationAgent
from backend.app.orchestrator.orchestrator import VerificationOrchestrator


def test_ssrf_url_validation_rejection():
    """Verifies that SSRF attack vectors and private IP ranges are strictly rejected."""
    # Localhost and loopback
    with pytest.raises(SecurityValidationError):
        validate_safe_url("http://localhost:8000/admin")

    with pytest.raises(SecurityValidationError):
        validate_safe_url("http://127.0.0.1/secret")

    with pytest.raises(SecurityValidationError):
        validate_safe_url("http://127.0.0.1:22")

    # Private RFC 1918 networks
    with pytest.raises(SecurityValidationError):
        validate_safe_url("http://192.168.1.1/router")

    with pytest.raises(SecurityValidationError):
        validate_safe_url("http://10.0.0.5:5432")

    # AWS/GCP cloud metadata IP
    with pytest.raises(SecurityValidationError):
        validate_safe_url("http://169.254.169.254/latest/meta-data/")

    # Invalid protocols
    with pytest.raises(SecurityValidationError):
        validate_safe_url("file:///etc/passwd")

    with pytest.raises(SecurityValidationError):
        validate_safe_url("ftp://internal.repo/data")


def test_agent1_input_processing():
    """Tests factual claim decomposition and entity detection."""
    agent = InputProcessingAgent()
    text = "NASA launched the Europa Clipper spacecraft on Monday to explore Jupiter's icy moon."
    result = agent.process(text)

    assert len(result.claims) >= 1
    assert "NASA" in result.claims[0].named_entities or "Europa Clipper" in result.claims[0].named_entities
    assert result.claims[0].is_verifiable is True


def test_agent2_evidence_retrieval_dual_queries():
    """Tests dual retrieval path (supporting and contradicting search)."""
    agent = EvidenceRetrievalAgent(provider="mock")
    claim = ExtractedClaim(
        claim_id="claim_test_1",
        claim_text="NASA confirmed the Moon is hollow and artificial radio transmitters were installed inside.",
        claim_type=ClaimType.FACTUAL_ASSERTION,
    )

    sources = agent.retrieve_for_claim(claim)
    assert len(sources) > 0

    intents = {s.query_intent for s in sources}
    assert "supporting_search" in intents or "contradicting_search" in intents


def test_agent3_evidence_analysis_and_syndication():
    """Tests stance classification and source syndication clustering."""
    agent = EvidenceAnalysisAgent()
    claim = ExtractedClaim(
        claim_id="c1",
        claim_text="Drinking boiled lemon water cures cancer.",
    )

    source1 = RetrievedEvidenceSource(
        source_id="s1",
        claim_id="c1",
        url="https://www.reuters.com/fact-check/lemon-water-cancer-hoax",
        publisher="Reuters",
        title="Fact Check: No scientific evidence that lemon water cures cancer",
        relevant_excerpt="Medical oncologists confirm this is a persistent online hoax with zero clinical basis.",
        retrieval_query="lemon water cancer hoax",
        query_intent="contradicting_search",
        source_credibility_tier="tier_1",
    )
    # Syndicated duplicate of source1
    source2 = RetrievedEvidenceSource(
        source_id="s2",
        claim_id="c1",
        url="https://www.yahoo.com/news/reuters-lemon-cancer-hoax",
        publisher="Yahoo News (Reuters Syndicate)",
        title="Fact Check: No scientific evidence that lemon water cures cancer",
        relevant_excerpt="Medical oncologists confirm this is a persistent online hoax with zero clinical basis.",
        retrieval_query="lemon water cancer hoax",
        query_intent="contradicting_search",
        source_credibility_tier="tier_2",
    )

    analysis = agent.analyze_claim_evidence(claim, [source1, source2])
    assert analysis.stance == StanceType.CONTRADICTS
    assert analysis.stance_confidence > 0.7
    assert len(analysis.contradicting_sources) == 2
    # Check that syndication was detected
    assert len(analysis.syndication_groups) == 1


def test_agent4_adjudication_verdicts():
    """Tests deterministic verdict rules and citation transparency."""
    agent = AdjudicationAgent()

    # Case A: Strong contradiction -> FALSE
    from backend.app.schemas.agent_schemas import EvidenceAnalysisResult

    contra_src = RetrievedEvidenceSource(
        source_id="s_contra",
        claim_id="c1",
        url="https://www.who.int/debunk",
        publisher="WHO",
        title="WHO statement",
        relevant_excerpt="Claim is disproven and fabricated.",
        retrieval_query="query",
        query_intent="contradicting_search",
        source_credibility_tier="tier_1",
    )
    analysis_contra = EvidenceAnalysisResult(
        claim_id="c1",
        stance=StanceType.CONTRADICTS,
        stance_confidence=0.9,
        contradicting_sources=[contra_src],
    )
    verdict_false = agent.adjudicate(
        ml_prediction={"predicted_class": "fake", "confidence": 0.85},
        claim_analyses=[analysis_contra],
    )
    assert verdict_false.verdict == VerdictType.FALSE
    assert verdict_false.ml_agreement == "agrees"
    assert len(verdict_false.recorded_citations) == 1
    assert verdict_false.recorded_citations[0]["url"] == "https://www.who.int/debunk"

    # Case B: Zero evidence -> UNVERIFIABLE
    verdict_empty = agent.adjudicate(
        ml_prediction={"predicted_class": "real", "confidence": 0.60},
        claim_analyses=[],
    )
    assert verdict_empty.verdict == VerdictType.UNVERIFIABLE
    assert verdict_empty.ml_agreement == "unresolved"


def test_verification_orchestrator_end_to_end():
    """Runs complete deterministic orchestrator pipeline."""
    orchestrator = VerificationOrchestrator()
    events_logged = []

    def callback(ev):
        events_logged.append(ev["event_type"])

    result = orchestrator.run_verification(
        text="NASA's James Webb Space Telescope reveals atmospheric composition of exoplanet.",
        ml_prediction={"predicted_class": "real", "confidence": 0.90},
        event_callback=callback,
    )

    assert "adjudication" in result
    assert result["adjudication"]["verdict"] in [v.value for v in VerdictType]
    assert "extraction_started" in events_logged
    assert "adjudication_completed" in events_logged
