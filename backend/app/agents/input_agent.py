"""Agent 1: Input Processing Agent.

Responsibilities:
- Normalize incoming news text.
- Validate submitted URL.
- Extract candidate factual claims with Pydantic validation.
- Identify named entities, dates, and locations.
- Distinguish factual claims from opinions, predictions, and satire.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import re
import json
import logging
from urllib.parse import urlparse

from backend.app.schemas.agent_schemas import (
    ClaimType,
    ExtractedClaim,
    InputProcessingResult,
)
from ml.preprocessing.text import clean_text, extract_claims_heuristic

from backend.app.agents.retrieval_agent import is_gemini_available, disable_gemini

logger = logging.getLogger(__name__)


class InputProcessingAgent:
    """Isolates testable factual claims from headline, article body, or URL."""

    def __init__(self, gemini_api_key: Optional[str] = None):
        use_gemini = os.getenv("USE_GEMINI", "true").lower() not in ("false", "0", "no")
        self.api_key = (gemini_api_key or os.getenv("GEMINI_API_KEY")) if use_gemini else None

    def _extract_domain(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return None

    def process(
        self,
        text: str,
        submitted_url: Optional[str] = None,
    ) -> InputProcessingResult:
        """Processes and decomposes news item into verifiable claims."""
        normalized = clean_text(text, preserve_urls=True)
        domain = self._extract_domain(submitted_url)

        # If Gemini API key is available, attempt structured extraction via LLM
        if self.api_key and is_gemini_available():
            try:
                claims, is_satire = self._llm_extract(normalized)
                if claims:
                    return InputProcessingResult(
                        normalized_text=normalized,
                        submitted_url=submitted_url,
                        url_domain=domain,
                        claims=claims,
                        summary_of_context=f"Analyzed {len(claims)} claim(s) from article input.",
                        opinion_or_satire_detected=is_satire,
                    )
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    logger.warning("Gemini API quota exhausted; disabling Gemini extraction for remaining session.")
                    disable_gemini()
                else:
                    logger.warning("Gemini LLM extraction failed, using deterministic fallback: %s", e)

        # Deterministic rule-based extraction fallback
        return self._deterministic_extract(normalized, submitted_url, domain)

    def _deterministic_extract(
        self,
        text: str,
        url: Optional[str],
        domain: Optional[str],
    ) -> InputProcessingResult:
        """Deterministic heuristic fallback when external LLM is offline or unconfigured."""
        candidate_sentences = extract_claims_heuristic(text)
        claims: List[ExtractedClaim] = []

        for idx, sentence in enumerate(candidate_sentences, start=1):
            # Extract simple capitalized word entities
            entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence)
            # Filter common stopword capitalizations
            entities = [e for e in entities if e.lower() not in {"the", "a", "an", "this", "that", "in", "on"}]

            # Detect year / date patterns
            dates = re.findall(r"\b(19\d\d|20\d\d|january|february|march|april|may|june|july|august|september|october|november|december)\b", sentence, re.IGNORECASE)
            temp_anchor = dates[0] if dates else None

            claims.append(
                ExtractedClaim(
                    claim_id=f"claim_{idx}",
                    claim_text=sentence,
                    claim_type=ClaimType.FACTUAL_ASSERTION,
                    named_entities=list(set(entities)),
                    temporal_anchor=temp_anchor,
                    geographic_anchor=None,
                    is_verifiable=True,
                )
            )

        is_satire_cue = bool(re.search(r"\b(satire|parody|the onion|babylon bee|humor)\b", text, re.IGNORECASE))

        return InputProcessingResult(
            normalized_text=text,
            submitted_url=url,
            url_domain=domain,
            claims=claims,
            summary_of_context=f"Extracted {len(claims)} claim(s) via deterministic heuristic.",
            opinion_or_satire_detected=is_satire_cue,
        )

    def _llm_extract(self, text: str) -> Tuple[List[ExtractedClaim], bool]:
        """Uses Gemini API to extract claims with strict JSON schema."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        prompt = (
            "You are the Input Processing Agent in a fact-checking pipeline. "
            "Analyze the following text and extract specific, checkable factual claims. "
            "Do NOT invent claims that are absent from the text.\n\n"
            f"TEXT:\n{text}\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            '  "is_satire_or_opinion": false,\n'
            '  "claims": [\n'
            '    {\n'
            '      "claim_id": "claim_1",\n'
            '      "claim_text": "verbatim or concise factual proposition",\n'
            '      "claim_type": "factual_assertion",\n'
            '      "named_entities": ["Entity1"],\n'
            '      "temporal_anchor": "2024",\n'
            '      "geographic_anchor": "Location",\n'
            '      "is_verifiable": true\n'
            '    }\n'
            '  ]\n'
            "}"
        )

        models_to_try = ["gemini-2.5-flash", "gemini-3.8-flash"]
        response = None
        for m in models_to_try:
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                if response and response.text:
                    break
            except Exception as e:
                logger.warning("Gemini model %s extraction failed (%s), trying next", m, e)

        if not response or not response.text:
            disable_gemini()
            raise RuntimeError("All Gemini models failed or quota exhausted.")

        parsed = json.loads(response.text)
        claims = [ExtractedClaim(**c) for c in parsed.get("claims", [])]
        is_satire = parsed.get("is_satire_or_opinion", False)
        return claims, is_satire
