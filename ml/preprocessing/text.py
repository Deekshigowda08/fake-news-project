"""Text preprocessing and cleaning utilities for fake news detection."""

import re
import unicodedata
from typing import Dict, List, Optional
import html


def clean_text(text: str, preserve_urls: bool = False) -> str:
    """Cleans raw text while preserving semantic cues.
    
    - Unescapes HTML entities.
    - Normalizes unicode characters (NFKC).
    - Removes zero-width characters and excessive whitespace.
    - Optionally strips raw http/https links or normalizes them.
    """
    if not text:
        return ""

    # Unescape HTML
    text = html.unescape(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove zero-width spaces and control characters (except newline/tab)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))

    if not preserve_urls:
        # Normalize URLs to generic token to prevent overfitting to specific domains
        text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

    # Collapse repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)

    return text.strip()


def extract_claims_heuristic(text: str) -> List[str]:
    """Lightweight rule-based sentence segmentation for candidate factual claims."""
    cleaned = clean_text(text, preserve_urls=True)
    if not cleaned:
        return []

    # Split by sentence terminators
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    candidate_claims = []

    for s in sentences:
        s_clean = s.strip()
        # Keep declarative sentences with at least 4 words
        words = s_clean.split()
        if len(words) >= 4 and not s_clean.endswith("?"):
            candidate_claims.append(s_clean)

    # If no sentences met criteria, return original cleaned text
    if not candidate_claims and len(cleaned.split()) >= 3:
        candidate_claims.append(cleaned)

    return candidate_claims
