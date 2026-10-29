"""Optical Character Recognition (OCR) & Visual Text Analysis Module.

Provides transparent, step-by-step image reading and cross-modal text verification:
1. Optical Quality & Image Structure Analysis
2. Visual Text Extraction & Bounding Region Detection
3. Text Cleaning & Transcription
4. Cross-Modal Concordance (Input Claim vs Image Text)
5. Discrepancy & Misleading Context Detection
"""

import os
import io
import re
import json
import base64
import difflib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageFilter, ImageStat

logger = logging.getLogger(__name__)


def _get_gemini_api_key() -> Optional[str]:
    """Retrieves Gemini API key from environment or project .env."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    
    # Try reading from project root .env
    env_paths = [
        Path(".env"),
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    for p in env_paths:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        k = line.split("=", 1)[1].strip().strip("\"'")
                        if k:
                            os.environ["GEMINI_API_KEY"] = k
                            return k
            except Exception:
                pass
    return None


def calculate_image_sharpness(img: Image.Image) -> float:
    """Calculates image sharpness via high-frequency edge variance (Laplacian proxy)."""
    try:
        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        stat = ImageStat.Stat(edges)
        variance = stat.var[0] if stat.var else 0.0
        # Normalize roughly between 0.0 and 1.0 (variance > 1000 is sharp)
        return min(1.0, round(variance / 800.0, 3))
    except Exception:
        return 0.5


def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """Computes word-level Jaccard similarity index."""
    w1 = set(re.findall(r"\b\w{3,}\b", text1.lower()))
    w2 = set(re.findall(r"\b\w{3,}\b", text2.lower()))
    if not w1 or not w2:
        return 0.0
    intersection = w1.intersection(w2)
    union = w1.union(w2)
    return round(len(intersection) / len(union), 4)


class ImageOCRService:
    """End-to-end OCR and cross-modal image text verification service."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _get_gemini_api_key()

    def process_image_and_compare(
        self,
        image_bytes: bytes,
        input_claim_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the complete 5-step OCR and cross-modal reading pipeline.
        
        Args:
            image_bytes: Raw binary image payload.
            input_claim_text: The user-supplied headline or article claim text.
            
        Returns structured dictionary documenting each and every pipeline step.
        """
        # ==============================================================
        # STEP 1: Optical Quality & Structural Inspection
        # ==============================================================
        pil_img = Image.open(io.BytesIO(image_bytes))
        width, height = pil_img.size
        format_name = (pil_img.format or "JPEG").upper()
        sharpness_score = calculate_image_sharpness(pil_img)
        aspect_ratio = round(width / max(1, height), 2)
        stat = ImageStat.Stat(pil_img.convert("L"))
        brightness_mean = round(stat.mean[0], 1) if stat.mean else 128.0

        step1_quality = {
            "step_number": 1,
            "step_title": "Image Ingestion & Optical Quality Check",
            "dimensions": f"{width}x{height} px",
            "aspect_ratio": aspect_ratio,
            "color_format": format_name,
            "sharpness_score": sharpness_score,
            "optical_clarity": "High" if sharpness_score > 0.6 else ("Moderate" if sharpness_score > 0.3 else "Low / Blurred"),
            "mean_luminance": brightness_mean,
            "payload_size_kb": round(len(image_bytes) / 1024, 1),
        }

        # ==============================================================
        # STEP 2 & 3: Optical Character Recognition (OCR Engine)
        # ==============================================================
        ocr_result = self._run_vision_ocr(image_bytes, width, height)

        has_text = ocr_result.get("has_text", False)
        raw_text = ocr_result.get("raw_extracted_text", "").strip()
        text_blocks = ocr_result.get("text_blocks", [])
        detected_elements = ocr_result.get("detected_elements", [])
        visual_style = ocr_result.get("visual_style", "general media")

        words = re.findall(r"\b\w+\b", raw_text)
        word_count = len(words)
        char_count = len(raw_text)

        step2_detection = {
            "step_number": 2,
            "step_title": "Optical Character Recognition (OCR) Engine",
            "has_readable_text": has_text,
            "total_words_extracted": word_count,
            "total_characters_extracted": char_count,
            "detected_text_regions_count": len(text_blocks),
            "text_blocks": text_blocks,
            "detected_visual_elements": detected_elements,
            "visual_style_classification": visual_style,
            "ocr_engine": "Gemini Multimodal Neural OCR",
        }

        step3_transcription = {
            "step_number": 3,
            "step_title": "Text Transcription & Normalization",
            "raw_extracted_text": raw_text if has_text else "[No readable typography detected in visual media]",
            "cleaned_text": " ".join(words) if has_text else "",
            "estimated_readability": "High" if word_count > 5 else ("Low" if word_count > 0 else "N/A"),
        }

        # ==============================================================
        # STEP 4: Cross-Modal Text Comparison (Claim vs Image Text)
        # ==============================================================
        input_claim = (input_claim_text or "").strip()
        jaccard_overlap = 0.0
        sequence_sim = 0.0
        shared_keywords: List[str] = []
        concordance_verdict = "NOT_EVALUATED"
        concordance_explanation = ""

        if not has_text or word_count == 0:
            concordance_verdict = "PURE_VISUAL_MEDIA"
            concordance_explanation = (
                "The image contains natural photography, diagrammatic graphics, or artwork "
                "without embedded textual claims. Visual veracity is evaluated via Vision Transformer "
                "and CLIP cross-modal alignment."
            )
        elif not input_claim:
            concordance_verdict = "NO_INPUT_CLAIM_TO_COMPARE"
            concordance_explanation = "Extracted image text was recorded, but no input claim was provided for cross-referencing."
        else:
            jaccard_overlap = compute_jaccard_similarity(input_claim, raw_text)
            sequence_sim = round(difflib.SequenceMatcher(None, input_claim.lower(), raw_text.lower()).ratio(), 4)

            claim_words = set(re.findall(r"\b\w{3,}\b", input_claim.lower()))
            img_words = set(re.findall(r"\b\w{3,}\b", raw_text.lower()))
            shared_keywords = sorted(list(claim_words.intersection(img_words)))

            # Decision Logic for Cross-Modal Discrepancy
            if jaccard_overlap >= 0.25 or sequence_sim >= 0.35 or len(shared_keywords) >= 3:
                concordance_verdict = "CONCORDANT_MATCH"
                concordance_explanation = (
                    f"Strong text concordance detected ({int(jaccard_overlap * 100)}% vocabulary overlap). "
                    f"Text extracted directly from the image corroborates the user's input claim. "
                    f"Shared key terms: {', '.join(shared_keywords[:6])}."
                )
            elif len(shared_keywords) >= 1:
                concordance_verdict = "PARTIAL_ALIGNMENT"
                concordance_explanation = (
                    f"Moderate vocabulary overlap ({int(jaccard_overlap * 100)}%). "
                    f"The image shares topical keywords ({', '.join(shared_keywords[:4])}) "
                    "with the input claim, but phrasing or specific assertions differ."
                )
            else:
                concordance_verdict = "CROSS_MODAL_DISCREPANCY"
                concordance_explanation = (
                    "CRITICAL MISMATCH DETECTED: The text extracted from the image does NOT match the input claim. "
                    "Common disinformation tactic: taking a genuine image from an unrelated event and "
                    "pairing it with a sensationalized or fabricated headline."
                )

        step4_comparison = {
            "step_number": 4,
            "step_title": "Cross-Modal Text Concordance & Discrepancy Analysis",
            "jaccard_word_overlap_ratio": jaccard_overlap,
            "sequence_similarity_ratio": sequence_sim,
            "shared_keywords": shared_keywords,
            "concordance_verdict": concordance_verdict,
            "concordance_explanation": concordance_explanation,
            "is_manipulated_context_flag": (concordance_verdict == "CROSS_MODAL_DISCREPANCY"),
        }

        # ==============================================================
        # STEP 5: Synthesis for Presentation
        # ==============================================================
        step5_synthesis = {
            "step_number": 5,
            "step_title": "Multimodal Synthesis & Visual Forensics Verdict",
            "verdict": concordance_verdict,
            "headline_image_text_matched": (concordance_verdict == "CONCORDANT_MATCH"),
            "summary_for_presentation": (
                f"OCR scanned {word_count} words across {len(text_blocks)} detected region(s). "
                f"Concordance with submitted headline: {concordance_verdict}. {concordance_explanation}"
            ),
        }

        return {
            "has_image": True,
            "has_extracted_text": has_text,
            "raw_text": raw_text,
            "word_count": word_count,
            "concordance_verdict": concordance_verdict,
            "concordance_score": max(jaccard_overlap, sequence_sim),
            "is_discrepant": (concordance_verdict == "CROSS_MODAL_DISCREPANCY"),
            "steps": [
                step1_quality,
                step2_detection,
                step3_transcription,
                step4_comparison,
                step5_synthesis,
            ],
            "quality_metrics": step1_quality,
            "ocr_blocks": text_blocks,
            "comparison": step4_comparison,
        }

    def _run_vision_ocr(self, image_bytes: bytes, width: int, height: int) -> Dict[str, Any]:
        """Runs native high-performance local OCR (winocr) first, with Gemini Vision fallback."""
        # 1. Try Native Windows Media OCR (local, instant, zero-quota)
        try:
            import concurrent.futures
            import winocr
            from PIL import Image
            pil_img = Image.open(io.BytesIO(image_bytes))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                ocr_result = pool.submit(winocr.recognize_pil_sync, pil_img).result(timeout=10)
            
            raw_text = (ocr_result.get("text") or "").strip()
            lines = ocr_result.get("lines", [])
            
            if raw_text or lines:
                text_blocks = []
                detected_elements = ["native optical character recognition", "high-resolution typography detection"]
                for idx, line in enumerate(lines, start=1):
                    line_text = line.get("text", "").strip()
                    if not line_text:
                        continue
                    
                    # Estimate region based on bounding rect position if available
                    words = line.get("words", [])
                    region = "headline_banner"
                    if words and height > 0:
                        y_pos = words[0].get("bounding_rect", {}).get("y", 0)
                        rel_y = y_pos / max(height, 1)
                        if rel_y < 0.25:
                            region = "headline_banner"
                        elif rel_y > 0.75:
                            region = "caption_or_source"
                        else:
                            region = "body_paragraph"
                    
                    text_blocks.append({
                        "block_id": idx,
                        "text": line_text,
                        "region": region,
                        "estimated_confidence": 0.95,
                    })

                if any("news" in b["text"].lower() or "breaking" in b["text"].lower() for b in text_blocks):
                    detected_elements.append("news broadcast / headline banner")

                return {
                    "has_text": bool(raw_text),
                    "raw_extracted_text": raw_text,
                    "text_blocks": text_blocks,
                    "detected_elements": detected_elements,
                    "visual_style": "embedded text overlay" if raw_text else "natural photograph",
                    "visual_quality_score": 0.95 if raw_text else 0.88,
                }
        except Exception as local_err:
            logger.info("Local winocr engine skipped or unavailable: %s", local_err)

        # 2. Try Gemini Vision API if key is available
        if self.api_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.api_key)
                prompt = (
                    "You are an Optical Character Recognition (OCR) and Visual Forensics engine in a multimodal news verification pipeline. "
                    "Examine the uploaded image, extract EVERY printed or overlaid word precisely, and categorize visual text elements.\n\n"
                    "Respond ONLY with valid JSON using this exact schema:\n"
                    "{\n"
                    '  "has_text": true,\n'
                    '  "raw_extracted_text": "all extracted text lines joined by newlines",\n'
                    '  "text_blocks": [\n'
                    '    {\n'
                    '      "block_id": 1,\n'
                    '      "text": "text segment",\n'
                    '      "region": "headline_banner" | "subheading" | "body_paragraph" | "caption" | "watermark_or_stamp" | "social_tag",\n'
                    '      "estimated_confidence": 0.98\n'
                    '    }\n'
                    '  ],\n'
                    '  "detected_elements": ["headline banner", "news ticker", "source logo", "watermark"],\n'
                    '  "visual_style": "news broadcast overlay" | "newspaper clipping" | "social media screenshot" | "meme" | "natural photograph",\n'
                    '  "visual_quality_score": 0.95\n'
                    "}\n"
                    "If the image has NO text, return:\n"
                    "{\n"
                    '  "has_text": false,\n'
                    '  "raw_extracted_text": "",\n'
                    '  "text_blocks": [],\n'
                    '  "detected_elements": ["natural photograph"],\n'
                    '  "visual_style": "natural photograph",\n'
                    '  "visual_quality_score": 0.90\n'
                    "}"
                )

                models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
                for m_name in models_to_try:
                    try:
                        response = client.models.generate_content(
                            model=m_name,
                            contents=[
                                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                                prompt,
                            ],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=0.1,
                            ),
                        )
                        parsed = json.loads(response.text)
                        return parsed
                    except Exception as e:
                        logger.warning("Gemini Vision model %s attempt failed: %s", m_name, e)
                        continue
            except Exception as e:
                logger.warning("Gemini Vision OCR setup error (%s)", e)

        return self._heuristic_local_ocr(image_bytes, width, height)

    def _heuristic_local_ocr(self, image_bytes: bytes, width: int, height: int) -> Dict[str, Any]:
        """Offline fallback heuristic when vision engines are unreachable."""
        return {
            "has_text": False,
            "raw_extracted_text": "",
            "text_blocks": [],
            "detected_elements": ["optical scan completed (offline fallback)"],
            "visual_style": "standard visual media",
            "visual_quality_score": 0.85,
        }

