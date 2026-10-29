"""FastAPI request and response schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TextInferenceRequest(BaseModel):
    text: str = Field(..., description="News headline or article text to classify", min_length=3)


class AnalysisMode(str):
    FULL = "full"
    ML_ONLY = "ml_only"
    VERIFY_ONLY = "verify_only"


class AnalyzeResponse(BaseModel):
    analysis_id: str
    status: str
    input_text: str
    has_image: bool
    submitted_url: Optional[str] = None
    available_modalities: List[str]
    ml_prediction: Optional[Dict[str, Any]] = None
    evidence_assessment: Optional[Dict[str, Any]] = None
    semantic_alignment: Optional[Dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    database: str
    gpu_available: bool
    device: str
    version: str = "1.0.0"


class ModelInfoItem(BaseModel):
    name: str
    type: str
    backbone: str
    trainable_parameters: Optional[int] = None
    status: str


class FeedbackRequest(BaseModel):
    analysis_id: str
    feedback_type: str = Field(..., description="'agree', 'disagree', 'report_citation'")
    feedback_text: Optional[str] = None
