"""SQLModel Database Entities for Persistence and Auditability.

Follows Section 15 database design specifications.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
from sqlmodel import SQLModel, Field, Column, JSON


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    password_hash: str
    role: str = Field(default="researcher")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Analysis(SQLModel, table=True):
    __tablename__ = "analyses"

    id: str = Field(primary_key=True, index=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    input_text: str
    input_url: Optional[str] = None
    image_storage_reference: Optional[str] = None
    dataset_or_model_configuration: Optional[str] = None
    ml_prediction: Optional[str] = None
    ml_probabilities: Optional[Dict[str, float]] = Field(default=None, sa_column=Column(JSON))
    ensemble_method: Optional[str] = None
    evidence_verdict: Optional[str] = None
    analysis_status: str = Field(default="pending", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class ModelPredictionRecord(SQLModel, table=True):
    __tablename__ = "model_predictions"

    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: str = Field(foreign_key="analyses.id", index=True)
    model_name: str = Field(index=True)
    model_version: str = Field(default="1.0.0")
    predicted_class: str
    class_probabilities: Dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    calibration_version: Optional[str] = None


class ClaimRecord(SQLModel, table=True):
    __tablename__ = "claims"

    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: str = Field(foreign_key="analyses.id", index=True)
    claim_text: str
    claim_type: str = Field(default="factual_assertion")
    extraction_status: str = Field(default="extracted")


class EvidenceSourceRecord(SQLModel, table=True):
    __tablename__ = "evidence_sources"

    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: str = Field(foreign_key="analyses.id", index=True)
    claim_id: Optional[str] = None
    source_url: str
    publisher: str
    publication_date: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    relevant_excerpt: str
    stance: str
    source_type: str = Field(default="web")
    independence_group: Optional[str] = None
    retrieval_status: str = Field(default="retrieved")


class AgentEventRecord(SQLModel, table=True):
    __tablename__ = "agent_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: str = Field(foreign_key="analyses.id", index=True)
    agent_name: str
    event_type: str
    status: str
    duration_ms: int = Field(default=0)
    structured_details: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackRecord(SQLModel, table=True):
    __tablename__ = "feedback"

    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: str = Field(foreign_key="analyses.id", index=True)
    feedback_type: str
    feedback_text: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
