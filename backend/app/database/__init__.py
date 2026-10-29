"""Database module initialization."""

from backend.app.database.session import engine, get_session, create_db_and_tables
from backend.app.database.models import (
    User,
    Analysis,
    ModelPredictionRecord,
    ClaimRecord,
    EvidenceSourceRecord,
    AgentEventRecord,
    FeedbackRecord,
)

__all__ = [
    "engine",
    "get_session",
    "create_db_and_tables",
    "User",
    "Analysis",
    "ModelPredictionRecord",
    "ClaimRecord",
    "EvidenceSourceRecord",
    "AgentEventRecord",
    "FeedbackRecord",
]
