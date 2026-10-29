"""Database engine configuration and session provider."""

import os
from pathlib import Path
from typing import Generator
from sqlmodel import SQLModel, Session, create_engine

# Default to SQLite for zero-setup local research development; override with PostgreSQL in production
DEFAULT_DB_FILE = Path("data/fake_news_research.db")
DEFAULT_DB_FILE.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_FILE}")

# SQLite requires check_same_thread=False
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args=connect_args,
)


def create_db_and_tables():
    """Initializes schema and tables."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency yielding database session."""
    with Session(engine) as session:
        yield session
