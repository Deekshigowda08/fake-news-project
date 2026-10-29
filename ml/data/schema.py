"""Data schema and validation models for multimodal fake news detection."""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict


class BinaryLabel(int, Enum):
    REAL = 0
    FAKE = 1


# Explicit mapping documentation for multi-class Fakeddit labels
# Fakeddit 6-way categories:
# 0: True (Real)
# 1: Satire / Parody (Fake)
# 2: Misleading Content (Fake)
# 3: Manipulated Content (Fake)
# 4: False Connection (Fake)
# 5: Completely Fabricated (Fake)
FAKEDDIT_6WAY_TO_BINARY = {
    0: BinaryLabel.REAL.value,
    1: BinaryLabel.FAKE.value,
    2: BinaryLabel.FAKE.value,
    3: BinaryLabel.FAKE.value,
    4: BinaryLabel.FAKE.value,
    5: BinaryLabel.FAKE.value,
}

# Twitter / Weibo binary text labels
STRING_LABEL_TO_BINARY = {
    "real": BinaryLabel.REAL.value,
    "true": BinaryLabel.REAL.value,
    "0": BinaryLabel.REAL.value,
    0: BinaryLabel.REAL.value,
    "fake": BinaryLabel.FAKE.value,
    "false": BinaryLabel.FAKE.value,
    "1": BinaryLabel.FAKE.value,
    1: BinaryLabel.FAKE.value,
}


class ProcessedSample(BaseModel):
    """Canonical schema for every sample in the multimodal pipeline."""
    sample_id: str = Field(..., description="Unique sample identifier")
    text: str = Field(..., description="News headline or article body")
    image_path: Optional[str] = Field(None, description="Path to local image file if available")
    label: int = Field(..., description="0 for REAL, 1 for FAKE")
    dataset_source: str = Field(default="unknown", description="Source dataset e.g. fakeddit, twitter, weibo, benchmark")
    split: str = Field(default="train", description="Dataset split: train, val, or test")
    has_image: bool = Field(default=False, description="Whether a valid readable image exists on disk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata such as username, timestamp, URL")

    model_config = ConfigDict(frozen=False)
