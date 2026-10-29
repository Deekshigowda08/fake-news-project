"""Unit and integration tests for Phase 1: Data, Preprocessing, Calibration, and Baselines."""

from pathlib import Path
import io
import pytest
import numpy as np
from PIL import Image
import torch

from ml.data.schema import (
    BinaryLabel,
    FAKEDDIT_6WAY_TO_BINARY,
    STRING_LABEL_TO_BINARY,
    ProcessedSample,
)
from ml.data.loader import (
    ModularNewsDataset,
    analyze_dataset_statistics,
    compute_text_hash,
    load_dataset_from_records,
)
from ml.data.sample_generator import generate_benchmark_dataset
from ml.preprocessing.text import clean_text, extract_claims_heuristic
from ml.preprocessing.image import (
    validate_and_load_image,
    preprocess_image_for_vit,
    ImageValidationError,
)
from ml.calibration.metrics import (
    compute_expected_calibration_error,
    compute_comprehensive_metrics,
)
from ml.calibration.temperature import TemperatureScaler
from ml.models.bert_classifier import BertNewsClassifier, BertInferencePipeline
from ml.models.vit_classifier import ViTNewsClassifier, ViTInferencePipeline


def test_schema_and_fakeddit_mapping():
    """Validates binary label mapping for standard and 6-way Fakeddit taxonomy."""
    assert BinaryLabel.REAL.value == 0
    assert BinaryLabel.FAKE.value == 1

    # Fakeddit 6-way taxonomy
    assert FAKEDDIT_6WAY_TO_BINARY[0] == 0  # True -> Real
    assert FAKEDDIT_6WAY_TO_BINARY[1] == 1  # Satire -> Fake
    assert FAKEDDIT_6WAY_TO_BINARY[2] == 1  # Misleading -> Fake
    assert FAKEDDIT_6WAY_TO_BINARY[5] == 1  # Fabricated -> Fake

    # String mapping
    assert STRING_LABEL_TO_BINARY["real"] == 0
    assert STRING_LABEL_TO_BINARY["fake"] == 1


def test_clean_text():
    """Tests text normalization, unicode cleaning, and URL masking."""
    raw = "Breaking: &quot;Major discovery&quot; at https://example.com/news !   Details inside.  "
    cleaned = clean_text(raw, preserve_urls=False)
    assert '[URL]' in cleaned
    assert '"Major discovery"' in cleaned
    assert '   ' not in cleaned


def test_extract_claims_heuristic():
    """Tests claim extraction heuristic."""
    text = "NASA discovered liquid water on Mars. The rover captured panoramic photographs."
    claims = extract_claims_heuristic(text)
    assert len(claims) == 2
    assert "NASA discovered liquid water on Mars." in claims


def test_image_validation_valid_and_invalid(tmp_path):
    """Tests image loading, corruption rejection, and format security."""
    # 1. Valid PNG
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img_path = tmp_path / "valid.png"
    img.save(img_path, format="PNG")

    loaded = validate_and_load_image(img_path)
    assert loaded.size == (100, 100)

    # 2. Corrupted file
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"NOT_AN_IMAGE_HEADER_CORRUPTED_BYTES")
    with pytest.raises(ImageValidationError):
        validate_and_load_image(corrupt_path)

    # 3. Missing file
    with pytest.raises(FileNotFoundError):
        validate_and_load_image(tmp_path / "non_existent.jpg")


def test_benchmark_dataset_generation_and_loading(tmp_path):
    """Verifies synthetic benchmark generator produces valid splits and images."""
    bench_dir = generate_benchmark_dataset(tmp_path / "benchmark")
    assert (bench_dir / "train.jsonl").is_file()
    assert (bench_dir / "val.jsonl").is_file()
    assert (bench_dir / "test.jsonl").is_file()

    samples = load_dataset_from_records(bench_dir / "train.jsonl")
    assert len(samples) > 0

    stats = analyze_dataset_statistics(samples)
    assert stats["total_samples"] == len(samples)
    assert stats["real_count"] > 0
    assert stats["fake_count"] > 0
    assert stats["samples_with_image"] == len(samples)


def test_expected_calibration_error():
    """Tests ECE and Brier score computation."""
    y_true = np.array([0, 0, 1, 1])
    # Perfectly confident and correct
    y_prob_good = np.array([0.05, 0.10, 0.90, 0.95])
    metrics_good = compute_comprehensive_metrics(y_true, [0, 0, 1, 1], y_prob_good)
    assert metrics_good["accuracy"] == 1.0
    assert metrics_good["brier_score"] < 0.05
    assert metrics_good["ece"] < 0.15

    # Completely wrong high confidence
    y_prob_bad = np.array([0.95, 0.95, 0.05, 0.05])
    metrics_bad = compute_comprehensive_metrics(y_true, [1, 1, 0, 0], y_prob_bad)
    assert metrics_bad["accuracy"] == 0.0
    assert metrics_bad["brier_score"] > 0.8
    assert metrics_bad["ece"] > 0.8


def test_temperature_scaler():
    """Tests temperature optimization on validation logits."""
    scaler = TemperatureScaler(initial_temp=1.0)
    # Overconfident uncalibrated logits
    logits = np.array([[5.0, -5.0], [4.0, -4.0], [-4.0, 4.0], [-5.0, 5.0]])
    labels = np.array([0, 0, 1, 1])

    scaler.fit(logits, labels)
    probs = scaler.predict_proba(logits)
    assert probs.shape == (4, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_vit_inference_pipeline_error_handling():
    """Tests ViT inference pipeline with missing and valid image scenarios."""
    # Initialize mock model structure
    model = ViTNewsClassifier(pretrained_model_name="google/vit-base-patch16-224", freeze_encoder=True)
    pipeline = ViTInferencePipeline(model=model, device=torch.device("cpu"))

    # Case 1: Missing image
    res_missing = pipeline.predict(None)
    assert res_missing["status"] == "missing_image"
    assert res_missing["class_probabilities"]["real"] == 0.5
    assert res_missing["class_probabilities"]["fake"] == 0.5

    # Case 2: Corrupt image bytes
    res_corrupt = pipeline.predict(b"corrupt_image_data")
    assert res_corrupt["status"] == "invalid_image"

    # Case 3: Valid PIL image
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    res_valid = pipeline.predict(img)
    assert res_valid["status"] == "success"
    assert res_valid["model_name"] == "vit"
    assert "real" in res_valid["class_probabilities"]
    assert "fake" in res_valid["class_probabilities"]
    assert round(res_valid["class_probabilities"]["real"] + res_valid["class_probabilities"]["fake"], 2) == 1.0
