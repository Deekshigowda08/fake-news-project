"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
import pytest
import io
from PIL import Image

from backend.app.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_api_health_endpoint(client):
    """Tests GET /api/v1/health."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "gpu_available" in data


def test_api_models_endpoint(client):
    """Tests GET /api/v1/models."""
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    data = res.json()
    model_names = [m["name"] for m in data]
    assert "bert_text_classifier" in model_names
    assert "vit_image_classifier" in model_names
    assert "multimodal_fusion_classifier" in model_names
    assert "ensemble_learner" in model_names


def test_api_metrics_endpoint(client):
    """Tests GET /api/v1/metrics."""
    res = client.get("/api/v1/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "experiments" in data
    assert "markdown_report" in data
    assert "EXP-5" in data["experiments"]


def test_api_analyze_text_only(client):
    """Tests POST /api/v1/analyze/text."""
    payload = {"text": "Scientific breakthrough confirms quantum coherence at room temperature."}
    res = client.post("/api/v1/analyze/text", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "bert"
    assert "real" in data["class_probabilities"]
    assert "fake" in data["class_probabilities"]


def test_api_analyze_multimodal_endpoint(client):
    """Tests POST /api/v1/analyze with text and image file."""
    # Generate in-memory PNG
    img = Image.new("RGB", (224, 224), color=(30, 60, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"image": ("test.jpg", buf, "image/jpeg")}
    data = {
        "text": "Secret Antarctic pyramid discovered housing alien energy crystals.",
        "mode": "full",
    }

    res = client.post("/api/v1/analyze", data=data, files=files)
    assert res.status_code == 200
    resp_data = res.json()

    assert resp_data["status"] == "completed"
    assert resp_data["has_image"] is True
    assert "bert_prediction" in resp_data["ml_prediction"]
    assert "vit_prediction" in resp_data["ml_prediction"]
    assert "multimodal_fusion_prediction" in resp_data["ml_prediction"]
    assert "ensemble_prediction" in resp_data["ml_prediction"]

    # Evidence assessment output
    assert resp_data["evidence_assessment"] is not None
    adjudication = resp_data["evidence_assessment"]["adjudication"]
    assert adjudication["verdict"] in ["TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"]
    assert adjudication["ml_predicted_class"] in ["REAL_CLASSIFIED", "FAKE_CLASSIFIED"]

    # Test feedback on this valid analysis_id
    analysis_id = resp_data["analysis_id"]
    fb_res = client.post(
        "/api/v1/feedback",
        json={
            "analysis_id": analysis_id,
            "feedback_type": "agree",
            "feedback_text": "Correct evaluation.",
        },
    )
    assert fb_res.status_code == 200
    assert fb_res.json()["status"] == "feedback_recorded"
