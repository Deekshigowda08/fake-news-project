"""Tests for Phase 2 (Multimodal Fusion) and Phase 3 (CLIP Alignment & Ensembles)."""

import pytest
import numpy as np
from PIL import Image
import torch

from ml.models.multimodal_fusion import MultimodalFusionClassifier, MultimodalInferencePipeline
from ml.models.alignment_model import ClipSemanticAligner
from ml.models.ensemble import SoftVotingEnsemble, StackingEnsemble


def test_multimodal_fusion_modality_routing():
    """Verifies that MultimodalFusionClassifier dynamically routes between multimodal and text-only."""
    # Build model (frozen for fast test execution)
    model = MultimodalFusionClassifier(
        bert_model_name="bert-base-uncased",
        vit_model_name="google/vit-base-patch16-224",
        freeze_vit=True,
    )
    pipeline = MultimodalInferencePipeline(model=model, device=torch.device("cpu"))

    # 1. Text + Valid Image
    img = Image.new("RGB", (224, 224), color=(50, 100, 150))
    res_mm = pipeline.predict("NASA James Webb Telescope reveals distant galaxy.", img)
    assert res_mm["model_name"] == "multimodal_fusion"
    assert res_mm["modality_used"] == "multimodal"
    assert "real" in res_mm["class_probabilities"]
    assert "fake" in res_mm["class_probabilities"]

    # 2. Text Only (Missing Image)
    res_text_only = pipeline.predict("NASA James Webb Telescope reveals distant galaxy.", None)
    assert res_text_only["modality_used"] == "text_only"
    assert "real" in res_text_only["class_probabilities"]
    assert "fake" in res_text_only["class_probabilities"]


def test_soft_voting_ensemble():
    """Tests weighted soft voting and dynamic re-normalization when ViT is omitted."""
    ensemble = SoftVotingEnsemble(weights={"bert": 0.4, "vit": 0.2, "multimodal_fusion": 0.4})

    model_preds = {
        "bert": {"class_probabilities": {"real": 0.20, "fake": 0.80}},
        "vit": {"class_probabilities": {"real": 0.10, "fake": 0.90}},
        "multimodal_fusion": {"class_probabilities": {"real": 0.15, "fake": 0.85}},
    }

    # With Image
    res = ensemble.predict_sample(model_preds, has_image=True)
    assert res["predicted_class"] == "fake"
    assert res["class_probabilities"]["fake"] > 0.80
    assert "vit" in res["contributing_models"]

    # Without Image (ViT skipped)
    res_no_img = ensemble.predict_sample(model_preds, has_image=False)
    assert "vit" not in res_no_img["contributing_models"]
    assert "bert" in res_no_img["contributing_models"]
    assert "multimodal_fusion" in res_no_img["contributing_models"]
    assert round(res_no_img["class_probabilities"]["real"] + res_no_img["class_probabilities"]["fake"], 2) == 1.0


def test_stacking_ensemble():
    """Tests out-of-fold meta-classifier fitting and inference."""
    stacker = StackingEnsemble()

    # Synthetic predictions from [bert, vit, fusion]
    X_train = np.array([
        [0.85, 0.90, 0.88],  # fake
        [0.78, 0.82, 0.80],  # fake
        [0.12, 0.15, 0.10],  # real
        [0.05, 0.10, 0.08],  # real
    ])
    y_train = np.array([1, 1, 0, 0])

    stacker.fit(X_train, y_train, ["bert", "vit", "multimodal_fusion"])

    pred_fake = stacker.predict_sample([0.80, 0.85, 0.82], ["bert", "vit", "multimodal_fusion"])
    assert pred_fake["predicted_class"] == "fake"

    pred_real = stacker.predict_sample([0.10, 0.08, 0.12], ["bert", "vit", "multimodal_fusion"])
    assert pred_real["predicted_class"] == "real"
