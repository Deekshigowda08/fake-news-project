"""Unified ML Inference Service for serving predictions to FastAPI backend.

Coordinates:
- BERT text classifier
- ViT visual classifier
- Multimodal fusion classifier
- CLIP semantic alignment
- Soft voting & Stacking ensembles
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import io
from PIL import Image

from ml.models.bert_classifier import BertInferencePipeline, BertNewsClassifier
from ml.models.vit_classifier import ViTInferencePipeline, ViTNewsClassifier
from ml.models.multimodal_fusion import MultimodalInferencePipeline, MultimodalFusionClassifier
from ml.models.alignment_model import ClipSemanticAligner
from ml.models.ensemble import SoftVotingEnsemble, StackingEnsemble
from ml.preprocessing.ocr_service import ImageOCRService

logger = logging.getLogger(__name__)


class MultimodalInferenceService:
    """Singleton-ready service managing model lifetimes and orchestrating multi-model inferences."""

    def __init__(
        self,
        bert_pipeline: Optional[BertInferencePipeline] = None,
        vit_pipeline: Optional[ViTInferencePipeline] = None,
        fusion_pipeline: Optional[MultimodalInferencePipeline] = None,
        aligner: Optional[ClipSemanticAligner] = None,
        ensemble: Optional[SoftVotingEnsemble] = None,
        ocr_service: Optional[ImageOCRService] = None,
    ):
        self.bert_pipeline = bert_pipeline
        self.vit_pipeline = vit_pipeline
        self.fusion_pipeline = fusion_pipeline
        self.aligner = aligner or ClipSemanticAligner()
        self.ensemble = ensemble or SoftVotingEnsemble()
        self.ocr_service = ocr_service or ImageOCRService()


    @classmethod
    def create_default(cls, device=None) -> "MultimodalInferenceService":
        """Instantiates default pipelines with pretrained backbones."""
        import torch

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing MultimodalInferenceService on device: %s", dev)

        bert_checkpoint = Path("checkpoints/bert_finetuned")
        if bert_checkpoint.exists() and (bert_checkpoint / "pytorch_model.bin").is_file():
            logger.info("Loading fine-tuned BERT checkpoint from %s", bert_checkpoint)
            bert_pipe = BertInferencePipeline.load(bert_checkpoint, device=dev)
        else:
            bert_model = BertNewsClassifier(pretrained_model_name="bert-base-uncased")
            bert_pipe = BertInferencePipeline(model=bert_model, device=dev)

        vit_checkpoint = Path("checkpoints/vit_finetuned")
        if vit_checkpoint.exists() and (vit_checkpoint / "pytorch_model.bin").is_file():
            logger.info("Loading fine-tuned ViT checkpoint from %s", vit_checkpoint)
            vit_pipe = ViTInferencePipeline.load(vit_checkpoint, device=dev)
        else:
            vit_model = ViTNewsClassifier(pretrained_model_name="google/vit-base-patch16-224")
            vit_pipe = ViTInferencePipeline(model=vit_model, device=dev)

        fusion_model = MultimodalFusionClassifier(
            bert_model_name="bert-base-uncased",
            vit_model_name="google/vit-base-patch16-224",
        )
        fusion_pipe = MultimodalInferencePipeline(model=fusion_model, device=dev)

        aligner = ClipSemanticAligner(device=dev)
        ensemble = SoftVotingEnsemble()

        return cls(
            bert_pipeline=bert_pipe,
            vit_pipeline=vit_pipe,
            fusion_pipeline=fusion_pipe,
            aligner=aligner,
            ensemble=ensemble,
        )

    def analyze(
        self,
        text: str,
        image_source: Optional[Any] = None,
        run_alignment: bool = True,
    ) -> Dict[str, Any]:
        """Runs complete multimodal analysis returning individual and ensemble predictions.
        
        Returns unified dictionary:
        {
            "headline_or_text": str,
            "has_image": bool,
            "bert_prediction": {...},
            "vit_prediction": {...},
            "multimodal_fusion_prediction": {...},
            "ensemble_prediction": {...},
            "semantic_alignment": {...},
            "calibrated_model_confidence": float,
            "available_modalities": ["text", "image"] | ["text"]
        }
        """
        has_image = image_source is not None

        # 1. Optical Character Recognition (OCR) & Visual Text Analysis
        ocr_res = None
        if has_image and self.ocr_service:
            try:
                img_raw = image_source
                if hasattr(img_raw, "read"):
                    img_raw = img_raw.read()
                elif isinstance(img_raw, io.BytesIO):
                    img_raw = img_raw.getvalue()
                elif isinstance(img_raw, Image.Image):
                    b = io.BytesIO()
                    img_raw.save(b, format="JPEG")
                    img_raw = b.getvalue()
                elif isinstance(img_raw, (str, Path)) and Path(img_raw).is_file():
                    img_raw = Path(img_raw).read_bytes()

                if isinstance(img_raw, bytes) and len(img_raw) > 0:
                    ocr_res = self.ocr_service.process_image_and_compare(img_raw, input_claim_text=text)
            except Exception as e:
                logger.warning("OCR processing error: %s", e)

        # 2. BERT text inference
        bert_res = self.bert_pipeline.predict(text) if self.bert_pipeline else None

        # 3. ViT image inference
        vit_res = self.vit_pipeline.predict(image_source) if self.vit_pipeline else None

        # 4. Multimodal fusion inference
        fusion_res = self.fusion_pipeline.predict(text, image_source) if self.fusion_pipeline else None
        if not has_image and bert_res and fusion_res:
            fusion_res["class_probabilities"] = bert_res["class_probabilities"]
            fusion_res["predicted_class"] = bert_res["predicted_class"]
            fusion_res["confidence"] = bert_res["confidence"]

        # 5. CLIP semantic alignment
        alignment_res = None
        if run_alignment and self.aligner and has_image:
            alignment_res = self.aligner.compute_alignment(text, image_source)

        # 6. Ensemble combination
        model_preds = {}
        if bert_res:
            model_preds["bert"] = bert_res
        if vit_res and vit_res.get("status") == "success":
            model_preds["vit"] = vit_res
        if fusion_res:
            model_preds["multimodal_fusion"] = fusion_res

        ensemble_res = self.ensemble.predict_sample(model_preds, has_image=has_image)

        modalities = ["text"]
        if has_image and vit_res and vit_res.get("status") == "success":
            modalities.append("image")

        # 7. Comparative Score Matrix for Professors
        weights_applied = ensemble_res.get("weights_applied", {})
        comparative_scores = []
        if bert_res:
            comparative_scores.append({
                "model_key": "bert",
                "model_name": "BERT Contextual Classifier",
                "modality": "Text (WordPiece)",
                "predicted_class": bert_res.get("predicted_class", "unknown"),
                "p_real": bert_res.get("class_probabilities", {}).get("real", 0.5),
                "p_fake": bert_res.get("class_probabilities", {}).get("fake", 0.5),
                "confidence": bert_res.get("confidence", 0.5),
                "fine_grained_class": bert_res.get("fine_grained_class"),
                "ensemble_weight": round(float(weights_applied.get("bert", 0.35)), 2),
                "backbone": "bert-base-uncased (110M params)",
            })

        if vit_res and vit_res.get("status") == "success":
            comparative_scores.append({
                "model_key": "vit",
                "model_name": "Vision Transformer (ViT)",
                "modality": "Image (196 Patches)",
                "predicted_class": vit_res.get("predicted_class", "unknown"),
                "p_real": vit_res.get("class_probabilities", {}).get("real", 0.5),
                "p_fake": vit_res.get("class_probabilities", {}).get("fake", 0.5),
                "confidence": vit_res.get("confidence", 0.5),
                "fine_grained_class": None,
                "ensemble_weight": round(float(weights_applied.get("vit", 0.20)), 2),
                "backbone": "google/vit-base-patch16-224 (86M params)",
            })

        if fusion_res:
            comparative_scores.append({
                "model_key": "multimodal_fusion",
                "model_name": "Multimodal Feature Fusion",
                "modality": "Cross-Modal Concatenation (1536-d)" if has_image else "Text-Only Dynamic Routing",
                "predicted_class": fusion_res.get("predicted_class", "unknown"),
                "p_real": fusion_res.get("class_probabilities", {}).get("real", 0.5),
                "p_fake": fusion_res.get("class_probabilities", {}).get("fake", 0.5),
                "confidence": fusion_res.get("confidence", 0.5),
                "fine_grained_class": None,
                "ensemble_weight": round(float(weights_applied.get("multimodal_fusion", 0.45)), 2),
                "backbone": "BERT [768-d] + ViT [768-d] -> MLP",
            })

        comparative_scores.append({
            "model_key": "ensemble",
            "model_name": "Ensemble Soft-Voting Consensus",
            "modality": "Weighted Meta-Consensus",
            "predicted_class": ensemble_res.get("predicted_class", "unknown"),
            "p_real": ensemble_res.get("class_probabilities", {}).get("real", 0.5),
            "p_fake": ensemble_res.get("class_probabilities", {}).get("fake", 0.5),
            "confidence": ensemble_res.get("confidence", 0.5),
            "fine_grained_class": None,
            "ensemble_weight": 1.0,
            "backbone": "Soft Voting Probability Distribution",
        })

        return {
            "headline_or_text": text,
            "has_image": has_image,
            "available_modalities": modalities,
            "bert_prediction": bert_res,
            "vit_prediction": vit_res,
            "multimodal_fusion_prediction": fusion_res,
            "ensemble_prediction": ensemble_res,
            "semantic_alignment": alignment_res,
            "ocr_analysis": ocr_res,
            "comparative_scores": comparative_scores,
            "calibrated_model_confidence": ensemble_res.get("confidence", 0.5),
            "disclaimer": (
                "ML model predictions reflect learned patterns in training data and do not "
                "constitute definitive real-world proof of factual authenticity."
            ),
        }

