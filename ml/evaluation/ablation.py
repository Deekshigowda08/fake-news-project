"""Ablation Study and Research Evaluation Framework.

Executes and logs Experiments 1 through 8:
- Exp 1: BERT text only
- Exp 2: ViT image only
- Exp 3: Multimodal Fusion (BERT + ViT)
- Exp 4: Soft-Voting Ensemble
- Exp 5: Stacking Ensemble
- Exp 6: Best ML Model + Evidence Verification
- Exp 7: Full System (Ensemble + Evidence Verification)
- Exp 8: Ablation comparison matrix
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging
import numpy as np

from ml.calibration.metrics import compute_comprehensive_metrics

logger = logging.getLogger(__name__)


class ResearchEvaluationMatrix:
    """Manages evaluation records and formats ablation comparison tables."""

    def __init__(self):
        self.experiments: Dict[str, Dict[str, Any]] = {}

    def record_experiment(
        self,
        exp_id: str,
        name: str,
        metrics: Dict[str, Any],
        description: str,
        modalities: List[str],
    ):
        """Records metrics for an experimental run."""
        self.experiments[exp_id] = {
            "name": name,
            "description": description,
            "modalities": modalities,
            "metrics": metrics,
        }

    def generate_markdown_report(self) -> str:
        """Generates a clean Markdown table comparing all recorded experiments."""
        lines = [
            "# Multimodal Fake News Detection - Research Evaluation Matrix",
            "",
            "| Exp ID | Model / Configuration | Modalities | Accuracy | Precision | Recall | F1-Score | Brier Score | ECE |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for exp_id, data in self.experiments.items():
            name = data["name"]
            mods = "+".join(data["modalities"])
            m = data["metrics"]
            acc = f"{m.get('accuracy', 0.0):.4f}"
            prec = f"{m.get('precision', 0.0):.4f}"
            rec = f"{m.get('recall', 0.0):.4f}"
            f1 = f"{m.get('f1_score', 0.0):.4f}"
            brier = f"{m.get('brier_score', 0.0):.4f}" if m.get("brier_score") is not None else "N/A"
            ece = f"{m.get('ece', 0.0):.4f}" if m.get("ece") is not None else "N/A"

            lines.append(
                f"| `{exp_id}` | {name} | {mods} | {acc} | {prec} | {rec} | {f1} | {brier} | {ece} |"
            )

        lines.extend([
            "",
            "> [!NOTE]",
            "> All metrics are computed strictly on the held-out evaluation test set.",
            "> Models were selected and calibrated using validation data only, preventing data leakage.",
        ])

        return "\n".join(lines)

    def save_json(self, output_path: Union[str, Path]):
        """Saves experimental matrix to JSON."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.experiments, f, indent=2)
