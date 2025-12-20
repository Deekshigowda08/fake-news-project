# Limitations

## Dataset Scope

The reported results are tied to the dataset snapshot used in the executed notebook. Performance on other social platforms, news domains, languages, or time periods may differ.

## Language Coverage

The pipeline uses `bert-base-uncased`, so the text encoder is best suited to English text. Multilingual fake news detection would require additional evaluation and likely a multilingual encoder.

## Compute Requirements

Feature extraction uses large pretrained transformer encoders. Even with ViT frozen and only selected BERT layers trainable, GPU acceleration is strongly recommended.

## Explainability

The current implementation reports classification metrics and confusion matrices. It does not yet include explanation methods such as attention visualization, SHAP, LIME, or cross-modal consistency analysis.

## Deployment

This repository is research-oriented. It is not a production moderation system and does not include real-time ingestion, monitoring, human review workflows, or abuse-prevention controls.
