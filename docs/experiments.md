# Experiments

The canonical experiment is the executed notebook:

```text
notebooks/multimodal_fake_news_detection_bert_vit_svm.ipynb
```

The notebook already contains saved outputs and should not be rerun just to inspect reported results.

## Pipeline

1. Load multimodal post records from CSV.
2. Drop rows with missing text, image ID, or label.
3. Tokenize text with BERT-compatible preprocessing.
4. Load and normalize corresponding images.
5. Extract BERT and ViT `[CLS]` embeddings.
6. Fuse text and image embeddings with text-dominant weighting.
7. Train a linear SVM on fused embeddings.
8. Evaluate on validation and test splits.

## Split Strategy

The executed notebook uses group-aware splitting by `image_id`. This prevents the same image from appearing across train, validation, and test sets, reducing the risk of image leakage.

The notebook reported:

| Split | Rows |
| --- | ---: |
| Training | 8,631 |
| Validation | 2,919 |
| Test | 2,121 |

## Hyperparameters

| Parameter | Value |
| --- | --- |
| BERT model | `bert-base-uncased` |
| ViT model | `google/vit-base-patch16-224` |
| Max sequence length | 128 |
| Image size | `224 x 224` |
| Batch size | 8 |
| Image feature weight | 0.5 |
| SVM `C` | 1.0 |
| SVM class weight | `balanced` |
| SVM max iterations | 5000 |

## Ablation Context

The research framing compares:

- text-only BERT features
- image-only ViT features
- multimodal BERT + ViT features

The final reported model is the text-dominant multimodal configuration.
