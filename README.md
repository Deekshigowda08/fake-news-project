# Multimodal Fake News Detection with BERT and Vision Transformer

This repository contains a research-oriented implementation of multimodal fake news detection for social media posts. The approach combines textual features from BERT with visual features from a Vision Transformer (ViT), then trains a linear SVM classifier on the fused representation.

The canonical experiment is preserved as an already-executed notebook in `notebooks/multimodal_fake_news_detection_bert_vit_svm.ipynb`. The notebook should be treated as the reported experiment artifact; it does not need to be rerun to view the results.

The project builds on the original multimodal fake news detection paper used as the baseline for this work. That paper reported approximately `88%` test accuracy (`88.90%` in the comparison table); this implementation improves the reported held-out accuracy to `95.43%` with a BERT + Vision Transformer feature pipeline.

## Highlights

- Text encoder: `bert-base-uncased`
- Image encoder: `google/vit-base-patch16-224`
- Fusion: text embedding concatenated with `0.5 * image_embedding`
- Classifier: `StandardScaler` + `LinearSVC`
- Test accuracy: `95.43%`
- Fake-news recall: `96.73%`
- Fake-news F1-score: `96.06%`

## Repository Structure

```text
.
├── docs/                         # MkDocs research documentation
├── notebooks/                    # Canonical executed experiment notebook
├── src/                          # Reusable implementation modules
├── data/README.md                # Dataset layout and access notes
├── mkdocs.yml                    # Documentation configuration
├── requirements.txt              # Python dependencies
└── LICENSE
```

## Dataset

The dataset is intentionally not committed to GitHub. It contains social media text, user metadata, labels, and associated images. The final local dataset used for this project is expected directly under `data/`.

Expected files include:

- `data/df_train2.csv`
- `data/df_test2.csv`
- `data/train_posts.csv`
- `data/test_posts.csv`
- `data/train_posts_clean.csv`
- `data/images_train/`
- `data/images_test/`

The dataset can be provided on request.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the Code Path

The notebook already contains the reported outputs. To run the modular script later, place the dataset under `data/` and run:

```bash
python src/train.py --data-dir data
```

## Documentation

View the hosted documentation:

```text
https://CuriousSensie.github.io/fake-news-detection/
```

Build the MkDocs site locally with:

```bash
mkdocs serve
```

Then open the local URL printed by MkDocs.

## License

This project is released under the MIT License.
