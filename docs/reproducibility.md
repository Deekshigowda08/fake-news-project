# Reproducibility

The executed notebook already contains the reported outputs. Use it as the primary artifact for inspecting the completed experiment.

## Environment

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Placement

Place the local dataset directly under `data/`:

```text
data/
├── df_train2.csv
├── df_test2.csv
├── images_train/
└── images_test/
```

Additional CSVs such as `train_posts.csv`, `test_posts.csv`, and `train_posts_clean.csv` may also be present for traceability.

## Notebook

Open the canonical notebook:

```text
notebooks/multimodal_fake_news_detection_bert_vit_svm.ipynb
```

The notebook is already executed and contains saved results. Avoid rerunning it unless you intentionally want to reproduce the full experiment.

## Scripted Pipeline

After placing the dataset locally, run:

```bash
python src/train.py --data-dir data
```

The script extracts fused BERT + ViT embeddings and trains a `LinearSVC` classifier. It may download public pretrained model weights from Hugging Face if they are not already cached.

## Documentation

The GitHub Pages documentation URL is intended to be:

```text
https://<github-username>.github.io/<repo-name>/
```

Build the research documentation with:

```bash
mkdocs build
```

Preview locally with:

```bash
mkdocs serve
```
