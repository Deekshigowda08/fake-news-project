# Dataset

The dataset is local-only and is not included in the public repository. It can be provided on request.

## Expected Layout

Place the final dataset directly under `data/`:

```text
data/
├── df_train2.csv
├── df_test2.csv
├── train_posts.csv
├── test_posts.csv
├── train_posts_clean.csv
├── images_train/
└── images_test/
```

The repository ignores these files by default so the project can be published without committing private or large data artifacts.

## Schema

| Column | Role |
| --- | --- |
| `post_id` | Unique identifier for the social media post |
| `post_text` | Text input for BERT |
| `user_id` | Source platform user identifier |
| `image_id` | Image filename stem used to load the associated image |
| `username` | Source platform username |
| `timestamp` | Original post timestamp |
| `label` | Ground-truth class: `real` or `fake` |

## Local Dataset Snapshot

The final local data snapshot inspected during repo preparation contained:

| File or folder | Count |
| --- | ---: |
| `df_train2.csv` | 13,072 rows |
| `df_test2.csv` | 3,268 rows |
| `train_posts.csv` | 13,671 rows |
| `test_posts.csv` | 1,091 rows |
| `train_posts_clean.csv` | 13,366 rows |
| `images_train/` | 515 files |
| `images_test/` | 515 files |

## Data Handling

The experiment uses `post_text`, `image_id`, and `label` as critical fields. Rows missing these fields are excluded before modeling.

Image IDs are resolved against `images_train/` and `images_test/`. Supported image extensions in the reusable source code are `.jpg`, `.jpeg`, `.png`, and `.gif`.
