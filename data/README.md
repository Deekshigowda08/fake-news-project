# Dataset Layout

The dataset used for the final experiment is local-only and is not committed to GitHub. It can be provided on request.

Place the final dataset files directly in this `data/` directory:

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

## CSV Schema

The expected CSV columns are:

| Column | Description |
| --- | --- |
| `post_id` | Unique social media post identifier |
| `post_text` | Text content associated with the post |
| `user_id` | User identifier from the source platform |
| `image_id` | Image filename without extension |
| `username` | Username from the source platform |
| `timestamp` | Original post timestamp |
| `label` | Binary label: `real` or `fake` |

## Observed Local Counts

At cleanup time, the local final dataset contained:

| File or folder | Count |
| --- | ---: |
| `df_train2.csv` | 13,072 rows |
| `df_test2.csv` | 3,268 rows |
| `train_posts.csv` | 13,671 rows |
| `test_posts.csv` | 1,091 rows |
| `train_posts_clean.csv` | 13,366 rows |
| `images_train/` | 515 files |
| `images_test/` | 515 files |

The public repository ignores all dataset contents except this README.
