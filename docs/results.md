# Results

The final model was evaluated on the held-out test set in the executed notebook.

## Test Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 95.43% |
| Precision, fake | 95.40% |
| Recall, fake | 96.73% |
| F1-score, fake | 96.06% |

## Confusion Matrix

Rows are actual labels and columns are predicted labels.

|  | Predicted real | Predicted fake |
| --- | ---: | ---: |
| Actual real | 841 | 57 |
| Actual fake | 40 | 1183 |

The model correctly identified 1,183 fake posts and missed 40 fake posts. It incorrectly flagged 57 real posts as fake.

## Interpretation

The strongest result is fake-news recall. A recall of 96.73% means the model detected most fake posts in the held-out test set. This is important for fake news detection because false negatives represent fake content that passes through undetected.

The false-positive count is also relatively low: 57 out of 898 real posts were flagged as fake. This suggests the final classifier did not gain fake-news recall by simply labeling most content as fake.

## Baseline Comparison

The research framing compares the final BERT + ViT approach against the original multimodal fake news detection paper used as this project's baseline. The original paper used a BERT + EfficientNet-style multimodal setup and reported approximately 88% accuracy (`88.90%` here). The final model improves that result to `95.43%` accuracy.

| Metric | Baseline | Final model | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 88.90% | 95.43% | +6.53 |
| Precision, fake | 94.00% | 95.40% | +1.40 |
| Recall, fake | 71.00% | 96.73% | +25.73 |
| F1-score, fake | 81.00% | 96.06% | +15.06 |

The main improvement is recall for fake news, which is the most consequential metric when the goal is to reduce missed misinformation.
