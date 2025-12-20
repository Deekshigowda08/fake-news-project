# Methodology

The project uses late fusion between transformer-based text and image encoders.

## Text Representation

Text is tokenized with the BERT tokenizer using:

| Parameter | Value |
| --- | --- |
| Model | `bert-base-uncased` |
| Maximum length | 128 tokens |
| Padding | `max_length` |
| Truncation | enabled |

The BERT `[CLS]` output is used as a 768-dimensional representation of each post's text.

## Image Representation

Images are resized to `224 x 224`, converted to RGB tensors, and normalized with mean and standard deviation `[0.5, 0.5, 0.5]`.

The Vision Transformer encoder is `google/vit-base-patch16-224`. Its `[CLS]` output is used as a 768-dimensional representation of the image.

## Encoder Freezing

The final notebook uses a conservative training strategy:

- ViT is frozen completely.
- BERT is mostly frozen.
- Only BERT encoder layers 10 and 11 are left trainable.

This setup reduces overfitting risk while still allowing the text encoder to adapt to the fake news detection task.

## Feature Fusion

The final representation concatenates full-strength text features with down-weighted image features:

```text
fused = [text_cls, 0.5 * image_cls]
```

The result is a 1536-dimensional feature vector.

## Classifier

The fused feature vectors are passed into a scikit-learn pipeline:

```text
StandardScaler -> LinearSVC
```

The SVM uses `class_weight="balanced"` and `max_iter=5000`.
