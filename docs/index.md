# Multimodal Fake News Detection

This project studies fake news detection as a multimodal classification problem. Each social media post is represented by two complementary signals:

- textual content from the post
- visual content from the associated image

The final experiment combines BERT text embeddings with Vision Transformer image embeddings and trains a linear support vector machine on the fused representation.

This work explicitly compares against the original multimodal fake news detection paper that motivated the project. The paper's baseline reported approximately 88% accuracy (`88.90%` in the detailed comparison), while the final model here reaches `95.43%` accuracy on the held-out test set.

View the documentation here: 

```text
https://CuriousSensie.github.io/fake-news-detection/
```

## Research Question

Can transformer-based text and image representations improve fake news detection compared with relying on a single modality?

## Final Approach

The implemented pipeline uses:

| Component | Choice |
| --- | --- |
| Text encoder | `bert-base-uncased` |
| Image encoder | `google/vit-base-patch16-224` |
| Text feature | BERT `[CLS]` embedding |
| Image feature | ViT `[CLS]` embedding |
| Fusion | `[text_features, 0.5 * image_features]` |
| Classifier | `StandardScaler` + `LinearSVC` |

## Reported Result

On the held-out test set, the final multimodal model achieved:

| Metric | Value |
| --- | ---: |
| Accuracy | 95.43% |
| Precision, fake | 95.40% |
| Recall, fake | 96.73% |
| F1-score, fake | 96.06% |

The canonical executed notebook is available at `notebooks/multimodal_fake_news_detection_bert_vit_svm.ipynb`.
