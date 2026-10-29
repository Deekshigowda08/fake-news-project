"""Training module initialization."""

from ml.training.trainer import train_epoch, evaluate_model

def __getattr__(name: str):
    if name in ("train_bert_model", "TextClassificationDataset"):
        from ml.training.train_bert import train_bert_model, TextClassificationDataset
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["train_epoch", "evaluate_model", "train_bert_model", "TextClassificationDataset"]
