"""CNN-LSTM and Multimodal CNN-LSTM Models for Fake News Detection.

Implements architectures following the IEEE Access ELD-FN paper:
1. TextCNNLSTM:
   Input Text Tokens -> Embedding (GloVe/trainable) -> Parallel 1D CNNs (k=[2,3,4,5]) ->
   AdaptiveMaxPool1d -> Concat (400-d) -> Bidirectional LSTM -> Last Hidden -> Dense -> Output

2. MultimodalCNNLSTM:
   Text Branch: CNN-LSTM text representation
   Image Branch: ConvNet/ResNet feature extractor
   Fusion: Concatenation / Sentiment integration -> Classifier
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torchvision.models as models


class TextCNNLSTM(nn.Module):
    """Text-only CNN + BiLSTM model matching paper's design."""

    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 100,
        num_classes: int = 2,
        num_filters: int = 100,
        kernel_sizes: Optional[List[int]] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            )
            for k in kernel_sizes
        ])

        cnn_out_dim = len(kernel_sizes) * num_filters
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len] -> [batch, seq_len, embed_dim] -> [batch, embed_dim, seq_len]
        emb = self.embedding(input_ids).transpose(1, 2)
        conv_outs = [conv(emb) for conv in self.conv_branches]
        # Concat along channel dimension: [batch, cnn_out_dim, seq_len] -> [batch, seq_len, cnn_out_dim]
        cnn_feat = torch.cat(conv_outs, dim=1).transpose(1, 2)
        lstm_out, (hn, _) = self.lstm(cnn_feat)
        # Concatenate forward and backward final hidden states
        feat = torch.cat([hn[-2], hn[-1]], dim=-1)
        return feat

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(input_ids)
        return self.classifier(self.dropout(features))


class MultimodalCNNLSTM(nn.Module):
    """Multimodal Fake News Detector fusing CNN+LSTM text features and Visual features."""

    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 100,
        num_classes: int = 2,
        num_filters: int = 100,
        kernel_sizes: Optional[List[int]] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        image_feat_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.text_encoder = TextCNNLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )

        # Image CNN Backbone (ResNet18 / ConvNet)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.image_encoder = resnet

        fused_dim = (lstm_hidden * 2) + image_feat_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, image: Optional[torch.Tensor] = None) -> torch.Tensor:
        text_feat = self.text_encoder.extract_features(input_ids)
        if image is not None:
            img_feat = self.image_encoder(image)
            fused = torch.cat([text_feat, img_feat], dim=-1)
            return self.classifier(self.dropout(fused))
        return self.text_encoder.classifier(self.dropout(text_feat))


# Export aliases matching existing src/ imports
MultimodalEncoder = MultimodalCNNLSTM
