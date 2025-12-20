import torch
import torch.nn as nn
from transformers import BertModel, ViTModel


class MultimodalEncoder(nn.Module):
    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        vit_model_name="google/vit-base-patch16-224",
        image_weight=0.5,
        trainable_bert_layers=(10, 11),
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.image_weight = image_weight

        for parameter in self.vit.parameters():
            parameter.requires_grad = False

        trainable_layer_names = tuple(f"encoder.layer.{layer}" for layer in trainable_bert_layers)
        for name, parameter in self.bert.named_parameters():
            parameter.requires_grad = any(layer_name in name for layer_name in trainable_layer_names)

    def forward(self, input_ids, attention_mask, image):
        text_features = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]

        image_features = self.vit(pixel_values=image).last_hidden_state[:, 0, :]
        return torch.cat([text_features, self.image_weight * image_features], dim=1)
