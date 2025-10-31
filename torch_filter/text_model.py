# torch_filter/text_model.py
# DistilRoBERTa binary text classifier
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class DistilRobertaBinary(nn.Module):
    def __init__(self, model_name: str = "distilroberta-base"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, 1)  # unsafe logit

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        x = self.dropout(pooled)
        return self.head(x).squeeze(1)
