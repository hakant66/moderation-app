# torch_filter/text_dataset.py
# 
import json
from dataclasses import dataclass
from typing import List, Dict
from transformers import AutoTokenizer
import torch

@dataclass
class TextItem:
    text: str
    label: int

class JsonlTextDataset:
    def __init__(self, path: str):
        self.items: List[TextItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append(TextItem(obj["text"], int(obj["label"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> TextItem:
        return self.items[idx]

class Collator:
    def __init__(self, model_name: str = "distilroberta-base", max_len: int = 256):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_len = max_len

    def __call__(self, batch: List[TextItem]) -> Dict:
        texts = [b.text for b in batch]
        labels = [b.label for b in batch]
        out = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        out["labels"] = torch.tensor(labels, dtype=torch.float)
        return out
