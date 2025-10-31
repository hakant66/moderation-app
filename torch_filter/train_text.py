# torch_filter/train_text.py
import os, json, math, random
from dataclasses import dataclass
from typing import Dict, List
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = os.getenv("MODEL_NAME", "distilroberta-base")
TRAIN = os.getenv("TRAIN_PATH", "data/text_train.jsonl")
VAL = os.getenv("VAL_PATH", "data/text_val.jsonl")
OUT = os.getenv("OUT_DIR", "runs/text_prefilter")
EPOCHS = int(os.getenv("EPOCHS", "3"))
LR = float(os.getenv("LR", "2e-5"))
BATCH = int(os.getenv("BATCH", "16"))
MAXLEN = int(os.getenv("MAXLEN", "256"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

class JsonlText(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(x) for x in open(path, "r", encoding="utf-8")]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = tok(r["text"], truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
        item = {k:v.squeeze(0) for k,v in enc.items()}
        item["label"] = torch.tensor(int(r["label"]), dtype=torch.long)
        return item

train_ds = JsonlText(TRAIN)
val_ds   = JsonlText(VAL)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=BATCH)

class TextHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(dim, 1)
        )
    def forward(self, x): return self.fc(x)

class TextPrefilter(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        hidden = self.backbone.config.hidden_size
        self.head = TextHead(hidden)
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.head(cls).squeeze(-1)
        return logits

model = TextPrefilter(MODEL_NAME).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
bce = nn.BCEWithLogitsLoss()

def evaluate():
    model.eval()
    n, loss_sum, correct = 0, 0.0, 0
    with torch.no_grad():
        for batch in val_dl:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y = batch["label"].float().to(DEVICE)
            logits = model(ids, mask)
            loss = bce(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = (torch.sigmoid(logits) >= 0.5).long()
            correct += (pred == y.long()).sum().item()
            n += y.size(0)
    return loss_sum / max(n,1), correct / max(n,1)

best_acc = 0.0
for ep in range(1, EPOCHS+1):
    model.train()
    for batch in train_dl:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        y = batch["label"].float().to(DEVICE)
        logits = model(ids, mask)
        loss = bce(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
    val_loss, val_acc = evaluate()
    print(f"epoch {ep} val_loss={val_loss:.4f} acc={val_acc:.3f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUT, "text_best.pt"))
        print("saved:", os.path.join(OUT, "text_best.pt"))

# Export TorchScript
model.load_state_dict(torch.load(os.path.join(OUT, "text_best.pt"), map_location="cpu"))
model.eval().to("cpu")
ex = tok("example", return_tensors="pt", padding="max_length", truncation=True, max_length=MAXLEN)
ts = torch.jit.trace(model, (ex["input_ids"], ex["attention_mask"]))
os.makedirs("filters", exist_ok=True)
ts.save("filters/text_model.ts")
print("Exported -> filters/text_model.ts")
