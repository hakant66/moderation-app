# torch_filter/train_image.py
import os
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

# ----------------------------
# Config (env overrides)
# ----------------------------
DATA_DIR = Path(os.getenv("IMG_DATA", "data/image"))
OUT_DIR = Path(os.getenv("OUT_DIR", "runs/image_prefilter"))
EPOCHS = int(os.getenv("EPOCHS", "6"))
LR = float(os.getenv("LR", "1e-3"))
BATCH = int(os.getenv("BATCH", "32"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))  # 0 is safe on Windows
PATIENCE = int(os.getenv("PATIENCE", "3"))        # early stop patience (epochs)
SEED = int(os.getenv("SEED", "1337"))
IM_SIZE = int(os.getenv("IM_SIZE", "224"))
USE_SAMPLER = os.getenv("USE_SAMPLER", "1") == "1"  # weighted sampler for class imbalance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seeds(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders():
    """
    Expect directory layout:
      data/image/train/safe|unsafe/...
      data/image/val/safe|unsafe/...
    """
    # Use the official MobileNet weights to get the correct normalization
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    mean, std = weights.meta["mean"], weights.meta["std"]

    train_tf = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_root = DATA_DIR / "train"
    val_root = DATA_DIR / "val"

    train_ds = datasets.ImageFolder(str(train_root), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_root), transform=val_tf)

    # Sanity: ensure class mapping is what we expect
    # ImageFolder maps classes alphabetically, so "safe"->0, "unsafe"->1 (correct for BCE)
    expected = {"safe": 0, "unsafe": 1}
    if any(k not in train_ds.class_to_idx for k in expected):
        raise RuntimeError(f"Expected folders 'safe' & 'unsafe' under {train_root}")
    if train_ds.class_to_idx != expected:
        # If your folder names differ (e.g., benign/toxic), change policy accordingly
        print("WARN: class_to_idx is", train_ds.class_to_idx, "(expected {'safe':0,'unsafe':1})")

    # Optional weighted sampler (helps with imbalance)
    if USE_SAMPLER:
        counts = [0, 0]
        for _, y in train_ds.samples:
            counts[y] += 1
        # weight inversely proportional to class frequency
        class_weights = [0, 0]
        total = sum(counts) if sum(counts) > 0 else 1
        for c in (0, 1):
            class_weights[c] = total / max(counts[c], 1)
        sample_weights = [class_weights[y] for _, y in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_dl = DataLoader(
        train_ds, batch_size=BATCH, shuffle=shuffle, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )
    return train_ds, val_ds, train_dl, val_dl


def build_model():
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    m = models.mobilenet_v3_small(weights=weights)
    # Replace classifier head with binary logit
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, 1)
    return m


@torch.no_grad()
def evaluate(model, dl, loss_fn):
    model.eval()
    n, loss_sum, correct = 0, 0.0, 0
    for x, y in dl:
        x = x.to(DEVICE)
        y = y.float().to(DEVICE)
        logits = model(x).squeeze(1)
        loss = loss_fn(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = (torch.sigmoid(logits) >= 0.5).long()
        correct += (pred == y.long()).sum().item()
        n += y.size(0)
    return (loss_sum / max(n, 1)), (correct / max(n, 1))


def main():
    set_seeds(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, train_dl, val_dl = make_dataloaders()
    print("Classes:", train_ds.classes, "class_to_idx:", train_ds.class_to_idx)
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)} | Device: {DEVICE}")

    model = build_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    epochs_no_improve = 0
    best_path = OUT_DIR / "image_best.pt"

    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for x, y in train_dl:
            x = x.to(DEVICE)
            y = y.float().to(DEVICE)
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

        val_loss, val_acc = evaluate(model, val_dl, loss_fn)
        dt = time.time() - t0
        print(f"epoch {ep} val_loss={val_loss:.4f} acc={val_acc:.3f} ({dt:.1f}s)")

        # Early stopping on accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path.as_posix())
            print("saved:", best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stop after {ep} epochs (no improve {PATIENCE}x)")
                break

    # ----------------------------
    # Export TorchScript
    # ----------------------------
    # Load best (CPU), set to eval, trace with normalized dummy
    model_cpu = build_model()
    model_cpu.load_state_dict(torch.load(best_path, map_location="cpu"))
    model_cpu.eval()

    # Use the same normalization for the dummy input (via weights meta)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    mean, std = weights.meta["mean"], weights.meta["std"]
    # Create a dummy tensor that *would be* post-normalization shape [1,3,224,224]
    # (Tracing only needs shape, not exact pixel stats)
    example = torch.randn(1, 3, IM_SIZE, IM_SIZE)

    ts = torch.jit.trace(model_cpu, example)
    filters_dir = Path("filters")
    filters_dir.mkdir(parents=True, exist_ok=True)
    ts_path = filters_dir / "model.ts"
    ts.save(ts_path.as_posix())
    print(f"Exported -> {ts_path}")


if __name__ == "__main__":
    main()
