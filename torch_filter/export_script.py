# torch_filter/export_script.py
import os
import torch
from model import ImageFilter

OUT = os.getenv("OUT_TS", "filters/model.ts")

if __name__ == "__main__":
    m = ImageFilter().eval()
    example = torch.randn(1, 3, 224, 224)
    ts = torch.jit.trace(m, example)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ts.save(OUT)
    print("saved TorchScript ->", OUT)
