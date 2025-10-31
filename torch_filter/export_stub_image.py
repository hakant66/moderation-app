# torch_filter/export_stub_image.py
# Generates filters/model.ts – image stub (always "unsafe" ~ high prob)
import os, torch, torch.nn as nn

class ImageStub(nn.Module):
    def forward(self, x):
        # x shape (N, 3, 224, 224) typically; ignore and return high logit
        return torch.ones(x.size(0)) * 10.0  # logit ~ +10 -> prob ~ 0.99995

os.makedirs("filters", exist_ok=True)
m = ImageStub().eval()
example = torch.randn(1, 3, 224, 224)
ts = torch.jit.trace(m, example)
ts.save("filters/model.ts")
print("wrote filters/model.ts (image stub)")
