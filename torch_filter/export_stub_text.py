# torch_filter/export_stub_text.py
# Generates filters/text_model.ts – text stub (always "safe" ~ low prob)
import os, torch, torch.nn as nn

class TextStub(nn.Module):
    def forward(self, input_ids, attention_mask):
        batch = input_ids.size(0)
        return torch.ones(batch) * -10.0  # logit ~ -10 -> prob ~ 0.00005

os.makedirs("filters", exist_ok=True)
m = TextStub().eval()
# Example inputs: batch=1, seq=16
ids = torch.ones(1, 16, dtype=torch.long)
mask = torch.ones(1, 16, dtype=torch.long)
ts = torch.jit.trace(m, (ids, mask))
ts.save("filters/text_model.ts")
print("wrote filters/text_model.ts (text stub)")
