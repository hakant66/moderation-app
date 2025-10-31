# torch_filter/export_text.py
# TorchScript export for text model
import os, torch
from text_model import DistilRobertaBinary

MODEL_NAME = os.getenv("TEXT_MODEL", "distilroberta-base")
WEIGHTS = os.getenv("WEIGHTS", "filters/text_model.pt")
OUT = os.getenv("OUT", "filters/text_model.ts")
MAXLEN = int(os.getenv("MAXLEN", 256))

if __name__ == "__main__":
    m = DistilRobertaBinary(MODEL_NAME)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    m.eval()

    example_ids = torch.ones(1, MAXLEN, dtype=torch.long)
    example_mask = torch.ones(1, MAXLEN, dtype=torch.long)
    ts = torch.jit.trace(m, (example_ids, example_mask))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ts.save(OUT)
    print(f"saved TorchScript -> {OUT}")
