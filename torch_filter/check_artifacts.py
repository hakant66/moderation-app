# torch_filter/check_artifacts.py
import torch

m_img = torch.jit.load("filters/model.ts", map_location="cpu")
m_txt = torch.jit.load("filters/text_model.ts", map_location="cpu")
print("image ok:", callable(m_img), "text ok:", callable(m_txt))
