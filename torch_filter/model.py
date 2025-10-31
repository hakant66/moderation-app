# torch_filter/model.py
import torch
import torch.nn as nn
from torchvision import models

class ImageFilter(nn.Module):
    """
    Binary image classifier built on MobileNetV3-Small.
    MobileNetV3-Small produces 576-dim features before the classifier.
    We pool to 1x1, then run a small head to a single logit.
    """
    def __init__(self):
        super().__init__()
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = m.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_f = m.classifier[0].in_features  # 576 for MobileNetV3-Small

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_f, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),  # single logit (unsafe probability will be sigmoid in serving code)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)                 # [N, 576, 1, 1]
        x = self.head(x)                    # [N, 1]
        return x.squeeze(1)                 # [N]
