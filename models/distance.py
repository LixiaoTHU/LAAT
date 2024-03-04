import torch
from torch import nn


class L2Similarity(nn.Module):
    def __init__(self, gts: torch.Tensor):
        super().__init__()
        self.register_buffer("gts", gts, False)

    def forward(self, image_features: torch.Tensor):
        x_expand = image_features.unsqueeze(1)
        gts = self.gts.unsqueeze(0)
        logits = -torch.sum((x_expand - gts).square(), axis=-1).sqrt()
        return logits
