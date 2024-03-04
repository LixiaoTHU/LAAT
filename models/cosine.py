import torch
from torch import nn


class CosineSimilarity(nn.Module):
    def __init__(self, gts: torch.Tensor, tau=1.0, use_acos=False):
        super().__init__()
        gts = gts / gts.norm(dim=-1, keepdim=True)
        self.register_buffer("gts", gts, False)
        self.tau = tau
        self.use_acos = use_acos

    def forward(self, image_features: torch.Tensor):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.gts.T
        if self.use_acos:
            logits = torch.acos(logits)
        return logits / self.tau
