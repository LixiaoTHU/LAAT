import torch
from torch import nn
import torch.nn.functional as F

def dot_loss(logits, y, reduction="mean"):
    mask = F.one_hot(y, num_classes=logits.shape[1])
    if reduction == "mean":
        return -torch.sum(mask * logits) / logits.shape[0]
    elif reduction == "sum":
        return -torch.sum(mask * logits)
    elif reduction == "none":
        return -torch.sum(mask * logits, dim=-1)
    else:
        raise ValueError("Unknown reduction '{}'".format(reduction))

class DotLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, X, y):
        return dot_loss(X, y, self.reduction)
