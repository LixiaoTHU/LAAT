import torch
from torch import nn
import torch.nn.functional as F

from .adversary import Adverasry

class PGD(Adverasry):
    def __init__(self, eps=8 / 255, alpha=2 / 255, steps=10):
        super().__init__(eps)
        self.alpha = alpha
        self.steps = steps

    def _attack(self, model, X, y):
        delta = torch.empty_like(X)
        delta.uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(X + delta, min=0, max=1).detach()
        for _ in range(self.steps):
            x_adv.requires_grad_()
            output = model(x_adv)
            loss = F.cross_entropy(output, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv.requires_grad_(False)
            delta = x_adv - X
            delta = torch.add(delta, torch.sign(grad), alpha=self.alpha)
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(X + delta, min=0, max=1).detach()
        return x_adv


class PGDTrades(Adverasry):
    def __init__(self, eps=8 / 255, alpha=2 / 255, steps=10):
        super().__init__(eps)
        self.alpha = alpha
        self.steps = steps

    def _attack(self, model, X, y):
        delta = torch.empty_like(X)
        delta.uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(X + delta, min=0, max=1).detach()
        output_clean = model(X)
        softmax_clean = F.softmax(output_clean, dim=1)
        for _ in range(self.steps):
            x_adv.requires_grad_()
            output = model(x_adv)
            loss = F.kl_div(
                F.log_softmax(output, dim=1), softmax_clean, reduction="batchmean"
            )
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv.requires_grad_(False)
            delta = x_adv - X
            delta = torch.add(delta, torch.sign(grad), alpha=self.alpha)
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(X + delta, min=0, max=1).detach()
        return x_adv
