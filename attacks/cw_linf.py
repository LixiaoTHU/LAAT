import torch
from torch import nn
import torch.nn.functional as F

from .adversary import Adverasry

class CWLinf(Adverasry):
    def __init__(self, eps=8 / 255, alpha=0.8 / 255, steps=30, kappa=0):
        super().__init__(eps)
        self.alpha = alpha
        self.steps = steps
        self.kappa = kappa

    def _attack(self, model, X, y):
        delta = torch.empty_like(X)
        delta.uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(X + delta, min=0, max=1).detach()
        for _ in range(self.steps):
            x_adv.requires_grad_()
            output = model(x_adv)
            loss = -self.f(output, y).mean()
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv.requires_grad_(False)
            delta = x_adv - X
            delta = torch.add(delta, torch.sign(grad), alpha=self.alpha)
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(X + delta, min=0, max=1).detach()
        return x_adv

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]), device=outputs.device)[labels]

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

        return torch.clamp((j-i), min=-self.kappa)
