import torch
from torch import nn
import torch.nn.functional as F

from .adversary import Adverasry


class FGSM(Adverasry):
    def __init__(self, eps=8 / 255):
        super().__init__(eps)

    def _attack(self, model, X, y):
        x_adv = X.detach().requires_grad_()
        output = model(x_adv)
        loss = F.cross_entropy(output, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        delta = self.eps * torch.sign(grad)
        x_adv = torch.clamp(X + delta, min=0, max=1).detach()
        return x_adv
