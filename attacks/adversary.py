import torch
from torch import nn
import torch.nn.functional as F


class Adverasry:
    def __init__(self, eps):
        self.eps = eps

    def _attack(self, model, X, y):
        raise NotImplementedError

    def validate(self, X_adv, X):
        max_perturb = (X_adv - X).abs().max()
        lower_bound = X_adv.min()
        upper_bound = X_adv.max()
        return max_perturb <= self.eps and lower_bound >= 0 and upper_bound <= 1

    @torch.enable_grad()
    def attack(self, model, X, y=None):
        model.requires_grad_(False)
        training = model.training
        if training:
            model.eval()
        x_adv = self._attack(model, X, y)
        if training:
            model.train()
        model.requires_grad_()
        return x_adv
