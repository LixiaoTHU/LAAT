import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(
        model_state_dict.items(), proxy_state_dict.items()
    ):
        if len(old_w.size()) <= 1:
            continue
        if "weight" in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + _EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturb(object):
    def __init__(self, model, criterion, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.criterion = criterion
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss = -self.criterion(self.proxy(inputs_adv), targets)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


class AdvWeightPerturb(object):
    def __init__(self, model, criterion, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.criterion = criterion
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss = -self.criterion(self.proxy(inputs_adv), targets)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


class TradesCosAWP(AdvWeightPerturb):
    def __init__(self, model, criterion, proxy, proxy_optim, gamma):
        super().__init__(model, criterion, proxy, proxy_optim, gamma)

    def calc_awp(self, inputs_adv, inputs_clean, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        feat_clean = self.proxy[:-1](inputs_clean)
        feat_adv = self.proxy[:-1](inputs_adv)
        logits_clean = self.proxy[-1](feat_clean)
        logits_adv = self.proxy[-1](feat_adv)
        loss_clean = self.criterion(logits_adv, targets)
        feat_clean = feat_clean / feat_clean.norm(dim=-1, keepdim=True)
        feat_adv = feat_adv / feat_adv.norm(dim=-1, keepdim=True)
        loss_kl = -torch.sum(feat_clean * feat_adv, dim=-1).mean()
        loss = -(loss_clean + beta * loss_kl)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff
