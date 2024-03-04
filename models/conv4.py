import math
import torch.nn as nn


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# Simple Conv Block
class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, channels=64, postprocess="flatten"):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else channels
            outdim = channels
            # only pooling for fist 4 layers
            B = ConvBlock(indim, outdim, pool=(i < 4))
            trunk.append(B)
        
        self.postprocess = postprocess
        if postprocess == "flatten":
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.channels = channels
        self.final_feat_dim = 4 * channels

    def forward(self, x):
        x = self.trunk(x)
        if self.postprocess == "avgpool":
            x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x


def conv4():
    return ConvNet(4)


def conv4_512():
    return ConvNet(4, 512)


if __name__ == "__main__":
    import torch

    net = conv4_512()
    print(net)
    x = torch.randn([10, 3, 32, 32])
    y = net(x)
    print(y.shape)
    print(torch.sum(y < 0))
