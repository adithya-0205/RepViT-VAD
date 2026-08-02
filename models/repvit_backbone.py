import torch
import torch.nn as nn

from model.repvit import repvit_m1_0


class RepViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = repvit_m1_0()

    def forward(self, x):
        for layer in self.model.features:
            x = layer(x)

        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.flatten(1)

        return x