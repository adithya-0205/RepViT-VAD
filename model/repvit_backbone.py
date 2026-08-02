import torch
import torch.nn as nn

from model.repvit import repvit_m0_9


class RepViTBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = repvit_m0_9()

    def forward(self, x):

        # Pass through all RepViT feature blocks
        for layer in self.model.features:
            x = layer(x)

        # Global Average Pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)

        # Shape:
        # (B,C,1,1) -> (B,C)
        x = x.flatten(1)

        return x