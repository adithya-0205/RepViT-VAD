import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class TCN(nn.Module):
    def __init__(self, input_dim=448):

        super().__init__()

        self.network = nn.Sequential(

            TemporalBlock(input_dim, 256, dilation=1),

            TemporalBlock(256, 256, dilation=2),

            TemporalBlock(256, 128, dilation=4)

        )

    def forward(self, x):

        # Input:
        # (Batch, Frames, Features)

        x = x.permute(0, 2, 1)

        x = self.network(x)

        x = x.mean(dim=2)

        return x