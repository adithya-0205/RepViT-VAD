import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub

from models.repvit_backbone import RepViTBackbone
from models.tcn import TCN


class RepViTTCN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Quantization stubs for QAT / INT8 conversion
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # RepViT-M1.0 backbone produces 448-dimensional features
        self.backbone = RepViTBackbone()
        self.tcn = TCN(input_dim=448)

        # Anomaly Head: MLP + Sigmoid
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x shape:
        (Batch, Frames, Channels, Height, Width)
        Example: (B, 16, 3, 224, 224)
        """
        x = self.quant(x)

        B, T, C, H, W = x.shape

        # Merge batch and time dimensions for frame feature extraction
        x = x.view(B * T, C, H, W)

        # Extract spatial features via RepViT-M1.0 backbone
        x = self.backbone(x)

        # Reshape to (Batch, Frames, Feature_Dim=448)
        x = x.view(B, T, -1)

        # Temporal modeling & Global Average Pooling (GAP across time)
        x = self.tcn(x)

        # Anomaly Classification Head (MLP + Sigmoid)
        x = self.classifier(x)

        x = self.dequant(x)

        return x