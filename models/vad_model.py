import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
import os

from models.repvit_backbone import RepViTBackbone
from models.tcn import TCN


class RepViTTCN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Quantization stubs for QAT / INT8 conversion
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # RepViT-M1.0 backbone produces 448-dimensional features
        self.backbone = RepViTBackbone()
        self.tcn = TCN(input_dim=448)

        # Multi-class Anomaly Head: MLP outputting raw logits
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def expand_classifier(self, new_num_classes):
        """
        Dynamically resizes classifier head to new_num_classes while preserving
        already learned weights for previous classes.
        """
        old_linear = self.classifier[3]
        old_classes = old_linear.out_features
        if new_num_classes == old_classes:
            return

        new_linear = nn.Linear(64, new_num_classes)
        with torch.no_grad():
            min_classes = min(old_classes, new_num_classes)
            new_linear.weight[:min_classes] = old_linear.weight[:min_classes]
            new_linear.bias[:min_classes] = old_linear.bias[:min_classes]

        self.classifier[3] = new_linear

    def load_continual_checkpoint(self, checkpoint_path, device="cpu"):
        """
        Loads state dict or dictionary checkpoint from teammates safely, preserving
        learned anomaly weights even if number of classes differs.
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return None

        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            saved_classes = checkpoint.get("classes", None)
        else:
            state_dict = checkpoint
            saved_classes = None

        # Check classifier final weight shape in checkpoint
        weight_key = "classifier.3.weight"
        bias_key = "classifier.3.bias"

        if weight_key in state_dict:
            ckpt_num_classes = state_dict[weight_key].shape[0]
            current_num_classes = self.classifier[3].out_features

            if ckpt_num_classes != current_num_classes:
                target_classes = max(ckpt_num_classes, current_num_classes)
                self.expand_classifier(target_classes)

                # Copy matching classifier weights
                ckpt_weight = state_dict.pop(weight_key)
                ckpt_bias = state_dict.pop(bias_key)

                with torch.no_grad():
                    min_c = min(ckpt_num_classes, self.classifier[3].out_features)
                    self.classifier[3].weight[:min_c] = ckpt_weight[:min_c]
                    self.classifier[3].bias[:min_c] = ckpt_bias[:min_c]

        self.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded continual checkpoint from '{checkpoint_path}'!")
        return saved_classes

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

        # Anomaly Classification Head (MLP)
        x = self.classifier(x)

        x = self.dequant(x)

        return x