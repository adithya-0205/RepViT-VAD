
import os

import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub

from models.repvit_backbone import RepViTBackbone
from models.tcn import TCN


# ============================================================
# RepViT-M1.0 + TCN VIDEO ANOMALY DETECTION MODEL
# ============================================================

class RepViTTCN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # ----------------------------------------------------
        # Quantization stubs
        # ----------------------------------------------------
        # These allow later QAT / INT8 conversion.
        # They do not change normal FP32 training behavior.

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # ----------------------------------------------------
        # RepViT-M1.0 BACKBONE
        # ----------------------------------------------------
        #
        # RepViTBackbone internally uses:
        #
        #     repvit_m1_0()
        #
        # Output:
        #
        #     448 features per frame
        #

        self.backbone = RepViTBackbone()

        # ----------------------------------------------------
        # TCN
        # ----------------------------------------------------
        #
        # Input:
        #     448 features/frame
        #
        # Output:
        #     128-dimensional temporal representation
        #

        self.tcn = TCN(
            input_dim=448
        )

        # ----------------------------------------------------
        # CLASSIFICATION HEAD
        # ----------------------------------------------------
        #
        # 128 -> 64 -> num_classes
        #
        # Initial stage:
        #
        #     normal   = 0
        #     fighting = 1
        #
        # Future:
        #
        #     assault  = 2
        #     arson    = 3
        #     etc.
        #

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    # ========================================================
    # CLASSIFIER EXPANSION
    # ========================================================

    def expand_classifier(self, new_num_classes):
        """
        Expand the final classifier while preserving
        previously learned class weights.

        Example:

            Old model:
                2 classes
                normal = 0
                fighting = 1

            New model:
                3 classes
                normal = 0
                fighting = 1
                assault = 2

        The weights for normal and fighting are copied.
        The new class weights are initialized normally
        by nn.Linear.
        """

        old_linear = self.classifier[3]

        old_num_classes = (
            old_linear.out_features
        )

        if new_num_classes == old_num_classes:
            return

        if new_num_classes < old_num_classes:
            raise ValueError(
                "Cannot shrink classifier from "
                f"{old_num_classes} classes to "
                f"{new_num_classes} classes. "
                "Class indices must remain stable "
                "during continual learning."
            )

        new_linear = nn.Linear(
            old_linear.in_features,
            new_num_classes
        )

        # ----------------------------------------------------
        # Preserve old class weights
        # ----------------------------------------------------

        with torch.no_grad():

            new_linear.weight[
                :old_num_classes
            ].copy_(
                old_linear.weight
            )

            new_linear.bias[
                :old_num_classes
            ].copy_(
                old_linear.bias
            )

        # Replace only the final classifier layer.

        self.classifier[3] = new_linear

        print(
            f"[Classifier] Expanded from "
            f"{old_num_classes} -> "
            f"{new_num_classes} classes."
        )

    # ========================================================
    # LOAD CONTINUAL CHECKPOINT
    # ========================================================

    def load_continual_checkpoint(
        self,
        checkpoint_path,
        device="cpu"
    ):
        """
        Load a checkpoint from a previous teammate/stage.

        Example:

            Previous checkpoint:
                normal
                fighting

            Current model:
                normal
                fighting
                assault

        The old two-class classifier weights are copied into
        the first two rows of the new three-class classifier.

        RepViT-M1.0 and TCN weights are also restored.
        """

        if not checkpoint_path:

            return None

        if not os.path.exists(
            checkpoint_path
        ):

            print(
                f"[Checkpoint] Not found: "
                f"{checkpoint_path}"
            )

            return None

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        # ----------------------------------------------------
        # Extract state dictionary
        # ----------------------------------------------------

        if (
            isinstance(checkpoint, dict)
            and "state_dict" in checkpoint
        ):

            state_dict = checkpoint[
                "state_dict"
            ]

            saved_classes = checkpoint.get(
                "classes",
                None
            )

        else:

            state_dict = checkpoint
            saved_classes = None

        # Make a copy so the original checkpoint dictionary
        # isn't modified accidentally.

        state_dict = dict(
            state_dict
        )

        # ----------------------------------------------------
        # Check saved classifier
        # ----------------------------------------------------

        weight_key = (
            "classifier.3.weight"
        )

        bias_key = (
            "classifier.3.bias"
        )

        if weight_key in state_dict:

            saved_weight = state_dict[
                weight_key
            ]

            saved_bias = state_dict[
                bias_key
            ]

            saved_num_classes = (
                saved_weight.shape[0]
            )

            current_num_classes = (
                self.classifier[3].out_features
            )

            # ------------------------------------------------
            # Current model has more classes
            # ------------------------------------------------

            if (
                current_num_classes
                > saved_num_classes
            ):

                print(
                    f"[Checkpoint] Previous model: "
                    f"{saved_num_classes} classes"
                )

                print(
                    f"[Checkpoint] Current model: "
                    f"{current_num_classes} classes"
                )

                print(
                    "[Checkpoint] Expanding classifier "
                    "for new anomaly classes."
                )

                # Create the larger classifier.

                old_linear = (
                    self.classifier[3]
                )

                new_linear = nn.Linear(
                    old_linear.in_features,
                    current_num_classes
                )

                # Copy old checkpoint weights.

                with torch.no_grad():

                    copy_classes = min(
                        saved_num_classes,
                        current_num_classes
                    )

                    new_linear.weight[
                        :copy_classes
                    ].copy_(
                        saved_weight[
                            :copy_classes
                        ]
                    )

                    new_linear.bias[
                        :copy_classes
                    ].copy_(
                        saved_bias[
                            :copy_classes
                        ]
                    )

                self.classifier[3] = (
                    new_linear
                )

                # Remove classifier entries from the
                # state dictionary because their dimensions
                # no longer match.

                state_dict.pop(
                    weight_key,
                    None
                )

                state_dict.pop(
                    bias_key,
                    None
                )

                # Load the rest of the network.

                self.load_state_dict(
                    state_dict,
                    strict=False
                )

                print(
                    "[Checkpoint] Previous "
                    "classifier weights preserved."
                )

                print(
                    "[Checkpoint] New class weights "
                    "initialized for training."
                )

                # Copy old classifier weights again after
                # loading the remaining model parameters.

                with torch.no_grad():

                    self.classifier[3].weight[
                        :saved_num_classes
                    ].copy_(
                        saved_weight
                    )

                    self.classifier[3].bias[
                        :saved_num_classes
                    ].copy_(
                        saved_bias
                    )

            # ------------------------------------------------
            # Same number of classes
            # ------------------------------------------------

            elif (
                current_num_classes
                == saved_num_classes
            ):

                self.load_state_dict(
                    state_dict,
                    strict=False
                )

            # ------------------------------------------------
            # Current model has fewer classes
            # ------------------------------------------------

            else:

                saved_names = (
                    saved_classes
                    if saved_classes is not None
                    else "unknown"
                )

                raise ValueError(
                    "\n[Checkpoint ERROR]\n"
                    f"Checkpoint contains "
                    f"{saved_num_classes} classes,\n"
                    f"but current model has only "
                    f"{current_num_classes} classes.\n"
                    f"Checkpoint classes: "
                    f"{saved_names}\n\n"
                    "Do not train a smaller class set "
                    "from a larger continual-learning "
                    "checkpoint."
                )

        else:

            # ------------------------------------------------
            # Older checkpoint without classifier information
            # ------------------------------------------------

            self.load_state_dict(
                state_dict,
                strict=False
            )

        print(
            f"Successfully loaded continual "
            f"checkpoint from "
            f"'{checkpoint_path}'!"
        )

        if saved_classes is not None:

            print(
                f"[Checkpoint] Previous classes: "
                f"{saved_classes}"
            )

        return saved_classes

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):
        """
        Input:

            x =
            (B, T, C, H, W)

        Example:

            (2, 16, 3, 224, 224)

        Processing:

            Video
                ↓
            RepViT-M1.0
                ↓
            448 features/frame
                ↓
            TCN
                ↓
            128 temporal features
                ↓
            Classifier
                ↓
            class logits

        Output:

            (B, num_classes)
        """

        # ----------------------------------------------------
        # Quantization stub
        # ----------------------------------------------------

        x = self.quant(x)

        # ----------------------------------------------------
        # Video dimensions
        # ----------------------------------------------------

        B, T, C, H, W = x.shape

        # ----------------------------------------------------
        # Merge batch + temporal dimensions
        # ----------------------------------------------------
        #
        # (B, T, C, H, W)
        #
        # becomes:
        #
        # (B*T, C, H, W)
        #

        x = x.reshape(
            B * T,
            C,
            H,
            W
        )

        # ----------------------------------------------------
        # RepViT-M1.0
        # ----------------------------------------------------
        #
        # Each frame independently becomes:
        #
        # 448-dimensional feature
        #

        x = self.backbone(x)

        # ----------------------------------------------------
        # Restore temporal dimension
        # ----------------------------------------------------
        #
        # (B*T, 448)
        #
        # becomes:
        #
        # (B, T, 448)
        #

        x = x.reshape(
            B,
            T,
            448
        )

        # ----------------------------------------------------
        # TCN
        # ----------------------------------------------------
        #
        # Temporal modeling across the 16 frames.
        #
        # Expected output:
        #
        # (B, 128)
        #

        x = self.tcn(x)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        x = self.classifier(x)

        # ----------------------------------------------------
        # Dequantization stub
        # ----------------------------------------------------

        x = self.dequant(x)

        return x
