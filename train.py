import os
import json
import random
import shutil
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    Dataset
)

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from video_dataset import VideoDataset, ReplayDataset
from models.vad_model import RepViTTCN


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = device.type == "cuda"


if device.type == "cuda":

    torch.backends.cudnn.benchmark = True

else:

    cpu_count = os.cpu_count() or 4

    torch.set_num_threads(
        max(2, min(cpu_count, 8))
    )

    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass


print("=" * 75)
print("RepViT-M1.0 + TCN CONTINUAL ANOMALY LEARNING")
print("=" * 75)

print(f"[Device] {device}")
print(f"[AMP]    {USE_AMP}")

if device.type == "cpu":

    print(
        f"[CPU] Threads: {torch.get_num_threads()}"
    )


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_PATH = "best_model.pth"

RECOVERY_PATH = "training_recovery.pth"

CLASSES_JSON = "classes.json"

REPLAY_ROOT = "replay_buffer"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 10

BATCH_SIZE = 2

CLIP_LEN = 8

LR = 1e-3

WEIGHT_DECAY = 1e-4


# ============================================================
# CONTINUAL LEARNING
# ============================================================

# Elastic Weight Consolidation
EWC_LAMBDA = 100.0

# Knowledge Distillation
KD_ALPHA = 0.5

KD_TEMP = 4.0

# Number of clips retained per class
REPLAY_PER_CLASS = 30

# Fisher estimate
FISHER_BATCHES = 3

FISHER_LAYERS = (
    "tcn",
    "classifier"
)


# ============================================================
# INITIAL CLASSES
# ============================================================

INITIAL_CLASSES = [
    "normal",
    "fighting"
]


# ============================================================
# COMMAND LINE
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Continual training of RepViT-M1.0 + TCN "
        "with dynamically added anomaly classes."
    )
)

parser.add_argument(
    "--new-class",
    type=str,
    required=True,
    help=(
        "New anomaly class to add. "
        "Examples: fighting, assault, arson, robbery"
    )
)

args = parser.parse_args()

NEW_CLASS = args.new_class.strip().lower()


if not NEW_CLASS:

    raise ValueError(
        "New class name cannot be empty."
    )


# ============================================================
# CLASS NORMALIZATION
# ============================================================

def normalize_classes(classes):

    return [
        str(x).strip().lower()
        for x in classes
    ]


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(path):

    if not os.path.exists(path):

        return None

    try:

        checkpoint = torch.load(
            path,
            map_location="cpu"
        )

        if not isinstance(
            checkpoint,
            dict
        ):

            print(
                f"[Checkpoint] Invalid format: {path}"
            )

            return None

        return checkpoint

    except Exception as e:

        print(
            f"[Checkpoint] Failed to load "
            f"{path}: {e}"
        )

        return None


# ============================================================
# FIND BEST AVAILABLE CHECKPOINT
# ============================================================

def get_available_checkpoint():

    # --------------------------------------------------------
    # Priority 1:
    # best_model.pth
    # --------------------------------------------------------

    if os.path.exists(
        CHECKPOINT_PATH
    ):

        checkpoint = load_checkpoint(
            CHECKPOINT_PATH
        )

        if checkpoint is not None:

            print(
                f"[Checkpoint] Using "
                f"{CHECKPOINT_PATH}"
            )

            return checkpoint, CHECKPOINT_PATH

    # --------------------------------------------------------
    # Priority 2:
    # training_recovery.pth
    #
    # Your current situation is:
    #
    # training_recovery.pth exists
    # best_model.pth does not exist
    #
    # Therefore recovery becomes the old model.
    # --------------------------------------------------------

    if os.path.exists(
        RECOVERY_PATH
    ):

        recovery = load_checkpoint(
            RECOVERY_PATH
        )

        if recovery is not None:

            print(
                "[Checkpoint] best_model.pth "
                "not found."
            )

            print(
                "[Checkpoint] Using "
                "training_recovery.pth "
                "as fallback."
            )

            return recovery, RECOVERY_PATH

    return None, None


# ============================================================
# DETERMINE EXISTING CLASSES
# ============================================================

def get_existing_classes():

    checkpoint, checkpoint_path = (
        get_available_checkpoint()
    )

    if checkpoint is not None:

        classes = checkpoint.get(
            "classes"
        )

        if classes:

            classes = normalize_classes(
                classes
            )

            print(
                f"[Checkpoint] Existing classes: "
                f"{classes}"
            )

            return classes

    print(
        "[Checkpoint] No previous model found."
    )

    print(
        f"[Checkpoint] Initial classes: "
        f"{INITIAL_CLASSES}"
    )

    return INITIAL_CLASSES.copy()


# ============================================================
# DETERMINE CURRENT CLASSES
# ============================================================

def build_current_classes(
    existing_classes,
    checkpoint_exists
):

    # --------------------------------------------------------
    # No previous model:
    #
    # First stage must be normal + fighting.
    # --------------------------------------------------------

    if not checkpoint_exists:

        if NEW_CLASS not in INITIAL_CLASSES:

            raise RuntimeError(
                "\nNo previous checkpoint exists.\n"
                "The first training stage must use:\n"
                "  --new-class fighting\n\n"
                f"Received:\n"
                f"  --new-class {NEW_CLASS}\n"
            )

        return INITIAL_CLASSES.copy()

    # --------------------------------------------------------
    # Continual stage
    # --------------------------------------------------------

    current_classes = existing_classes.copy()

    if NEW_CLASS in current_classes:

        raise RuntimeError(
            f"\nClass '{NEW_CLASS}' already exists.\n\n"
            f"Existing classes:\n"
            f"{current_classes}\n\n"
            "Use a different anomaly class."
        )

    current_classes.append(
        NEW_CLASS
    )

    return current_classes


# ============================================================
# DATASET CHECK
# ============================================================

def check_new_class_exists():

    train_path = os.path.join(
        "dataset",
        "train",
        NEW_CLASS
    )

    val_path = os.path.join(
        "dataset",
        "val",
        NEW_CLASS
    )

    if not os.path.isdir(
        train_path
    ):

        raise FileNotFoundError(
            "\nNew class training directory does not exist:\n"
            f"  {train_path}\n\n"
            "Create:\n"
            f"  dataset/train/{NEW_CLASS}/"
        )

    if not os.path.isdir(
        val_path
    ):

        raise FileNotFoundError(
            "\nNew class validation directory does not exist:\n"
            f"  {val_path}\n\n"
            "Create:\n"
            f"  dataset/val/{NEW_CLASS}/"
        )

    print(
        f"[Dataset] Found training data: "
        f"{train_path}"
    )

    print(
        f"[Dataset] Found validation data: "
        f"{val_path}"
    )


# ============================================================
# FREEZE REPVIT
# ============================================================

def freeze_repvit(model):

    trainable = 0

    frozen = 0

    for name, parameter in model.named_parameters():

        lname = name.lower()

        if (
            "tcn" in lname
            or "classifier" in lname
        ):

            parameter.requires_grad = True

            trainable += parameter.numel()

        else:

            parameter.requires_grad = False

            frozen += parameter.numel()

    print()
    print("[MODEL]")

    print(
        "Architecture : RepViT-M1.0 + TCN"
    )

    print(
        f"Trainable    : {trainable:,}"
    )

    print(
        f"Frozen       : {frozen:,}"
    )

    if trainable == 0:

        raise RuntimeError(
            "No trainable parameters found."
        )


# ============================================================
# DATA LOADER
# ============================================================

def make_loader(
    dataset,
    shuffle
):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=False
    )


# ============================================================
# REPLAY PATHS
# ============================================================

def collect_replay_paths(
    classes
):

    replay_paths = {}

    for class_name in classes:

        class_dir = os.path.join(
            REPLAY_ROOT,
            class_name
        )

        if not os.path.isdir(
            class_dir
        ):

            continue

        clips = []

        for name in os.listdir(
            class_dir
        ):

            path = os.path.join(
                class_dir,
                name
            )

            if os.path.isdir(
                path
            ):

                clips.append(
                    os.path.normpath(
                        path
                    )
                )

        if clips:

            replay_paths[
                class_name
            ] = clips

    return replay_paths


# ============================================================
# RELATIVE REPLAY PATHS
# ============================================================

def make_relative_replay_paths(
    replay_paths
):

    result = {}

    for class_name, paths in replay_paths.items():

        result[class_name] = []

        for path in paths:

            path = os.path.normpath(
                path
            )

            if os.path.exists(
                path
            ):

                try:

                    relative = os.path.relpath(
                        path,
                        os.getcwd()
                    )

                except Exception:

                    relative = path

                result[class_name].append(
                    relative
                )

    return result


# ============================================================
# SAVE NEW CLASS REPLAY
# ============================================================

def add_class_to_replay(
    class_name,
    dataset
):

    class_dir = os.path.join(
        REPLAY_ROOT,
        class_name
    )

    os.makedirs(
        class_dir,
        exist_ok=True
    )

    samples = []

    if not hasattr(
        dataset,
        "samples"
    ):

        print(
            f"[Replay] Dataset for "
            f"{class_name} has no samples attribute."
        )

        return

    for path, label in dataset.samples:

        if (
            label < len(
                dataset.classes
            )
            and dataset.classes[label]
            == class_name
        ):

            samples.append(
                path
            )

    if not samples:

        print(
            f"[Replay] No samples found "
            f"for '{class_name}'."
        )

        return

    selected = random.sample(
        samples,
        min(
            REPLAY_PER_CLASS,
            len(samples)
        )
    )

    print()
    print(
        f"[Replay] Saving "
        f"{len(selected)} clips "
        f"for '{class_name}'"
    )

    for index, source in enumerate(
        selected
    ):

        source = os.path.normpath(
            source
        )

        base_name = os.path.basename(
            source
        )

        destination = os.path.join(
            class_dir,
            base_name
        )

        if os.path.exists(
            destination
        ):

            destination = os.path.join(
                class_dir,
                f"{base_name}_{index}"
            )

        try:

            shutil.copytree(
                source,
                destination,
                dirs_exist_ok=True
            )

        except Exception as e:

            print(
                f"[Replay] Failed to copy "
                f"{source}: {e}"
            )


# ============================================================
# REMAPPED DATASET
# ============================================================

class RemappedDataset(
    Dataset
):

    def __init__(
        self,
        base_dataset,
        global_label
    ):

        self.base_dataset = base_dataset

        self.global_label = global_label

    def __len__(self):

        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        index
    ):

        frames, _ = self.base_dataset[
            index
        ]

        return (
            frames,
            self.global_label
        )


# ============================================================
# BUILD TRAINING DATA
# ============================================================

def build_training_dataset(
    current_classes,
    previous_classes,
    is_first_stage
):

    # ========================================================
    # FIRST STAGE
    # ========================================================

    if is_first_stage:

        train_dataset = VideoDataset(
            train=True,
            classes=current_classes,
            clip_len=CLIP_LEN
        )

        val_dataset = VideoDataset(
            train=False,
            classes=current_classes,
            clip_len=CLIP_LEN
        )

        return (
            train_dataset,
            val_dataset
        )

    # ========================================================
    # NEW CLASS
    # ========================================================

    new_train_base = VideoDataset(
        train=True,
        classes=[NEW_CLASS],
        clip_len=CLIP_LEN
    )

    new_val_base = VideoDataset(
        train=False,
        classes=[NEW_CLASS],
        clip_len=CLIP_LEN
    )

    new_global_index = current_classes.index(
        NEW_CLASS
    )

    new_train_dataset = RemappedDataset(
        new_train_base,
        new_global_index
    )

    new_val_dataset = RemappedDataset(
        new_val_base,
        new_global_index
    )

    # ========================================================
    # OLD CLASS REPLAY
    # ========================================================

    replay_paths = collect_replay_paths(
        previous_classes
    )

    replay_dataset = None

    if replay_paths:

        print()
        print(
            "[Replay] Existing classes:"
        )

        for class_name, paths in replay_paths.items():

            print(
                f"  {class_name}: "
                f"{len(paths)} clips"
            )

        replay_dataset = ReplayDataset(
            replay_paths,
            {
                class_name: index
                for index, class_name
                in enumerate(
                    current_classes
                )
            },
            clip_len=CLIP_LEN
        )

    else:

        print()
        print(
            "[Replay] WARNING: "
            "No previous replay data found."
        )

        print(
            "[Replay] Old-class retention "
            "will rely on the teacher/EWC only."
        )

    # ========================================================
    # COMBINE
    # ========================================================

    if replay_dataset is not None:

        train_dataset = ConcatDataset(
            [
                new_train_dataset,
                replay_dataset
            ]
        )

        val_dataset = ConcatDataset(
            [
                new_val_dataset,
                replay_dataset
            ]
        )

    else:

        train_dataset = new_train_dataset

        val_dataset = new_val_dataset

    return (
        train_dataset,
        val_dataset
    )


# ============================================================
# EXPAND CLASSIFIER
# ============================================================

def expand_model_classifier(
    model,
    new_num_classes
):

    if not hasattr(
        model,
        "expand_classifier"
    ):

        raise RuntimeError(
            "\nRepViTTCN must provide "
            "expand_classifier().\n\n"
            "Add this method to "
            "models/vad_model.py."
        )

    model.expand_classifier(
        new_num_classes
    )

    return model


# ============================================================
# LOAD OLD WEIGHTS INTO NEW MODEL
# ============================================================

def load_old_weights_into_expanded_model(
    model,
    checkpoint
):

    old_state = checkpoint.get(
        "state_dict"
    )

    if old_state is None:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'state_dict'."
        )

    # --------------------------------------------------------
    # First try the model's continual loader.
    # --------------------------------------------------------

    if hasattr(
        model,
        "load_continual_checkpoint"
    ):

        try:

            model.load_continual_checkpoint(
                CHECKPOINT_PATH,
                device=device
            )

            print(
                "[Checkpoint] Loaded using "
                "load_continual_checkpoint()."
            )

            return model

        except Exception as e:

            print(
                "[Checkpoint] Continual loader "
                f"failed: {e}"
            )

    # --------------------------------------------------------
    # Fallback:
    # load every compatible parameter.
    # --------------------------------------------------------

    current_state = model.state_dict()

    compatible_state = {}

    for name, tensor in old_state.items():

        if (
            name in current_state
            and current_state[name].shape
            == tensor.shape
        ):

            compatible_state[name] = tensor

    missing, unexpected = model.load_state_dict(
        compatible_state,
        strict=False
    )

    print(
        f"[Checkpoint] Loaded "
        f"{len(compatible_state)} "
        f"compatible tensors."
    )

    if missing:

        print(
            f"[Checkpoint] Missing tensors: "
            f"{len(missing)}"
        )

    if unexpected:

        print(
            f"[Checkpoint] Unexpected tensors: "
            f"{len(unexpected)}"
        )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    current_classes,
    checkpoint
):

    model = RepViTTCN(
        num_classes=len(
            current_classes
        )
    )

    # --------------------------------------------------------
    # Existing model
    # --------------------------------------------------------

    if checkpoint is not None:

        saved_classes = normalize_classes(
            checkpoint.get(
                "classes",
                []
            )
        )

        print()
        print(
            "[Checkpoint] Previous classes:"
        )

        print(
            f"  {saved_classes}"
        )

        print(
            "[Checkpoint] Current classes:"
        )

        print(
            f"  {current_classes}"
        )

        # ----------------------------------------------------
        # Load old weights.
        # ----------------------------------------------------

        model = load_old_weights_into_expanded_model(
            model,
            checkpoint
        )

        # ----------------------------------------------------
        # Ensure classifier has current size.
        # ----------------------------------------------------

        if hasattr(
            model,
            "expand_classifier"
        ):

            try:

                model.expand_classifier(
                    len(current_classes)
                )

            except Exception:

                # It may already be expanded.
                pass

    model = model.to(
        device
    )

    return model


# ============================================================
# CREATE TEACHER
# ============================================================

def create_teacher(
    previous_classes,
    checkpoint
):

    if checkpoint is None:

        return None

    old_num_classes = len(
        previous_classes
    )

    print()
    print(
        "[KD] Creating teacher model..."
    )

    teacher = RepViTTCN(
        num_classes=old_num_classes
    )

    old_state = checkpoint.get(
        "state_dict"
    )

    if old_state is None:

        raise RuntimeError(
            "Previous checkpoint does not "
            "contain state_dict."
        )

    # --------------------------------------------------------
    # Teacher exactly matches old class count.
    # --------------------------------------------------------

    try:

        teacher.load_state_dict(
            old_state,
            strict=True
        )

    except RuntimeError as e:

        print(
            "[KD] Strict teacher loading failed."
        )

        print(
            "[KD] Trying compatible tensors..."
        )

        current_state = teacher.state_dict()

        compatible_state = {}

        for name, tensor in old_state.items():

            if (
                name in current_state
                and current_state[name].shape
                == tensor.shape
            ):

                compatible_state[name] = tensor

        teacher.load_state_dict(
            compatible_state,
            strict=False
        )

    teacher = teacher.to(
        device
    )

    teacher.eval()

    for parameter in teacher.parameters():

        parameter.requires_grad = False

    print(
        f"[KD] Teacher classes: "
        f"{previous_classes}"
    )

    return teacher


# ============================================================
# EWC PENALTY
# ============================================================

def ewc_penalty(
    model,
    fisher,
    optimal_params
):

    if not fisher:

        return torch.tensor(
            0.0,
            device=device
        )

    penalty = torch.tensor(
        0.0,
        device=device
    )

    for name, parameter in model.named_parameters():

        if (
            parameter.requires_grad
            and name in fisher
            and name in optimal_params
        ):

            fisher_tensor = fisher[
                name
            ].to(device)

            optimal_tensor = optimal_params[
                name
            ].to(device)

            # ------------------------------------------------
            # Safety check for classifier expansion.
            # ------------------------------------------------

            if (
                fisher_tensor.shape
                != parameter.shape
                or optimal_tensor.shape
                != parameter.shape
            ):

                continue

            penalty += (
                fisher_tensor
                * (
                    parameter
                    - optimal_tensor
                ).pow(2)
            ).sum()

    return (
        0.5
        * EWC_LAMBDA
        * penalty
    )


# ============================================================
# KNOWLEDGE DISTILLATION
# ============================================================

def distillation_loss(
    new_logits,
    old_logits
):

    new_log_probs = F.log_softmax(
        new_logits / KD_TEMP,
        dim=1
    )

    old_probs = F.softmax(
        old_logits / KD_TEMP,
        dim=1
    )

    return (
        F.kl_div(
            new_log_probs,
            old_probs,
            reduction="batchmean"
        )
        * KD_TEMP
        * KD_TEMP
    )


# ============================================================
# FISHER
# ============================================================

def compute_fisher(
    model,
    dataset,
    criterion,
    optimizer,
    previous_fisher
):

    print()
    print(
        "[Fisher] Computing Fisher information..."
    )

    fisher = {}

    for name, parameter in model.named_parameters():

        if (
            parameter.requires_grad
            and any(
                layer in name.lower()
                for layer in FISHER_LAYERS
            )
        ):

            fisher[name] = torch.zeros_like(
                parameter
            )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model.train()

    processed = 0

    for images, labels in loader:

        if processed >= FISHER_BATCHES:

            break

        images = images.to(
            device
        )

        labels = labels.long().to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=USE_AMP
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

        loss.backward()

        for name, parameter in model.named_parameters():

            if (
                name in fisher
                and parameter.grad is not None
            ):

                fisher[name] += (
                    parameter.grad.detach()
                    .pow(2)
                )

        processed += 1

    if processed > 0:

        for name in fisher:

            fisher[name] /= processed

            if name in previous_fisher:

                previous = previous_fisher[
                    name
                ].to(device)

                if (
                    previous.shape
                    == fisher[name].shape
                ):

                    fisher[name] = (
                        fisher[name]
                        + previous
                    ) / 2.0

    print(
        f"[Fisher] Processed "
        f"{processed} batches."
    )

    return fisher


# ============================================================
# TRAIN
# ============================================================

def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scaler,
    teacher,
    fisher,
    optimal_params,
    current_classes,
    previous_classes
):

    best_acc = -1.0

    best_state = None

    print()
    print("=" * 75)
    print("TRAINING")
    print("=" * 75)

    print(
        "Architecture : RepViT-M1.0 + TCN"
    )

    print(
        f"Classes      : {current_classes}"
    )

    print(
        f"Previous     : {previous_classes}"
    )

    print(
        f"New anomaly  : {NEW_CLASS}"
    )

    print(
        f"Epochs       : {EPOCHS}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Clip length  : {CLIP_LEN}"
    )

    print(
        "RepViT       : FROZEN"
    )

    print(
        "TCN          : TRAINABLE"
    )

    print(
        "Classifier   : TRAINABLE"
    )

    print(
        f"EWC lambda   : {EWC_LAMBDA}"
    )

    print(
        f"KD alpha     : {KD_ALPHA}"
    )

    print("=" * 75)

    for epoch in range(
        EPOCHS
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS}"
            )
        )

        for images, labels in progress:

            images = images.to(
                device
            )

            labels = labels.long().to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=USE_AMP
            ):

                outputs = model(
                    images
                )

                ce_loss = criterion(
                    outputs,
                    labels
                )

                loss = ce_loss

                # ============================================
                # EWC
                # ============================================

                if (
                    fisher
                    and optimal_params
                ):

                    loss = (
                        loss
                        + ewc_penalty(
                            model,
                            fisher,
                            optimal_params
                        )
                    )

                # ============================================
                # KNOWLEDGE DISTILLATION
                # ============================================

                if teacher is not None:

                    with torch.no_grad():

                        old_logits = teacher(
                            images
                        )

                    old_num_classes = (
                        old_logits.shape[1]
                    )

                    new_old_logits = outputs[
                        :,
                        :old_num_classes
                    ]

                    kd_loss = distillation_loss(
                        new_old_logits,
                        old_logits
                    )

                    loss = (
                        (1.0 - KD_ALPHA)
                        * loss
                        +
                        KD_ALPHA
                        * kd_loss
                    )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            loss_value = loss.item()

            running_loss += loss_value

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                acc=(
                    f"{100 * correct / max(total, 1):.2f}%"
                )
            )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        predictions_all = []

        labels_all = []

        with torch.inference_mode():

            for images, labels in val_loader:

                images = images.to(
                    device
                )

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=USE_AMP
                ):

                    outputs = model(
                        images
                    )

                predictions_all.extend(
                    outputs.argmax(
                        dim=1
                    ).cpu().tolist()
                )

                labels_all.extend(
                    labels.tolist()
                )

        if labels_all:

            val_acc = accuracy_score(
                labels_all,
                predictions_all
            )

        else:

            val_acc = 0.0

        train_acc = (
            correct
            / max(total, 1)
        )

        avg_loss = (
            running_loss
            / max(
                len(train_loader),
                1
            )
        )

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss : {avg_loss:.4f}"
        )

        print(
            f"Train Acc  : "
            f"{train_acc * 100:.2f}%"
        )

        print(
            f"Val Acc    : "
            f"{val_acc * 100:.2f}%"
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        if (
            best_state is None
            or val_acc > best_acc
        ):

            best_acc = val_acc

            best_state = deepcopy(
                model.state_dict()
            )

            print(
                f"[BEST] "
                f"{best_acc * 100:.2f}%"
            )

        # ====================================================
        # RECOVERY
        # ====================================================

        recovery = {

            "state_dict":
                model.state_dict(),

            "classes":
                list(current_classes),

            "num_classes":
                len(current_classes),

            "epoch":
                epoch + 1,

            "best_acc":
                best_acc
        }

        torch.save(
            recovery,
            RECOVERY_PATH
        )

        print(
            f"[Recovery] Saved epoch "
            f"{epoch + 1}"
        )

    return (
        best_acc,
        best_state
    )


# ============================================================
# SAVE FINAL CHECKPOINT
# ============================================================

def save_final_checkpoint(
    model,
    best_state,
    current_classes,
    best_acc,
    fisher,
    replay_paths
):

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Save EWC parameters only for trainable TCN/classifier.
    # --------------------------------------------------------

    ewc_params = {}

    for name, parameter in model.named_parameters():

        if (
            parameter.requires_grad
            and any(
                layer in name.lower()
                for layer in FISHER_LAYERS
            )
        ):

            ewc_params[name] = (
                parameter.detach().cpu()
            )

    checkpoint = {

        "state_dict":
            {
                name:
                tensor.detach().cpu()
                for name, tensor
                in best_state.items()
            },

        "classes":
            list(current_classes),

        "num_classes":
            len(current_classes),

        "best_acc":
            best_acc,

        "epoch":
            EPOCHS,

        "ewc_fisher":
            {
                name:
                tensor.detach().cpu()
                for name, tensor
                in fisher.items()
            },

        "ewc_params":
            ewc_params,

        "replay_paths":
            replay_paths
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )

    print()
    print(
        f"[Checkpoint] Saved: "
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

def save_classes_json(
    current_classes
):

    class_mapping = {
        class_name: index
        for index, class_name
        in enumerate(
            current_classes
        )
    }

    with open(
        CLASSES_JSON,
        "w"
    ) as file:

        json.dump(
            class_mapping,
            file,
            indent=2
        )

    print(
        f"[Classes] Saved: "
        f"{CLASSES_JSON}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("CONTINUAL LEARNING STAGE")
    print("=" * 75)

    print(
        f"New class: {NEW_CLASS}"
    )

    # ========================================================
    # PREVIOUS CHECKPOINT
    # ========================================================

    checkpoint, checkpoint_path = (
        get_available_checkpoint()
    )

    checkpoint_exists = (
        checkpoint is not None
    )

    # ========================================================
    # PREVIOUS CLASSES
    # ========================================================

    previous_classes = get_existing_classes()

    # ========================================================
    # FIRST / CONTINUAL STAGE
    # ========================================================

    is_first_stage = (
        not checkpoint_exists
    )

    current_classes = build_current_classes(
        previous_classes,
        checkpoint_exists
    )

    print()
    print(
        f"Previous classes: "
        f"{previous_classes}"
    )

    print(
        f"Current classes : "
        f"{current_classes}"
    )

    print(
        f"New class       : "
        f"{NEW_CLASS}"
    )

    print(
        f"Stage           : "
        f"{'INITIAL' if is_first_stage else 'CONTINUAL'}"
    )

    if checkpoint_path:

        print(
            f"Source model    : "
            f"{checkpoint_path}"
        )

    # ========================================================
    # CHECK NEW DATASET
    # ========================================================

    check_new_class_exists()

    # ========================================================
    # BUILD DATASETS
    # ========================================================

    train_dataset, val_dataset = (
        build_training_dataset(
            current_classes,
            previous_classes,
            is_first_stage
        )
    )

    print()
    print("[DATASET]")

    print(
        f"Training samples   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples : "
        f"{len(val_dataset)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_model(
        current_classes,
        checkpoint
    )

    # ========================================================
    # FREEZE REPVIT
    # ========================================================

    freeze_repvit(
        model
    )

    # ========================================================
    # TEACHER
    # ========================================================

    teacher = None

    if checkpoint is not None:

        teacher = create_teacher(
            previous_classes,
            checkpoint
        )

    # ========================================================
    # EWC
    # ========================================================

    previous_fisher = {}

    optimal_params = {}

    if checkpoint is not None:

        saved_fisher = checkpoint.get(
            "ewc_fisher",
            {}
        )

        previous_fisher = {

            name:
            tensor.to(device)

            for name, tensor
            in saved_fisher.items()
        }

        saved_params = checkpoint.get(
            "ewc_params",
            {}
        )

        optimal_params = {

            name:
            tensor.to(device)

            for name, tensor
            in saved_params.items()
        }

        print()
        print(
            f"[EWC] Loaded "
            f"{len(previous_fisher)} "
            f"Fisher tensors."
        )

        print(
            f"[EWC] Loaded "
            f"{len(optimal_params)} "
            f"optimal parameters."
        )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = make_loader(
        train_dataset,
        shuffle=True
    )

    val_loader = make_loader(
        val_dataset,
        shuffle=False
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    trainable_parameters = [

        parameter

        for parameter
        in model.parameters()

        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # AMP
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    # ========================================================
    # TRAIN
    # ========================================================

    best_acc, best_state = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        teacher=teacher,
        fisher=previous_fisher,
        optimal_params=optimal_params,
        current_classes=current_classes,
        previous_classes=previous_classes
    )

    if best_state is None:

        best_state = deepcopy(
            model.state_dict()
        )

    model.load_state_dict(
        best_state
    )

    # ========================================================
    # FISHER FOR NEW MODEL
    # ========================================================

    new_fisher = compute_fisher(
        model=model,
        dataset=train_dataset,
        criterion=criterion,
        optimizer=optimizer,
        previous_fisher=previous_fisher
    )

    # ========================================================
    # SAVE NEW CLASS TO REPLAY
    # ========================================================

    print()
    print(
        "[Replay] Updating replay buffer..."
    )

    new_class_dataset = VideoDataset(
        train=True,
        classes=[NEW_CLASS],
        clip_len=CLIP_LEN
    )

    add_class_to_replay(
        NEW_CLASS,
        new_class_dataset
    )

    # ========================================================
    # COLLECT COMPLETE REPLAY BUFFER
    # ========================================================

    replay_paths = collect_replay_paths(
        current_classes
    )

    print()
    print(
        "[Replay] Complete replay buffer:"
    )

    for class_name, paths in replay_paths.items():

        print(
            f"  {class_name}: "
            f"{len(paths)} clips"
        )

    # ========================================================
    # CONVERT PATHS
    # ========================================================

    replay_paths = (
        make_relative_replay_paths(
            replay_paths
        )
    )

    # ========================================================
    # SAVE FINAL CHECKPOINT
    # ========================================================

    save_final_checkpoint(
        model=model,
        best_state=best_state,
        current_classes=current_classes,
        best_acc=best_acc,
        fisher=new_fisher,
        replay_paths=replay_paths
    )

    # ========================================================
    # SAVE CLASSES.JSON
    # ========================================================

    save_classes_json(
        current_classes
    )

    # ========================================================
    # SAVE FISHER
    # ========================================================

    torch.save(
        {
            name:
            tensor.detach().cpu()
            for name, tensor
            in new_fisher.items()
        },
        "fisher.pkl"
    )

    print(
        "[EWC] Saved: fisher.pkl"
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 75)
    print("CONTINUAL TRAINING COMPLETE")
    print("=" * 75)

    print(
        "Architecture : RepViT-M1.0 + TCN"
    )

    print(
        f"Classes      : {current_classes}"
    )

    print(
        f"Number       : {len(current_classes)}"
    )

    print(
        f"New class    : {NEW_CLASS}"
    )

    print(
        f"Best Val Acc : "
        f"{best_acc * 100:.2f}%"
    )

    print()
    print(
        "Files generated:"
    )

    print(
        f"  {CHECKPOINT_PATH}"
    )

    print(
        f"  {CLASSES_JSON}"
    )

    print(
        f"  {RECOVERY_PATH}"
    )

    print(
        f"  {REPLAY_ROOT}/"
    )

    print(
        "  fisher.pkl"
    )

    print()
    print(
        "Next stage can add another anomaly."
    )

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()