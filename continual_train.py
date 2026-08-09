import os
import json
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, ConcatDataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from tqdm import tqdm

from video_dataset import (
    VideoDataset,
    ReplayDataset,
)

from models.vad_model import (
    RepViTTCN,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

print("=" * 60)
print("CONTINUAL LEARNING - VANDALISM")
print("=" * 60)

print(f"[DEBUG] Script location: {__file__}")
print(f"[DEBUG] Project root: {PROJECT_ROOT}")


# ============================================================
# DEVICES
# ============================================================

STUDENT_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEACHER_DEVICE = torch.device("cpu")

USE_AMP = STUDENT_DEVICE.type == "cuda"

LOW_VRAM_MODE = STUDENT_DEVICE.type == "cuda"

if STUDENT_DEVICE.type == "cuda":

    torch.backends.cudnn.benchmark = False

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:32",
    )

print(f"[Student Device] {STUDENT_DEVICE}")
print(f"[Teacher Device] {TEACHER_DEVICE}")
print(f"[AMP] {USE_AMP}")

if LOW_VRAM_MODE:
    print("[GPU] Low-VRAM mode enabled")


# ============================================================
# PATHS
# ============================================================

VANDALISM_DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "vandalism_dataset",
)

REPLAY_ROOT = os.path.join(
    PROJECT_ROOT,
    "replay_buffer",
)

REPLAY_TRAIN_ROOT = os.path.join(
    REPLAY_ROOT,
    "train",
)

REPLAY_VAL_ROOT = os.path.join(
    REPLAY_ROOT,
    "val",
)

# ------------------------------------------------------------
# IMPORTANT
#
# Original 2-class model is ONLY the source.
# It will NEVER be overwritten.
# ------------------------------------------------------------

SOURCE_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "best_model_before_vandalism.pth",
)

OUTPUT_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "best_model_continual_vandalism.pth",
)

CLASSES_JSON = os.path.join(
    PROJECT_ROOT,
    "classes.json",
)


# ============================================================
# CLASSES
# ============================================================

OLD_CLASSES = [
    "normal",
    "fighting",
]

CLASSES = [
    "normal",
    "fighting",
    "vandalism",
]

CLASS_MAPPING = {
    "normal": 0,
    "fighting": 1,
    "vandalism": 2,
}

OLD_CLASS_COUNT = 2
NEW_CLASS_COUNT = 3


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 15

BATCH_SIZE = 1
VAL_BATCH_SIZE = 1

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5

# ------------------------------------------------------------
# Knowledge Distillation
# ------------------------------------------------------------

KD_ALPHA = 0.20
KD_TEMP = 4.0

# ------------------------------------------------------------
# EWC
# ------------------------------------------------------------

EWC_LAMBDA = 10.0

FISHER_BATCHES = 20
FISHER_BATCH_SIZE = 1

# ------------------------------------------------------------
# Video
# ------------------------------------------------------------

CLIP_LEN = 16

# ------------------------------------------------------------
# Gradient clipping
# ------------------------------------------------------------

MAX_GRAD_NORM = 1.0

# ------------------------------------------------------------
# Replay
# ------------------------------------------------------------

REPLAY_SAMPLES_PER_CLASS = 30


# ============================================================
# MEMORY HELPERS
# ============================================================

def clear_cuda():

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_cuda_memory(prefix=""):

    if not torch.cuda.is_available():
        return

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 2)
    )

    print(
        f"{prefix}[CUDA] "
        f"allocated={allocated:.1f} MB | "
        f"reserved={reserved:.1f} MB"
    )


# ============================================================
# DEVICE HELPERS
# ============================================================

def force_model_device(model, device):

    return model.to(device)


def get_model_devices(model):

    devices = set()

    for parameter in model.parameters():
        devices.add(str(parameter.device))

    for buffer in model.buffers():
        devices.add(str(buffer.device))

    return devices


def verify_model_device(
    model,
    expected_device,
    name="model",
):

    devices = get_model_devices(model)

    print(
        f"[Device Check] "
        f"{name}: {devices}"
    )

    if len(devices) != 1:

        raise RuntimeError(
            f"{name} has parameters "
            f"on multiple devices: "
            f"{devices}"
        )

    actual = torch.device(
        next(iter(devices))
    )

    expected = torch.device(
        expected_device
    )

    if actual.type != expected.type:

        raise RuntimeError(
            f"{name} is on {actual}, "
            f"expected {expected}"
        )

    if actual.type == "cuda":

        actual_index = (
            actual.index
            if actual.index is not None
            else torch.cuda.current_device()
        )

        expected_index = (
            expected.index
            if expected.index is not None
            else torch.cuda.current_device()
        )

        if actual_index != expected_index:

            raise RuntimeError(
                f"{name} is on {actual}, "
                f"expected {expected}"
            )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def load_checkpoint_safe(path):

    try:

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    return checkpoint


def get_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

    return checkpoint


def get_checkpoint_num_classes(checkpoint):

    if isinstance(checkpoint, dict):

        if "num_classes" in checkpoint:

            return int(
                checkpoint["num_classes"]
            )

        classes = checkpoint.get("classes")

        if isinstance(
            classes,
            (list, tuple),
        ):

            return len(classes)

        class_mapping = checkpoint.get(
            "class_mapping"
        )

        if isinstance(
            class_mapping,
            dict,
        ):

            return len(class_mapping)

    state_dict = get_state_dict(
        checkpoint
    )

    if isinstance(state_dict, dict):

        for key, tensor in state_dict.items():

            if (
                key.endswith(
                    "classifier.3.weight"
                )
                and tensor.ndim == 2
            ):

                return tensor.shape[0]

            if (
                key.endswith(
                    "classifier.3.bias"
                )
                and tensor.ndim == 1
            ):

                return tensor.shape[0]

    return None


# ============================================================
# SAFE 2-CLASS -> 3-CLASS MODEL LOADING
# ============================================================

def load_old_weights_into_new_student(
    student,
    checkpoint,
):

    print()
    print("[Model Transfer]")
    print(
        "Loading old 2-class weights "
        "into new 3-class model..."
    )

    old_state = get_state_dict(
        checkpoint
    )

    if not isinstance(
        old_state,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint does not contain "
            "a valid state_dict."
        )

    new_state = student.state_dict()

    copied = 0
    partially_copied = 0
    skipped = 0

    for name, new_tensor in new_state.items():

        if name not in old_state:

            skipped += 1
            continue

        old_tensor = old_state[name]

        # ----------------------------------------------------
        # Exact shape match
        # ----------------------------------------------------

        if old_tensor.shape == new_tensor.shape:

            new_state[name] = (
                old_tensor.detach()
                .clone()
                .to(new_tensor.dtype)
            )

            copied += 1
            continue

        # ----------------------------------------------------
        # Classifier expansion
        # ----------------------------------------------------

        if "classifier" in name:

            if (
                old_tensor.ndim
                == new_tensor.ndim
                and old_tensor.ndim >= 1
            ):

                if (
                    old_tensor.shape[0]
                    == OLD_CLASS_COUNT
                    and new_tensor.shape[0]
                    == NEW_CLASS_COUNT
                    and old_tensor.shape[1:]
                    == new_tensor.shape[1:]
                ):

                    expanded_tensor = (
                        new_tensor.clone()
                    )

                    expanded_tensor[
                        :OLD_CLASS_COUNT
                    ] = old_tensor

                    new_state[name] = (
                        expanded_tensor
                    )

                    partially_copied += 1
                    continue

        skipped += 1

    student.load_state_dict(
        new_state,
        strict=True,
    )

    print(
        f"Exact tensors copied        : "
        f"{copied}"
    )

    print(
        f"Classifier tensors expanded : "
        f"{partially_copied}"
    )

    print(
        f"Skipped tensors             : "
        f"{skipped}"
    )

    print()
    print(
        "[Class Transfer]"
    )

    print(
        "Old class 0 -> normal"
    )

    print(
        "Old class 1 -> fighting"
    )

    print(
        "New class 2 -> vandalism "
        "(newly initialized)"
    )

    return student


# ============================================================
# REPLAY LIMIT
# ============================================================

def limit_replay_samples(
    dataset,
    samples_per_class=30,
):

    selected = []

    for class_name in OLD_CLASSES:

        class_label = CLASS_MAPPING[
            class_name
        ]

        class_samples = [
            sample
            for sample in dataset.samples
            if sample[1] == class_label
        ]

        if len(class_samples) < samples_per_class:

            raise RuntimeError(
                f"Not enough replay samples "
                f"for {class_name}. "
                f"Found {len(class_samples)}, "
                f"required {samples_per_class}."
            )

        class_samples = sorted(
            class_samples,
            key=lambda x: x[0],
        )

        selected.extend(
            class_samples[:samples_per_class]
        )

    dataset.samples = selected

    normal_count = sum(
        1
        for _, label in dataset.samples
        if label == CLASS_MAPPING["normal"]
    )

    fighting_count = sum(
        1
        for _, label in dataset.samples
        if label == CLASS_MAPPING["fighting"]
    )

    print()
    print("[Replay] Limited replay dataset")
    print(f"Normal   : {normal_count}")
    print(f"Fighting : {fighting_count}")
    print(f"Total    : {len(dataset.samples)}")

    return dataset


# ============================================================
# KNOWLEDGE DISTILLATION
# ============================================================

def distillation_loss(
    student_logits,
    teacher_logits,
    temperature=KD_TEMP,
):

    student_log_probs = F.log_softmax(
        student_logits / temperature,
        dim=1,
    )

    teacher_probs = F.softmax(
        teacher_logits / temperature,
        dim=1,
    )

    loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    )

    return loss * (
        temperature ** 2
    )


# ============================================================
# EWC
# ============================================================

def get_shared_parameters(model):

    params = {}

    for name, parameter in model.named_parameters():

        if "classifier" not in name:

            params[name] = parameter

    return params


def ewc_penalty(
    model,
    fisher,
    optimal_params,
):

    if not fisher:

        return torch.zeros(
            (),
            device=STUDENT_DEVICE,
        )

    penalty = torch.zeros(
        (),
        device=STUDENT_DEVICE,
    )

    for name, parameter in model.named_parameters():

        if name not in fisher:
            continue

        if name not in optimal_params:
            continue

        fisher_tensor = fisher[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )

        optimal_tensor = optimal_params[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )

        if parameter.shape != fisher_tensor.shape:
            continue

        if parameter.shape != optimal_tensor.shape:
            continue

        penalty += (
            fisher_tensor
            * (
                parameter
                - optimal_tensor
            ).pow(2)
        ).sum()

    return (
        EWC_LAMBDA
        / 2.0
        * penalty
    )


# ============================================================
# FISHER INFORMATION
# ============================================================

def compute_fisher(
    teacher,
    replay_dataset,
):

    print()
    print("[EWC] Computing Fisher information...")

    sample_limit = min(
        len(replay_dataset),
        FISHER_BATCHES,
    )

    print(
        f"[EWC] Samples: {sample_limit}"
    )

    teacher = force_model_device(
        teacher,
        TEACHER_DEVICE,
    )

    teacher.eval()

    # --------------------------------------------------------
    # Freeze everything first
    # --------------------------------------------------------

    for parameter in teacher.parameters():

        parameter.requires_grad = False

    # --------------------------------------------------------
    # Enable gradients for shared backbone
    # --------------------------------------------------------

    shared_parameters = (
        get_shared_parameters(
            teacher
        )
    )

    if not shared_parameters:

        raise RuntimeError(
            "No shared backbone parameters "
            "were found for EWC."
        )

    for parameter in shared_parameters.values():

        parameter.requires_grad = True

    # --------------------------------------------------------
    # Initialize Fisher
    # --------------------------------------------------------

    fisher = {}
    optimal_params = {}

    for name, parameter in shared_parameters.items():

        fisher[name] = torch.zeros_like(
            parameter,
            device="cpu",
        )

        optimal_params[name] = (
            parameter.detach()
            .clone()
            .cpu()
        )

    loader = DataLoader(
        replay_dataset,
        batch_size=FISHER_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    processed = 0

    # --------------------------------------------------------
    # Fisher calculation
    # --------------------------------------------------------

    for images, labels in loader:

        if processed >= sample_limit:
            break

        images = images.to(
            TEACHER_DEVICE,
            dtype=torch.float32,
        )

        labels = labels.to(
            TEACHER_DEVICE,
            dtype=torch.long,
        )

        teacher.zero_grad(
            set_to_none=True
        )

        with torch.enable_grad():

            outputs = teacher(images)

            outputs = outputs[
                :, :OLD_CLASS_COUNT
            ]

            loss = F.cross_entropy(
                outputs,
                labels,
            )

            loss.backward()

        for (
            name,
            parameter,
        ) in shared_parameters.items():

            if parameter.grad is None:
                continue

            fisher[name] += (
                parameter.grad.detach()
                .pow(2)
                .cpu()
            )

        processed += 1

        print(
            f"\r[EWC] Fisher "
            f"{processed}/{sample_limit}",
            end="",
        )

        teacher.zero_grad(
            set_to_none=True
        )

        del images
        del labels
        del outputs
        del loss

    print()

    if processed == 0:

        raise RuntimeError(
            "Fisher processed zero samples."
        )

    # --------------------------------------------------------
    # Average Fisher
    # --------------------------------------------------------

    for name in fisher:

        fisher[name] /= float(
            processed
        )

        fisher[name] = (
            fisher[name]
            .detach()
            .cpu()
        )

    # --------------------------------------------------------
    # Freeze teacher again
    # --------------------------------------------------------

    for parameter in teacher.parameters():

        parameter.requires_grad = False

    print(
        "[EWC] Fisher completed."
    )

    print(
        f"[EWC] Fisher tensors: "
        f"{len(fisher)}"
    )

    return (
        fisher,
        optimal_params,
    )


# ============================================================
# DATASETS
# ============================================================

def create_datasets():

    print()
    print("=" * 60)
    print("[DATASET SETUP]")
    print("=" * 60)

    # ========================================================
    # VANDALISM TRAIN
    # ========================================================

    vandalism_train = VideoDataset(
        root=VANDALISM_DATASET_ROOT,
        train=True,
        classes=["vandalism"],
        class_to_idx=CLASS_MAPPING,
        clip_len=CLIP_LEN,
    )

    # ========================================================
    # VANDALISM VALIDATION
    # ========================================================

    vandalism_val = VideoDataset(
        root=VANDALISM_DATASET_ROOT,
        train=False,
        classes=["vandalism"],
        class_to_idx=CLASS_MAPPING,
        clip_len=CLIP_LEN,
    )

    print(
        f"[Vandalism] Train videos: "
        f"{len(vandalism_train)}"
    )

    print(
        f"[Vandalism] Val videos: "
        f"{len(vandalism_val)}"
    )

    if len(vandalism_train) == 0:

        raise RuntimeError(
            "Vandalism training dataset is empty."
        )

    if len(vandalism_val) == 0:

        raise RuntimeError(
            "Vandalism validation dataset is empty."
        )

    # ========================================================
    # REPLAY TRAIN
    # ========================================================

    replay_train = ReplayDataset(
        root=REPLAY_TRAIN_ROOT,
        class_to_idx=CLASS_MAPPING,
        classes=[
            "normal",
            "fighting",
        ],
        clip_len=CLIP_LEN,
        random_clip=True,
    )

    if len(replay_train) == 0:

        raise RuntimeError(
            "Replay training dataset is empty."
        )

    replay_train = limit_replay_samples(
        replay_train,
        samples_per_class=REPLAY_SAMPLES_PER_CLASS,
    )

    # ========================================================
    # REPLAY VALIDATION
    # ========================================================

    replay_val = ReplayDataset(
        root=REPLAY_VAL_ROOT,
        class_to_idx=CLASS_MAPPING,
        classes=[
            "normal",
            "fighting",
        ],
        clip_len=CLIP_LEN,
        random_clip=False,
    )

    if len(replay_val) == 0:

        raise RuntimeError(
            "Replay validation dataset is empty."
        )

    # ========================================================
    # COUNTS
    # ========================================================

    normal_train = sum(
        1
        for _, label in replay_train.samples
        if label == CLASS_MAPPING["normal"]
    )

    fighting_train = sum(
        1
        for _, label in replay_train.samples
        if label == CLASS_MAPPING["fighting"]
    )

    print()
    print("[Replay Train Count]")
    print(f"Normal   : {normal_train}")
    print(f"Fighting : {fighting_train}")

    # ========================================================
    # COMBINE
    # ========================================================

    train_dataset = ConcatDataset(
        [
            replay_train,
            vandalism_train,
        ]
    )

    val_dataset = ConcatDataset(
        [
            replay_val,
            vandalism_val,
        ]
    )

    print()
    print(
        f"[Dataset] Combined train: "
        f"{len(train_dataset)}"
    )

    print(
        f"[Dataset] Combined val: "
        f"{len(val_dataset)}"
    )

    print()
    print("[Dataset] Labels:")
    print("0 = normal")
    print("1 = fighting")
    print("2 = vandalism")

    return (
        vandalism_train,
        vandalism_val,
        replay_train,
        replay_val,
        train_dataset,
        val_dataset,
    )


# ============================================================
# TEACHER
# ============================================================

def create_teacher(checkpoint):

    print()
    print("[KD] Creating old 2-class teacher...")

    teacher = RepViTTCN(
        num_classes=OLD_CLASS_COUNT
    )

    state_dict = get_state_dict(
        checkpoint
    )

    teacher.load_state_dict(
        state_dict,
        strict=True,
    )

    teacher = force_model_device(
        teacher,
        TEACHER_DEVICE,
    )

    teacher.eval()

    for parameter in teacher.parameters():

        parameter.requires_grad = False

    print(
        "[KD] Teacher output classes: "
        f"{OLD_CLASS_COUNT}"
    )

    verify_model_device(
        teacher,
        TEACHER_DEVICE,
        "Teacher",
    )

    return teacher


# ============================================================
# STUDENT
# ============================================================

def create_student(checkpoint):

    print()
    print("[Model] Creating 3-class student...")

    student = RepViTTCN(
        num_classes=NEW_CLASS_COUNT
    )

    student = force_model_device(
        student,
        STUDENT_DEVICE,
    )

    student = load_old_weights_into_new_student(
        student,
        checkpoint,
    )

    student = force_model_device(
        student,
        STUDENT_DEVICE,
    )

    print()
    print("[Model] Previous classes:")
    print("    normal   = 0")
    print("    fighting = 1")

    print()
    print("[Model] New class:")
    print("    vandalism = 2")

    print()
    print("[Model] Classifier:")
    print(student.classifier)

    verify_model_device(
        student,
        STUDENT_DEVICE,
        "Student",
    )

    return student


# ============================================================
# LOADERS
# ============================================================

def create_loaders(
    train_dataset,
    val_dataset,
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return (
        train_loader,
        val_loader,
    )


# ============================================================
# SETUP
# ============================================================

def setup_training():

    # --------------------------------------------------------
    # Save class mapping
    # --------------------------------------------------------

    with open(
        CLASSES_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            CLASS_MAPPING,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        vandalism_train,
        vandalism_val,
        replay_train,
        replay_val,
        train_dataset,
        val_dataset,
    ) = create_datasets()

    # --------------------------------------------------------
    # SOURCE CHECKPOINT
    # --------------------------------------------------------

    if not os.path.isfile(
        SOURCE_CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            f"Source checkpoint not found:\n"
            f"{SOURCE_CHECKPOINT_PATH}"
        )

    print()
    print("[Checkpoint] Loading SOURCE:")
    print(SOURCE_CHECKPOINT_PATH)

    checkpoint = load_checkpoint_safe(
        SOURCE_CHECKPOINT_PATH
    )

    # --------------------------------------------------------
    # Verify old checkpoint
    # --------------------------------------------------------

    checkpoint_num_classes = (
        get_checkpoint_num_classes(
            checkpoint
        )
    )

    print(
        f"[Checkpoint] Detected classes: "
        f"{checkpoint_num_classes}"
    )

    if checkpoint_num_classes != OLD_CLASS_COUNT:

        raise RuntimeError(
            "\nThe SOURCE checkpoint is not the "
            "original 2-class model.\n\n"
            f"Detected classes: "
            f"{checkpoint_num_classes}\n"
            f"Expected: "
            f"{OLD_CLASS_COUNT}\n\n"
            "Use best_model_before_vandalism.pth "
            "containing normal + fighting."
        )

    # --------------------------------------------------------
    # Teacher
    # --------------------------------------------------------

    teacher = create_teacher(
        checkpoint
    )

    # --------------------------------------------------------
    # Student
    # --------------------------------------------------------

    clear_cuda()

    student = create_student(
        checkpoint
    )

    # --------------------------------------------------------
    # Device verification
    # --------------------------------------------------------

    print()
    print("[Device Check] Final placement")

    verify_model_device(
        student,
        STUDENT_DEVICE,
        "Student",
    )

    verify_model_device(
        teacher,
        TEACHER_DEVICE,
        "Teacher",
    )

    # --------------------------------------------------------
    # EWC
    # --------------------------------------------------------

    saved_fisher = {}
    saved_params = {}

    if isinstance(
        checkpoint,
        dict,
    ):

        saved_fisher = checkpoint.get(
            "ewc_fisher",
            {},
        )

        saved_params = checkpoint.get(
            "ewc_params",
            {},
        )

    if saved_fisher and saved_params:

        print(
            "[EWC] Saved Fisher found."
        )

        fisher = {
            name: tensor.detach().cpu()
            for name, tensor
            in saved_fisher.items()
        }

        optimal_params = {
            name: tensor.detach().cpu()
            for name, tensor
            in saved_params.items()
        }

    else:

        print(
            "[EWC] Computing Fisher "
            "from replay data..."
        )

        (
            fisher,
            optimal_params,
        ) = compute_fisher(
            teacher,
            replay_train,
        )

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
    ) = create_loaders(
        train_dataset,
        val_dataset,
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = torch.tensor(
        [
            1.5,  # normal
            1.5,  # fighting
            1.0,  # vandalism
        ],
        dtype=torch.float32,
        device=STUDENT_DEVICE,
    )

    print(
        "[Loss] Class weights: "
        f"{class_weights.tolist()}"
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    return {
        "model": student,
        "old_model": teacher,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "scaler": scaler,
        "ewc_fisher": fisher,
        "ewc_optimal_params": optimal_params,
    }


# ============================================================
# TRAINING
# ============================================================

def run_training(state):

    model = state["model"]

    teacher = state["old_model"]

    train_loader = state["train_loader"]

    val_loader = state["val_loader"]

    criterion = state["criterion"]

    optimizer = state["optimizer"]

    scaler = state["scaler"]

    ewc_fisher = state["ewc_fisher"]

    ewc_optimal_params = state[
        "ewc_optimal_params"
    ]

    model = force_model_device(
        model,
        STUDENT_DEVICE,
    )

    teacher = force_model_device(
        teacher,
        TEACHER_DEVICE,
    )

    teacher.eval()

    for parameter in teacher.parameters():

        parameter.requires_grad = False

    best_acc = -1.0

    best_state_dict = None

    # ========================================================
    # EPOCHS
    # ========================================================

    for epoch in range(EPOCHS):

        print()
        print("=" * 60)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print("=" * 60)

        model.train()

        total_loss = 0.0

        total_ce = 0.0

        total_kd = 0.0

        total_ewc = 0.0

        total_samples = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS}"
            ),
        )

        # ====================================================
        # TRAIN
        # ====================================================

        for images, labels in progress:

            # ------------------------------------------------
            # Student data
            # ------------------------------------------------

            student_images = images.to(
                STUDENT_DEVICE,
                dtype=torch.float32,
            )

            student_labels = labels.to(
                STUDENT_DEVICE,
                dtype=torch.long,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Student forward
            # ------------------------------------------------

            with torch.amp.autocast(
                device_type=STUDENT_DEVICE.type,
                enabled=USE_AMP,
            ):

                student_outputs = model(
                    student_images
                )

                ce_loss = criterion(
                    student_outputs,
                    student_labels,
                )

            # ------------------------------------------------
            # EWC
            # ------------------------------------------------

            ewc_loss = torch.zeros(
                (),
                device=STUDENT_DEVICE,
            )

            if ewc_fisher:

                ewc_loss = ewc_penalty(
                    model,
                    ewc_fisher,
                    ewc_optimal_params,
                )

                if not torch.isfinite(
                    ewc_loss
                ):

                    ewc_loss = torch.zeros(
                        (),
                        device=STUDENT_DEVICE,
                    )

            # ------------------------------------------------
            # Knowledge Distillation
            # ------------------------------------------------

            kd_loss = torch.zeros(
                (),
                device=STUDENT_DEVICE,
            )

            old_class_mask = (
                student_labels < OLD_CLASS_COUNT
            )

            if (
                KD_ALPHA > 0
                and old_class_mask.any()
            ):

                # --------------------------------------------
                # Student old-class logits
                # --------------------------------------------

                old_student_outputs = (
                    student_outputs[
                        old_class_mask,
                        :OLD_CLASS_COUNT
                    ]
                )

                # --------------------------------------------
                # CPU mask
                # --------------------------------------------

                old_class_mask_cpu = (
                    old_class_mask.cpu()
                )

                # --------------------------------------------
                # Get old-class images
                # --------------------------------------------

                old_images = images[
                    old_class_mask_cpu
                ]

                # --------------------------------------------
                # Send to CPU teacher
                # --------------------------------------------

                teacher_images = old_images.to(
                    TEACHER_DEVICE,
                    dtype=torch.float32,
                )

                # --------------------------------------------
                # Teacher forward
                # --------------------------------------------

                with torch.no_grad():

                    teacher_outputs = teacher(
                        teacher_images
                    )

                # --------------------------------------------
                # Teacher logits to student device
                # --------------------------------------------

                teacher_outputs = (
                    teacher_outputs.to(
                        STUDENT_DEVICE
                    )
                )

                # --------------------------------------------
                # KD
                # --------------------------------------------

                kd_loss = distillation_loss(
                    old_student_outputs,
                    teacher_outputs[
                        :, :OLD_CLASS_COUNT
                    ],
                )

                # --------------------------------------------
                # Cleanup
                # --------------------------------------------

                del old_student_outputs
                del old_class_mask_cpu
                del old_images
                del teacher_images
                del teacher_outputs

            # ------------------------------------------------
            # FINAL LOSS
            #
            # Correct formulation:
            #
            # Loss =
            #   (1-alpha) * CE
            #   + alpha * KD
            #   + EWC
            #
            # EWC is NOT multiplied by KD_ALPHA.
            # ------------------------------------------------

            loss = (
                (1.0 - KD_ALPHA)
                * ce_loss
                + KD_ALPHA
                * kd_loss
                + ewc_loss
            )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            batch_size = (
                student_labels.size(0)
            )

            loss_value = (
                loss.detach().item()
            )

            ce_value = (
                ce_loss.detach().item()
            )

            kd_value = (
                kd_loss.detach().item()
            )

            ewc_value = (
                ewc_loss.detach().item()
            )

            total_loss += (
                loss_value
                * batch_size
            )

            total_ce += (
                ce_value
                * batch_size
            )

            total_kd += (
                kd_value
                * batch_size
            )

            total_ewc += (
                ewc_value
                * batch_size
            )

            total_samples += batch_size

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                ce=f"{ce_value:.4f}",
                kd=f"{kd_value:.4f}",
                ewc=f"{ewc_value:.4f}",
            )

            # ------------------------------------------------
            # Cleanup
            # ------------------------------------------------

            del images
            del labels

            del student_images
            del student_labels

            del student_outputs
            del loss
            del ce_loss
            del kd_loss
            del ewc_loss

            clear_cuda()

        # ====================================================
        # TRAIN LOSS
        # ====================================================

        denominator = max(
            total_samples,
            1,
        )

        average_loss = (
            total_loss
            / denominator
        )

        average_ce = (
            total_ce
            / denominator
        )

        average_kd = (
            total_kd
            / denominator
        )

        average_ewc = (
            total_ewc
            / denominator
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        predictions = []

        ground_truth = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    STUDENT_DEVICE,
                    dtype=torch.float32,
                )

                with torch.amp.autocast(
                    device_type=STUDENT_DEVICE.type,
                    enabled=USE_AMP,
                ):

                    outputs = model(
                        images
                    )

                preds = (
                    outputs
                    .argmax(dim=1)
                    .cpu()
                    .tolist()
                )

                targets = (
                    labels
                    .cpu()
                    .tolist()
                )

                predictions.extend(
                    preds
                )

                ground_truth.extend(
                    targets
                )

                del images
                del outputs

                clear_cuda()

        # ====================================================
        # ACCURACY
        # ====================================================

        accuracy = accuracy_score(
            ground_truth,
            predictions,
        )

        print()

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {average_loss:.4f} | "
            f"CE: {average_ce:.4f} | "
            f"KD: {average_kd:.4f} | "
            f"EWC: {average_ewc:.4f} | "
            f"Val Acc: {accuracy:.4f}"
        )

        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        print()

        print(
            "[Validation Classification Report]"
        )

        print(
            classification_report(
                ground_truth,
                predictions,
                labels=[
                    0,
                    1,
                    2,
                ],
                target_names=[
                    "normal",
                    "fighting",
                    "vandalism",
                ],
                zero_division=0,
            )
        )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(
            ground_truth,
            predictions,
            labels=[
                0,
                1,
                2,
            ],
        )

        print(
            "[Confusion Matrix]"
        )

        print(
            "             Predicted"
        )

        print(
            "             N    F    V"
        )

        print(
            f"Actual N   "
            f"{cm[0, 0]:3d} "
            f"{cm[0, 1]:3d} "
            f"{cm[0, 2]:3d}"
        )

        print(
            f"Actual F   "
            f"{cm[1, 0]:3d} "
            f"{cm[1, 1]:3d} "
            f"{cm[1, 2]:3d}"
        )

        print(
            f"Actual V   "
            f"{cm[2, 0]:3d} "
            f"{cm[2, 1]:3d} "
            f"{cm[2, 2]:3d}"
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        if accuracy > best_acc:

            best_acc = accuracy

            best_state_dict = deepcopy(
                model.state_dict()
            )

            print()

            print(
                f"[BEST] New best accuracy: "
                f"{best_acc:.4f}"
            )

    return (
        best_acc,
        best_state_dict,
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    state,
    best_acc,
    best_state_dict,
):

    if best_state_dict is None:

        best_state_dict = deepcopy(
            state["model"].state_dict()
        )

    # --------------------------------------------------------
    # Model weights -> CPU
    # --------------------------------------------------------

    cpu_state_dict = {}

    for name, tensor in best_state_dict.items():

        cpu_state_dict[name] = (
            tensor.detach()
            .cpu()
        )

    # --------------------------------------------------------
    # Fisher -> CPU
    # --------------------------------------------------------

    cpu_fisher = {}

    for name, tensor in state[
        "ewc_fisher"
    ].items():

        cpu_fisher[name] = (
            tensor.detach()
            .cpu()
        )

    # --------------------------------------------------------
    # Optimal parameters -> CPU
    # --------------------------------------------------------

    cpu_optimal = {}

    for name, tensor in state[
        "ewc_optimal_params"
    ].items():

        cpu_optimal[name] = (
            tensor.detach()
            .cpu()
        )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = {

        "state_dict":
            cpu_state_dict,

        "classes":
            CLASSES,

        "class_mapping":
            CLASS_MAPPING,

        "num_classes":
            NEW_CLASS_COUNT,

        "best_acc":
            best_acc,

        "ewc_fisher":
            cpu_fisher,

        "ewc_params":
            cpu_optimal,
    }

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Save to NEW output file.
    # Do NOT overwrite source checkpoint.
    # --------------------------------------------------------

    torch.save(
        checkpoint,
        OUTPUT_CHECKPOINT_PATH,
    )

    print()
    print("=" * 60)
    print("[CHECKPOINT]")
    print("=" * 60)

    print(
        f"Saved: {OUTPUT_CHECKPOINT_PATH}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_acc:.4f}"
    )

    print(
        f"Classes: {CLASS_MAPPING}"
    )

    print(
        f"EWC tensors: "
        f"{len(cpu_fisher)}"
    )


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_paths():

    print()
    print("=" * 60)
    print("[PATH CHECK]")
    print("=" * 60)

    print(
        f"Vandalism dataset:\n"
        f"{VANDALISM_DATASET_ROOT}"
    )

    print(
        f"Replay train:\n"
        f"{REPLAY_TRAIN_ROOT}"
    )

    print(
        f"Replay validation:\n"
        f"{REPLAY_VAL_ROOT}"
    )

    print(
        f"Source checkpoint:\n"
        f"{SOURCE_CHECKPOINT_PATH}"
    )

    print(
        f"Output checkpoint:\n"
        f"{OUTPUT_CHECKPOINT_PATH}"
    )

    required_dirs = [
        VANDALISM_DATASET_ROOT,
        REPLAY_TRAIN_ROOT,
        REPLAY_VAL_ROOT,
    ]

    missing = []

    for path in required_dirs:

        if not os.path.isdir(path):

            missing.append(path)

    # --------------------------------------------------------
    # ONLY source checkpoint must already exist.
    # Output checkpoint will be created later.
    # --------------------------------------------------------

    if not os.path.isfile(
        SOURCE_CHECKPOINT_PATH
    ):

        missing.append(
            SOURCE_CHECKPOINT_PATH
        )

    if missing:

        print()
        print("[FAIL] Missing:")

        for path in missing:

            print(f"  - {path}")

        raise FileNotFoundError(
            "Required paths are missing."
        )

    print()
    print(
        "[PASS] Required paths exist."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # Validate paths
        # ----------------------------------------------------

        validate_paths()

        # ----------------------------------------------------
        # Setup
        # ----------------------------------------------------

        state = setup_training()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        (
            best_acc,
            best_state_dict,
        ) = run_training(
            state
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_checkpoint(
            state,
            best_acc,
            best_state_dict,
        )

        # ----------------------------------------------------
        # Final
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print(
            "CONTINUAL LEARNING COMPLETED"
        )
        print("=" * 60)

        print(
            f"Model: "
            f"{OUTPUT_CHECKPOINT_PATH}"
        )

        print(
            f"Best accuracy: "
            f"{best_acc:.4f}"
        )

        print()
        print("Final classes:")

        print("0 -> normal")
        print("1 -> fighting")
        print("2 -> vandalism")

        print()
        print(
            "[Original 2-class model]"
        )

        print(
            SOURCE_CHECKPOINT_PATH
        )

        print()
        print(
            "[New 3-class model]"
        )

        print(
            OUTPUT_CHECKPOINT_PATH
        )

    except torch.cuda.OutOfMemoryError:

        print()
        print("=" * 60)
        print("[ERROR] CUDA OUT OF MEMORY")
        print("=" * 60)

        print(
            "Student = GPU"
        )

        print(
            "Teacher = CPU"
        )

        print(
            f"Batch size = {BATCH_SIZE}"
        )

        print(
            f"Clip length = {CLIP_LEN}"
        )

        clear_cuda()

        raise

    except Exception as error:

        print()
        print("=" * 60)
        print("[ERROR]")
        print("=" * 60)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()