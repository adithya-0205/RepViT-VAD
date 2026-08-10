import os
import json
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

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
print("CONTINUAL LEARNING - ARSON  (4 -> 5 classes)")
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

ARSON_DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
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
# Original 4-class model is ONLY the source.
# It will NEVER be overwritten.
# ------------------------------------------------------------

SOURCE_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "best_model.pth",
)

OUTPUT_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "best_model_arson.pth",
)

CLASSES_JSON = os.path.join(
    PROJECT_ROOT,
    "classes.json",
)


# ============================================================
# CLASSES
#
# CHANGE vs continual_train_assault.py:
#   OLD_CLASSES now contains all 4 old classes (not 3).
#   CLASSES now contains all 5 classes.
#   OLD_CLASS_COUNT = 4  (was 3)
#   NEW_CLASS_COUNT = 5  (was 4)
# ============================================================

OLD_CLASSES = [
    "normal",
    "fighting",
    "vandalism",
    "assault",
]

CLASSES = [
    "normal",
    "fighting",
    "vandalism",
    "assault",
    "arson",
]

CLASS_MAPPING = {
    "normal":    0,
    "fighting":  1,
    "vandalism": 2,
    "assault":   3,
    "arson":     4,
}

OLD_CLASS_COUNT = 4   # was 3 in assault stage
NEW_CLASS_COUNT = 5   # was 4 in assault stage


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 10

BATCH_SIZE = 1
VAL_BATCH_SIZE = 1

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5

# ------------------------------------------------------------
# Knowledge Distillation
# (same values as assault stage)
# ------------------------------------------------------------

KD_ALPHA = 0.20
KD_TEMP = 4.0

# ------------------------------------------------------------
# EWC
# (same values as assault stage)
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
# SAFE 4-CLASS -> 5-CLASS MODEL LOADING
# ============================================================

def load_old_weights_into_new_student(
    student,
    checkpoint,
):

    print()
    print("[Model Transfer]")
    print(
        "Loading old 4-class weights "
        "into new 5-class model..."
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
        # Classifier expansion: 4-class -> 5-class
        # Old rows 0..3 are preserved exactly.
        # Only row 4 (arson) is newly initialized.
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

    print("Old class 0 -> normal")
    print("Old class 1 -> fighting")
    print("Old class 2 -> vandalism")
    print("Old class 3 -> assault")
    print("New class 4 -> arson (newly initialized)")

    # --------------------------------------------------------
    # Verify old rows are identical
    # --------------------------------------------------------

    old_w = old_state["classifier.3.weight"]
    new_w = student.classifier[3].weight.detach()
    old_b = old_state["classifier.3.bias"]
    new_b = student.classifier[3].bias.detach()

    w_match = torch.equal(new_w[:OLD_CLASS_COUNT], old_w)
    b_match = torch.equal(new_b[:OLD_CLASS_COUNT], old_b)

    print()
    print(
        f"[Verify] Weight rows [:4] preserved exactly: {w_match}"
    )
    print(
        f"[Verify] Bias   rows [:4] preserved exactly: {b_match}"
    )

    if not w_match or not b_match:
        raise RuntimeError(
            "Old classifier weights were NOT preserved correctly!"
        )

    return student


# ============================================================
# FISHER INFORMATION  (fresh computation fallback)
# ============================================================

def compute_fisher_fresh(
    teacher,
    arson_train_dataset,
):
    """
    Compute Fisher Information on the *teacher* model's parameters
    that correspond to the current RepViTTCN naming scheme, using
    the Arson training data as a proxy.

    This is triggered when the saved EWC Fisher in best_model.pth
    has ZERO overlap with the student's parameter names (i.e. it
    was computed on an older model architecture).
    """

    print()
    print("[EWC] Computing FRESH Fisher on Arson data (saved Fisher unusable)...")

    sample_limit = min(
        len(arson_train_dataset),
        FISHER_BATCHES,
    )

    print(f"[EWC] Samples used: {sample_limit}")

    teacher = force_model_device(teacher, TEACHER_DEVICE)
    teacher.eval()

    # Collect shared (non-classifier) parameters on the teacher
    shared = get_shared_parameters(teacher)

    if not shared:
        raise RuntimeError("No shared parameters found for EWC.")

    for p in shared.values():
        p.requires_grad = True

    fisher = {n: torch.zeros_like(p, device="cpu") for n, p in shared.items()}
    optimal = {n: p.detach().clone().cpu()          for n, p in shared.items()}

    loader = DataLoader(
        arson_train_dataset,
        batch_size=FISHER_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    processed = 0
    for images, _ in loader:
        if processed >= sample_limit:
            break

        images = images.to(TEACHER_DEVICE, dtype=torch.float32)
        teacher.zero_grad(set_to_none=True)

        with torch.enable_grad():
            outputs = teacher(images)
            # Use a uniform target (class 0) so we get gradient signal
            labels = torch.zeros(images.size(0), dtype=torch.long, device=TEACHER_DEVICE)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

        for name, p in shared.items():
            if p.grad is not None:
                fisher[name] += p.grad.detach().pow(2).cpu()

        teacher.zero_grad(set_to_none=True)
        processed += 1
        print(f"\r[EWC] Fisher {processed}/{sample_limit}", end="")
        del images, outputs, loss

    print()

    if processed == 0:
        raise RuntimeError("Fisher computed zero samples.")

    for name in fisher:
        fisher[name] = (fisher[name] / float(processed)).detach().cpu()

    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[EWC] Fresh Fisher computed. Tensors: {len(fisher)}")
    return fisher, optimal


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
# DATASETS
# ============================================================

def create_datasets():

    print()
    print("=" * 60)
    print("[DATASET SETUP]")
    print("=" * 60)

    # ========================================================
    # ARSON TRAIN
    # ========================================================

    arson_train = VideoDataset(
        root=ARSON_DATASET_ROOT,
        train=True,
        classes=["arson"],
        class_to_idx=CLASS_MAPPING,
        clip_len=CLIP_LEN,
    )

    # ========================================================
    # ARSON VALIDATION
    # ========================================================

    arson_val = VideoDataset(
        root=ARSON_DATASET_ROOT,
        train=False,
        classes=["arson"],
        class_to_idx=CLASS_MAPPING,
        clip_len=CLIP_LEN,
    )

    print(
        f"[Arson] Train clips: "
        f"{len(arson_train)}"
    )

    print(
        f"[Arson] Val clips: "
        f"{len(arson_val)}"
    )

    if len(arson_train) == 0:
        raise RuntimeError("Arson training dataset is empty.")

    if len(arson_val) == 0:
        raise RuntimeError("Arson validation dataset is empty.")

    # ========================================================
    # REPLAY TRAIN (all 4 old classes from replay_buffer/)
    # ========================================================

    replay_train = ReplayDataset(
        root=REPLAY_TRAIN_ROOT,
        class_to_idx=CLASS_MAPPING,
        classes=OLD_CLASSES,
        clip_len=CLIP_LEN,
        random_clip=True,
    )

    # ========================================================
    # REPLAY VALIDATION
    # ========================================================

    replay_val = ReplayDataset(
        root=REPLAY_VAL_ROOT,
        class_to_idx=CLASS_MAPPING,
        classes=OLD_CLASSES,
        clip_len=CLIP_LEN,
        random_clip=False,
    )

    print(f"[Replay] train clips: {len(replay_train)}")
    print(f"[Replay] val   clips: {len(replay_val)}")

    if len(replay_train) == 0:
        raise RuntimeError("Replay training dataset is empty.")

    if len(replay_val) == 0:
        raise RuntimeError("Replay validation dataset is empty.")

    # Per-class replay train counts
    for cls in OLD_CLASSES:
        cnt = sum(1 for _, lbl in replay_train.samples if lbl == CLASS_MAPPING[cls])
        print(f"  Replay train {cls}: {cnt}")

    # ========================================================
    # COMBINE: replay (4 old classes) + arson
    # ========================================================

    from torch.utils.data import ConcatDataset

    train_dataset = ConcatDataset([replay_train, arson_train])
    val_dataset   = ConcatDataset([replay_val,   arson_val])

    print()
    print(f"[Dataset] Combined train: {len(train_dataset)}")
    print(f"[Dataset] Combined val:   {len(val_dataset)}")
    print()
    print("[Dataset] Labels:")
    print("0 = normal")
    print("1 = fighting")
    print("2 = vandalism")
    print("3 = assault")
    print("4 = arson")

    return (
        arson_train,
        arson_val,
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
    print("[KD] Creating old 4-class teacher...")

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
    print("[Model] Creating 5-class student...")

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
    print("[Model] Previous classes (preserved):")
    print("    normal    = 0")
    print("    fighting  = 1")
    print("    vandalism = 2")
    print("    assault   = 3")

    print()
    print("[Model] New class:")
    print("    arson = 4")

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
    # Save class mapping (5 classes)
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
        arson_train,
        arson_val,
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
    # Verify old checkpoint has exactly 4 classes
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
            "expected 4-class model.\n\n"
            f"Detected classes: "
            f"{checkpoint_num_classes}\n"
            f"Expected: "
            f"{OLD_CLASS_COUNT}\n\n"
            "Use best_model.pth "
            "containing normal + fighting + vandalism + assault."
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
    # EWC — load from checkpoint (no replay needed)
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

    # --------------------------------------------------------
    # Check whether saved Fisher is compatible with the
    # current student parameter names.
    # If overlap == 0 the saved Fisher is from an older
    # architecture and must be recomputed.
    # --------------------------------------------------------

    student_param_names = {
        n for n, _ in student.named_parameters()
        if "classifier" not in n
    }

    saved_fisher_names = set(saved_fisher.keys())
    overlap = len(saved_fisher_names & student_param_names)

    if saved_fisher and saved_params and overlap > 0:

        print(
            "[EWC] Saved Fisher found in checkpoint and compatible."
        )
        print(
            f"[EWC] Fisher tensors: {len(saved_fisher)}, matching: {overlap}"
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
            f"[EWC] Saved Fisher has {overlap} matching params — "
            "incompatible (older architecture). Computing fresh Fisher..."
        )

        fisher, optimal_params = compute_fisher_fresh(
            teacher,
            arson_train,
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
    # All arson-only training; equal weights for the 5 classes.
    # Old classes are not present in training batches —
    # their knowledge is retained via KD + EWC.
    # --------------------------------------------------------

    class_weights = torch.tensor(
        [
            1.0,  # normal    (not in training data)
            1.0,  # fighting  (not in training data)
            1.0,  # vandalism (not in training data)
            1.0,  # assault   (not in training data)
            1.0,  # arson     (active training class)
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

    best_epoch = -1

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
            #
            # Since training data is ONLY arson (label 4),
            # old_class_mask will be all False.
            # KD is therefore applied to EVERY batch:
            # we always pass the batch through the frozen
            # teacher and distill its 4-class knowledge
            # into the student's first 4 outputs.
            # ------------------------------------------------

            kd_loss = torch.zeros(
                (),
                device=STUDENT_DEVICE,
            )

            if KD_ALPHA > 0:

                # --------------------------------------------
                # Student old-class logits [:, :4]
                # --------------------------------------------

                student_old_logits = (
                    student_outputs[
                        :, :OLD_CLASS_COUNT
                    ]
                )

                # --------------------------------------------
                # Send images to CPU teacher
                # --------------------------------------------

                teacher_images = images.to(
                    TEACHER_DEVICE,
                    dtype=torch.float32,
                )

                # --------------------------------------------
                # Teacher forward (4 outputs)
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
                # KD: student[:, :4] vs teacher[:, :4]
                # --------------------------------------------

                kd_loss = distillation_loss(
                    student_old_logits,
                    teacher_outputs[
                        :, :OLD_CLASS_COUNT
                    ],
                )

                # --------------------------------------------
                # Cleanup
                # --------------------------------------------

                del student_old_logits
                del teacher_images
                del teacher_outputs

            # ------------------------------------------------
            # FINAL LOSS
            #
            # Loss =
            #   (1 - alpha) * CE
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

        arson_gt = [g for g, p in zip(ground_truth, predictions) if g == 4]
        arson_pd = [p for g, p in zip(ground_truth, predictions) if g == 4]
        arson_acc = accuracy_score(arson_gt, arson_pd) if arson_gt else 0.0

        print()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {average_loss:.4f} | "
            f"CE: {average_ce:.4f} | "
            f"KD: {average_kd:.4f} | "
            f"EWC: {average_ewc:.4f} | "
            f"Val Acc (Total 5-class): {accuracy:.4f} | "
            f"Arson Acc: {arson_acc:.4f}"
        )

        # ====================================================
        # CLASSIFICATION REPORT (All 5 classes)
        # ====================================================

        print()

        print(
            "[Validation Classification Report — All 5 Classes]"
        )

        print(
            classification_report(
                ground_truth,
                predictions,
                labels=[0, 1, 2, 3, 4],
                target_names=CLASSES,
                zero_division=0,
            )
        )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(
            ground_truth,
            predictions,
            labels=[0, 1, 2, 3, 4],
        )

        print("[Confusion Matrix]")
        print("             Predicted")
        print("             N    F    V    A    AR")
        for idx, cls_name in enumerate(["N ", "F ", "V ", "A ", "AR"]):
            row_str = " ".join(f"{cm[idx, j]:4d}" for j in range(5))
            print(f"Actual {cls_name}  {row_str}")

        # ====================================================
        # BEST MODEL
        # ====================================================

        if accuracy > best_acc:

            best_acc = accuracy

            best_state_dict = deepcopy(
                model.state_dict()
            )

            best_epoch = epoch + 1

            print()

            print(
                f"[BEST] New best Total 5-class accuracy: "
                f"{best_acc:.4f} "
                f"(epoch {best_epoch})"
            )

    return (
        best_acc,
        best_state_dict,
        best_epoch,
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    state,
    best_acc,
    best_state_dict,
    best_epoch,
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

        "best_epoch":
            best_epoch,

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
        f"Best Arson validation accuracy: "
        f"{best_acc:.4f} (epoch {best_epoch})"
    )

    print(
        f"Classes: {CLASS_MAPPING}"
    )

    print(
        f"num_classes in checkpoint: {NEW_CLASS_COUNT}"
    )

    print(
        f"EWC tensors: "
        f"{len(cpu_fisher)}"
    )

    # --------------------------------------------------------
    # Verify source checkpoint was NOT overwritten
    # --------------------------------------------------------

    import hashlib
    orig_hash = hashlib.sha256(
        open(SOURCE_CHECKPOINT_PATH, "rb").read()
    ).hexdigest()

    print()
    print("[Integrity Check]")
    print(
        f"Source checkpoint SHA-256: {orig_hash}"
    )

    expected_hash = "723a77dbf1c78edeb23141e53658cfe80a4ba1cd3c5088e0d32452e9a6bf8e48"

    if orig_hash == expected_hash:
        print("[PASS] best_model.pth is unmodified.")
    else:
        print(
            "[FAIL] WARNING: best_model.pth hash has changed! "
            "This should never happen."
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
        f"Arson dataset:\n"
        f"{ARSON_DATASET_ROOT}"
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
        ARSON_DATASET_ROOT,
    ]

    missing = []

    for path in required_dirs:

        if not os.path.isdir(path):

            missing.append(path)

    if not os.path.isfile(
        SOURCE_CHECKPOINT_PATH
    ):

        missing.append(
            SOURCE_CHECKPOINT_PATH
        )

    # --------------------------------------------------------
    # Verify output path does NOT point to source checkpoint
    # --------------------------------------------------------

    if os.path.abspath(OUTPUT_CHECKPOINT_PATH) == \
       os.path.abspath(SOURCE_CHECKPOINT_PATH):

        raise RuntimeError(
            "OUTPUT_CHECKPOINT_PATH must not be the same as "
            "SOURCE_CHECKPOINT_PATH. "
            "best_model.pth must not be overwritten."
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

    print(
        "[PASS] Output checkpoint is different from source checkpoint."
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
            best_epoch,
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
            best_epoch,
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
            f"Best Arson validation accuracy: "
            f"{best_acc:.4f} (epoch {best_epoch})"
        )

        print()
        print("Final classes:")

        print("0 -> normal")
        print("1 -> fighting")
        print("2 -> vandalism")
        print("3 -> assault")
        print("4 -> arson")

        print()
        print("Experience Replay   = unavailable")
        print("Synthetic Replay    = not used")
        print("KD                  = used")
        print("EWC                 = used")
        print(
            "Old-class retention = "
            "qualitatively protected via KD + EWC; "
            "quantitative measurement unavailable "
            "(no real old-class validation data)."
        )

        print()
        print(
            "[Source 4-class model]"
        )

        print(
            SOURCE_CHECKPOINT_PATH
        )

        print()
        print(
            "[New 5-class model]"
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