import json
import os
import random
import shutil
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from video_dataset import VideoDataset, ReplayDataset
from models.vad_model import RepViTTCN

# ════════════════════════════════════════════════════════════════════════════
# Device + speed flags
# ════════════════════════════════════════════════════════════════════════════
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"            # Mixed-precision only on GPU

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # Fastest cuDNN kernel auto-select
else:
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(max(num_cores // 2, 1))

print(f"  [Device] Running on: {device}  |  AMP: {USE_AMP}")

# ════════════════════════════════════════════════════════════════════════════
# Continual-Learning Hyperparameters & Paths
# ════════════════════════════════════════════════════════════════════════════
EWC_LAMBDA             = 500   # EWC regularisation strength
KD_ALPHA               = 0.5   # KD loss weight  (0 = pure CE, 1 = pure KD)
KD_TEMP                = 4.0   # Soft-label temperature
REPLAY_PER_CLASS       = 40    # Max old-class clips in replay buffer
FISHER_LAYERS          = ("classifier", "tcn")
FISHER_BATCHES         = 20    # Batches used to estimate Fisher
EPOCHS                 = 10
BATCH_SIZE             = 8
CLASSES                = ["normal", "fighting", "vandalism"]
CLASS_MAPPING          = {"normal": 0, "fighting": 1, "vandalism": 2}
CHECKPOINT_PATH        = "best_model.pth"
OUTPUT_CHECKPOINT_PATH = "best_model.pth"
CLASSES_JSON           = "classes.json"
REPLAY_ROOT            = r"D:\RepViT_VAD_Data\replay_buffer"
VANDALISM_DATASET_ROOT = r"D:\RepViT_VAD_Data\vandalism_dataset"

# ════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════════════════════════

def ewc_penalty(model, ewc_fisher, ewc_optimal_params) -> torch.Tensor:
    """EWC: penalise drift from previously optimal weights."""
    if not ewc_fisher:
        return torch.tensor(0., device=device)
    loss = torch.tensor(0., device=device)
    for name, param in model.named_parameters():
        if name in ewc_fisher and name in ewc_optimal_params:
            fisher_tensor = ewc_fisher[name].to(device)
            opt_tensor    = ewc_optimal_params[name].to(device)
            p = param
            if p.shape != fisher_tensor.shape and p.shape[0] >= fisher_tensor.shape[0]:
                p = p[:fisher_tensor.shape[0]]
            loss += (fisher_tensor * (p - opt_tensor).pow(2)).sum()
    return (EWC_LAMBDA / 2) * loss


def distillation_loss(new_logits: torch.Tensor, old_logits: torch.Tensor) -> torch.Tensor:
    """KD: KL-divergence soft-target loss with temperature scaling for old classes."""
    p_new = F.log_softmax(new_logits / KD_TEMP, dim=1)
    p_old = F.softmax(    old_logits / KD_TEMP, dim=1)
    return F.kl_div(p_new, p_old, reduction="batchmean") * (KD_TEMP ** 2)


def compute_fisher_on_dataset(model, dataset, optimizer, criterion) -> dict:
    """Computes diagonal Fisher Information Matrix on old knowledge dataset."""
    # Ensure parameters have requires_grad=True temporarily for backward pass
    orig_grad_flags = {n: p.requires_grad for n, p in model.named_parameters()}
    for name, param in model.named_parameters():
        if any(layer in name for layer in FISHER_LAYERS):
            param.requires_grad = True

    fisher = {
        n: torch.zeros_like(p)
        for n, p in model.named_parameters()
        if p.requires_grad and any(layer in n for layer in FISHER_LAYERS)
    }

    fisher_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    model.train()

    for i, (images, labels) in enumerate(fisher_loader):
        if i >= FISHER_BATCHES:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()

        for name, param in model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)

    count = max(min(FISHER_BATCHES, len(fisher_loader)), 1)
    for name in fisher:
        fisher[name] /= count

    # Restore original grad flags
    for name, param in model.named_parameters():
        param.requires_grad = orig_grad_flags[name]

    return fisher


def setup_training():
    print("  [Setup] Initializing Vandalism continual learning datasets...")
    
    # Save target class mapping
    with open(CLASSES_JSON, "w") as f:
        json.dump(CLASS_MAPPING, f, indent=2)
        
    # 1. Base Vandalism Dataset (Using CLASS_MAPPING so vandalism gets label 2)
    vandalism_train_ds = VideoDataset(root=VANDALISM_DATASET_ROOT, train=True, classes=["vandalism"], class_to_idx=CLASS_MAPPING)
    vandalism_val_ds   = VideoDataset(root=VANDALISM_DATASET_ROOT, train=False, classes=["vandalism"], class_to_idx=CLASS_MAPPING)

    # 2. External Replay Buffer (Normal + Fighting)
    replay_paths = {
        "normal": [
            os.path.join(REPLAY_ROOT, "normal", d)
            for d in os.listdir(os.path.join(REPLAY_ROOT, "normal"))
            if os.path.isdir(os.path.join(REPLAY_ROOT, "normal", d))
        ],
        "fighting": [
            os.path.join(REPLAY_ROOT, "fighting", d)
            for d in os.listdir(os.path.join(REPLAY_ROOT, "fighting"))
            if os.path.isdir(os.path.join(REPLAY_ROOT, "fighting", d))
        ]
    }

    replay_train_ds = ReplayDataset(replay_paths, CLASS_MAPPING, clip_len=16, random_clip=True)
    replay_val_ds   = ReplayDataset(replay_paths, CLASS_MAPPING, clip_len=16, random_clip=False)

    train_dataset = ConcatDataset([vandalism_train_ds, replay_train_ds])
    val_dataset   = ConcatDataset([vandalism_val_ds, replay_val_ds])

    print(f"  [Dataset] Combined Train set: {len(train_dataset)} samples (Vandalism + Normal/Fighting Replay)")
    print(f"  [Dataset] Combined Val set:   {len(val_dataset)} samples")

    # 3. Model setup
    num_classes = len(CLASSES) # 3
    model = RepViTTCN(num_classes=num_classes).to(device)

    old_model          = None
    ewc_fisher         = {}
    ewc_optimal_params = {}

    if os.path.exists(CHECKPOINT_PATH):
        raw_ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        
        # Load 2-class teacher BEFORE classifier expansion
        old_model = RepViTTCN(num_classes=2).to(device)
        ckpt_sd = raw_ckpt.get("state_dict", raw_ckpt)
        old_model.load_state_dict(ckpt_sd, strict=False)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False
        print("  [KD]     Frozen 2-class teacher model loaded.")

        # Expand student model to 3 classes
        model.load_continual_checkpoint(CHECKPOINT_PATH, device=device)
        print("  [Model]  Student model expanded to 3 classes (normal=0, fighting=1, vandalism=2).")

        # EWC setup
        if isinstance(raw_ckpt, dict):
            ewc_fisher         = {k: v.to(device) for k, v in raw_ckpt.get("ewc_fisher", {}).items()}
            ewc_optimal_params = {k: v.to(device) for k, v in raw_ckpt.get("ewc_params",  {}).items()}

        # If EWC Fisher is empty in checkpoint, compute it on old knowledge replay dataset
        if not ewc_fisher:
            print("  [EWC]    Computing Fisher Information Matrix on old Normal/Fighting knowledge...")
            temp_opt = torch.optim.Adam(old_model.parameters(), lr=1e-5)
            temp_crit = nn.CrossEntropyLoss()
            ewc_fisher = compute_fisher_on_dataset(old_model, replay_val_ds, temp_opt, temp_crit)
            ewc_optimal_params = {
                n: p.detach().cpu()
                for n, p in old_model.named_parameters()
                if any(layer in n for layer in FISHER_LAYERS)
            }
            print(f"  [EWC]    Fisher matrix computed ({len(ewc_fisher)} param tensors).")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "scaler": scaler,
        "model": model,
        "old_model": old_model,
        "ewc_fisher": ewc_fisher,
        "ewc_optimal_params": ewc_optimal_params,
        "train_dataset": train_dataset,
        "replay_paths": replay_paths
    }


def run_training(state: dict):
    model              = state["model"]
    train_loader       = state["train_loader"]
    val_loader         = state["val_loader"]
    old_model          = state["old_model"]
    ewc_fisher         = state["ewc_fisher"]
    ewc_optimal_params = state["ewc_optimal_params"]
    criterion          = state["criterion"]
    optimizer          = state["optimizer"]
    scaler             = state["scaler"]

    best_acc        = 0.0
    best_state_dict = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
                outputs = model(images)

                # 1. Cross-entropy loss on 3 classes
                loss = criterion(outputs, labels)

                # 2. EWC regularisation
                loss = loss + ewc_penalty(model, ewc_fisher, ewc_optimal_params)

                # 3. Knowledge Distillation on OLD classes only (normal=0, fighting=1)
                if old_model is not None and KD_ALPHA > 0:
                    with torch.no_grad():
                        old_logits = old_model(images)
                    # Strictly compare logits[:, :2]
                    kd_loss = distillation_loss(outputs[:, :2], old_logits[:, :2])
                    loss    = (1 - KD_ALPHA) * loss + KD_ALPHA * kd_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Validation
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
                    outputs = model(images.to(device, non_blocking=True))
                preds.extend(outputs.argmax(dim=1).cpu().numpy())
                gts.extend(labels.cpu().numpy())

        acc = accuracy_score(gts, preds)
        print(f"\nEpoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state_dict = deepcopy(model.state_dict())
            print(f"→ New best accuracy: {best_acc:.4f}")

    return best_acc, best_state_dict

if __name__ == "__main__":
    state = setup_training()
    best_acc, best_state_dict = run_training(state)

    if best_state_dict is None:
        best_state_dict = deepcopy(state["model"].state_dict())

    checkpoint = {
        "state_dict":   best_state_dict,
        "classes":      CLASSES,
        "num_classes":  len(CLASSES),
        "best_acc":     best_acc,
        "ewc_fisher":   {k: v.cpu() for k, v in state["ewc_fisher"].items()},
        "ewc_params":   {n: p.detach().cpu() for n, p in state["ewc_optimal_params"].items()},
        "replay_paths": state["replay_paths"]
    }

    torch.save(checkpoint, OUTPUT_CHECKPOINT_PATH)
    print(f"\nSaved continual learning model to {OUTPUT_CHECKPOINT_PATH} with best accuracy: {best_acc:.4f}")
