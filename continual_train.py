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
    # CPU: use all available cores for inter/intra-op parallelism
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(max(num_cores // 2, 1))

print(f"  [Device] Running on: {device}  |  AMP: {USE_AMP}")

# ════════════════════════════════════════════════════════════════════════════
# Continual-Learning Hyperparameters
# ════════════════════════════════════════════════════════════════════════════
EWC_LAMBDA       = 500   # EWC regularisation strength
KD_ALPHA         = 0.5   # KD loss weight  (0 = pure CE, 1 = pure KD)
KD_TEMP          = 4.0   # Soft-label temperature
REPLAY_PER_CLASS = 40    # Max old-class clips in replay buffer
FISHER_LAYERS    = ("classifier", "tcn")
FISHER_BATCHES   = 20    # Batches used to estimate Fisher
EPOCHS           = 10
CLASSES          = ["normal", "fighting"]
CHECKPOINT_PATH  = "best_model.pth"
CLASSES_JSON     = "classes.json"
REPLAY_ROOT      = "replay_buffer"
FIRST_MEMBER     = os.environ.get("FIRST_MEMBER", "false").lower() in ("1", "true", "yes") or not os.path.exists(CHECKPOINT_PATH)
# ════════════════════════════════════════════════════════════════════════════

def setup_training():
    # ─── Base datasets ──────────────────────────────────────────────────────────
    if FIRST_MEMBER:
        base_train_dataset = VideoDataset(train=True, classes=CLASSES)
        val_dataset        = VideoDataset(train=False, classes=CLASSES)
    else:
        base_train_dataset = VideoDataset(train=True)
        val_dataset        = VideoDataset(train=False)

    num_classes = len(base_train_dataset.classes)
    model = RepViTTCN(num_classes=num_classes).to(device)

    old_model          = None
    ewc_fisher         = {}
    ewc_optimal_params = {}
    train_dataset      = base_train_dataset

    if not FIRST_MEMBER and os.path.exists(CHECKPOINT_PATH):
        raw_ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_continual_checkpoint(CHECKPOINT_PATH, device=device)

        old_model = deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False
        print("  [KD]     Frozen old model loaded as teacher.")

        if isinstance(raw_ckpt, dict):
            ewc_fisher         = {k: v.to(device) for k, v in raw_ckpt.get("ewc_fisher", {}).items()}
            ewc_optimal_params = {k: v.to(device) for k, v in raw_ckpt.get("ewc_params",  {}).items()}
            if ewc_fisher:
                print(f"  [EWC]    Fisher matrix restored ({len(ewc_fisher)} param tensors).")

            saved_replay = raw_ckpt.get("replay_paths", {})
            valid_replay = {}
            for cls, paths in saved_replay.items():
                alive = [p for p in paths if os.path.exists(p)]
                if alive:
                    valid_replay[cls] = alive
                    print(f"  [Replay] Restored {len(alive)} clips for class '{cls}'")

            if valid_replay:
                replay_ds     = ReplayDataset(valid_replay, base_train_dataset.class_to_idx, clip_len=16)
                train_dataset = ConcatDataset([base_train_dataset, replay_ds])
                print(f"  [Replay] Combined dataset: {len(train_dataset)} total samples\n")
            else:
                train_dataset = base_train_dataset
    else:
        print("  [INFO] No previous checkpoint loaded. Training scratch model on Normal + Fighting.")
        train_dataset = base_train_dataset

    # ─── DataLoaders ────────────────────────────────────────────────────────────
    workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)   # AMP gradient scaler

    return {
        "base_train_dataset": base_train_dataset,
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
    }

# ════════════════════════════════════════════════════════════════════════════
# Continual-Learning Helper Functions
# ════════════════════════════════════════════════════════════════════════════

def ewc_penalty(model, ewc_fisher, ewc_optimal_params) -> torch.Tensor:
    """EWC: penalise drift from previously optimal weights."""
    if not ewc_fisher:
        return torch.tensor(0., device=device)
    loss = torch.tensor(0., device=device)
    for name, param in model.named_parameters():
        if name in ewc_fisher:
            loss += (ewc_fisher[name] * (param - ewc_optimal_params[name]).pow(2)).sum()
    return (EWC_LAMBDA / 2) * loss


def distillation_loss(new_logits: torch.Tensor, old_logits: torch.Tensor) -> torch.Tensor:
    """KD: KL-divergence soft-target loss with temperature scaling."""
    p_new = F.log_softmax(new_logits / KD_TEMP, dim=1)
    p_old = F.softmax(    old_logits / KD_TEMP, dim=1)
    return F.kl_div(p_new, p_old, reduction="batchmean") * (KD_TEMP ** 2)


def compute_fisher(model, base_train_dataset, optimizer, criterion, ewc_fisher) -> dict:
    """
    Computes diagonal Fisher Information Matrix.
    Called ONCE after training on the best model.
    """

    fisher = {
        n: torch.zeros_like(p)
        for n, p in model.named_parameters()
        if p.requires_grad and any(layer in n for layer in FISHER_LAYERS)
    }

    fisher_loader = DataLoader(
        base_train_dataset,
        batch_size=8,
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

        with torch.amp.autocast(
            device_type=device.type,
            enabled=USE_AMP
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # IMPORTANT:
        # Compute gradients only. Do NOT call optimizer.step() here.
        loss.backward()

        for name, param in model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)

    count = max(min(FISHER_BATCHES, len(fisher_loader)), 1)

    for name in fisher:
        fisher[name] /= count

        if name in ewc_fisher:
            fisher[name] = (fisher[name] + ewc_fisher[name]) / 2

    return fisher

def build_replay_paths(base_train_dataset) -> dict:
    """Samples up to REPLAY_PER_CLASS paths per class for next run's replay buffer."""
    buckets: dict = {}
    for path, label in base_train_dataset.samples:
        buckets.setdefault(label, []).append(path)
    result: dict = {}
    for label, paths in buckets.items():
        cls_name         = base_train_dataset.classes[label] if label < len(base_train_dataset.classes) else str(label)
        result[cls_name] = random.sample(paths, min(REPLAY_PER_CLASS, len(paths)))
    return result


def save_class_mapping(class_to_idx: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(class_to_idx, f, indent=2)


def save_replay_buffer(replay_paths: dict, replay_root: str) -> None:
    os.makedirs(replay_root, exist_ok=True)
    for cls_name, paths in replay_paths.items():
        class_dir = os.path.join(replay_root, cls_name)
        os.makedirs(class_dir, exist_ok=True)

        for idx, src in enumerate(paths):
            if not os.path.isdir(src):
                print(f"  [Replay] Skipping invalid source path: {src}")
                continue

            base_name = os.path.basename(os.path.normpath(src))
            dest = os.path.join(class_dir, base_name)
            if os.path.exists(dest):
                dest = os.path.join(class_dir, f"{base_name}_{idx}")

            shutil.copytree(src, dest, dirs_exist_ok=True)


# ════════════════════════════════════════════════════════════════════
# Training Loop
# ════════════════════════════════════════════════════════════════════

def run_training(state: dict):
    model             = state["model"]
    train_loader      = state["train_loader"]
    val_loader        = state["val_loader"]
    base_train_dataset = state["base_train_dataset"]
    old_model         = state["old_model"]
    ewc_fisher        = state["ewc_fisher"]
    ewc_optimal_params= state["ewc_optimal_params"]
    criterion         = state["criterion"]
    optimizer         = state["optimizer"]
    scaler            = state["scaler"]

    best_acc        = 0.0
    best_state_dict = None   # Keep best weights in memory; save once at the end

    for epoch in range(EPOCHS):

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)   # Faster than zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
                outputs = model(images)

                # 1. Cross-entropy loss
                loss = criterion(outputs, labels)

                # 2. EWC regularisation
                loss = loss + ewc_penalty(model, ewc_fisher, ewc_optimal_params)

                # 3. Knowledge Distillation
                if old_model is not None and KD_ALPHA > 0:
                    with torch.no_grad():
                        old_logits = old_model(images)
                    min_c   = min(outputs.shape[1], old_logits.shape[1])
                    kd_loss = distillation_loss(outputs[:, :min_c], old_logits[:, :min_c])
                    loss    = (1 - KD_ALPHA) * loss + KD_ALPHA * kd_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # ── Validation ─────────────────────────────────────────────────────────
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

        # Keep best weights in memory (no Fisher cost here)
        if acc > best_acc:
            best_acc = acc
            best_state_dict = deepcopy(model.state_dict())
            
            replay_paths = build_replay_paths(base_train_dataset)

            torch.save(
                {
                    "state_dict": best_state_dict,
                    "classes": base_train_dataset.classes,
                    "num_classes": len(base_train_dataset.classes),
                    "best_acc": best_acc,
                    "ewc_fisher": {k: v.cpu() for k, v in ewc_fisher.items()},
                    "ewc_params": {
                        n: p.detach().cpu()
                        for n, p in model.named_parameters()
                        if p.requires_grad and any(layer in n for layer in FISHER_LAYERS)
                    },
                    "replay_paths": replay_paths,
                },
                "best_model_epoch.pth",
            )

            print(f"→ New best: {best_acc:.4f}")
            
    return best_acc, best_state_dict

if __name__ == "__main__":
    # 1. Setup Data and Model
    state = setup_training()
    
    # 2. Run Training Loop
    best_acc, best_state_dict = run_training(state)
    
    # 3. Post-Training: Compute Fisher ONCE on best model, then save
    print("\nComputing Fisher Information Matrix on best model...")
    model = state["model"]
    base_train_dataset = state["base_train_dataset"]
    
    if best_state_dict is None:
        best_state_dict = deepcopy(model.state_dict())
        
    model.load_state_dict(best_state_dict)   # Restore best weights
    
    new_fisher = compute_fisher(
        model, 
        base_train_dataset, 
        state["optimizer"], 
        state["criterion"], 
        state["ewc_fisher"]
    )
    
    replay_paths = build_replay_paths(base_train_dataset)

    checkpoint = {
        "state_dict":   best_state_dict,
        "classes":      base_train_dataset.classes,
        "num_classes":  len(base_train_dataset.classes),
        "best_acc":     best_acc,
        "ewc_fisher":   {k: v.cpu() for k, v in new_fisher.items()},
        "ewc_params":   {n: p.detach().cpu()
                         for n, p in model.named_parameters()
                         if p.requires_grad and any(lyr in n for lyr in FISHER_LAYERS)},
        "replay_paths": replay_paths,
    }
    
    torch.save(checkpoint, CHECKPOINT_PATH)
    save_class_mapping(base_train_dataset.class_to_idx, CLASSES_JSON)
    torch.save(new_fisher, "fisher.pkl")
    save_replay_buffer(replay_paths, REPLAY_ROOT)
    
    print(f"\nBest continual learning model saved!  Accuracy: {best_acc:.4f}")
    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    print(f"Saved class mapping: {CLASSES_JSON}")
    print("Saved Fisher matrix: fisher.pkl")
    print(f"Saved replay buffer: {REPLAY_ROOT}")
    print("Training Finished!")
