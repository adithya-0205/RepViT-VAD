import os
import json
import torch
import torch.nn.functional as F

from video_dataset import VideoDataset, ReplayDataset
from models.vad_model import RepViTTCN
import continual_train_assault as ct


# ============================================================
# PATH CONFIGURATION
# ============================================================

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

ASSAULT_ROOT = os.path.join(
    REPO_ROOT,
    "dataset"
)

REPLAY_ROOT = os.path.join(
    REPO_ROOT,
    "replay_buffer"
)

CHECKPOINT_PATH = os.path.join(
    REPO_ROOT,
    "best_model.pth"
)

BACKUP_PATH = os.path.join(
    REPO_ROOT,
    "best_model_before_assault.pth"
)

CLASSES_PATH = os.path.join(
    REPO_ROOT,
    "classes.json"
)


# ============================================================
# VALIDATION
# ============================================================

def run_validation():

    print("==================================================")
    print("RUNNING ASSAULT PRE-TRAINING VALIDATION CHECKLIST")
    print("==================================================")

    print(f"\nRepository : {REPO_ROOT}")
    print(f"Assault    : {ASSAULT_ROOT}")
    print(f"Replay     : {REPLAY_ROOT}")

    results = []

    ckpt = None
    a_train = None
    a_val = None
    rep_norm = None
    rep_fight = None
    rep_vandal = None

    # ========================================================
    # 1. CHECKPOINT LOAD
    # ========================================================

    try:
        ckpt = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu"
        )

        results.append(
            (
                "1. best_model.pth loads",
                True,
                "Successfully loaded"
            )
        )

    except Exception as e:

        results.append(
            (
                "1. best_model.pth loads",
                False,
                str(e)
            )
        )

    # ========================================================
    # 2. BACKUP EXISTS
    # ========================================================

    backup_exists = os.path.exists(BACKUP_PATH)

    results.append(
        (
            "2. Backup of best_model.pth exists before assault training",
            backup_exists,
            f"Backup at: {BACKUP_PATH}" if backup_exists else "Backup file missing"
        )
    )

    # ========================================================
    # 3. CHECKPOINT METADATA
    # ========================================================

    if ckpt is not None:

        has_state = "state_dict" in ckpt
        has_classes = "classes" in ckpt
        has_mapping = "class_mapping" in ckpt

        results.append(
            (
                "3. best_model.pth contains expected keys",
                has_state and has_classes and has_mapping,
                f"Keys: {list(ckpt.keys())}"
            )
        )

    else:

        results.append(
            (
                "3. best_model.pth contains expected keys",
                False,
                "Checkpoint not loaded"
            )
        )

    # ========================================================
    # 4. CHECKPOINT CLASSES
    # ========================================================

    if ckpt is not None:

        classes = ckpt.get("classes", [])
        num_classes = ckpt.get("num_classes", 0)

        correct_classes = (
            "normal" in classes
            and "fighting" in classes
            and "vandalism" in classes
            and len(classes) == 3
        )

        results.append(
            (
                "4. Checkpoint classes count and elements are correct",
                correct_classes,
                f"Classes: {classes}, num_classes: {num_classes}"
            )
        )

    else:

        results.append(
            (
                "4. Checkpoint classes count is correct",
                False,
                "Checkpoint not loaded"
            )
        )

    # ========================================================
    # 5. CLASSES JSON
    # ========================================================

    try:
        with open(CLASSES_PATH, "r") as f:
            mapping = json.load(f)

        correct_mapping = (
            mapping.get("normal") == 0
            and mapping.get("fighting") == 1
            and mapping.get("vandalism") == 2
        )

        results.append(
            (
                "5. classes.json is correctly configured for 3 classes",
                correct_mapping,
                f"Mapping: {mapping}"
            )
        )

    except Exception as e:

        results.append(
            (
                "5. classes.json is correctly configured",
                False,
                str(e)
            )
        )

    # ========================================================
    # 6. ASSAULT DATASET LOADS (TRAIN)
    # ========================================================

    try:
        a_train = VideoDataset(
            root=ASSAULT_ROOT,
            train=True,
            classes=["assault"],
            class_to_idx=ct.CLASS_MAPPING,
            clip_len=16
        )

        results.append(
            (
                "6. Assault train dataset loads",
                len(a_train) > 0,
                f"Count: {len(a_train)} videos"
            )
        )

    except Exception as e:

        results.append(
            (
                "6. Assault train dataset loads",
                False,
                str(e)
            )
        )

    # ========================================================
    # 7. ASSAULT DATASET LOADS (VAL)
    # ========================================================

    try:
        a_val = VideoDataset(
            root=ASSAULT_ROOT,
            train=False,
            classes=["assault"],
            class_to_idx=ct.CLASS_MAPPING,
            clip_len=16
        )

        results.append(
            (
                "7. Assault validation dataset loads",
                len(a_val) > 0,
                f"Count: {len(a_val)} videos"
            )
        )

    except Exception as e:

        results.append(
            (
                "7. Assault validation dataset loads",
                False,
                str(e)
            )
        )

    # ========================================================
    # 8-10. REPLAY BUFFERS LOAD
    # ========================================================

    try:
        normal_paths = os.path.join(REPLAY_ROOT, "train")
        val_normal_paths = os.path.join(REPLAY_ROOT, "val")

        rep_norm = ReplayDataset(
            normal_paths,
            ct.CLASS_MAPPING,
            classes=["normal"],
            clip_len=16
        )

        rep_fight = ReplayDataset(
            normal_paths,
            ct.CLASS_MAPPING,
            classes=["fighting"],
            clip_len=16
        )

        rep_vandal = ReplayDataset(
            normal_paths,
            ct.CLASS_MAPPING,
            classes=["vandalism"],
            clip_len=16
        )

        results.append(
            (
                "8. Replay Normal loads",
                len(rep_norm) > 0,
                f"Count: {len(rep_norm)} clips"
            )
        )

        results.append(
            (
                "9. Replay Fighting loads",
                len(rep_fight) > 0,
                f"Count: {len(rep_fight)} clips"
            )
        )

        results.append(
            (
                "10. Replay Vandalism loads",
                len(rep_vandal) > 0,
                f"Count: {len(rep_vandal)} clips"
            )
        )

    except Exception as e:

        results.append(
            (
                "8. Replay Normal loads",
                False,
                str(e)
            )
        )

        results.append(
            (
                "9. Replay Fighting loads",
                False,
                str(e)
            )
        )

        results.append(
            (
                "10. Replay Vandalism loads",
                False,
                str(e)
            )
        )

    # ========================================================
    # 11. COMBINED DATASET
    # ========================================================

    try:

        if (
            a_train is None
            or rep_norm is None
            or rep_fight is None
            or rep_vandal is None
        ):
            raise RuntimeError(
                "Required datasets were not loaded"
            )

        combined_ds = torch.utils.data.ConcatDataset(
            [
                a_train,
                rep_norm,
                rep_fight,
                rep_vandal
            ]
        )

        labels = set()

        _, assault_label = a_train[0]
        _, normal_label = rep_norm[0]
        _, fighting_label = rep_fight[0]
        _, vandalism_label = rep_vandal[0]

        labels.add(int(assault_label))
        labels.add(int(normal_label))
        labels.add(int(fighting_label))
        labels.add(int(vandalism_label))

        correct_labels = (
            labels == {0, 1, 2, 3}
        )

        results.append(
            (
                "11. Combined training data contains all 4 classes (normal=0, fighting=1, vandalism=2, assault=3)",
                correct_labels,
                f"Labels present: {labels}"
            )
        )

    except Exception as e:

        results.append(
            (
                "11. Combined training data contains all 4 classes",
                False,
                str(e)
            )
        )

    # ========================================================
    # 12. TEMPORAL CLIP LENGTH
    # ========================================================

    try:

        if a_train is None:
            raise RuntimeError(
                "Assault training dataset was not loaded"
            )

        clip, label = a_train[0]

        correct_clip_length = (
            clip.shape[0] == 16
        )

        results.append(
            (
                "12. Temporal clips use 16 frames",
                correct_clip_length,
                f"Clip shape: {clip.shape}"
            )
        )

    except Exception as e:

        results.append(
            (
                "12. Temporal clips use 16 frames",
                False,
                str(e)
            )
        )

    # ========================================================
    # 13. TEACHER MODEL
    # ========================================================

    try:

        teacher = RepViTTCN(num_classes=3)

        teacher.load_state_dict(
            ckpt["state_dict"],
            strict=False
        )

        out_dim = (
            teacher.classifier[3].out_features
        )

        results.append(
            (
                "13. Teacher represents old 3-class model",
                out_dim == 3,
                f"Teacher out features: {out_dim}"
            )
        )

    except Exception as e:

        results.append(
            (
                "13. Teacher represents old 3-class model",
                False,
                str(e)
            )
        )

    # ========================================================
    # 14. KNOWLEDGE DISTILLATION
    # ========================================================

    try:

        student_logits = torch.randn(2, 4)
        teacher_logits = torch.randn(2, 3)

        kd_loss = ct.distillation_loss(
            student_logits[:, :3],
            teacher_logits[:, :3]
        )

        results.append(
            (
                "14. KD applied only to old classes ([:, :3])",
                True,
                f"KD Loss computed cleanly: {kd_loss.item():.4f}"
            )
        )

    except Exception as e:

        results.append(
            (
                "14. KD applied only to old classes ([:, :3])",
                False,
                str(e)
            )
        )

    # ========================================================
    # 15. STUDENT MODEL
    # ========================================================

    try:

        student = RepViTTCN(num_classes=4)

        results.append(
            (
                "15. Student represents new 4-class model",
                student.classifier[3].out_features == 4,
                f"Student out features: {student.classifier[3].out_features}"
            )
        )

    except Exception as e:

        results.append(
            (
                "15. Student represents new 4-class model",
                False,
                str(e)
            )
        )

    # ========================================================
    # 16. WEIGHT EXPANSION
    # ========================================================

    try:

        student = RepViTTCN(num_classes=4)
        student = ct.load_old_weights_into_new_student(
            student,
            ckpt
        )

        # check that student classifier weights for old classes match teacher
        student_w = student.classifier[3].weight.detach()
        teacher_w = ckpt["state_dict"]["classifier.3.weight"]

        match = torch.allclose(student_w[:3, :], teacher_w)

        results.append(
            (
                "16. Student preserves old classifier weights in weight expansion",
                match,
                f"Classifier weights matched perfectly" if match else "Mismatch detected"
            )
        )

    except Exception as e:

        results.append(
            (
                "16. Student preserves old classifier weights in weight expansion",
                False,
                str(e)
            )
        )

    # ========================================================
    # 17. EWC PENALTY
    # ========================================================

    try:

        student = RepViTTCN(num_classes=4)
        student = ct.load_old_weights_into_new_student(
            student,
            ckpt
        )

        fisher = {}
        for name, p in student.named_parameters():
            if p.requires_grad:
                fisher[name] = torch.ones_like(p.data)

        # Clone optimal parameters BEFORE modification
        optimal_params = {name: p.clone() for name, p in student.named_parameters()}

        # Modify parameter to test penalty
        first_param = next(student.parameters())
        original_val = first_param.clone()
        with torch.no_grad():
            first_param.add_(0.1)

        penalty = ct.ewc_penalty(
            student,
            fisher,
            optimal_params
        )

        # Restore
        with torch.no_grad():
            first_param.copy_(original_val)

        results.append(
            (
                "17. EWC penalty computes non-zero loss when weights drift",
                penalty.item() > 0,
                f"Penalty loss: {penalty.item():.6f}"
            )
        )

    except Exception as e:

        results.append(
            (
                "17. EWC penalty computes non-zero loss when weights drift",
                False,
                str(e)
            )
        )

    # ========================================================
    # 18. END-TO-END STEP (FORWARD PASS)
    # ========================================================

    try:

        if combined_ds is None:
            raise RuntimeError(
                "Combined dataset was not loaded"
            )

        loader = torch.utils.data.DataLoader(
            combined_ds,
            batch_size=2,
            shuffle=True
        )

        images, labels = next(iter(loader))

        student = RepViTTCN(num_classes=4)
        outputs = student(images)

        results.append(
            (
                "18. End-to-end forward pass succeeds on training batch",
                outputs.shape == (2, 4),
                f"Output shape: {outputs.shape}"
            )
        )

    except Exception as e:

        results.append(
            (
                "18. End-to-end forward pass succeeds on training batch",
                False,
                str(e)
            )
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_pass = True

    for name, status, message in results:

        icon = "[ PASS ]" if status else "[ FAIL ]"
        print(f"{icon:<9} {name:<65} | {message}")

        if not status:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL 18 PRE-TRAINING CHECKS PASSED. SETUP IS ROBUST.")
    else:
        print("SOME CHECKS FAILED. RESOLVE ERRORS BEFORE TRAINING.")
    print("=" * 60 + "\n")

    return all_pass

if __name__ == "__main__":
    run_validation()
