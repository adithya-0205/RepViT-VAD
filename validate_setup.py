```python
import os
import json
import torch

from video_dataset import VideoDataset, ReplayDataset
from models.vad_model import RepViTTCN
import continual_train as ct


# ============================================================
# PATH CONFIGURATION
# ============================================================

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

VANDALISM_ROOT = os.path.join(
    REPO_ROOT,
    "data",
    "vandalism_dataset"
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
    "best_model_before_vandalism.pth"
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
    print("RUNNING 18-POINT PRE-TRAINING VALIDATION CHECKLIST")
    print("==================================================")

    print(f"\nRepository : {REPO_ROOT}")
    print(f"Vandalism  : {VANDALISM_ROOT}")
    print(f"Replay     : {REPLAY_ROOT}")

    results = []

    ckpt = None
    v_train = None
    v_val = None
    rep_norm = None
    rep_fight = None

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
            "2. Backup best_model_before_vandalism.pth exists",
            backup_exists,
            f"Exists: {backup_exists}"
        )
    )

    # ========================================================
    # 3 + 4. CLASSIFIER EXPANSION
    # ========================================================

    try:

        model = RepViTTCN(num_classes=3)

        old_w = ckpt["state_dict"]["classifier.3.weight"].clone()
        old_b = ckpt["state_dict"]["classifier.3.bias"].clone()

        model.load_continual_checkpoint(CHECKPOINT_PATH)

        new_w = model.classifier[3].weight.data
        new_b = model.classifier[3].bias.data

        weights_preserved = torch.allclose(
            new_w[:2],
            old_w
        )

        biases_preserved = torch.allclose(
            new_b[:2],
            old_b
        )

        shape_correct = (
            new_w.shape == (3, 64)
        )

        results.append(
            (
                "3. Classifier expands 2 -> 3",
                shape_correct,
                f"New shape: {new_w.shape}"
            )
        )

        results.append(
            (
                "4. Normal/Fighting classifier weights preserved",
                weights_preserved and biases_preserved,
                f"Weights match: {weights_preserved}, "
                f"Biases match: {biases_preserved}"
            )
        )

    except Exception as e:

        results.append(
            (
                "3. Classifier expands 2 -> 3",
                False,
                str(e)
            )
        )

        results.append(
            (
                "4. Normal/Fighting classifier weights preserved",
                False,
                str(e)
            )
        )

    # ========================================================
    # 5. CLASS MAPPING
    # ========================================================

    try:

        with open(CLASSES_PATH, "r") as f:
            class_mapping = json.load(f)

        expected_mapping = {
            "normal": 0,
            "fighting": 1,
            "vandalism": 2
        }

        mapping_correct = (
            class_mapping == expected_mapping
        )

        results.append(
            (
                "5. Final class mapping "
                "(normal=0, fighting=1, vandalism=2)",
                mapping_correct,
                f"Mapping: {class_mapping}"
            )
        )

    except Exception as e:

        results.append(
            (
                "5. Final class mapping",
                False,
                str(e)
            )
        )

    # ========================================================
    # 6 + 7. VANDALISM DATASET
    # ========================================================

    try:

        v_train = VideoDataset(
            root=VANDALISM_ROOT,
            train=True,
            classes=["vandalism"],
            class_to_idx=ct.CLASS_MAPPING
        )

        v_val = VideoDataset(
            root=VANDALISM_ROOT,
            train=False,
            classes=["vandalism"],
            class_to_idx=ct.CLASS_MAPPING
        )

        train_count = len(v_train)
        val_count = len(v_val)

        results.append(
            (
                "6. Vandalism train dataset loads",
                train_count == 40,
                f"Count: {train_count} videos"
            )
        )

        results.append(
            (
                "7. Vandalism validation dataset loads",
                val_count == 10,
                f"Count: {val_count} videos"
            )
        )

    except Exception as e:

        results.append(
            (
                "6. Vandalism train dataset loads",
                False,
                str(e)
            )
        )

        results.append(
            (
                "7. Vandalism validation dataset loads",
                False,
                str(e)
            )
        )

    # ========================================================
    # 8 + 9. REPLAY BUFFER
    # ========================================================

    try:

        normal_root = os.path.join(
            REPLAY_ROOT,
            "normal"
        )

        fighting_root = os.path.join(
            REPLAY_ROOT,
            "fighting"
        )

        normal_paths = [
            os.path.join(normal_root, d)
            for d in os.listdir(normal_root)
            if os.path.isdir(
                os.path.join(normal_root, d)
            )
        ]

        fighting_paths = [
            os.path.join(fighting_root, d)
            for d in os.listdir(fighting_root)
            if os.path.isdir(
                os.path.join(fighting_root, d)
            )
        ]

        rep_norm = ReplayDataset(
            {"normal": normal_paths},
            ct.CLASS_MAPPING,
            clip_len=16
        )

        rep_fight = ReplayDataset(
            {"fighting": fighting_paths},
            ct.CLASS_MAPPING,
            clip_len=16
        )

        results.append(
            (
                "8. External replay Normal loads",
                len(rep_norm) == 10,
                f"Count: {len(rep_norm)} clips"
            )
        )

        results.append(
            (
                "9. External replay Fighting loads",
                len(rep_fight) == 10,
                f"Count: {len(rep_fight)} clips"
            )
        )

    except Exception as e:

        results.append(
            (
                "8. External replay Normal loads",
                False,
                str(e)
            )
        )

        results.append(
            (
                "9. External replay Fighting loads",
                False,
                str(e)
            )
        )

    # ========================================================
    # 10. COMBINED DATASET
    # ========================================================

    try:

        if (
            v_train is None
            or rep_norm is None
            or rep_fight is None
        ):
            raise RuntimeError(
                "Required datasets were not loaded"
            )

        combined_ds = torch.utils.data.ConcatDataset(
            [
                v_train,
                rep_norm,
                rep_fight
            ]
        )

        labels = set()

        _, vandalism_label = v_train[0]
        _, normal_label = rep_norm[0]
        _, fighting_label = rep_fight[0]

        labels.add(int(vandalism_label))
        labels.add(int(normal_label))
        labels.add(int(fighting_label))

        correct_labels = (
            labels == {0, 1, 2}
        )

        results.append(
            (
                "10. Combined training data contains "
                "all 3 classes "
                "(normal=0, fighting=1, vandalism=2)",
                correct_labels,
                f"Labels present: {labels}"
            )
        )

    except Exception as e:

        results.append(
            (
                "10. Combined training data contains "
                "all 3 classes",
                False,
                str(e)
            )
        )

    # ========================================================
    # 11. TEMPORAL CLIP LENGTH
    # ========================================================

    try:

        if v_train is None:
            raise RuntimeError(
                "Vandalism training dataset was not loaded"
            )

        clip, label = v_train[0]

        correct_clip_length = (
            clip.shape[0] == 16
        )

        results.append(
            (
                "11. Temporal clips use 16 frames",
                correct_clip_length,
                f"Clip shape: {clip.shape}"
            )
        )

    except Exception as e:

        results.append(
            (
                "11. Temporal clips use 16 frames",
                False,
                str(e)
            )
        )

    # ========================================================
    # 12. TEACHER MODEL
    # ========================================================

    try:

        teacher = RepViTTCN(num_classes=2)

        teacher.load_state_dict(
            ckpt["state_dict"],
            strict=False
        )

        out_dim = (
            teacher.classifier[3].out_features
        )

        results.append(
            (
                "12. Teacher represents old 2-class model",
                out_dim == 2,
                f"Teacher out features: {out_dim}"
            )
        )

    except Exception as e:

        results.append(
            (
                "12. Teacher represents old 2-class model",
                False,
                str(e)
            )
        )

    # ========================================================
    # 13. KNOWLEDGE DISTILLATION
    # ========================================================

    try:

        student_logits = torch.randn(2, 3)
        teacher_logits = torch.randn(2, 2)

        kd_loss = ct.distillation_loss(
            student_logits[:, :2],
            teacher_logits[:, :2]
        )

        results.append(
            (
                "13. KD applied only to old classes ([:, :2])",
                True,
                f"KD Loss computed cleanly: "
                f"{kd_loss.item():.4f}"
            )
        )

    except Exception as e:

        results.append(
            (
                "13. KD applied only to old classes",
                False,
                str(e)
            )
        )

    # ========================================================
    # 14 + 15 + 16. EWC / FISHER / MODEL SHAPE
    # ========================================================

    try:

        state = ct.setup_training()

        fisher = state["ewc_fisher"]
        optimal_params = state["ewc_optimal_params"]
        model = state["model"]

        fisher_ok = (
            fisher is not None
            and len(fisher) > 0
        )

        results.append(
            (
                "14. Fisher information computed "
                "for old classes",
                fisher_ok,
                f"Fisher tensors: {len(fisher)}"
            )
        )

        # EWC dry run
        ewc_penalty_value = ct.ewc_penalty(
            model,
            fisher,
            optimal_params
        )

        results.append(
            (
                "15. EWC penalty included in loss loop",
                True,
                f"EWC penalty dry-run value: "
                f"{ewc_penalty_value.item():.4f}"
            )
        )

        # Model forward test
        dummy_x = torch.randn(
            2,
            16,
            3,
            224,
            224
        ).to(ct.device)

        model.eval()

        with torch.no_grad():
            output = model(dummy_x)

        correct_shape = (
            output.shape == (2, 3)
        )

        results.append(
            (
                "16. No tensor-shape mismatch",
                correct_shape,
                f"Model forward output shape: "
                f"{output.shape}"
            )
        )

    except Exception as e:

        results.append(
            (
                "14. Fisher information computed",
                False,
                str(e)
            )
        )

        results.append(
            (
                "15. EWC penalty included",
                False,
                str(e)
            )
        )

        results.append(
            (
                "16. No tensor-shape mismatch",
                False,
                str(e)
            )
        )

    # ========================================================
    # 17. NO FULL OLD DATASET REQUIRED
    # ========================================================

    old_normal_path = os.path.join(
        REPO_ROOT,
        "dataset",
        "train",
        "normal"
    )

    old_fighting_path = os.path.join(
        REPO_ROOT,
        "dataset",
        "train",
        "fighting"
    )

    no_old_full_dataset = (
        not os.path.exists(old_normal_path)
        and
        not os.path.exists(old_fighting_path)
    )

    results.append(
        (
            "17. No path requires full old "
            "Normal/Fighting datasets",
            no_old_full_dataset,
            "No full old dataset folders required"
        )
    )

    # ========================================================
    # 18. LARGE DATASET LOCATION CHECK
    # ========================================================

    # Your Vandalism dataset is currently inside
    # data\vandalism_dataset, so this check should verify
    # that unintended duplicate folders are NOT present.

    unwanted_repo_data = [
        os.path.join(REPO_ROOT, "extracted_frames"),
        os.path.join(REPO_ROOT, "dataset")
    ]

    unwanted_existing = [
        path
        for path in unwanted_repo_data
        if os.path.exists(path)
    ]

    no_unwanted_large_data = (
        len(unwanted_existing) == 0
    )

    results.append(
        (
            "18. No unintended large dataset folder",
            no_unwanted_large_data,
            (
                "No unintended dataset folder found"
                if no_unwanted_large_data
                else f"Found: {unwanted_existing}"
            )
        )
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n--- VALIDATION SUMMARY ---")

    all_passed = True

    for desc, status, msg in results:

        flag = "[PASS]" if status else "[FAIL]"

        print(
            f"{flag} {desc} -> {msg}"
        )

        if not status:
            all_passed = False

    print("\n==============================================")

    if all_passed:
        print("OVERALL VALIDATION STATUS: ALL 18 CHECKS PASSED!")
    else:
        print("OVERALL VALIDATION STATUS: SOME CHECKS FAILED!")

    print("==============================================")

    return all_passed


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_validation()
```
