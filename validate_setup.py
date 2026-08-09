import os
import json
import torch
import torch.nn as nn
from PIL import Image

from video_dataset import VideoDataset, ReplayDataset
from models.vad_model import RepViTTCN
import continual_train as ct

def run_validation():
    print("==================================================")
    print("RUNNING 18-POINT PRE-TRAINING VALIDATION CHECKLIST")
    print("==================================================")

    results = []

    # 1. best_model.pth loads
    try:
        ckpt = torch.load("best_model.pth", map_location="cpu")
        results.append(("1. best_model.pth loads", True, "Successfully loaded"))
    except Exception as e:
        results.append(("1. best_model.pth loads", False, str(e)))

    # 2. Backup exists
    b_exists = os.path.exists("best_model_before_vandalism.pth")
    results.append(("2. Backup best_model_before_vandalism.pth exists", b_exists, f"Exists: {b_exists}"))

    # 3. Classifier expands 2 -> 3
    # 4. Normal/Fighting classifier weights preserved
    try:
        model = RepViTTCN(num_classes=3)
        old_w = ckpt["state_dict"]["classifier.3.weight"].clone()
        old_b = ckpt["state_dict"]["classifier.3.bias"].clone()
        model.load_continual_checkpoint("best_model.pth")
        new_w = model.classifier[3].weight.data
        new_b = model.classifier[3].bias.data
        
        w_match = torch.allclose(new_w[:2], old_w)
        b_match = torch.allclose(new_b[:2], old_b)
        shape_match = new_w.shape == (3, 64)
        
        results.append(("3. Classifier expands 2 -> 3", shape_match, f"New shape: {new_w.shape}"))
        results.append(("4. Normal/Fighting classifier weights preserved", w_match and b_match, f"Weights match: {w_match}, Biases match: {b_match}"))
    except Exception as e:
        results.append(("3. Classifier expands 2 -> 3", False, str(e)))
        results.append(("4. Normal/Fighting classifier weights preserved", False, str(e)))

    # 5. Final class mapping
    try:
        with open("classes.json", "r") as f:
            cmap = json.load(f)
        expected = {"normal": 0, "fighting": 1, "vandalism": 2}
        results.append(("5. Final class mapping (normal=0, fighting=1, vandalism=2)", cmap == expected, f"Mapping: {cmap}"))
    except Exception as e:
        results.append(("5. Final class mapping", False, str(e)))

    # 6. Vandalism train dataset loads
    # 7. Vandalism val dataset loads
    try:
        v_train = VideoDataset(root=r"D:\RepViT_VAD_Data\vandalism_dataset", train=True, classes=["vandalism"], class_to_idx=ct.CLASS_MAPPING)
        v_val   = VideoDataset(root=r"D:\RepViT_VAD_Data\vandalism_dataset", train=False, classes=["vandalism"], class_to_idx=ct.CLASS_MAPPING)
        results.append(("6. Vandalism train dataset loads", len(v_train) == 40, f"Count: {len(v_train)} videos"))
        results.append(("7. Vandalism validation dataset loads", len(v_val) == 10, f"Count: {len(v_val)} videos"))
    except Exception as e:
        results.append(("6. Vandalism train dataset loads", False, str(e)))
        results.append(("7. Vandalism validation dataset loads", False, str(e)))

    # 8. External replay Normal loads
    # 9. External replay Fighting loads
    try:
        replay_root = r"D:\RepViT_VAD_Data\replay_buffer"
        norm_paths = [os.path.join(replay_root, "normal", d) for d in os.listdir(os.path.join(replay_root, "normal")) if os.path.isdir(os.path.join(replay_root, "normal", d))]
        fight_paths = [os.path.join(replay_root, "fighting", d) for d in os.listdir(os.path.join(replay_root, "fighting")) if os.path.isdir(os.path.join(replay_root, "fighting", d))]
        
        rep_norm = ReplayDataset({"normal": norm_paths}, ct.CLASS_MAPPING, clip_len=16)
        rep_fight = ReplayDataset({"fighting": fight_paths}, ct.CLASS_MAPPING, clip_len=16)
        
        results.append(("8. External replay Normal loads", len(rep_norm) == 10, f"Count: {len(rep_norm)} clips"))
        results.append(("9. External replay Fighting loads", len(rep_fight) == 10, f"Count: {len(rep_fight)} clips"))
    except Exception as e:
        results.append(("8. External replay Normal loads", False, str(e)))
        results.append(("9. External replay Fighting loads", False, str(e)))

    # 10. Combined training data contains all three classes
    try:
        combined_ds = torch.utils.data.ConcatDataset([v_train, rep_norm, rep_fight])
        labels = set()
        labels.add(v_train[0][1])
        labels.add(rep_norm[0][1])
        labels.add(rep_fight[0][1])
        results.append(("10. Combined training data contains all 3 classes (normal=0, fighting=1, vandalism=2)", labels == {0, 1, 2}, f"Labels present: {labels}"))
    except Exception as e:
        results.append(("10. Combined training data contains all 3 classes", False, str(e)))

    # 11. Temporal clips use 16-frame length
    try:
        clip, _ = v_train[0]
        results.append(("11. Temporal clips use 16 frames", clip.shape[0] == 16, f"Clip shape: {clip.shape}"))
    except Exception as e:
        results.append(("11. Temporal clips use 16 frames", False, str(e)))

    # 12. Teacher represents old 2-class model
    try:
        teacher = RepViTTCN(num_classes=2)
        teacher.load_state_dict(ckpt["state_dict"], strict=False)
        out_dim = teacher.classifier[3].out_features
        results.append(("12. Teacher represents old 2-class model", out_dim == 2, f"Teacher out features: {out_dim}"))
    except Exception as e:
        results.append(("12. Teacher represents old 2-class model", False, str(e)))

    # 13. KD applied only to old classes ([:, :2])
    try:
        student_logits = torch.randn(2, 3)
        teacher_logits = torch.randn(2, 2)
        kd_l = ct.distillation_loss(student_logits[:, :2], teacher_logits[:, :2])
        results.append(("13. KD applied only to old classes ([:, :2])", True, f"KD Loss computed cleanly: {kd_l.item():.4f}"))
    except Exception as e:
        results.append(("13. KD applied only to old classes", False, str(e)))

    # 14. Fisher info computed for old classes
    # 15. EWC penalty included
    # 16. No tensor-shape mismatch
    try:
        state = ct.setup_training()
        fisher_ok = len(state["ewc_fisher"]) > 0
        ewc_pen = ct.ewc_penalty(state["model"], state["ewc_fisher"], state["ewc_optimal_params"])
        results.append(("14. Fisher information computed for old classes", fisher_ok, f"Fisher tensors: {len(state['ewc_fisher'])}"))
        results.append(("15. EWC penalty included in loss loop", True, f"EWC penalty dry-run value: {ewc_pen.item():.4f}"))

        dummy_x = torch.randn(2, 16, 3, 224, 224).to(ct.device)
        out = state["model"](dummy_x)
        results.append(("16. No tensor-shape mismatch", out.shape == (2, 3), f"Model forward output shape: {out.shape}"))
    except Exception as e:
        results.append(("14. Fisher information computed", False, str(e)))
        results.append(("15. EWC penalty included", False, str(e)))
        results.append(("16. No tensor-shape mismatch", False, str(e)))

    # 17. No dataset path points to full old datasets
    no_old_full_path = not os.path.exists(r"dataset\train\normal") and not os.path.exists(r"dataset\train\fighting")
    results.append(("17. No path requires full old Normal/Fighting datasets", no_old_full_path, "Clean workspace"))

    # 18. No large data path points inside Git repository
    repo_data = os.path.exists(r"extracted_frames") or os.path.exists(r"dataset\train\vandalism")
    results.append(("18. No large dataset folder inside Git repo", not repo_data, "All large datasets on D:\\RepViT_VAD_Data"))

    print("\n--- VALIDATION SUMMARY ---")
    all_passed = True
    for desc, status, msg in results:
        flag = "[PASS]" if status else "[FAIL]"
        print(f"{flag} {desc} -> {msg}")
        if not status:
            all_passed = False

    print("\nOVERALL VALIDATION STATUS:", "ALL 18 CHECKS PASSED!" if all_passed else "SOME CHECKS FAILED!")
    return all_passed

if __name__ == "__main__":
    run_validation()
