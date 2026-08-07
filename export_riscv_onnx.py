import torch
import torch.nn as nn
import os
import sys
import argparse

# Ensure standard output can print Unicode characters without crashing on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from models.vad_model import RepViTTCN

def export_to_riscv_onnx(output_onnx_path="repvit_m1_0_tcn_riscv.onnx", static_shape=False, num_classes=3):
    print("--- Exporting RepViT-M1.0 + TCN Anomaly Model to ONNX for RISC-V Accelerator ---")
    
    device = torch.device("cpu")
    model = RepViTTCN(num_classes=num_classes).to(device)
    model.eval()

    # Load weights if available and matching shape
    checkpoint_path = "repvit_m1_0_tcn_int8.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "best_model.pth"

    if os.path.exists(checkpoint_path):
        print(f"Checking checkpoint weights from '{checkpoint_path}'...")
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded model weights.")
        except Exception as e:
            print("Exporting structure with initialized weights.")
    else:
        print("No existing checkpoint found. Exporting model architecture with initialized weights.")

    # Create example dummy input (Batch=1, Frames=16, Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 16, 3, 224, 224, device=device)

    # Dynamic axes configuration (if static_shape=False)
    dynamic_axes = None if static_shape else {
        "video_input": {0: "batch_size", 1: "num_frames"},
        "class_logits": {0: "batch_size"}
    }

    print(f"Exporting to '{output_onnx_path}' (Static Shapes: {static_shape})...")
    
    # Export using legacy PyTorch ONNX exporter (dynamo=False) for maximum NPU/RISC-V toolchain compatibility
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["video_input"],
            output_names=["class_logits"],
            dynamic_axes=dynamic_axes,
            dynamo=False
        )
    except TypeError:
        # Fallback if dynamo argument is not supported in current torch version
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["video_input"],
            output_names=["class_logits"],
            dynamic_axes=dynamic_axes
        )

    print(f"Successfully exported ONNX model to: '{output_onnx_path}'")

    # Verify ONNX model if onnx package is available
    try:
        import onnx
        onnx_model = onnx.load(output_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX Model Checker: Graph is valid and well-formed!")
    except ImportError:
        print("Install 'onnx' library to perform graph structure verification.")
    except Exception as e:
        print(f"ONNX Checker notice: {e}")

    # Test ONNX Runtime execution if available
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(output_onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        print(f"ONNX Runtime Verification: Output shape = {ort_outputs[0].shape}, Output score = {ort_outputs[0]}")
    except ImportError:
        print("Install 'onnxruntime' library for ONNX Runtime verification.")
    except Exception as e:
        print(f"ONNX Runtime Execution notice: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RepViT-M1.0 + TCN model to ONNX for RISC-V deployment")
    parser.add_argument("--output", type=str, default="repvit_m1_0_tcn_riscv.onnx", help="Output ONNX filename")
    parser.add_argument("--static", action="store_true", help="Enforce static input shapes for hardware NPU compilers")
    args = parser.parse_args()

    export_to_riscv_onnx(output_onnx_path=args.output, static_shape=args.static)

