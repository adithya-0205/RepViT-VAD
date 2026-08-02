import torch

from models.vad_model import RepViTTCN

model = RepViTTCN()
model.eval()

dummy = torch.randn(1,16,3,224,224)

torch.onnx.export(
    model,
    dummy,
    "repvit_tcn.onnx",
    opset_version=18,
    input_names=["video"],
    output_names=["prediction"]
)

print("ONNX exported successfully.")