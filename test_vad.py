import torch

from models.vad_model import RepViTTCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RepViTTCN().to(device)

# Batch=2, Frames=16
x = torch.randn(2, 16, 3, 224, 224).to(device)

with torch.no_grad():
    y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)