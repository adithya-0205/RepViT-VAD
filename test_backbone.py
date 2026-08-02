import torch
from models.repvit_backbone import RepViTBackbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RepViTBackbone().to(device)
model.eval()

x = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    y = model(x)

print("Feature shape:", y.shape)