import torch

from models.vad_model import RepViTTCN

model = RepViTTCN()

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameters :", total)
print("Trainable :", trainable)
print("Model size (Million):", total / 1e6)