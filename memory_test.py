import torch

from models.vad_model import RepViTTCN

model = RepViTTCN()

total = 0

for p in model.parameters():
    total += p.nelement() * p.element_size()

print("Model Memory:", total/1024**2, "MB")