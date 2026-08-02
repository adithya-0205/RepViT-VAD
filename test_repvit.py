import torch

from model.repvit import repvit_m0_9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = repvit_m0_9().to(device)
model.eval()

x = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    y = model(x)

print("Output shape:", y.shape)