import torch

from models.tcn import TCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TCN().to(device)

x = torch.randn(2, 16, 384).to(device)

with torch.no_grad():
    y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)