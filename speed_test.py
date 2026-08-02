import time
import torch

from models.vad_model import RepViTTCN

device = torch.device("cpu")

model = RepViTTCN().to(device)
model.eval()

x = torch.randn(1,16,3,224,224).to(device)

# Warm-up
for _ in range(5):
    with torch.no_grad():
        _ = model(x)

runs = 20

start = time.time()

for _ in range(runs):
    with torch.no_grad():
        _ = model(x)

end = time.time()

avg = (end-start)/runs

print(f"Average inference time : {avg:.4f} sec")
print(f"FPS (clips/sec)        : {1/avg:.2f}")