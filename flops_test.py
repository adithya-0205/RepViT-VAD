import torch
from thop import profile

from models.vad_model import RepViTTCN

model = RepViTTCN()

x = torch.randn(1,16,3,224,224)

flops, params = profile(model, inputs=(x,), verbose=False)

print("FLOPs :", flops/1e9, "GFLOPs")
print("Params:", params/1e6, "Million")