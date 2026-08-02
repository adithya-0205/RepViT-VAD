import torch
from torchinfo import summary

from models.vad_model import RepViTTCN

device = torch.device("cpu")

model = RepViTTCN().to(device)

summary(
    model,
    input_size=(1,16,3,224,224),
    device="cpu"
)