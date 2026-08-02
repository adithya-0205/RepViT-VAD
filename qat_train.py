import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
import os

from video_dataset import VideoDataset
from models.vad_model import RepViTTCN

def run_qat_training(epochs=1, batch_size=2, lr=1e-4, backend="fbgemm"):
    device = torch.device("cpu") # PyTorch QAT prepare/convert runs on CPU

    print(f"--- Initializing RepViT-M1.0 + TCN QAT Training (Backend: {backend}) ---")
    
    # Instantiate RepViT-M1.0 + TCN Anomaly Model
    model = RepViTTCN(num_classes=1).to(device)

    # Configure Quantization-Aware Training (QAT) for INT8
    model.qconfig = get_default_qat_qconfig(backend)
    model.train()
    
    # Prepare model for QAT (inserts FakeQuantize modules)
    prepared_model = prepare_qat(model, inplace=False)
    print("QAT FakeQuantize modules inserted successfully.")

    # Datasets & Dataloaders
    try:
        train_dataset = VideoDataset(train=True)
        val_dataset = VideoDataset(train=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        has_real_data = True
    except Exception as e:
        print(f"Warning: Could not load real video dataset ({e}). Using synthetic forward pass for QAT check.")
        has_real_data = False

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(prepared_model.parameters(), lr=lr)

    if has_real_data:
        for epoch in range(epochs):
            prepared_model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                outputs = prepared_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_loss = train_loss / len(train_loader)
            print(f"QAT Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
    else:
        # Synthetic batch forward/backward to calibrate fake quantization
        dummy_input = torch.randn(2, 8, 3, 224, 224, device=device)
        dummy_labels = torch.ones(2, 1, device=device)
        optimizer.zero_grad()
        out = prepared_model(dummy_input)
        loss = criterion(out, dummy_labels)
        loss.backward()
        optimizer.step()
        print("Calibrated QAT with synthetic sequence forward pass.")

    # Convert fake-quantized model into actual INT8 quantized model
    prepared_model.eval()
    quantized_model = convert(prepared_model, inplace=False)
    print("Model successfully converted to INT8 quantized format!")

    torch.save(quantized_model.state_dict(), "repvit_m1_0_tcn_int8.pth")
    print("Saved INT8 quantized model checkpoint to 'repvit_m1_0_tcn_int8.pth'.")
    return quantized_model

if __name__ == "__main__":
    run_qat_training(epochs=1)
