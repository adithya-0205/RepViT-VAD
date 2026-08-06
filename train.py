import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from video_dataset import VideoDataset
from models.vad_model import RepViTTCN

import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_dataset = VideoDataset(train=True)
val_dataset = VideoDataset(train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)


model = RepViTTCN().to(device)
# --- LOAD TEAMMATE'S WEIGHTS ---
if os.path.exists("best_model_arson_stage1.pth"):
    model.load_state_dict(
        torch.load("best_model_arson_stage1.pth", map_location=device)
    )
    print("Loaded Stage 1 model successfully!")

# Unfreeze the RepViT backbone
for param in model.backbone.parameters():
    param.requires_grad = True

print("RepViT backbone unfrozen. Fine-tuning entire model.")




criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-5
)

best_acc = 0

EPOCHS = 5

for epoch in range(EPOCHS):

    ########################
    # TRAIN
    ########################

    model.train()

    train_loss = 0

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        labels = labels.float().view_as(outputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    ########################
    # VALIDATION
    ########################

    model.eval()

    preds = []
    gts = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            outputs = model(images)

            pred = (outputs > 0.5).long().squeeze(1)

            preds.extend(pred.cpu().numpy())

            gts.extend(labels.cpu().numpy())

    acc = accuracy_score(gts, preds)

    print()

    print(f"Epoch {epoch+1}")

    print(f"Train Loss : {train_loss/len(train_loader):.4f}")

    print(f"Validation Accuracy : {acc:.4f}")

    if acc > best_acc:

        best_acc = acc

        torch.save(model.state_dict(), "best_model_final.pth")

        print("Best model saved!")

print()

print("Training Finished!")

print("Best Accuracy :", best_acc)