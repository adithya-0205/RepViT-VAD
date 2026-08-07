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

num_classes = len(train_dataset.classes) if hasattr(train_dataset, "classes") and len(train_dataset.classes) > 0 else 3
model = RepViTTCN(num_classes=num_classes).to(device)

# --- LOAD TEAMMATE'S WEIGHTS CONTINUALLY ---
if os.path.exists("best_model.pth"):
    model.load_continual_checkpoint("best_model.pth", device=device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-5  # Lowered LR to preserve previously learned anomaly features
)

best_acc = 0

EPOCHS = 10

for epoch in range(EPOCHS):

    ########################
    # TRAIN
    ########################

    model.train()

    train_loss = 0

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()

        outputs = model(images)

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

            pred = outputs.argmax(dim=1)

            preds.extend(pred.cpu().numpy())

            gts.extend(labels.cpu().numpy())

    acc = accuracy_score(gts, preds)

    print()

    print(f"Epoch {epoch+1}")

    print(f"Train Loss : {train_loss/len(train_loader):.4f}")

    print(f"Validation Accuracy : {acc:.4f}")

    if acc > best_acc:

        best_acc = acc

        checkpoint = {
            "state_dict": model.state_dict(),
            "classes": train_dataset.classes,
            "num_classes": len(train_dataset.classes),
            "epoch": epoch + 1,
            "best_acc": best_acc
        }
        torch.save(checkpoint, "best_model.pth")

        print("Best continual learning model saved successfully!")

print()

print("Training Finished!")

print("Best Accuracy :", best_acc)