import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VideoDataset(Dataset):

    def __init__(self,
                 root="dataset",
                 train=True,
                 clip_len=16):

        random.seed(42)

        self.clip_len = clip_len

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

        videos = []

        # --- FIX: Determine whether to pull from dataset/train or dataset/val ---
        split_folder = "train" if train else "val"

        for label_name, label in [("normal", 0), ("fighting", 1)]:

            # --- FIX: Correct path to dataset/train/normal, dataset/train/fighting, etc. ---
            folder = os.path.join(root, split_folder, label_name)

            if not os.path.exists(folder):
                continue

            names = os.listdir(folder)
            names.sort()

            for n in names:
                video_path = os.path.join(folder, n)
                if os.path.isdir(video_path):
                    videos.append((video_path, label))

        self.samples = videos

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        video_folder, label = self.samples[index]

        frames = sorted(os.listdir(video_folder))

        if len(frames) >= self.clip_len:

            start = random.randint(0, len(frames) - self.clip_len)

            frames = frames[start:start + self.clip_len]

        else:

            frames += [frames[-1]] * (self.clip_len - len(frames))

        imgs = []

        for f in frames:

            img = Image.open(os.path.join(video_folder, f)).convert("RGB")

            imgs.append(self.transform(img))

        imgs = torch.stack(imgs)

        return imgs, label