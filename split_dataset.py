import os
import random
import shutil

random.seed(42)

SRC = "extracted_frames/train"
DST = "dataset"

classes = ["normal", "arson"]

for cls in classes:

    src_folder = os.path.join(SRC, cls)

    if not os.path.exists(src_folder):
        print(f"Folder not found: {src_folder}")
        continue

    videos = sorted(os.listdir(src_folder))
    random.shuffle(videos)

    split = int(0.8 * len(videos))

    train = videos[:split]
    val = videos[split:]

    print(f"{cls}: {len(train)} train, {len(val)} val")

    for subset, vids in [("train", train), ("val", val)]:

        save = os.path.join(DST, subset, cls)
        os.makedirs(save, exist_ok=True)

        for v in vids:

            src = os.path.join(src_folder, v)
            dst = os.path.join(save, v)

            if os.path.exists(dst):
                shutil.rmtree(dst)

            shutil.copytree(src, dst)

print("\nDataset split completed!")