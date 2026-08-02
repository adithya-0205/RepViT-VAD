import os
import random
import shutil

random.seed(42)

SRC = "extracted_frames/train"

DST = "dataset"

classes = ["fighting", "normal"]

for cls in classes:

    videos = os.listdir(os.path.join(SRC, cls))
    random.shuffle(videos)

    split = int(0.8 * len(videos))

    train = videos[:split]
    val = videos[split:]

    for subset, vids in [("train", train), ("val", val)]:

        save = os.path.join(DST, subset, cls)
        os.makedirs(save, exist_ok=True)

        for v in vids:

            src = os.path.join(SRC, cls, v)
            dst = os.path.join(save, v)

            shutil.copytree(src, dst)

print("Dataset split completed.")