import os
import cv2
import random
import shutil

FRAME_SKIP = 10         # Save every 10th frame for efficiency
IMG_SIZE = (224, 224)

ASSAULT_SRC = "Assault"
EXTRACTED_DIR = "extracted_frames/train/assault"
DATASET_TRAIN = "dataset/train/assault"
DATASET_VAL = "dataset/val/assault"

def extract_video_frames(video_path, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % FRAME_SKIP == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            filename = os.path.join(save_folder, f"frame_{saved:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1
    cap.release()
    return saved

if __name__ == "__main__":
    if not os.path.exists(ASSAULT_SRC):
        print(f"Error: {ASSAULT_SRC} does not exist.")
        exit(1)

    videos = sorted([
        f for f in os.listdir(ASSAULT_SRC)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ])

    print(f"Found {len(videos)} Assault videos. Extracting frames for a training subset...")

    # Process first 10 videos for quick demonstration & fast training
    selected_videos = videos[:10]

    for v in selected_videos:
        v_name = os.path.splitext(v)[0]
        v_path = os.path.join(ASSAULT_SRC, v)
        out_path = os.path.join(EXTRACTED_DIR, v_name)
        saved = extract_video_frames(v_path, out_path)
        print(f"Extracted {saved} frames from {v}")

    # Split into train / val
    random.seed(42)
    extracted_vids = sorted(os.listdir(EXTRACTED_DIR))
    random.shuffle(extracted_vids)

    split = int(0.8 * len(extracted_vids))
    train_vids = extracted_vids[:split]
    val_vids = extracted_vids[split:]

    for subset, vids in [("train", train_vids), ("val", val_vids)]:
        target_dir = DATASET_TRAIN if subset == "train" else DATASET_VAL
        os.makedirs(target_dir, exist_ok=True)
        for v in vids:
            src = os.path.join(EXTRACTED_DIR, v)
            dst = os.path.join(target_dir, v)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    print(f"\nSuccessfully prepared Assault dataset: {len(train_vids)} train videos, {len(val_vids)} val videos!")
