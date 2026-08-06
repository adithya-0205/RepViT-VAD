import os
import cv2

# -----------------------------
# SETTINGS
# -----------------------------
FRAME_SKIP = 5          # Save every 5th frame
IMG_SIZE = (224, 224)   # RepViT input size

# Dataset paths
NORMAL_DIR = "data/normal"
ARSON_DIR = "data/arson"

OUTPUT_DIR = "extracted_frames"


# -----------------------------
# Extract frames from one video
# -----------------------------
def extract(video_path, save_folder):
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

            filename = os.path.join(
                save_folder,
                f"frame_{saved:05d}.jpg"
            )

            cv2.imwrite(filename, frame)
            saved += 1

        count += 1

    cap.release()

    print(f"{os.path.basename(video_path)} -> {saved} frames")


# -----------------------------
# Process all videos
# -----------------------------
def process_folder(video_folder, output_folder):

    if not os.path.exists(video_folder):
        print(f"Folder not found: {video_folder}")
        return

    videos = sorted([
        f for f in os.listdir(video_folder)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ])

    print(f"Found {len(videos)} videos in {video_folder}")

    for video in videos:

        name = os.path.splitext(video)[0]

        extract(
            os.path.join(video_folder, video),
            os.path.join(output_folder, name)
        )


print("Processing Arson videos...")
process_folder(
    ARSON_DIR,
    os.path.join(OUTPUT_DIR, "train", "arson")
)

print()

print("Processing Normal videos...")
process_folder(
    NORMAL_DIR,
    os.path.join(OUTPUT_DIR, "train", "normal")
)

print("\nFrame extraction completed!")