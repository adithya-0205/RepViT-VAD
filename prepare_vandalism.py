import os
import io
import zipfile
import random
import argparse
from PIL import Image
import av

FRAME_SKIP = 10
IMG_SIZE = (224, 224)
ZIP_PATH = r"C:\Users\Lenovo\Downloads\Anomaly-Videos-Part-4.zip"
TARGET_DATASET_ROOT = r"D:\RepViT_VAD_Data\vandalism_dataset"

def extract_and_process_video_bytes(video_bytes, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    buf = io.BytesIO(video_bytes)
    container = av.open(buf)
    
    saved = 0
    count = 0
    for frame in container.decode(video=0):
        if count % FRAME_SKIP == 0:
            img = frame.to_image().resize(IMG_SIZE)
            filename = os.path.join(save_folder, f"frame_{saved:05d}.jpg")
            img.save(filename, "JPEG", quality=95)
            saved += 1
        count += 1
        
    container.close()
    return saved

def prepare_vandalism(single_test=False):
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Part 4 zip not found at {ZIP_PATH}")
        
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        all_names = z.namelist()
        vandalism_videos = sorted([
            n for n in all_names
            if n.lower().startswith("anomaly-videos-part-4/vandalism/")
            and n.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ])
        
        print(f"Found {len(vandalism_videos)} Vandalism videos in zip archive.")
        
        if single_test:
            vandalism_videos = vandalism_videos[:1]
            print("Running single-video safety test on:", vandalism_videos[0])
            
        random.seed(42)
        shuffled = list(vandalism_videos)
        random.shuffle(shuffled)
        
        if single_test:
            train_vids = shuffled
            val_vids = []
        else:
            split = int(0.8 * len(shuffled))
            train_vids = shuffled[:split]
            val_vids = shuffled[split:]
            print(f"Split plan: {len(train_vids)} train videos, {len(val_vids)} val videos.")
            
        for subset, vid_list in [("train", train_vids), ("val", val_vids)]:
            for vid_path in vid_list:
                vid_filename = os.path.basename(vid_path)
                vid_name = os.path.splitext(vid_filename)[0]
                
                target_folder = os.path.join(TARGET_DATASET_ROOT, subset, "vandalism", vid_name)
                
                video_bytes = z.read(vid_path)
                saved_frames = extract_and_process_video_bytes(video_bytes, target_folder)
                print(f"  [{subset}] Extracted {saved_frames} frames to {target_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-test", action="store_true", help="Extract only 1 video for safety validation")
    args = parser.parse_args()
    
    prepare_vandalism(single_test=args.single_test)
