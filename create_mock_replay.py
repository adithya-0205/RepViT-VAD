import os
import cv2
import numpy as np

def create_mock_class(root_dir, class_name, num_clips, frames_per_clip=16):
    class_path = os.path.join(root_dir, class_name)
    os.makedirs(class_path, exist_ok=True)
    print(f"Creating mock replay for {class_name} in {class_path}...")
    for i in range(num_clips):
        clip_name = f"mock_clip_{i:03d}"
        clip_path = os.path.join(class_path, clip_name)
        os.makedirs(clip_path, exist_ok=True)
        # Create solid gray frames
        for f in range(frames_per_clip):
            frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
            frame_path = os.path.join(clip_path, f"frame_{f:05d}.jpg")
            cv2.imwrite(frame_path, frame)

def extract_vandalism_replay(vandalism_src, target_root, num_train=30, num_val=10, frames_per_clip=16, frame_skip=10):
    if not os.path.exists(vandalism_src):
        print(f"Vandalism source not found: {vandalism_src}. Creating mock instead.")
        create_mock_class(os.path.join(target_root, "train"), "vandalism", num_train)
        create_mock_class(os.path.join(target_root, "val"), "vandalism", num_val)
        return

    videos = sorted([
        f for f in os.listdir(vandalism_src)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ])
    
    print(f"Found {len(videos)} Vandalism videos for replay extraction.")
    
    # Train replay extraction
    train_dest = os.path.join(target_root, "train", "vandalism")
    os.makedirs(train_dest, exist_ok=True)
    for i in range(min(num_train, len(videos))):
        v_path = os.path.join(vandalism_src, videos[i])
        clip_dest = os.path.join(train_dest, f"clip_{i:03d}")
        os.makedirs(clip_dest, exist_ok=True)
        
        cap = cv2.VideoCapture(v_path)
        saved = 0
        count = 0
        while saved < frames_per_clip:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_skip == 0:
                frame = cv2.resize(frame, (224, 224))
                cv2.imwrite(os.path.join(clip_dest, f"frame_{saved:05d}.jpg"), frame)
                saved += 1
            count += 1
        cap.release()
        
        # If video ended too early, repeat the last frame
        if saved > 0:
            last_frame_path = os.path.join(clip_dest, f"frame_{saved-1:05d}.jpg")
            last_frame = cv2.imread(last_frame_path)
            while saved < frames_per_clip:
                cv2.imwrite(os.path.join(clip_dest, f"frame_{saved:05d}.jpg"), last_frame)
                saved += 1
        else:
            # If no frame could be read, write mock
            for f in range(frames_per_clip):
                frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
                cv2.imwrite(os.path.join(clip_dest, f"frame_{f:05d}.jpg"), frame)
        print(f"Extracted vandalism train replay clip {i} from {videos[i]}")

    # Val replay extraction
    val_dest = os.path.join(target_root, "val", "vandalism")
    os.makedirs(val_dest, exist_ok=True)
    offset = num_train
    for i in range(num_val):
        v_idx = (offset + i) % len(videos)
        v_path = os.path.join(vandalism_src, videos[v_idx])
        clip_dest = os.path.join(val_dest, f"clip_{i:03d}")
        os.makedirs(clip_dest, exist_ok=True)
        
        cap = cv2.VideoCapture(v_path)
        saved = 0
        count = 0
        while saved < frames_per_clip:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_skip == 0:
                frame = cv2.resize(frame, (224, 224))
                cv2.imwrite(os.path.join(clip_dest, f"frame_{saved:05d}.jpg"), frame)
                saved += 1
            count += 1
        cap.release()
        
        if saved > 0:
            last_frame_path = os.path.join(clip_dest, f"frame_{saved-1:05d}.jpg")
            last_frame = cv2.imread(last_frame_path)
            while saved < frames_per_clip:
                cv2.imwrite(os.path.join(clip_dest, f"frame_{saved:05d}.jpg"), last_frame)
                saved += 1
        else:
            for f in range(frames_per_clip):
                frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
                cv2.imwrite(os.path.join(clip_dest, f"frame_{f:05d}.jpg"), frame)
        print(f"Extracted vandalism val replay clip {i} from {videos[v_idx]}")

if __name__ == "__main__":
    replay_root = "replay_buffer"
    train_root = os.path.join(replay_root, "train")
    val_root = os.path.join(replay_root, "val")
    
    # Create normal and fighting mock replay (since videos are not available)
    create_mock_class(train_root, "normal", num_clips=30)
    create_mock_class(train_root, "fighting", num_clips=30)
    create_mock_class(val_root, "normal", num_clips=10)
    create_mock_class(val_root, "fighting", num_clips=10)
    
    # Extract real vandalism replay (since we have vandalism videos)
    vandalism_src = r"c:\Users\USER\Downloads\Vandalism"
    extract_vandalism_replay(vandalism_src, replay_root, num_train=30, num_val=10)
    
    print("\nReplay buffer setup complete!")
