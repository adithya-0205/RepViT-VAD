from pathlib import Path
import shutil

ROOT = Path("replay_buffer")

TRAIN_COUNT = 8

for class_name in ["normal", "fighting"]:
    source = ROOT / class_name
    train = ROOT / "train" / class_name
    val = ROOT / "val" / class_name

    train.mkdir(parents=True, exist_ok=True)
    val.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        [x for x in source.iterdir() if x.is_dir()]
    )

    print(f"\n{class_name.upper()}: found {len(videos)} videos")

    for i, video in enumerate(videos):
        if i < TRAIN_COUNT:
            destination = train / video.name
        else:
            destination = val / video.name

        if destination.exists():
            print(f"Already exists: {destination}")
            continue

        shutil.move(str(video), str(destination))
        print(f"Moved: {video.name} -> {destination.parent}")

print("\nReplay buffer split completed.")