import os
import random
import argparse
import shutil

from PIL import Image
import av


# ============================================================
# Configuration
# ============================================================

SOURCE_DIR = os.path.join(
    "data",
    "Anomaly-Videos-Part-4",
    "Vandalism"
)

TARGET_DATASET_ROOT = os.path.join(
    "data",
    "vandalism_dataset"
)

FRAME_SKIP = 10
IMG_SIZE = (224, 224)

RANDOM_SEED = 42

EXPECTED_TOTAL = 50
TRAIN_COUNT = 40
VAL_COUNT = 10

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)


# ============================================================
# Extract frames from one video
# ============================================================

def extract_and_process_video(
    video_path,
    save_folder
):

    os.makedirs(
        save_folder,
        exist_ok=True
    )

    container = None

    try:

        container = av.open(video_path)

        saved = 0
        count = 0

        for frame in container.decode(video=0):

            if count % FRAME_SKIP == 0:

                img = frame.to_image()

                img = img.resize(
                    IMG_SIZE,
                    Image.Resampling.BILINEAR
                )

                filename = os.path.join(
                    save_folder,
                    f"frame_{saved:05d}.jpg"
                )

                img.save(
                    filename,
                    "JPEG",
                    quality=95
                )

                saved += 1

            count += 1

        return saved

    finally:

        if container is not None:
            container.close()


# ============================================================
# Find Vandalism videos
# ============================================================

def get_vandalism_videos():

    if not os.path.exists(SOURCE_DIR):

        raise FileNotFoundError(
            f"Vandalism source directory not found:\n"
            f"{os.path.abspath(SOURCE_DIR)}"
        )

    videos = []

    for filename in os.listdir(SOURCE_DIR):

        full_path = os.path.join(
            SOURCE_DIR,
            filename
        )

        if not os.path.isfile(full_path):
            continue

        if filename.lower().endswith(
            VIDEO_EXTENSIONS
        ):
            videos.append(full_path)

    return sorted(videos)


# ============================================================
# Prepare Vandalism dataset
# ============================================================

def prepare_vandalism(
    single_test=False,
    clean=False
):

    print("\n==============================================")
    print("VANDALISM DATASET PREPARATION")
    print("==============================================")

    print(
        f"Source: {os.path.abspath(SOURCE_DIR)}"
    )

    print(
        f"Target: {os.path.abspath(TARGET_DATASET_ROOT)}"
    )

    # --------------------------------------------------------
    # Find videos
    # --------------------------------------------------------

    videos = get_vandalism_videos()

    print(
        f"\nFound {len(videos)} Vandalism videos."
    )

    # --------------------------------------------------------
    # Check count
    # --------------------------------------------------------

    if len(videos) != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} Vandalism videos, "
            f"but found {len(videos)}."
        )

    # --------------------------------------------------------
    # Clean existing prepared dataset if requested
    # --------------------------------------------------------

    if clean and os.path.exists(
        TARGET_DATASET_ROOT
    ):

        print(
            "\n[Clean] Removing existing "
            "vandalism_dataset..."
        )

        shutil.rmtree(
            TARGET_DATASET_ROOT
        )

    # --------------------------------------------------------
    # Single-video test
    # --------------------------------------------------------

    if single_test:

        test_video = videos[0]

        video_name = os.path.splitext(
            os.path.basename(test_video)
        )[0]

        target_folder = os.path.join(
            TARGET_DATASET_ROOT,
            "train",
            "vandalism",
            video_name
        )

        print("\n[SINGLE TEST]")
        print(
            f"Video: {os.path.basename(test_video)}"
        )

        saved = extract_and_process_video(
            test_video,
            target_folder
        )

        print(
            f"Frames extracted: {saved}"
        )

        print(
            f"Output: {target_folder}"
        )

        print(
            "\nSingle-video test completed."
        )

        return

    # --------------------------------------------------------
    # Reproducible 40 / 10 split
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    shuffled = list(videos)

    random.shuffle(shuffled)

    train_videos = shuffled[:TRAIN_COUNT]

    val_videos = shuffled[TRAIN_COUNT:]

    print("\nSplit:")
    print(
        f"  Train: {len(train_videos)} videos"
    )
    print(
        f"  Val  : {len(val_videos)} videos"
    )

    if (
        len(train_videos) != 40
        or len(val_videos) != 10
    ):

        raise RuntimeError(
            "Train/validation split is not 40/10."
        )

    # --------------------------------------------------------
    # Process videos
    # --------------------------------------------------------

    total_frames = 0

    for subset, video_list in [
        ("train", train_videos),
        ("val", val_videos)
    ]:

        print(
            f"\n=============================================="
        )

        print(
            f"Processing {subset.upper()} "
            f"({len(video_list)} videos)"
        )

        print(
            f"=============================================="
        )

        for index, video_path in enumerate(
            video_list,
            start=1
        ):

            filename = os.path.basename(
                video_path
            )

            video_name = os.path.splitext(
                filename
            )[0]

            target_folder = os.path.join(
                TARGET_DATASET_ROOT,
                subset,
                "vandalism",
                video_name
            )

            print(
                f"\n[{index}/{len(video_list)}] "
                f"{filename}"
            )

            saved_frames = extract_and_process_video(
                video_path,
                target_folder
            )

            total_frames += saved_frames

            print(
                f"    Frames: {saved_frames}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        "DATASET PREPARATION COMPLETED"
    )

    print(
        "=============================================="
    )

    print(
        f"Total videos : {len(videos)}"
    )

    print(
        f"Train videos : {len(train_videos)}"
    )

    print(
        f"Val videos   : {len(val_videos)}"
    )

    print(
        f"Total frames : {total_frames}"
    )

    print(
        "\nDataset saved to:"
    )

    print(
        os.path.abspath(
            TARGET_DATASET_ROOT
        )
    )

    print(
        "=============================================="
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Prepare extracted Vandalism videos "
            "for continual learning."
        )
    )

    parser.add_argument(
        "--single-test",
        action="store_true",
        help=(
            "Process only the first video "
            "as a safety test."
        )
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete existing prepared dataset "
            "before processing."
        )
    )

    args = parser.parse_args()

    prepare_vandalism(
        single_test=args.single_test,
        clean=args.clean
    )