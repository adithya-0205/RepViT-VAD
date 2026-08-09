import os
import random
from typing import Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CLIP_LEN = 16
IMAGE_SIZE = 224

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def is_image_file(path: str) -> bool:
    return (
        os.path.isfile(path)
        and os.path.splitext(path)[1].lower()
        in IMAGE_EXTENSIONS
    )


def is_video_file(path: str) -> bool:
    return (
        os.path.isfile(path)
        and os.path.splitext(path)[1].lower()
        in VIDEO_EXTENSIONS
    )


def sorted_files(directory: str) -> List[str]:
    files = []

    if not os.path.isdir(directory):
        return files

    for name in os.listdir(directory):
        path = os.path.join(directory, name)

        if is_image_file(path):
            files.append(path)

    return sorted(files)


def resize_frame(frame):
    frame = cv2.resize(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    tensor = torch.from_numpy(
        frame.copy()
    ).float() / 255.0

    tensor = tensor.permute(
        2, 0, 1
    )

    return tensor


def read_image_frames(
    folder: str,
    clip_len: int,
    random_clip: bool,
):
    frame_paths = sorted_files(folder)

    if len(frame_paths) == 0:
        raise RuntimeError(
            f"No image frames found in:\n{folder}"
        )

    total = len(frame_paths)

    # --------------------------------------------------------
    # Select frames
    # --------------------------------------------------------

    if total >= clip_len:

        if random_clip:
            start = random.randint(
                0,
                total - clip_len,
            )
        else:
            start = (
                total - clip_len
            ) // 2

        selected = frame_paths[
            start:start + clip_len
        ]

    else:

        selected = frame_paths[:]

        # Repeat last frame
        while len(selected) < clip_len:
            selected.append(
                selected[-1]
            )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    frames = []

    for path in selected:

        frame = cv2.imread(path)

        if frame is None:
            raise RuntimeError(
                f"Could not read frame:\n{path}"
            )

        frames.append(
            resize_frame(frame)
        )

    return torch.stack(frames)


def read_video_frames(
    video_path: str,
    clip_len: int,
    random_clip: bool,
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:
        cap.release()

        raise RuntimeError(
            f"Video contains no frames:\n"
            f"{video_path}"
        )

    # --------------------------------------------------------
    # Determine frame indices
    # --------------------------------------------------------

    if total_frames >= clip_len:

        if random_clip:

            start = random.randint(
                0,
                total_frames - clip_len,
            )

        else:

            start = (
                total_frames - clip_len
            ) // 2

        indices = list(
            range(
                start,
                start + clip_len,
            )
        )

    else:

        indices = list(
            range(total_frames)
        )

        while len(indices) < clip_len:
            indices.append(
                indices[-1]
            )

    # --------------------------------------------------------
    # Read frames
    # --------------------------------------------------------

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            index,
        )

        success, frame = cap.read()

        if not success:
            cap.release()

            raise RuntimeError(
                f"Could not read frame "
                f"{index} from:\n"
                f"{video_path}"
            )

        frames.append(
            resize_frame(frame)
        )

    cap.release()

    return torch.stack(frames)


# ============================================================
# FIND CLIP SOURCES
# ============================================================

def find_clip_sources(
    root: str,
) -> List[str]:

    sources = []

    if not os.path.isdir(root):
        return sources

    for item in sorted(
        os.listdir(root)
    ):

        path = os.path.join(
            root,
            item,
        )

        # ----------------------------------------------------
        # A directory containing image frames
        # ----------------------------------------------------

        if os.path.isdir(path):

            image_files = sorted_files(
                path
            )

            if image_files:
                sources.append(path)
                continue

            # ------------------------------------------------
            # Directory containing video files
            # ------------------------------------------------

            video_files = [
                os.path.join(
                    path,
                    name,
                )
                for name in os.listdir(path)
                if is_video_file(
                    os.path.join(
                        path,
                        name,
                    )
                )
            ]

            sources.extend(
                sorted(video_files)
            )

            continue

        # ----------------------------------------------------
        # Direct video file
        # ----------------------------------------------------

        if is_video_file(path):
            sources.append(path)

    return sources


# ============================================================
# VIDEO DATASET
# ============================================================

class VideoDataset(Dataset):

    def __init__(
        self,
        root: str,
        train: bool = True,
        classes=None,
        class_to_idx=None,
        clip_len: int = DEFAULT_CLIP_LEN,
    ):

        self.root = root
        self.train = train
        self.clip_len = clip_len

        split = (
            "train"
            if train
            else "val"
        )

        self.split_root = os.path.join(
            root,
            split,
        )

        if not os.path.isdir(
            self.split_root
        ):

            raise FileNotFoundError(
                f"Dataset split not found:\n"
                f"{self.split_root}"
            )

        # ----------------------------------------------------
        # Classes
        # ----------------------------------------------------

        if classes is None:

            classes = sorted(
                [
                    name
                    for name in os.listdir(
                        self.split_root
                    )
                    if os.path.isdir(
                        os.path.join(
                            self.split_root,
                            name,
                        )
                    )
                ]
            )

        self.classes = list(classes)

        if class_to_idx is None:

            self.class_to_idx = {
                name: index
                for index, name
                in enumerate(
                    self.classes
                )
            }

        else:

            self.class_to_idx = dict(
                class_to_idx
            )

        # ----------------------------------------------------
        # Samples
        # ----------------------------------------------------

        self.samples = []

        for class_name in self.classes:

            class_root = os.path.join(
                self.split_root,
                class_name,
            )

            if not os.path.isdir(
                class_root
            ):
                continue

            label = self.class_to_idx[
                class_name
            ]

            sources = find_clip_sources(
                class_root
            )

            for source in sources:

                self.samples.append(
                    (
                        source,
                        label,
                    )
                )

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No samples found in:\n"
                f"{self.split_root}"
            )

        mode = (
            "TRAIN"
            if train
            else "VAL"
        )

        print(
            f"[VideoDataset] {mode}"
        )

        print(
            f"Classes : {self.classes}"
        )

        print(
            f"Samples : {len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self,
        index,
    ):

        source, label = (
            self.samples[index]
        )

        if os.path.isdir(source):

            frames = read_image_frames(
                source,
                self.clip_len,
                random_clip=self.train,
            )

        else:

            frames = read_video_frames(
                source,
                self.clip_len,
                random_clip=self.train,
            )

        return (
            frames,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )


# ============================================================
# REPLAY DATASET
# ============================================================

class ReplayDataset(Dataset):

    def __init__(
        self,
        root: str,
        class_to_idx: Dict[str, int],
        classes=None,
        clip_len: int = DEFAULT_CLIP_LEN,
        random_clip: bool = True,
    ):

        self.root = root
        self.class_to_idx = dict(
            class_to_idx
        )
        self.clip_len = clip_len
        self.random_clip = random_clip

        if classes is None:

            classes = [
                "normal",
                "fighting",
            ]

        self.classes = list(classes)

        self.samples = []

        # ----------------------------------------------------
        # Expected:
        #
        # replay_buffer/
        #     train/
        #         normal/
        #         fighting/
        #
        # OR
        #
        # replay_buffer/
        #     val/
        #         normal/
        #         fighting/
        # ----------------------------------------------------

        for class_name in self.classes:

            class_root = os.path.join(
                root,
                class_name,
            )

            if not os.path.isdir(
                class_root
            ):

                print(
                    f"[ReplayDataset] "
                    f"Warning: missing:\n"
                    f"{class_root}"
                )

                continue

            if class_name not in (
                self.class_to_idx
            ):
                raise KeyError(
                    f"Class '{class_name}' "
                    f"not found in class_to_idx."
                )

            label = self.class_to_idx[
                class_name
            ]

            sources = find_clip_sources(
                class_root
            )

            for source in sources:

                self.samples.append(
                    (
                        source,
                        label,
                    )
                )

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No replay clips found in:\n"
                f"{root}"
            )

        print(
            f"[ReplayDataset] Loaded "
            f"{len(self.samples)} replay clips "
            f"from {root}"
        )

        for class_name in self.classes:

            label = self.class_to_idx[
                class_name
            ]

            count = sum(
                1
                for _, sample_label
                in self.samples
                if sample_label == label
            )

            print(
                f"    {class_name}: "
                f"{count}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self,
        index,
    ):

        source, label = (
            self.samples[index]
        )

        if os.path.isdir(source):

            frames = read_image_frames(
                source,
                self.clip_len,
                random_clip=self.random_clip,
            )

        else:

            frames = read_video_frames(
                source,
                self.clip_len,
                random_clip=self.random_clip,
            )

        return (
            frames,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("VIDEO DATASET TEST")
    print("=" * 60)

    print(
        "This file provides:"
    )

    print(
        "  VideoDataset"
    )

    print(
        "  ReplayDataset"
    )

    print(
        f"Clip length: "
        f"{DEFAULT_CLIP_LEN}"
    )

    print(
        f"Image size: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )