
import os
import json
import random
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CLASSES = ["normal", "fighting"]

DEFAULT_CLIP_LEN = 16
IMAGE_SIZE = 224

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


# ============================================================
# COMMON TRANSFORM
# ============================================================

def create_transform():
    """
    Transform used by both VideoDataset and ReplayDataset.

    Output:
        Tensor shape = (3, 224, 224)
    """

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            NORMALIZE_MEAN,
            NORMALIZE_STD
        )
    ])


# ============================================================
# FRAME FILE CHECK
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def is_image_file(filename):
    """
    Returns True only for supported image files.
    """

    extension = os.path.splitext(filename)[1].lower()

    return extension in VALID_EXTENSIONS


# ============================================================
# GET FRAME LIST
# ============================================================

def get_frame_files(video_folder):
    """
    Returns sorted image-frame paths from a frame folder.

    Example:

        dataset/
        ├── train/
        │   ├── normal/
        │   │   ├── video001/
        │   │   │   ├── 001.jpg
        │   │   │   ├── 002.jpg
        │   │   │   └── ...
    """

    if not os.path.isdir(video_folder):
        return []

    try:
        names = os.listdir(video_folder)
    except OSError:
        return []

    frames = []

    for name in names:

        path = os.path.join(
            video_folder,
            name
        )

        if (
            os.path.isfile(path)
            and is_image_file(name)
        ):
            frames.append(path)

    # Natural-ish sorting based on filename.
    frames.sort()

    return frames


# ============================================================
# SELECT CLIP
# ============================================================

def select_clip(
    frames,
    clip_len,
    random_clip=True
):
    """
    Select exactly clip_len frames.

    If enough frames exist:
        random_clip=True  -> random temporal crop
        random_clip=False -> deterministic first clip

    If fewer frames exist:
        last frame is repeated.
    """

    if len(frames) == 0:
        raise RuntimeError(
            "Video folder contains no valid image frames."
        )

    # --------------------------------------------------------
    # Enough frames
    # --------------------------------------------------------

    if len(frames) >= clip_len:

        max_start = len(frames) - clip_len

        if random_clip and max_start > 0:

            start = random.randint(
                0,
                max_start
            )

        else:

            start = 0

        return frames[
            start:start + clip_len
        ]

    # --------------------------------------------------------
    # Too few frames
    # --------------------------------------------------------

    selected = list(frames)

    last_frame = frames[-1]

    while len(selected) < clip_len:

        selected.append(last_frame)

    return selected


# ============================================================
# LOAD CLIP
# ============================================================

def load_clip(
    frame_paths,
    transform
):
    """
    Loads all frames and returns:

        Tensor shape:
        (T, C, H, W)

    Example:

        (16, 3, 224, 224)
    """

    images = []

    for frame_path in frame_paths:

        try:

            with Image.open(frame_path) as img:

                img = img.convert("RGB")

                tensor = transform(img)

                images.append(tensor)

        except (
            OSError,
            UnidentifiedImageError
        ) as error:

            raise RuntimeError(
                f"Could not read frame:\n"
                f"{frame_path}\n"
                f"Error: {error}"
            )

    if not images:

        raise RuntimeError(
            "No valid frames could be loaded."
        )

    return torch.stack(images, dim=0)


# ============================================================
# VIDEO DATASET
# ============================================================

class VideoDataset(Dataset):
    """
    Dataset for video clips stored as frame folders.

    Expected structure:

        dataset/
        ├── train/
        │   ├── normal/
        │   │   ├── video_001/
        │   │   │   ├── 001.jpg
        │   │   │   ├── 002.jpg
        │   │   │   └── ...
        │   │   └── video_002/
        │   │
        │   └── fighting/
        │       ├── video_001/
        │       └── ...
        │
        └── val/
            ├── normal/
            └── fighting/

    Each returned sample:

        clip:
            (16, 3, 224, 224)

        label:
            integer class index
    """

    def __init__(
        self,
        root="dataset",
        train=True,
        clip_len=DEFAULT_CLIP_LEN,
        classes=None,
        class_to_idx=None,
        random_clip=True
    ):

        self.root = root
        self.train = train
        self.clip_len = clip_len
        self.random_clip = random_clip

        self.transform = create_transform()

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        split_folder = (
            "train"
            if train
            else "val"
        )

        self.split_dir = os.path.join(
            root,
            split_folder
        )

        # ----------------------------------------------------
        # Class mapping
        # ----------------------------------------------------

        self.classes_file = "classes.json"

        if class_to_idx is not None:
            self.class_to_idx = dict(class_to_idx)
            if classes is not None:
                self.classes = list(classes)
            else:
                self.classes = [k for k in sorted(self.class_to_idx, key=lambda x: self.class_to_idx[x])]
        elif classes is not None:

            # Explicit classes supplied by train.py.
            #
            # For your current stage:
            #
            # ["normal", "fighting"]
            #
            self.classes = list(classes)

            self.class_to_idx = {
                name: index
                for index, name
                in enumerate(self.classes)
            }

        else:

            # If no classes were explicitly supplied,
            # read existing mapping.

            self.classes = []
            self.class_to_idx = {}

            if os.path.exists(
                self.classes_file
            ):

                try:

                    with open(
                        self.classes_file,
                        "r",
                        encoding="utf-8"
                    ) as file:

                        saved = json.load(file)

                    if isinstance(
                        saved,
                        dict
                    ):

                        self.class_to_idx = {
                            str(k): int(v)
                            for k, v in saved.items()
                        }

                        # Sort by class index so that:
                        #
                        # 0 -> normal
                        # 1 -> fighting
                        # 2 -> assault
                        #
                        self.classes = [
                            name
                            for name, index
                            in sorted(
                                self.class_to_idx.items(),
                                key=lambda x: x[1]
                            )
                        ]

                    elif isinstance(
                        saved,
                        list
                    ):

                        self.classes = list(
                            saved
                        )

                        self.class_to_idx = {
                            name: index
                            for index, name
                            in enumerate(
                                self.classes
                            )
                        }

                except (
                    OSError,
                    json.JSONDecodeError,
                    ValueError
                ):

                    print(
                        "[Dataset] Warning: "
                        "Could not read classes.json."
                    )

            # No mapping found.
            if not self.classes:

                self.classes = list(
                    DEFAULT_CLASSES
                )

                self.class_to_idx = {
                    name: index
                    for index, name
                    in enumerate(
                        self.classes
                    )
                }

        # ----------------------------------------------------
        # Validate requested classes
        # ----------------------------------------------------

        if not os.path.isdir(
            self.split_dir
        ):

            raise FileNotFoundError(
                f"Dataset split not found:\n"
                f"{self.split_dir}"
            )

        disk_classes = sorted(
            [
                name
                for name in os.listdir(
                    self.split_dir
                )
                if os.path.isdir(
                    os.path.join(
                        self.split_dir,
                        name
                    )
                )
            ]
        )

        # ----------------------------------------------------
        # Explicit class mode
        # ----------------------------------------------------

        if classes is not None:

            missing = [
                cls
                for cls in self.classes
                if cls not in disk_classes
            ]

            if missing:

                raise ValueError(
                    f"Requested class(es) not found "
                    f"in {self.split_dir}: "
                    f"{missing}"
                )

        # ----------------------------------------------------
        # Build samples
        # ----------------------------------------------------

        self.samples = []

        for class_name in self.classes:

            class_dir = os.path.join(
                self.split_dir,
                class_name
            )

            if not os.path.isdir(
                class_dir
            ):
                continue

            label = self.class_to_idx[
                class_name
            ]

            try:

                video_names = sorted(
                    os.listdir(
                        class_dir
                    )
                )

            except OSError:

                continue

            for video_name in video_names:

                video_path = os.path.join(
                    class_dir,
                    video_name
                )

                if not os.path.isdir(
                    video_path
                ):
                    continue

                # Check that the folder actually
                # contains image frames.

                frame_files = get_frame_files(
                    video_path
                )

                if not frame_files:

                    print(
                        f"[Dataset] Skipping empty "
                        f"video folder:\n"
                        f"{video_path}"
                    )

                    continue

                self.samples.append(
                    (
                        video_path,
                        label
                    )
                )

        # ----------------------------------------------------
        # Final information
        # ----------------------------------------------------

        print(
            f"[VideoDataset] "
            f"{'TRAIN' if train else 'VAL'}"
        )

        print(
            f"    Classes : {self.classes}"
        )

        print(
            f"    Samples : {len(self.samples)}"
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.samples
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        video_folder, label = (
            self.samples[index]
        )

        frames = get_frame_files(
            video_folder
        )

        if not frames:

            raise RuntimeError(
                f"No image frames found in:\n"
                f"{video_folder}"
            )

        selected_frames = select_clip(
            frames,
            self.clip_len,
            random_clip=(
                self.random_clip
                and self.train
            )
        )

        clip = load_clip(
            selected_frames,
            self.transform
        )

        return clip, label


# ============================================================
# REPLAY DATASET
# ============================================================

class ReplayDataset(Dataset):
    """
    Experience Replay dataset.

    Loads previously saved frame-folder clips.

    replay_paths format:

        {
            "normal": [
                ".../video001",
                ".../video002"
            ],

            "fighting": [
                ".../video003",
                ".../video004"
            ]
        }

    The class_to_idx mapping comes from the CURRENT
    training stage.

    Example:

        {
            "normal": 0,
            "fighting": 1
        }

    Later teammate:

        {
            "normal": 0,
            "fighting": 1,
            "assault": 2
        }

    Existing replay labels therefore remain stable.
    """

    def __init__(
        self,
        replay_paths,
        class_to_idx,
        clip_len=DEFAULT_CLIP_LEN,
        random_clip=True
    ):

        self.clip_len = clip_len

        self.class_to_idx = dict(
            class_to_idx
        )

        self.random_clip = random_clip

        self.transform = create_transform()

        self.samples = []

        # ----------------------------------------------------
        # Build replay samples
        # ----------------------------------------------------

        for class_name, paths in (
            replay_paths.items()
        ):

            # Ignore unknown classes.

            if class_name not in (
                self.class_to_idx
            ):

                print(
                    f"[Replay] Ignoring unknown "
                    f"class: {class_name}"
                )

                continue

            label = self.class_to_idx[
                class_name
            ]

            if not isinstance(
                paths,
                (list, tuple)
            ):
                continue

            for path in paths:

                if not os.path.isdir(
                    path
                ):
                    print(
                        f"[Replay] Missing path: "
                        f"{path}"
                    )

                    continue

                frame_files = get_frame_files(
                    path
                )

                if not frame_files:

                    print(
                        f"[Replay] Empty clip: "
                        f"{path}"
                    )

                    continue

                self.samples.append(
                    (
                        path,
                        label
                    )
                )

        print(
            f"[ReplayDataset] "
            f"Loaded {len(self.samples)} replay clips."
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.samples
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        video_folder, label = (
            self.samples[index]
        )

        frames = get_frame_files(
            video_folder
        )

        if not frames:

            raise RuntimeError(
                f"Replay clip contains no "
                f"valid frames:\n"
                f"{video_folder}"
            )

        selected_frames = select_clip(
            frames,
            self.clip_len,
            random_clip=self.random_clip
        )

        clip = load_clip(
            selected_frames,
            self.transform
        )

        return clip, label

