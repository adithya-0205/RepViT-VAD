
import os
import json
import random
import shutil

import torch

from video_dataset import VideoDataset


# ============================================================
# CONFIG
# ============================================================

RECOVERY_PATH = "training_recovery.pth"
CHECKPOINT_PATH = "best_model.pth"

CLASSES_JSON = "classes.json"
REPLAY_ROOT = "replay_buffer"

CLIP_LEN = 8
REPLAY_PER_CLASS = 30

SEED = 42
random.seed(SEED)


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(RECOVERY_PATH):
    raise FileNotFoundError(
        f"{RECOVERY_PATH} was not found."
    )


# ============================================================
# LOAD RECOVERY
# ============================================================

print("=" * 75)
print("FIXING BEST MODEL CHECKPOINT")
print("=" * 75)

recovery = torch.load(
    RECOVERY_PATH,
    map_location="cpu"
)

if not isinstance(recovery, dict):
    raise RuntimeError(
        "training_recovery.pth is not a valid dictionary checkpoint."
    )


classes = recovery.get(
    "classes",
    ["normal", "fighting"]
)

classes = [
    str(x).strip().lower()
    for x in classes
]

state_dict = recovery.get(
    "state_dict"
)

if state_dict is None:
    raise RuntimeError(
        "training_recovery.pth does not contain 'state_dict'."
    )

best_acc = float(
    recovery.get(
        "best_acc",
        0.0
    )
)

epoch = int(
    recovery.get(
        "epoch",
        0
    )
)

num_classes = len(classes)


print()
print("[Recovery]")
print(f"Classes    : {classes}")
print(f"Num classes: {num_classes}")
print(f"Epoch      : {epoch}")
print(f"Best Acc   : {best_acc * 100:.2f}%")


# ============================================================
# CREATE REPLAY BUFFER
# ============================================================

os.makedirs(
    REPLAY_ROOT,
    exist_ok=True
)


def save_replay_for_class(class_name):

    print()
    print(
        f"[Replay] Processing '{class_name}'..."
    )

    source_dataset_path = os.path.join(
        "dataset",
        "train",
        class_name
    )

    if not os.path.isdir(
        source_dataset_path
    ):
        print(
            f"[Replay] WARNING: "
            f"{source_dataset_path} does not exist."
        )
        return []

    replay_class_dir = os.path.join(
        REPLAY_ROOT,
        class_name
    )

    os.makedirs(
        replay_class_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find clip directories.
    #
    # This assumes the VideoDataset uses:
    #
    # dataset/train/class_name/clip_folder/
    #
    # --------------------------------------------------------

    clips = []

    for name in os.listdir(
        source_dataset_path
    ):

        path = os.path.join(
            source_dataset_path,
            name
        )

        if os.path.isdir(path):
            clips.append(
                os.path.normpath(path)
            )

    if not clips:
        print(
            f"[Replay] No clip directories found for "
            f"'{class_name}'."
        )
        return []

    random.shuffle(clips)

    selected = clips[
        :min(
            REPLAY_PER_CLASS,
            len(clips)
        )
    ]

    print(
        f"[Replay] Found {len(clips)} clips."
    )

    print(
        f"[Replay] Retaining {len(selected)} clips."
    )

    saved = []

    for index, source in enumerate(
        selected
    ):

        base_name = os.path.basename(
            source
        )

        destination = os.path.join(
            replay_class_dir,
            base_name
        )

        # Avoid overwriting an existing replay clip.
        if os.path.exists(destination):

            destination = os.path.join(
                replay_class_dir,
                f"{base_name}_replay_{index}"
            )

        try:

            shutil.copytree(
                source,
                destination,
                dirs_exist_ok=True
            )

            saved.append(
                os.path.normpath(destination)
            )

        except Exception as e:

            print(
                f"[Replay] Failed to copy "
                f"{source}: {e}"
            )

    return saved


# ============================================================
# BUILD REPLAY
# ============================================================

replay_paths = {}

for class_name in classes:

    paths = save_replay_for_class(
        class_name
    )

    if paths:
        replay_paths[class_name] = paths


# ============================================================
# MAKE RELATIVE PATHS
# ============================================================

relative_replay_paths = {}

for class_name, paths in replay_paths.items():

    relative_replay_paths[class_name] = []

    for path in paths:

        try:

            relative = os.path.relpath(
                path,
                os.getcwd()
            )

        except Exception:

            relative = path

        relative_replay_paths[
            class_name
        ].append(
            relative
        )


# ============================================================
# CREATE INITIAL EWC STRUCTURES
# ============================================================
#
# We don't have Fisher information from the completed
# training_recovery.pth.
#
# Therefore the first checkpoint starts with empty EWC data.
#
# EWC will be calculated after the next continual-learning
# stage and carried forward from there.
#

ewc_fisher = {}

ewc_params = {}


# ============================================================
# CREATE BEST MODEL CHECKPOINT
# ============================================================

checkpoint = {

    "state_dict":
        state_dict,

    "classes":
        classes,

    "num_classes":
        num_classes,

    "best_acc":
        best_acc,

    "epoch":
        epoch,

    "ewc_fisher":
        ewc_fisher,

    "ewc_params":
        ewc_params,

    "replay_paths":
        relative_replay_paths
}


torch.save(
    checkpoint,
    CHECKPOINT_PATH
)


# ============================================================
# SAVE CLASSES.JSON
# ============================================================

class_mapping = {
    class_name: index
    for index, class_name
    in enumerate(classes)
}

with open(
    CLASSES_JSON,
    "w"
) as file:

    json.dump(
        class_mapping,
        file,
        indent=2
    )


# ============================================================
# VERIFY
# ============================================================

print()
print("=" * 75)
print("BEST MODEL CREATED")
print("=" * 75)

print(
    f"Checkpoint : {CHECKPOINT_PATH}"
)

print(
    f"Classes    : {classes}"
)

print(
    f"Num classes: {num_classes}"
)

print(
    f"Best Acc   : {best_acc * 100:.2f}%"
)

print(
    f"Epoch      : {epoch}"
)

print()
print("[Replay]")

for class_name, paths in relative_replay_paths.items():

    print(
        f"  {class_name}: {len(paths)} clips"
    )

print()
print(
    f"Created: {CHECKPOINT_PATH}"
)

print(
    f"Created: {CLASSES_JSON}"
)

print(
    f"Created/updated: {REPLAY_ROOT}/"
)

print("=" * 75)
