import os

with open("continual_train.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replacements
replacements = [
    # Header
    ('print("CONTINUAL LEARNING - VANDALISM")', 'print("CONTINUAL LEARNING - ASSAULT")'),
    
    # Dataset root
    ('VANDALISM_DATASET_ROOT = os.path.join(\n    PROJECT_ROOT,\n    "data",\n    "vandalism_dataset",\n)',
     'ASSAULT_DATASET_ROOT = os.path.join(\n    PROJECT_ROOT,\n    "dataset",\n)'),
     
    # Checkpoints
    ('SOURCE_CHECKPOINT_PATH = os.path.join(\n    PROJECT_ROOT,\n    "best_model_before_vandalism.pth",\n)',
     'SOURCE_CHECKPOINT_PATH = os.path.join(\n    PROJECT_ROOT,\n    "best_model.pth",\n)'),
     
    ('OUTPUT_CHECKPOINT_PATH = os.path.join(\n    PROJECT_ROOT,\n    "best_model_continual_vandalism.pth",\n)',
     'OUTPUT_CHECKPOINT_PATH = os.path.join(\n    PROJECT_ROOT,\n    "best_model_continual_assault.pth",\n)'),

    # Classes config
    ('OLD_CLASSES = [\n    "normal",\n    "fighting",\n]',
     'OLD_CLASSES = [\n    "normal",\n    "fighting",\n    "vandalism",\n]'),
     
    ('CLASSES = [\n    "normal",\n    "fighting",\n    "vandalism",\n]',
     'CLASSES = [\n    "normal",\n    "fighting",\n    "vandalism",\n    "assault",\n]'),
     
    ('CLASS_MAPPING = {\n    "normal": 0,\n    "fighting": 1,\n    "vandalism": 2,\n}',
     'CLASS_MAPPING = {\n    "normal": 0,\n    "fighting": 1,\n    "vandalism": 2,\n    "assault": 3,\n}'),
     
    ('OLD_CLASS_COUNT = 2\nNEW_CLASS_COUNT = 3',
     'OLD_CLASS_COUNT = 3\nNEW_CLASS_COUNT = 4'),

    # create_datasets - train loading
    ('    vandalism_train = VideoDataset(\n        root=VANDALISM_DATASET_ROOT,\n        train=True,\n        classes=["vandalism"],\n        class_to_idx=CLASS_MAPPING,\n        clip_len=CLIP_LEN,\n    )',
     '    assault_train = VideoDataset(\n        root=ASSAULT_DATASET_ROOT,\n        train=True,\n        classes=["assault"],\n        class_to_idx=CLASS_MAPPING,\n        clip_len=CLIP_LEN,\n    )'),

    # create_datasets - val loading
    ('    vandalism_val = VideoDataset(\n        root=VANDALISM_DATASET_ROOT,\n        train=False,\n        classes=["vandalism"],\n        class_to_idx=CLASS_MAPPING,\n        clip_len=CLIP_LEN,\n    )',
     '    assault_val = VideoDataset(\n        root=ASSAULT_DATASET_ROOT,\n        train=False,\n        classes=["assault"],\n        class_to_idx=CLASS_MAPPING,\n        clip_len=CLIP_LEN,\n    )'),

    # create_datasets - prints and checks
    ('    print(\n        f"[Vandalism] Train videos: "\n        f"{len(vandalism_train)}"\n    )',
     '    print(\n        f"[Assault] Train videos: "\n        f"{len(assault_train)}"\n    )'),
     
    ('    print(\n        f"[Vandalism] Val videos: "\n        f"{len(vandalism_val)}"\n    )',
     '    print(\n        f"[Assault] Val videos: "\n        f"{len(assault_val)}"\n    )'),
     
    ('    if len(vandalism_train) == 0:', '    if len(assault_train) == 0:'),
    ('            "Vandalism training dataset is empty."', '            "Assault training dataset is empty."'),
    ('    if len(vandalism_val) == 0:', '    if len(assault_val) == 0:'),
    ('            "Vandalism validation dataset is empty."', '            "Assault validation dataset is empty."'),

    # ReplayDataset classes
    ('        classes=[\n            "normal",\n            "fighting",\n        ],',
     '        classes=[\n            "normal",\n            "fighting",\n            "vandalism",\n        ],'),

    # Replay train counts
    ('    normal_train = sum(\n        1\n        for _, label in replay_train.samples\n        if label == CLASS_MAPPING["normal"]\n    )\n\n    fighting_train = sum(\n        1\n        for _, label in replay_train.samples\n        if label == CLASS_MAPPING["fighting"]\n    )\n\n    print()\n    print("[Replay Train Count]")\n    print(f"Normal   : {normal_train}")\n    print(f"Fighting : {fighting_train}")',
     '    normal_train = sum(\n        1\n        for _, label in replay_train.samples\n        if label == CLASS_MAPPING["normal"]\n    )\n\n    fighting_train = sum(\n        1\n        for _, label in replay_train.samples\n        if label == CLASS_MAPPING["fighting"]\n    )\n\n    vandalism_train_rep = sum(\n        1\n        for _, label in replay_train.samples\n        if label == CLASS_MAPPING["vandalism"]\n    )\n\n    print()\n    print("[Replay Train Count]")\n    print(f"Normal    : {normal_train}")\n    print(f"Fighting  : {fighting_train}")\n    print(f"Vandalism : {vandalism_train_rep}")'),

    # ConcatDataset
    ('    train_dataset = ConcatDataset(\n        [\n            replay_train,\n            vandalism_train,\n        ]\n    )',
     '    train_dataset = ConcatDataset(\n        [\n            replay_train,\n            assault_train,\n        ]\n    )'),
     
    ('    val_dataset = ConcatDataset(\n        [\n            replay_val,\n            vandalism_val,\n        ]\n    )',
     '    val_dataset = ConcatDataset(\n        [\n            replay_val,\n            assault_val,\n        ]\n    )'),

    ('    print("[Dataset] Labels:")\n    print("0 = normal")\n    print("1 = fighting")\n    print("2 = vandalism")',
     '    print("[Dataset] Labels:")\n    print("0 = normal")\n    print("1 = fighting")\n    print("2 = vandalism")\n    print("3 = assault")'),

    # return datasets
    ('    return (\n        vandalism_train,\n        vandalism_val,\n        replay_train,\n        replay_val,\n        train_dataset,\n        val_dataset,\n    )',
     '    return (\n        assault_train,\n        assault_val,\n        replay_train,\n        replay_val,\n        train_dataset,\n        val_dataset,\n    )'),

    # create_student prints
    ('    print()\n    print("[Model] Creating 3-class student...")',
     '    print()\n    print("[Model] Creating 4-class student...")'),

    ('    print()\n    print("[Model] Previous classes:")\n    print("    normal   = 0")\n    print("    fighting = 1")\n\n    print()\n    print("[Model] New class:")\n    print("    vandalism = 2")',
     '    print()\n    print("[Model] Previous classes:")\n    print("    normal    = 0")\n    print("    fighting  = 1")\n    print("    vandalism = 2")\n\n    print()\n    print("[Model] New class:")\n    print("    assault = 3")'),

    # Source check warning
    ('            "Use best_model_before_vandalism.pth "\n            "containing normal + fighting."',
     '            "Use best_model.pth "\n            "containing normal + fighting + vandalism."'),

    # Class weights
    ('    class_weights = torch.tensor(\n        [\n            1.5,  # normal\n            1.5,  # fighting\n            1.0,  # vandalism\n        ],',
     '    class_weights = torch.tensor(\n        [\n            1.5,  # normal\n            1.5,  # fighting\n            1.0,  # vandalism\n            1.0,  # assault\n        ],'),

    # setup_training unpack
    ('    (\n        vandalism_train,\n        vandalism_val,\n        replay_train,\n        replay_val,\n        train_dataset,\n        val_dataset,\n    ) = create_datasets()',
     '    (\n        assault_train,\n        assault_val,\n        replay_train,\n        replay_val,\n        train_dataset,\n        val_dataset,\n    ) = create_datasets()'),
]

for src, tgt in replacements:
    if src not in content:
        print(f"Warning: replacement string not found:\n{src[:100]}...")
    content = content.replace(src, tgt)

with open("continual_train_assault.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Created continual_train_assault.py successfully!")
