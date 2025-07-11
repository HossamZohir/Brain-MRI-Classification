import os
import shutil
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Source directory (update with your real path)
source_dir = "F:/hossam/work papers/Classification/archive"
classes = ["glioma", "healthy", "meningioma", "pituitary"]

# Destination directories
base_dir = "F:/hossam/work papers/Classification/dataset"
splits = ["train", "val", "test"]
split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

# Create directories
for split in splits:
    for cls in classes:
        Path(f"{base_dir}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

# Split data
for cls in classes:
    images = os.listdir(f"{source_dir}/{cls}")
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratios["train"])
    val_end = train_end + int(total * split_ratios["val"])

    for i, img in enumerate(images):
        src_path = os.path.join(source_dir, cls, img)

        if i < train_end:
            dst_path = f"{base_dir}/train/{cls}/{img}"
        elif i < val_end:
            dst_path = f"{base_dir}/val/{cls}/{img}"
        else:
            dst_path = f"{base_dir}/test/{cls}/{img}"

        shutil.copy(src_path, dst_path)

print("✅ Dataset successfully split!")
