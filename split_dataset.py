
import shutil
import random
from pathlib import Path

# === CONFIG START ===
random.seed(42)
source_dir = Path("F:/hossam/work papers/Classification/archive")
classes = ["glioma", "healthy", "meningioma", "pituitary"]
base_dir = Path("F:/hossam/work papers/Classification/dataset")
splits = ["train", "val", "test"]
split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
# === CONFIG END ===

# Create directory structure
for split in splits:
    for cls in classes:
        (base_dir / split / cls).mkdir(parents=True, exist_ok=True)

# Split data and copy files accordingly
for cls in classes:
    images = list((source_dir / cls).glob("*.*"))  # all files
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratios["train"])
    val_end = train_end + int(total * split_ratios["val"])

    for i, img_path in enumerate(images):
        if i < train_end:
            dst_dir = base_dir / "train" / cls
        elif i < val_end:
            dst_dir = base_dir / "val" / cls
        else:
            dst_dir = base_dir / "test" / cls

        shutil.copy(img_path, dst_dir / img_path.name)

print("✅ Dataset successfully split!")
