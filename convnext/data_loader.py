from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# === CONFIG START ===
DATA_DIR = "F:/hossam/work papers/Classification/dataset"
BATCH_SIZE = 32
NUM_WORKERS = 4
# === CONFIG END ===

def get_data_loaders():
    train_dir = Path(DATA_DIR) / "train"
    val_dir = Path(DATA_DIR) / "val"
    test_dir = Path(DATA_DIR) / "test"

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class_names = train_dataset.classes

    print(f"✅ Data loaders created with {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")
    print(f"Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names


