"""Dataset loading and transforms for the PlantVillage leaf-disease dataset.

Designed to run on Kaggle, where the dataset is mounted at
/kaggle/input/plantdisease (or similar) via
`Add Data -> emmarex/plantdisease`. Locally (no torch installed), only
`list_classes` and `dataset_stats` are usable, for quick sanity checks.

Usage on Kaggle:
    from leafguard.data import get_dataloaders
    train_loader, val_loader, classes = get_dataloaders(
        root="/kaggle/input/plantdisease/PlantVillage", batch_size=32
    )
"""

from pathlib import Path
from typing import Tuple, List


def list_classes(root: str) -> List[str]:
    """Return sorted class (folder) names under an ImageFolder-style root."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return sorted(p.name for p in root_path.iterdir() if p.is_dir())


def dataset_stats(root: str) -> dict:
    """Count images per class -- useful for spotting class imbalance."""
    root_path = Path(root)
    stats = {}
    for cls in list_classes(root):
        cls_dir = root_path / cls
        n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        stats[cls] = n
    return stats


def get_dataloaders(
    root: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    img_size: int = 224,
    seed: int = 42,
) -> Tuple["object", "object", List[str]]:
    """Build train/val DataLoaders with standard ImageNet-style augmentation.

    Only import torch/torchvision inside this function so the module still
    imports cleanly (and CI can smoke-test the rest of the file) on machines
    without the heavy deep-learning stack installed.
    """
    import torch
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms

    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = datasets.ImageFolder(root=root, transform=train_tfms)
    classes = full_dataset.classes

    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Validation set should use eval transforms, not train-time augmentation.
    val_ds.dataset.transform = eval_tfms

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, classes
