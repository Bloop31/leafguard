"""Dataset loading and transforms for the PlantVillage leaf-disease dataset.

Designed to run on Kaggle, where the dataset is mounted at
/kaggle/input/plantdisease (or similar) via
`Add Data -> emmarex/plantdisease`. Locally (no torch installed), only
`list_classes`, `dataset_stats` and `split_indices` are usable, for quick
sanity checks.

Usage on Kaggle:
    from leafguard.data import get_dataloaders
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        root="/kaggle/input/plantdisease/PlantVillage", batch_size=32
    )
"""

import random
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


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224) -> Tuple["object", "object"]:
    """Return (train_tfms, eval_tfms): augmented for training, deterministic for eval."""
    from torchvision import transforms

    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def split_indices(
    n_total: int,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Seeded, disjoint (train, val, test) index lists that together cover range(n_total).

    Pure Python (``random.Random(seed).shuffle``) so it runs without torch,
    is unit-tested in CI, and gives the same split on any machine. Val and
    test sizes are rounded down; train gets whatever is left over.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
    if not 0.0 < test_split < 1.0:
        raise ValueError(f"test_split must be between 0 and 1, got {test_split}")
    if val_split + test_split >= 1.0:
        raise ValueError(f"val_split + test_split must be below 1, got {val_split} + {test_split}")

    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            f"{n_total} images is too few for a {val_split}/{test_split} val/test split "
            f"(train={n_train}, val={n_val}, test={n_test})"
        )

    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def build_datasets(
    root: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    img_size: int = 224,
    seed: int = 42,
) -> Tuple["object", "object", "object", List[str]]:
    """Build seeded train/val/test splits; train is augmented, val/test use eval transforms.

    ``random_split`` returns Subsets that share one underlying dataset, so
    setting ``val_ds.dataset.transform`` would also strip augmentation from
    the training split. Instead we load the folder twice (one ImageFolder per
    transform pipeline) and take disjoint index sets from ``split_indices``.
    Val and test share the eval ImageFolder, which is safe because neither
    one is augmented.
    """
    from torch.utils.data import Subset
    from torchvision import datasets

    train_tfms, eval_tfms = build_transforms(img_size)
    train_base = datasets.ImageFolder(root=root, transform=train_tfms)
    eval_base = datasets.ImageFolder(root=root, transform=eval_tfms)
    if train_base.samples != eval_base.samples:
        raise RuntimeError(f"Dataset under {root} changed while it was being loaded")

    train_idx, val_idx, test_idx = split_indices(
        len(train_base), val_split=val_split, test_split=test_split, seed=seed
    )
    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(eval_base, val_idx)
    test_ds = Subset(eval_base, test_idx)
    return train_ds, val_ds, test_ds, train_base.classes


def get_dataloaders(
    root: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    img_size: int = 224,
    seed: int = 42,
    num_workers: int = 2,
) -> Tuple["object", "object", "object", List[str]]:
    """Build train/val/test DataLoaders (default 70/15/15) with ImageNet-style augmentation on train.

    Only import torch/torchvision inside this function so the module still
    imports cleanly (and CI can smoke-test the rest of the file) on machines
    without the heavy deep-learning stack installed.
    """
    from torch.utils.data import DataLoader

    train_ds, val_ds, test_ds, classes = build_datasets(
        root, val_split=val_split, test_split=test_split, img_size=img_size, seed=seed
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, classes


def imbalance_ratio(stats: dict) -> float:
    """Ratio of the largest class to the smallest class.

    A ratio close to 1.0 means classes are balanced. Anything above ~3
    is worth handling with class weights or oversampling later.
    """
    if not stats:
        raise ValueError("stats dict is empty")
    counts = list(stats.values())
    return round(max(counts) / min(counts), 2)


def top_n_classes(stats: dict, n: int = 5, largest: bool = True) -> list:
    """Return the n largest (or smallest) classes as (name, count) pairs.

    Useful for a quick "which classes need oversampling" check before
    training, without re-plotting the full distribution every time.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    ordered = sorted(stats.items(), key=lambda kv: kv[1], reverse=largest)

    return ordered[:n]
