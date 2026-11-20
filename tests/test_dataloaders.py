"""Dataloader tests on a tiny synthetic ImageFolder.

Skipped automatically where torch/torchvision aren't installed (CI and
laptops); run them on Kaggle or any machine with the deep-learning stack.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
Image = pytest.importorskip("PIL.Image")

from torch.utils.data import random_split  # noqa: E402

from leafguard.data import build_datasets, get_dataloaders  # noqa: E402

CLASSES = ["diseased", "healthy", "rust"]
IMAGES_PER_CLASS = 5


@pytest.fixture
def image_root(tmp_path):
    """3 classes x 5 non-square gradient images, so flips/rotations visibly change pixels."""
    for c, cls in enumerate(CLASSES):
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(IMAGES_PER_CLASS):
            img = Image.new("RGB", (40, 30))
            pixels = [((x * 6 + i * 10) % 256, (y * 8) % 256, c * 80) for y in range(30) for x in range(40)]
            img.putdata(pixels)
            img.save(cls_dir / f"img_{i}.png")
    return str(tmp_path)


def _transform_names(ds):
    return [type(t).__name__ for t in ds.dataset.transform.transforms]


def test_split_sizes_are_disjoint_and_cover_everything(image_root):
    train_ds, val_ds, classes = build_datasets(image_root, val_split=0.2, img_size=32)

    assert classes == CLASSES
    assert len(train_ds) == 12
    assert len(val_ds) == 3
    assert set(train_ds.indices).isdisjoint(val_ds.indices)
    assert sorted(train_ds.indices + val_ds.indices) == list(range(15))


def test_split_matches_previous_random_split(image_root):
    # Same seed must give the same membership as the old random_split code path.
    train_ds, val_ds, _ = build_datasets(image_root, val_split=0.2, img_size=32, seed=42)
    old_train, old_val = random_split(range(15), [12, 3], generator=torch.Generator().manual_seed(42))

    assert train_ds.indices == list(old_train.indices)
    assert val_ds.indices == list(old_val.indices)


def test_seed_controls_the_split(image_root):
    a_train, _, _ = build_datasets(image_root, img_size=32, seed=1)
    b_train, _, _ = build_datasets(image_root, img_size=32, seed=1)
    c_train, _, _ = build_datasets(image_root, img_size=32, seed=2)

    assert a_train.indices == b_train.indices
    assert a_train.indices != c_train.indices


def test_train_keeps_augmentation_and_val_uses_eval_transforms(image_root):
    # Regression: setting val_ds.dataset.transform used to strip augmentation from train too.
    train_ds, val_ds, _ = build_datasets(image_root, img_size=32)

    assert train_ds.dataset is not val_ds.dataset
    assert "RandomHorizontalFlip" in _transform_names(train_ds)
    assert "ColorJitter" in _transform_names(train_ds)
    assert not any(n.startswith("Random") or n == "ColorJitter" for n in _transform_names(val_ds))


def test_val_is_deterministic_and_train_is_augmented(image_root):
    train_ds, val_ds, _ = build_datasets(image_root, img_size=32)
    torch.manual_seed(0)

    assert torch.equal(val_ds[0][0], val_ds[0][0])
    draws = [train_ds[0][0] for _ in range(5)]
    assert any(not torch.equal(draws[0], d) for d in draws[1:])


def test_both_splits_map_to_the_same_image_files(image_root):
    train_ds, val_ds, _ = build_datasets(image_root, img_size=32)

    assert train_ds.dataset.samples == val_ds.dataset.samples


def test_get_dataloaders_batch_shapes_and_labels(image_root):
    train_loader, val_loader, classes = get_dataloaders(
        image_root, batch_size=4, img_size=64, num_workers=0
    )

    imgs, labels = next(iter(train_loader))
    assert imgs.shape == (4, 3, 64, 64)
    assert imgs.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert labels.min() >= 0 and labels.max() < len(classes)

    val_imgs, val_labels = next(iter(val_loader))
    assert val_imgs.shape == (3, 3, 64, 64)
    assert val_labels.shape == (3,)


@pytest.mark.parametrize("val_split", [0.0, 1.0, -0.1, 1.5])
def test_invalid_val_split_raises(image_root, val_split):
    with pytest.raises(ValueError):
        build_datasets(image_root, val_split=val_split)
