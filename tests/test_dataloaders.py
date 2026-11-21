"""Dataloader tests on a tiny synthetic ImageFolder.

Skipped automatically where torch/torchvision aren't installed (CI and
laptops); run them on Kaggle or any machine with the deep-learning stack.
The torch-free split logic itself is covered in tests/test_split.py, which
does run in CI.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
Image = pytest.importorskip("PIL.Image")

from leafguard.data import build_datasets, get_dataloaders, split_indices  # noqa: E402

CLASSES = ["diseased", "healthy", "rust"]
IMAGES_PER_CLASS = 5
N_IMAGES = len(CLASSES) * IMAGES_PER_CLASS  # 15 -> 70/15/15 gives 11 / 2 / 2


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


def _is_augmentation(name):
    return name.startswith("Random") or name == "ColorJitter"


def test_splits_are_disjoint_and_cover_everything(image_root):
    train_ds, val_ds, test_ds, classes = build_datasets(image_root, img_size=32)

    assert classes == CLASSES
    assert (len(train_ds), len(val_ds), len(test_ds)) == (11, 2, 2)
    train, val, test = set(train_ds.indices), set(val_ds.indices), set(test_ds.indices)
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert sorted(train | val | test) == list(range(N_IMAGES))


def test_split_can_be_rebuilt_from_seed_and_size_alone(image_root):
    # Day 4 evaluation relies on this: the test set is fully determined by
    # (n_total, val_split, test_split, seed), without loading any images.
    train_ds, val_ds, test_ds, _ = build_datasets(image_root, img_size=32, seed=7)
    train_idx, val_idx, test_idx = split_indices(N_IMAGES, seed=7)

    assert train_ds.indices == train_idx
    assert val_ds.indices == val_idx
    assert test_ds.indices == test_idx


def test_seed_controls_the_split(image_root):
    a = build_datasets(image_root, img_size=32, seed=1)
    b = build_datasets(image_root, img_size=32, seed=1)
    c = build_datasets(image_root, img_size=32, seed=2)

    for split in range(3):
        assert a[split].indices == b[split].indices
    assert [a[s].indices for s in range(3)] != [c[s].indices for s in range(3)]


def test_train_keeps_augmentation_and_val_test_use_eval_transforms(image_root):
    # Regression: setting val_ds.dataset.transform used to strip augmentation from train too.
    train_ds, val_ds, test_ds, _ = build_datasets(image_root, img_size=32)

    assert train_ds.dataset is not val_ds.dataset
    assert train_ds.dataset is not test_ds.dataset
    assert "RandomHorizontalFlip" in _transform_names(train_ds)
    assert "ColorJitter" in _transform_names(train_ds)
    assert not any(_is_augmentation(n) for n in _transform_names(val_ds))
    assert not any(_is_augmentation(n) for n in _transform_names(test_ds))


def test_val_and_test_are_deterministic_and_train_is_augmented(image_root):
    train_ds, val_ds, test_ds, _ = build_datasets(image_root, img_size=32)
    torch.manual_seed(0)

    assert torch.equal(val_ds[0][0], val_ds[0][0])
    assert torch.equal(test_ds[0][0], test_ds[0][0])
    draws = [train_ds[0][0] for _ in range(5)]
    assert any(not torch.equal(draws[0], d) for d in draws[1:])


def test_all_splits_map_to_the_same_image_files(image_root):
    train_ds, val_ds, test_ds, _ = build_datasets(image_root, img_size=32)

    assert train_ds.dataset.samples == val_ds.dataset.samples == test_ds.dataset.samples


def test_get_dataloaders_batch_shapes_and_labels(image_root):
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        image_root, batch_size=4, img_size=64, num_workers=0
    )

    imgs, labels = next(iter(train_loader))
    assert imgs.shape == (4, 3, 64, 64)
    assert imgs.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert labels.min() >= 0 and labels.max() < len(classes)

    for loader in (val_loader, test_loader):
        eval_imgs, eval_labels = next(iter(loader))
        assert eval_imgs.shape == (2, 3, 64, 64)
        assert eval_labels.shape == (2,)


@pytest.mark.parametrize("val_split, test_split", [(0.0, 0.15), (0.15, 0.0), (1.0, 0.15), (0.5, 0.5)])
def test_invalid_splits_raise(image_root, val_split, test_split):
    with pytest.raises(ValueError):
        build_datasets(image_root, val_split=val_split, test_split=test_split)
