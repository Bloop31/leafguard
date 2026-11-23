"""Unit tests for leafguard.data on tiny synthetic folders and images.

The torch-free helpers (list_classes, dataset_stats, imbalance_ratio,
top_n_classes) run everywhere, including CI. The transform tests need
torch/torchvision and skip cleanly where they aren't installed.

Not covered here (see the other test files): split logic (test_split.py),
dataset/dataloader splits and batch shapes (test_dataloaders.py).
"""

import random
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leafguard.data import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    dataset_stats,
    imbalance_ratio,
    list_classes,
    top_n_classes,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really an image")


def _save_image(path: Path, size=(12, 8), color=(30, 120, 60), mode="RGB", fmt=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path, format=fmt)


# --------------------------------------------------------------------------
# list_classes / dataset_stats
# --------------------------------------------------------------------------


def test_list_classes_returns_only_folders_sorted(tmp_path):
    for name in ["Tomato___healthy", "Apple___scab", "Corn___rust"]:
        (tmp_path / name).mkdir()
    _touch(tmp_path / "stray.jpg")
    _touch(tmp_path / "labels.csv")
    _touch(tmp_path / ".DS_Store")

    assert list_classes(str(tmp_path)) == ["Apple___scab", "Corn___rust", "Tomato___healthy"]


def test_list_classes_keeps_empty_class_folders(tmp_path):
    # An empty folder is still a class (ImageFolder sees it too), so it must
    # show up -- dataset_stats then reports it as 0 instead of hiding it.
    (tmp_path / "healthy").mkdir()
    (tmp_path / "rust").mkdir()
    _touch(tmp_path / "healthy" / "a.jpg")

    assert list_classes(str(tmp_path)) == ["healthy", "rust"]
    assert dataset_stats(str(tmp_path)) == {"healthy": 1, "rust": 0}


def test_dataset_stats_counts_image_extensions_case_insensitively(tmp_path):
    cls = tmp_path / "leaf"
    for name in ["IMG.JPG", "b.jpeg", "c.JPEG", "d.Png", "e.png", "f.jpg"]:
        _touch(cls / name)

    assert dataset_stats(str(tmp_path)) == {"leaf": 6}


def test_dataset_stats_ignores_non_images_and_root_files(tmp_path):
    cls = tmp_path / "leaf"
    _touch(cls / "real.jpg")
    for name in ["notes.txt", "labels.csv", ".DS_Store", "Thumbs.db", "archive.jpg.zip", "jpg"]:
        _touch(cls / name)
    # Loose files at the root are neither classes nor images of any class.
    _touch(tmp_path / "loose.jpg")
    _touch(tmp_path / "README.md")

    assert dataset_stats(str(tmp_path)) == {"leaf": 1}


def test_dataset_stats_on_empty_root_is_empty(tmp_path):
    assert dataset_stats(str(tmp_path)) == {}


# --------------------------------------------------------------------------
# imbalance_ratio
# --------------------------------------------------------------------------


def test_imbalance_ratio_balanced_is_one():
    assert imbalance_ratio({"a": 7, "b": 7, "c": 7}) == 1.0


def test_imbalance_ratio_skewed_is_largest_over_smallest_rounded():
    # 300 / 45 = 6.666... -> 6.67; the middle class doesn't matter.
    assert imbalance_ratio({"a": 300, "b": 120, "c": 45}) == 6.67


def test_imbalance_ratio_single_class_is_one():
    assert imbalance_ratio({"only": 12}) == 1.0


def test_imbalance_ratio_empty_dict_raises():
    with pytest.raises(ValueError, match="empty"):
        imbalance_ratio({})


def test_imbalance_ratio_zero_count_raises_naming_the_class():
    # Regression: an empty class folder used to raise ZeroDivisionError.
    with pytest.raises(ValueError, match="Corn___rust"):
        imbalance_ratio({"Apple___scab": 10, "Corn___rust": 0})


def test_imbalance_ratio_names_every_empty_class():
    with pytest.raises(ValueError) as exc:
        imbalance_ratio({"zeta": 0, "alpha": 5, "beta": 0})

    assert "beta, zeta" in str(exc.value)
    assert "alpha" not in str(exc.value)


# --------------------------------------------------------------------------
# top_n_classes
# --------------------------------------------------------------------------


def test_top_n_larger_than_number_of_classes_returns_all():
    stats = {"a": 3, "b": 9, "c": 1}

    assert top_n_classes(stats, n=10) == [("b", 9), ("a", 3), ("c", 1)]
    assert top_n_classes(stats, n=10, largest=False) == [("c", 1), ("a", 3), ("b", 9)]


@pytest.mark.parametrize("n", [0, -1])
def test_top_n_non_positive_raises(n):
    with pytest.raises(ValueError):
        top_n_classes({"a": 1}, n=n)


def test_top_n_ties_keep_input_order():
    # dataset_stats returns classes in sorted order, so stable ties mean the
    # report lists tied classes alphabetically in both directions.
    stats = {"a": 10, "b": 20, "c": 20, "d": 10}

    assert top_n_classes(stats, n=3) == [("b", 20), ("c", 20), ("a", 10)]
    assert top_n_classes(stats, n=3, largest=False) == [("a", 10), ("d", 10), ("b", 20)]


# --------------------------------------------------------------------------
# Class balance check on a folder of real (tiny) images
# --------------------------------------------------------------------------

UNEVEN_COUNTS = {"Apple___scab": 6, "Corn___rust": 2, "Tomato___healthy": 3}
EXTENSIONS = [(".jpg", "JPEG"), (".PNG", "PNG"), (".jpeg", "JPEG")]


@pytest.fixture
def uneven_image_root(tmp_path):
    """PlantVillage-style folder with deliberately uneven class sizes and mixed extensions."""
    for c, (cls, count) in enumerate(UNEVEN_COUNTS.items()):
        for i in range(count):
            ext, fmt = EXTENSIONS[i % len(EXTENSIONS)]
            path = tmp_path / cls / f"img_{i}{ext}"
            _save_image(path, size=(10 + i, 7), color=(c * 80, i * 20, 90), fmt=fmt)
        _touch(tmp_path / cls / "metadata.csv")
    _touch(tmp_path / "README.txt")
    return tmp_path


def test_synthetic_images_are_real_and_readable(uneven_image_root):
    for cls, count in UNEVEN_COUNTS.items():
        images = [p for p in (uneven_image_root / cls).iterdir() if p.suffix != ".csv"]
        assert len(images) == count
        for p in images:
            with Image.open(p) as img:
                img.verify()


def test_class_balance_check_on_uneven_folder(uneven_image_root):
    stats = dataset_stats(str(uneven_image_root))

    assert stats == UNEVEN_COUNTS
    assert sum(stats.values()) == 11
    assert imbalance_ratio(stats) == 3.0  # 6 / 2
    assert top_n_classes(stats, n=1) == [("Apple___scab", 6)]
    assert top_n_classes(stats, n=1, largest=False) == [("Corn___rust", 2)]


def test_class_balance_check_flags_an_empty_class(uneven_image_root):
    (uneven_image_root / "Potato___blight").mkdir()
    stats = dataset_stats(str(uneven_image_root))

    assert stats["Potato___blight"] == 0
    with pytest.raises(ValueError, match="Potato___blight"):
        imbalance_ratio(stats)


# --------------------------------------------------------------------------
# build_transforms (needs torch + torchvision; skipped in CI)
# --------------------------------------------------------------------------


@pytest.fixture
def torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    return torch


@pytest.fixture
def transforms(torch):
    from leafguard.data import build_transforms

    return build_transforms


def _gradient_image(size=(48, 20)):
    """Asymmetric gradient, so flips and rotations visibly change the pixels."""
    w, h = size
    img = Image.new("RGB", size)
    img.putdata([((x * 5) % 256, (y * 11) % 256, (x * y) % 256) for y in range(h) for x in range(w)])
    return img


def _noise_image(size=(40, 40), seed=0):
    rng = random.Random(seed)
    return Image.frombytes("RGB", size, bytes(rng.randrange(256) for _ in range(size[0] * size[1] * 3)))


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P"])
@pytest.mark.parametrize("src_size", [(48, 20), (15, 60), (300, 200)])
@pytest.mark.parametrize("img_size", [32, 64])
def test_transform_output_shape_and_dtype(torch, transforms, mode, src_size, img_size):
    # ImageFolder's default loader converts every image to RGB before the
    # transform runs, so grayscale / RGBA / palette leaves still give 3 channels.
    img = _gradient_image(src_size).convert(mode).convert("RGB")
    train_tfms, eval_tfms = transforms(img_size)

    for tfm in (train_tfms, eval_tfms):
        out = tfm(img)
        assert tuple(out.shape) == (3, img_size, img_size)
        assert out.dtype == torch.float32


def test_eval_transform_is_deterministic(torch, transforms):
    _, eval_tfms = transforms(32)
    img = _gradient_image()

    torch.manual_seed(0)
    a = eval_tfms(img)
    torch.manual_seed(1)
    b = eval_tfms(img)

    assert torch.equal(a, b)


def test_train_transform_is_random_but_reproducible_with_a_seed(torch, transforms):
    train_tfms, _ = transforms(32)
    img = _gradient_image()

    torch.manual_seed(123)
    first = train_tfms(img)
    torch.manual_seed(123)
    again = train_tfms(img)
    assert torch.equal(first, again)

    torch.manual_seed(0)
    draws = [train_tfms(img) for _ in range(5)]
    assert any(not torch.equal(draws[0], d) for d in draws[1:])


def test_eval_output_is_imagenet_normalized(torch, transforms):
    _, eval_tfms = transforms(32)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    # Random noise spans the full pixel range: output must leave [0, 1] on both sides.
    noisy = eval_tfms(_noise_image())
    assert noisy.min() < 0 and noisy.max() > 1
    assert abs(noisy.mean().item()) < 0.5

    # A solid image in the ImageNet mean colour should normalise to ~0 everywhere.
    mean_rgb = tuple(round(m * 255) for m in IMAGENET_MEAN)
    centred = eval_tfms(Image.new("RGB", (20, 20), mean_rgb))
    assert centred.abs().max() < 0.02

    # And a black image maps exactly to -mean / std per channel.
    black = eval_tfms(Image.new("RGB", (20, 20), (0, 0, 0)))
    assert torch.allclose(black, (-mean / std).expand_as(black), atol=1e-5)
