"""Tests for the torch-free train/val/test index split -- these run in CI."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leafguard.data import split_indices  # noqa: E402


def test_default_split_is_70_15_15_and_covers_everything():
    train, val, test = split_indices(1000)

    assert (len(train), len(val), len(test)) == (700, 150, 150)
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    assert sorted(train + val + test) == list(range(1000))


def test_rounding_leftovers_go_to_train():
    # 15 * 0.15 = 2.25 -> val and test get 2 each, train gets the other 11.
    train, val, test = split_indices(15)

    assert (len(train), len(val), len(test)) == (11, 2, 2)


def test_seed_controls_the_split():
    assert split_indices(100, seed=1) == split_indices(100, seed=1)
    assert split_indices(100, seed=1) != split_indices(100, seed=2)


def test_split_for_seed_42_is_pinned():
    # Changing the split algorithm would silently change which images are in the
    # held-out test set; if this fails on purpose, retrain before evaluating.
    train, val, test = split_indices(20, seed=42)

    assert train == [19, 5, 14, 4, 9, 13, 15, 18, 6, 12, 17, 10, 1, 11]
    assert val == [2, 16, 7]
    assert test == [8, 0, 3]


@pytest.mark.parametrize(
    "val_split, test_split",
    [(0.0, 0.15), (0.15, 0.0), (1.0, 0.15), (0.15, -0.1), (0.5, 0.5), (0.6, 0.6)],
)
def test_invalid_fractions_raise(val_split, test_split):
    with pytest.raises(ValueError):
        split_indices(100, val_split=val_split, test_split=test_split)


def test_too_few_images_raises():
    # 5 * 0.15 rounds down to 0 -> no val/test images at all.
    with pytest.raises(ValueError):
        split_indices(5)
