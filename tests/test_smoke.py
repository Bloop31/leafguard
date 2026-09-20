"""Smoke tests that run free on GitHub Actions (CPU, no torch installed).

These don't test model accuracy -- they just make sure the package imports
cleanly and the parts that don't need torch (data.list_classes, argument
parsing, etc.) behave correctly. Real training/eval correctness is checked
by hand on Kaggle and recorded in reports/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_package_imports():
    import leafguard  # noqa: F401
    import leafguard.data  # noqa: F401
    import leafguard.predict  # noqa: F401

    assert leafguard.__version__


def test_model_module_degrades_without_torch_or_imports_fine():
    # This should not raise even if torch isn't installed -- only calling
    # BaselineCNN()/build_transfer_model() should raise ImportError.
    import leafguard.model as model_mod

    assert hasattr(model_mod, "BaselineCNN")
    assert hasattr(model_mod, "build_transfer_model")


def test_list_classes_missing_root_raises(tmp_path):
    from leafguard.data import list_classes

    missing = tmp_path / "does_not_exist"
    try:
        list_classes(str(missing))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_list_classes_and_dataset_stats(tmp_path):
    from leafguard.data import list_classes, dataset_stats

    (tmp_path / "healthy").mkdir()
    (tmp_path / "diseased").mkdir()
    (tmp_path / "healthy" / "a.jpg").write_bytes(b"fake")
    (tmp_path / "healthy" / "b.png").write_bytes(b"fake")
    (tmp_path / "diseased" / "c.jpeg").write_bytes(b"fake")

    classes = list_classes(str(tmp_path))
    assert classes == ["diseased", "healthy"]

    stats = dataset_stats(str(tmp_path))
    assert stats == {"diseased": 1, "healthy": 2}
def test_imbalance_ratio():
    from leafguard.data import imbalance_ratio

    balanced = {"a": 100, "b": 100}
    assert imbalance_ratio(balanced) == 1.0

    skewed = {"a": 100, "b": 25}
    assert imbalance_ratio(skewed) == 4.0