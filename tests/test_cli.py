"""Torch-free checks of the train / predict command lines -- these run in CI.

The actual training and inference behind these flags is exercised on tiny
synthetic images in tests/test_pipeline.py (needs torch).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leafguard import predict, train  # noqa: E402
from leafguard.evaluate import build_parser as eval_parser, data_settings  # noqa: E402


def test_train_defaults_are_the_kaggle_settings():
    args = train.build_parser().parse_args(["--data-root", "/data/PlantVillage"])

    assert (args.backbone, args.epochs, args.batch_size, args.img_size, args.num_workers) == (
        "resnet18", 10, 32, 224, 2,
    )
    assert args.pretrained is True
    assert args.fast_dev_run is False
    assert args.max_samples is None


def test_train_fast_dev_run_flags():
    args = train.build_parser().parse_args([
        "--data-root", "d", "--backbone", "baseline", "--fast-dev-run", "--max-samples", "32",
        "--no-pretrained", "--num-workers", "0", "--img-size", "128",
    ])

    assert args.fast_dev_run and not args.pretrained
    assert (args.max_samples, args.num_workers, args.img_size) == (32, 0, 128)


def test_train_rejects_unknown_backbone_before_touching_data():
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["--data-root", "d", "--backbone", "vgg"])
    with pytest.raises(ValueError, match="Unknown backbone"):
        train.train("vgg", data_root="does-not-matter", out_dir="unused")


def test_evaluate_new_flags_default_to_the_training_run():
    args = eval_parser().parse_args(["--weights", "w.pt", "--data-root", "d"])

    assert (args.img_size, args.max_samples, args.num_workers) == (None, None, 2)


def test_data_settings_come_from_training_metrics():
    train_metrics = {
        "config": {"img_size": 96},
        "split": {"val_split": 0.2, "test_split": 0.1, "max_samples": 64},
    }

    assert data_settings(train_metrics) == {
        "val_split": 0.2, "test_split": 0.1, "img_size": 96, "max_samples": 64,
    }
    # Explicit values win over the recorded ones.
    assert data_settings(train_metrics, img_size=224, max_samples=10)["img_size"] == 224
    assert data_settings(train_metrics, img_size=224, max_samples=10)["max_samples"] == 10


def test_data_settings_without_metrics_match_train_defaults():
    # Older metrics files (before img_size / max_samples were recorded) get the same defaults.
    expected = {"val_split": 0.15, "test_split": 0.15, "img_size": 224, "max_samples": None}

    assert data_settings(None) == expected
    assert data_settings({"split": {"seed": 42}, "config": {"epochs": 10}}) == expected


def test_predict_cli_arguments():
    args = predict.build_parser().parse_args(["a.jpg", "b.png", "--weights", "models/baseline_best.pt"])

    assert args.images == ["a.jpg", "b.png"]
    assert (args.metrics, args.backbone, args.img_size, args.top_k) == (None, None, None, 3)


def test_find_metrics_file_from_checkpoint_name(tmp_path):
    (tmp_path / "metrics_baseline.json").write_text("{}")
    (tmp_path / "metrics_resnet18.json").write_text("{}")

    assert predict.find_metrics_file(str(tmp_path / "baseline_best.pt")).name == "metrics_baseline.json"
    assert predict.find_metrics_file(str(tmp_path / "x.pt"), "resnet18").name == "metrics_resnet18.json"
    with pytest.raises(FileNotFoundError, match="--metrics"):
        predict.find_metrics_file(str(tmp_path / "renamed.pt"))  # two candidates, no hint


def test_find_metrics_file_falls_back_to_the_only_one(tmp_path):
    (tmp_path / "metrics_efficientnet_b0.json").write_text("{}")

    assert predict.find_metrics_file(str(tmp_path / "renamed.pt")).name == "metrics_efficientnet_b0.json"
