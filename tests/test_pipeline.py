"""End-to-end check on CPU: train (--fast-dev-run) -> evaluate -> predict on synthetic images.

Needs torch + torchvision (+ matplotlib for the plots), so it runs in the CI
torch-tests job and skips in the lightweight one. It proves the pieces fit
together -- files written, test split rebuilt, checkpoint loadable offline --
not that the model is any good: accuracy on these images means nothing.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
Image = pytest.importorskip("PIL.Image")

from leafguard import evaluate, predict, train  # noqa: E402

CLASSES = ["Tomato_healthy", "Tomato_Late_blight", "Tomato_rust"]
IMAGES_PER_CLASS = 30  # 90 images > the 64-image fast-dev-run cap, so the subset kicks in
IMG_SIZE = 32  # small, so the whole file runs in seconds on CPU


@pytest.fixture(scope="module")
def image_root(tmp_path_factory):
    """3 classes x 30 small images; each class gets its own colour so the task is learnable."""
    root = tmp_path_factory.mktemp("PlantVillage")
    for c, cls in enumerate(CLASSES):
        cls_dir = root / cls
        cls_dir.mkdir()
        for i in range(IMAGES_PER_CLASS):
            img = Image.new("RGB", (40, 30), color=(c * 100, 200 - c * 60, (i * 7) % 256))
            img.save(cls_dir / f"img_{i:02d}.jpg")
    return root


@pytest.fixture(scope="module")
def baseline_run(image_root, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("outputs")
    metrics = train.train(
        "baseline", data_root=str(image_root), out_dir=str(out_dir), img_size=IMG_SIZE, fast_dev_run=True,
    )
    return out_dir, metrics


def test_fast_dev_run_trains_one_epoch_on_a_subset(baseline_run):
    out_dir, metrics = baseline_run

    assert metrics["config"]["fast_dev_run"] is True
    assert metrics["config"]["epochs"] == 1 and len(metrics["history"]["train_loss"]) == 1
    assert metrics["config"]["pretrained"] is False
    split = metrics["split"]
    assert split["max_samples"] == train.FAST_DEV_RUN_MAX_SAMPLES
    assert split["n_total"] == train.FAST_DEV_RUN_MAX_SAMPLES < len(CLASSES) * IMAGES_PER_CLASS
    assert min(split["n_train"], split["n_val"], split["n_test"]) >= 1
    assert metrics["classes"] == sorted(CLASSES)


def test_fast_dev_run_writes_checkpoint_metrics_and_curves(baseline_run):
    out_dir, metrics = baseline_run

    # Epoch 1 is always saved, even if val accuracy is 0 on a tiny run.
    assert metrics["best_epoch"] == 1
    assert (out_dir / "baseline_best.pt").is_file()
    assert json.loads((out_dir / "metrics_baseline.json").read_text())["split"]["max_samples"] == 64
    assert (out_dir / "training_curves_baseline.png").read_bytes()[:4] == b"\x89PNG"


def test_evaluate_rebuilds_the_subset_test_split_from_training_metrics(baseline_run, image_root):
    out_dir, train_metrics = baseline_run

    # No --img-size / --max-samples: both come from metrics_baseline.json next to the weights,
    # and the split check would raise if the rebuilt test indices didn't match.
    result = evaluate.evaluate(
        "baseline", weights_path=str(out_dir / "baseline_best.pt"), data_root=str(image_root),
        out_dir=str(out_dir / "eval"), num_workers=0,
    )

    assert result["split"]["check"] == "matches metrics_baseline.json"
    assert result["n_test"] == train_metrics["split"]["n_test"]
    assert result["split"]["max_samples"] == 64
    assert 0.0 <= result["accuracy"] <= 1.0
    assert {p.name for p in (out_dir / "eval").iterdir()} == {
        "eval_baseline.json", "baseline_eval.md", "confusion_matrix_baseline.png",
    }


def test_predict_cli_on_a_single_image(baseline_run, image_root, capsys):
    out_dir, _ = baseline_run
    image = image_root / CLASSES[1] / "img_00.jpg"

    results = predict.main([str(image), "--weights", str(out_dir / "baseline_best.pt"), "--top-k", "2"])

    (top,) = [r["top_k"] for r in results]
    assert len(top) == 2 and top[0][1] >= top[1][1]
    assert all(cls in CLASSES for cls, _ in top)
    assert all(0.0 <= p <= 1.0 for _, p in top)
    assert f"{image}: {top[0][0]}" in capsys.readouterr().out


def test_predict_image_matches_the_cli(baseline_run, image_root):
    out_dir, metrics = baseline_run
    image = str(image_root / CLASSES[0] / "img_05.jpg")

    cls, conf = predict.predict_image(
        image, str(out_dir / "baseline_best.pt"), metrics["classes"], backbone="baseline", img_size=IMG_SIZE,
    )
    top = predict.predict_topk(
        image, str(out_dir / "baseline_best.pt"), metrics["classes"], backbone="baseline", img_size=IMG_SIZE,
    )

    assert (cls, conf) == pytest.approx(top[0])


@pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0"])
def test_transfer_backbones_fast_dev_run_offline(backbone, image_root, tmp_path):
    # conftest blocks downloads, so this also proves --fast-dev-run never fetches ImageNet weights.
    metrics = train.train(
        backbone, data_root=str(image_root), out_dir=str(tmp_path), img_size=IMG_SIZE,
        batch_size=16, fast_dev_run=True,
    )
    assert metrics["config"]["pretrained"] is False

    results = predict.main([str(image_root / CLASSES[2] / "img_03.jpg"), "--weights",
                            str(tmp_path / f"{backbone}_best.pt")])
    assert results[0]["top_k"][0][0] in CLASSES


def test_max_samples_alone_keeps_the_other_settings(image_root, tmp_path):
    metrics = train.train(
        "baseline", data_root=str(image_root), out_dir=str(tmp_path), epochs=2, img_size=IMG_SIZE,
        num_workers=0, max_samples=20,
    )

    assert metrics["config"]["epochs"] == 2 and metrics["config"]["fast_dev_run"] is False
    assert metrics["split"]["n_total"] == 20
