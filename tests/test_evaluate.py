"""Tests for the torch-free evaluation helpers -- these run in CI.

The inference part of ``evaluate()`` needs torch and real weights, so it is
exercised on Kaggle (notebooks/day04_eval_baseline.ipynb); everything that
turns predictions into metrics and reports is checked here on tiny,
hand-made label lists.
"""

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leafguard.evaluate import (  # noqa: E402
    build_parser,
    check_against_training_metrics,
    confusion_matrix,
    format_markdown_report,
    per_class_report,
    pretty_label,
    row_normalise,
    save_reports,
    summarise,
    top_confusions,
    worst_classes,
)

CLASSES = ["healthy", "rust", "blight"]


@pytest.fixture(autouse=True)
def fail_on_numeric_warnings():
    """Any divide-by-zero / invalid-value warning in the helpers fails the test."""
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        yield


# --- confusion_matrix -------------------------------------------------------


def test_confusion_matrix_rows_are_true_columns_are_predicted():
    cm = confusion_matrix([0, 0, 1, 2, 2, 2], [0, 1, 1, 2, 2, 0], n_classes=3)

    assert cm.tolist() == [
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 2],
    ]
    assert cm.sum() == 6


def test_confusion_matrix_keeps_classes_missing_from_both_lists():
    cm = confusion_matrix([0, 0], [0, 0], n_classes=3)

    assert cm.shape == (3, 3)
    assert cm.tolist() == [[2, 0, 0], [0, 0, 0], [0, 0, 0]]


@pytest.mark.parametrize(
    "y_true, y_pred, n_classes",
    [([0, 1], [0], 2), ([0, 2], [0, 1], 2), ([0, -1], [0, 1], 2), ([0], [0], 0)],
)
def test_confusion_matrix_rejects_bad_input(y_true, y_pred, n_classes):
    with pytest.raises(ValueError):
        confusion_matrix(y_true, y_pred, n_classes)


# --- per_class_report -------------------------------------------------------


def test_perfect_predictions_score_one_everywhere():
    y = [0, 0, 1, 1, 1, 2]
    report = per_class_report(confusion_matrix(y, y, 3), CLASSES)

    assert report["accuracy"] == 1.0
    for m in report["per_class"].values():
        assert (m["precision"], m["recall"], m["f1"]) == (1.0, 1.0, 1.0)
    assert report["macro_avg"]["f1"] == 1.0
    assert report["weighted_avg"]["f1"] == 1.0
    assert [m["support"] for m in report["per_class"].values()] == [2, 3, 1]


def test_all_wrong_predictions_score_zero_without_nans():
    report = per_class_report(confusion_matrix([0, 1, 2, 0], [1, 2, 0, 2], 3), CLASSES)

    assert report["accuracy"] == 0.0
    for m in report["per_class"].values():
        assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 0.0)
    assert report["macro_avg"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 4}


def test_class_never_predicted_gets_zero_precision_not_divide_by_zero():
    # "blight" (2) appears in y_true but the model never predicts it.
    report = per_class_report(confusion_matrix([0, 0, 1, 1, 2], [0, 0, 0, 1, 1], 3), CLASSES)
    blight = report["per_class"]["blight"]

    assert blight == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert all(math.isfinite(v) for m in report["per_class"].values() for v in m.values())
    json.dumps(report)  # no NaN/inf or numpy types sneaking into the saved JSON


def test_class_absent_from_test_set_gets_zero_recall_and_support():
    report = per_class_report(confusion_matrix([0, 1], [0, 1], 3), CLASSES)

    assert report["per_class"]["blight"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    # Macro counts every class, so the empty one drags it down; weighted ignores it.
    assert report["macro_avg"]["f1"] == pytest.approx(2 / 3)
    assert report["weighted_avg"]["f1"] == 1.0


def test_known_precision_recall_f1_values():
    # cm = [[3, 1],
    #       [2, 4]]
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    cm = confusion_matrix(y_true, y_pred, 2)
    assert cm.tolist() == [[3, 1], [2, 4]]

    report = per_class_report(cm, ["a", "b"])
    a, b = report["per_class"]["a"], report["per_class"]["b"]

    assert report["accuracy"] == pytest.approx(0.7)
    assert a["precision"] == pytest.approx(3 / 5)
    assert a["recall"] == pytest.approx(3 / 4)
    assert a["f1"] == pytest.approx(2 / 3)
    assert b["precision"] == pytest.approx(4 / 5)
    assert b["recall"] == pytest.approx(4 / 6)
    assert b["f1"] == pytest.approx(8 / 11)
    assert report["macro_avg"]["precision"] == pytest.approx(0.7)
    assert report["macro_avg"]["f1"] == pytest.approx((2 / 3 + 8 / 11) / 2)
    assert report["weighted_avg"]["recall"] == pytest.approx(0.7)  # weighted recall == accuracy
    assert report["weighted_avg"]["f1"] == pytest.approx((4 * 2 / 3 + 6 * 8 / 11) / 10)
    assert report["macro_avg"]["support"] == report["weighted_avg"]["support"] == 10


def test_per_class_report_rejects_mismatched_classes():
    with pytest.raises(ValueError):
        per_class_report(np.zeros((3, 3), dtype=int), ["a", "b"])
    with pytest.raises(ValueError):
        per_class_report(np.zeros((2, 3), dtype=int), ["a", "b"])


def test_empty_predictions_do_not_divide_by_zero():
    report = per_class_report(confusion_matrix([], [], 3), CLASSES)

    assert report["accuracy"] == 0.0
    assert report["weighted_avg"]["f1"] == 0.0


# --- top_confusions ---------------------------------------------------------


def test_top_confusions_orders_by_count_and_skips_the_diagonal():
    cm = np.array([
        [50, 4, 1],
        [6, 40, 0],
        [4, 0, 30],
    ])
    top = top_confusions(cm, CLASSES, k=3)

    assert [(c["true"], c["predicted"], c["count"]) for c in top] == [
        ("rust", "healthy", 6),
        ("healthy", "rust", 4),  # ties (4 vs 4) go by class order
        ("blight", "healthy", 4),
    ]
    assert top[0]["rate"] == pytest.approx(6 / 46)


def test_top_confusions_returns_fewer_than_k_when_there_are_few_mistakes():
    assert top_confusions(np.diag([3, 3, 3]), CLASSES, k=5) == []
    assert len(top_confusions(np.array([[3, 1, 0], [0, 3, 0], [0, 0, 3]]), CLASSES, k=5)) == 1


def test_top_confusions_rejects_bad_k_and_shape():
    with pytest.raises(ValueError):
        top_confusions(np.diag([1, 1, 1]), CLASSES, k=0)
    with pytest.raises(ValueError):
        top_confusions(np.diag([1, 1]), CLASSES)


# --- row_normalise / labels -------------------------------------------------


def test_row_normalise_rows_sum_to_one_and_empty_rows_stay_zero():
    norm = row_normalise(np.array([[3, 1, 0], [0, 0, 0], [1, 1, 2]]))

    assert norm[0].tolist() == [0.75, 0.25, 0.0]
    assert norm[1].tolist() == [0.0, 0.0, 0.0]
    assert norm[2].sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "folder, label",
    [
        ("Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato YellowLeaf Curl Virus"),
        ("Pepper__bell___Bacterial_spot", "Pepper bell Bacterial spot"),
        ("Potato___healthy", "Potato healthy"),
        ("healthy", "healthy"),
    ],
)
def test_pretty_label(folder, label):
    assert pretty_label(folder) == label


# --- summary + reports ------------------------------------------------------


def test_summarise_has_everything_the_json_report_needs():
    metrics = summarise([0, 0, 1, 1, 2], [0, 0, 0, 1, 1], CLASSES)

    assert metrics["accuracy"] == pytest.approx(3 / 5)
    assert metrics["n_test"] == 5
    assert metrics["macro_f1"] == pytest.approx(metrics["macro_avg"]["f1"])
    assert list(metrics["per_class"]) == CLASSES
    assert metrics["confusion_matrix"] == [[2, 0, 0], [1, 1, 0], [0, 1, 0]]
    assert len(metrics["top_confusions"]) == 2
    assert worst_classes(metrics, n=1) == [("blight", 0.0)]
    json.dumps(metrics)


def test_markdown_report_lists_every_class_and_the_top_confusions():
    metrics = {"backbone": "baseline", **summarise([0, 0, 1, 1, 2], [0, 0, 0, 1, 1], CLASSES)}
    md = format_markdown_report(metrics)

    assert md.startswith("# baseline -- test-set evaluation")
    assert "| Class | Precision | Recall | F1 | Support |" in md
    for cls in CLASSES:
        assert f"| `{cls}` |" in md
    assert "**Macro avg**" in md and "**Weighted avg**" in md
    assert "`rust` -> `healthy`: 1 images (50.0% of `rust`)" in md


def test_markdown_report_with_no_mistakes():
    md = format_markdown_report(summarise([0, 1, 2], [0, 1, 2], CLASSES))

    assert "No misclassifications." in md


def test_save_reports_writes_json_markdown_and_figure(tmp_path):
    metrics = {"backbone": "baseline", **summarise([0, 0, 1, 1, 2], [0, 0, 0, 1, 1], CLASSES)}

    # matplotlib's own warnings / float handling aren't what the autouse fixture is guarding.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        paths = save_reports(metrics, tmp_path / "outputs")

    assert paths["json"].name == "eval_baseline.json"
    assert paths["markdown"].name == "baseline_eval.md"
    assert paths["figure"].name == "confusion_matrix_baseline.png"
    assert json.loads(paths["json"].read_text())["accuracy"] == pytest.approx(0.6)
    assert paths["figure"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


TRAIN_METRICS = {"classes": CLASSES, "split": {"seed": 42, "test_indices": [8, 0, 3]}}


def test_split_check_passes_when_test_set_matches_training():
    check_against_training_metrics(TRAIN_METRICS, [8, 0, 3], CLASSES, seed=42)


@pytest.mark.parametrize(
    "test_indices, classes, seed",
    [
        ([8, 0, 4], CLASSES, 42),  # dataset changed -> different test images
        ([8, 0, 3], ["healthy", "rust"], 42),  # class folders changed
        ([8, 0, 3], CLASSES, 7),  # wrong seed
    ],
)
def test_split_check_fails_loudly_on_mismatch(test_indices, classes, seed):
    with pytest.raises(ValueError):
        check_against_training_metrics(TRAIN_METRICS, test_indices, classes, seed)


def test_cli_arguments():
    args = build_parser().parse_args([
        "--backbone", "baseline",
        "--weights", "models/baseline_best.pt",
        "--data-root", "/data/PlantVillage",
    ])

    assert args.weights == "models/baseline_best.pt"
    assert args.out_dir == "/kaggle/working/outputs"
    assert (args.batch_size, args.seed, args.metrics) == (32, 42, None)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--backbone", "vgg", "--weights", "w.pt", "--data-root", "d"])


def test_module_imports_without_torch():
    import leafguard.evaluate as ev

    assert callable(ev.evaluate)
