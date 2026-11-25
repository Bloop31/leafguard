"""Test-set evaluation -- meant to run on Kaggle (GPU or CPU), from a notebook or:

    python -m leafguard.evaluate --backbone baseline --weights models/baseline_best.pt \
        --data-root /kaggle/input/plantdisease/PlantVillage \
        --out-dir /kaggle/working/outputs

Rebuilds the exact held-out test split that train.py set aside (same seed, same
70/15/15 ``split_indices`` call via ``get_dataloaders``), loads the checkpoint
and runs inference once. If ``metrics_{backbone}.json`` from training sits next
to the weights (or is passed with ``--metrics``), the rebuilt test indices and
class list are checked against it, so a changed dataset or seed fails loudly
instead of quietly scoring the model on images it trained on.

Saves, per backbone: ``eval_{backbone}.json`` (accuracy, macro/weighted F1,
per-class precision/recall/F1/support, top confusions, raw confusion matrix),
``confusion_matrix_{backbone}.png`` (row-normalised) and ``{backbone}_eval.md``.
Copy them back into this repo: .json and .md -> reports/, .png -> reports/figures/.

The metric helpers (``confusion_matrix``, ``per_class_report``,
``top_confusions``, ...) only need numpy, so CI unit-tests them without torch.
torch/torchvision are imported inside ``evaluate`` and matplotlib inside
``plot_confusion_matrix``.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

BACKBONES = ["baseline", "resnet18", "efficientnet_b0"]


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    """Counts matrix of shape (n_classes, n_classes): rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if n_classes < 1:
        raise ValueError(f"n_classes must be at least 1, got {n_classes}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred differ in length: {y_true.size} vs {y_pred.size}")
    for name, labels in (("y_true", y_true), ("y_pred", y_pred)):
        if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
            raise ValueError(f"{name} has labels outside 0..{n_classes - 1}")

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _safe_div(num: float, den: float) -> float:
    """num / den, or 0.0 when den is 0 (e.g. a class the model never predicts)."""
    return float(num) / float(den) if den else 0.0


def per_class_report(cm: np.ndarray, classes: Sequence[str]) -> dict:
    """Precision / recall / F1 / support per class, plus macro and weighted averages.

    Undefined ratios (no predictions for a class -> precision, no test images of
    a class -> recall) are reported as 0.0, like sklearn's ``zero_division=0``.
    Macro averages every class equally; weighted averages by support.
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion matrix must be square, got shape {cm.shape}")
    if cm.shape[0] != len(classes):
        raise ValueError(f"confusion matrix has {cm.shape[0]} rows but there are {len(classes)} classes")

    per_class = {}
    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())
        n_predicted = int(cm[:, i].sum())
        precision = _safe_div(tp, n_predicted)
        recall = _safe_div(tp, support)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    total = int(cm.sum())
    rows = list(per_class.values())
    macro_avg = {m: _safe_div(sum(r[m] for r in rows), len(rows)) for m in ("precision", "recall", "f1")}
    weighted_avg = {
        m: _safe_div(sum(r[m] * r["support"] for r in rows), total) for m in ("precision", "recall", "f1")
    }
    macro_avg["support"] = weighted_avg["support"] = total

    return {
        "accuracy": _safe_div(np.trace(cm), total),
        "n_samples": total,
        "per_class": per_class,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg,
    }


def top_confusions(cm: np.ndarray, classes: Sequence[str], k: int = 5) -> List[dict]:
    """The k most frequent (true, predicted) mistakes, most frequent first.

    Only off-diagonal cells with at least one image count, so fewer than k
    pairs come back when the model makes fewer kinds of mistake. ``rate`` is
    the share of that true class's test images that went to the wrong class.
    Ties are broken by class order so the list is deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    cm = np.asarray(cm)
    if cm.shape != (len(classes), len(classes)):
        raise ValueError(f"confusion matrix shape {cm.shape} does not match {len(classes)} classes")

    mistakes = [
        (int(cm[t, p]), t, p)
        for t in range(len(classes))
        for p in range(len(classes))
        if t != p and cm[t, p] > 0
    ]
    mistakes.sort(key=lambda m: (-m[0], m[1], m[2]))
    return [
        {
            "true": classes[t],
            "predicted": classes[p],
            "count": count,
            "rate": _safe_div(count, cm[t, :].sum()),
        }
        for count, t, p in mistakes[:k]
    ]


def row_normalise(cm: np.ndarray) -> np.ndarray:
    """Divide each row by its total so cell (i, j) = share of class i predicted as j.

    Rows with no images stay all-zero instead of becoming NaN.
    """
    cm = np.asarray(cm, dtype=np.float64)
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)


def pretty_label(name: str) -> str:
    """Readable axis label from a PlantVillage folder name.

    ``Tomato__Tomato_YellowLeaf__Curl_Virus`` -> ``Tomato YellowLeaf Curl Virus``:
    underscores become spaces and an immediately repeated word is dropped.
    """
    words = [w for w in re.split(r"_+", name) if w]
    kept = [w for i, w in enumerate(words) if i == 0 or w.lower() != words[i - 1].lower()]
    return " ".join(kept) or name


def summarise(y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[str], k: int = 5) -> dict:
    """All the torch-free metrics for one set of predictions, in the shape saved to eval_*.json."""
    classes = list(classes)
    cm = confusion_matrix(y_true, y_pred, len(classes))
    report = per_class_report(cm, classes)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_avg"]["f1"],
        "weighted_f1": report["weighted_avg"]["f1"],
        "n_test": report["n_samples"],
        "classes": classes,
        "per_class": report["per_class"],
        "macro_avg": report["macro_avg"],
        "weighted_avg": report["weighted_avg"],
        "top_confusions": top_confusions(cm, classes, k=k),
        "confusion_matrix": cm.tolist(),
    }


def worst_classes(metrics: dict, n: int = 3) -> List[tuple]:
    """The n classes with the lowest F1, as (class, f1) pairs, worst first."""
    ranked = sorted(metrics["per_class"].items(), key=lambda kv: (kv[1]["f1"], kv[0]))
    return [(cls, m["f1"]) for cls, m in ranked[:n]]


def _table_row(label: str, m: dict) -> str:
    return f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |"


def format_markdown_report(metrics: dict) -> str:
    """Markdown report: headline numbers, per-class table, worst classes, top confusions."""
    backbone = metrics.get("backbone", "model")
    lines = [f"# {backbone} -- test-set evaluation", ""]
    if "weights" in metrics:
        lines.append(f"- Weights: `{metrics['weights']}`")
    split = metrics.get("split", {})
    if split:
        lines.append(
            f"- Test split: {metrics['n_test']} images (seed {split.get('seed')}, "
            f"{split.get('val_split')} val / {split.get('test_split')} test, never used for training "
            "or checkpoint selection)"
        )
    else:
        lines.append(f"- Test images: {metrics['n_test']}")
    lines += [
        f"- Accuracy: **{metrics['accuracy']:.4f}**",
        f"- Macro F1: **{metrics['macro_f1']:.4f}**",
        f"- Weighted F1: {metrics['weighted_f1']:.4f}",
        "",
        "## Per-class metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls, m in metrics["per_class"].items():
        lines.append(_table_row(f"`{cls}`", m))
    lines.append(_table_row("**Macro avg**", metrics["macro_avg"]))
    lines.append(_table_row("**Weighted avg**", metrics["weighted_avg"]))

    lines += ["", "## Worst 3 classes by F1", ""]
    lines += [f"{i}. `{cls}` -- F1 {f1:.4f}" for i, (cls, f1) in enumerate(worst_classes(metrics), 1)]

    lines += ["", "## Most-confused pairs (true -> predicted)", ""]
    if metrics["top_confusions"]:
        lines += [
            f"{i}. `{c['true']}` -> `{c['predicted']}`: {c['count']} images "
            f"({c['rate']:.1%} of `{c['true']}`)"
            for i, c in enumerate(metrics["top_confusions"], 1)
        ]
    else:
        lines.append("No misclassifications.")
    return "\n".join(lines) + "\n"


def plot_confusion_matrix(cm: np.ndarray, classes: Sequence[str], out_path, title: str = "") -> None:
    """Save a row-normalised confusion-matrix heatmap with readable class labels."""
    import matplotlib.pyplot as plt

    norm = row_normalise(cm)
    labels = [pretty_label(c) for c in classes]
    n = len(classes)
    size = max(6.0, 0.6 * n + 3)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of true class")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=50, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title or "Confusion matrix (row-normalised)")

    for i in range(n):
        for j in range(n):
            if norm[i, j] >= 0.005:  # skip the near-empty cells so the mistakes stand out
                ax.text(
                    j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black",
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_reports(metrics: dict, out_dir) -> Dict[str, Path]:
    """Write eval_{backbone}.json, {backbone}_eval.md and confusion_matrix_{backbone}.png."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    backbone = metrics.get("backbone", "model")
    paths = {
        "json": out_path / f"eval_{backbone}.json",
        "markdown": out_path / f"{backbone}_eval.md",
        "figure": out_path / f"confusion_matrix_{backbone}.png",
    }
    with open(paths["json"], "w") as f:
        json.dump(metrics, f, indent=2)
    paths["markdown"].write_text(format_markdown_report(metrics), encoding="utf-8")
    plot_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        metrics["classes"],
        paths["figure"],
        title=f"{backbone} -- test confusion matrix (row-normalised, n={metrics['n_test']})",
    )
    return paths


def check_against_training_metrics(
    train_metrics: dict, test_indices: List[int], classes: List[str], seed: int
) -> None:
    """Raise if the rebuilt test split or class list differs from what train.py recorded."""
    split = train_metrics.get("split", {})
    if split.get("seed") is not None and split["seed"] != seed:
        raise ValueError(f"Training used seed {split['seed']} but evaluate() was called with seed {seed}")
    if train_metrics.get("classes") and list(train_metrics["classes"]) != list(classes):
        raise ValueError("Class folders differ from the ones the model was trained on")
    if split.get("test_indices") is not None and list(split["test_indices"]) != list(test_indices):
        raise ValueError(
            f"Rebuilt test split ({len(test_indices)} images) does not match the one recorded at training "
            f"time ({len(split['test_indices'])} images) -- has the dataset under data_root changed?"
        )


def data_settings(
    train_metrics: Optional[dict], img_size: Optional[int] = None, max_samples: Optional[int] = None
) -> dict:
    """The dataloader settings that rebuild the test split a training run used.

    Taken from the training ``metrics_{backbone}.json`` when there is one
    (split fractions, image size, ``max_samples`` subset), so evaluating a
    ``--fast-dev-run`` or ``--max-samples`` checkpoint needs no extra flags.
    ``img_size`` / ``max_samples`` passed explicitly win over the recorded
    values; without a metrics file the defaults match train.py's.
    """
    train_metrics = train_metrics or {}
    split = train_metrics.get("split", {})
    config = train_metrics.get("config", {})
    return {
        "val_split": split.get("val_split", 0.15),
        "test_split": split.get("test_split", 0.15),
        "img_size": img_size if img_size is not None else config.get("img_size", 224),
        "max_samples": max_samples if max_samples is not None else split.get("max_samples"),
    }


def evaluate(
    backbone: str,
    weights_path: str,
    data_root: str,
    out_dir: str,
    batch_size: int = 32,
    seed: int = 42,
    metrics_path: Optional[str] = None,
    img_size: Optional[int] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 2,
) -> dict:
    """Score a trained checkpoint on the held-out test split and save the reports.

    ``metrics_path`` is the training ``metrics_{backbone}.json`` used to
    cross-check the split; by default it is looked for next to the weights.
    Split fractions, image size and subset size come from it too (see
    ``data_settings``).
    """
    import torch

    from leafguard.data import get_dataloaders
    from leafguard.model import build_transfer_model, BaselineCNN

    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}, expected one of {BACKBONES}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, seed: {seed}")

    if metrics_path:
        metrics_file = Path(metrics_path)
    else:
        metrics_file = Path(weights_path).with_name(f"metrics_{backbone}.json")
    train_metrics = None
    if metrics_file.is_file():
        with open(metrics_file) as f:
            train_metrics = json.load(f)
    elif metrics_path:
        raise FileNotFoundError(f"Training metrics not found: {metrics_path}")
    settings = data_settings(train_metrics, img_size=img_size, max_samples=max_samples)

    # Same split settings as the training run, so split_indices hands back the same test set.
    _, _, test_loader, classes = get_dataloaders(
        data_root, batch_size=batch_size, seed=seed, num_workers=num_workers, **settings
    )
    test_indices = list(test_loader.dataset.indices)

    if train_metrics is not None:
        check_against_training_metrics(train_metrics, test_indices, classes, seed)
        split_check = f"matches {metrics_file.name}"
    else:
        split_check = "skipped (no training metrics file next to the weights)"
    print(f"Test split: {len(test_indices)} images, {len(classes)} classes; split check {split_check}")

    if backbone == "baseline":
        model = BaselineCNN(len(classes))
    else:
        # pretrained=False: the checkpoint replaces every weight, so don't download ImageNet ones.
        model = build_transfer_model(backbone, len(classes), freeze_backbone=True, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs.to(device))
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())

    metrics = {
        "backbone": backbone,
        "weights": Path(weights_path).name,
        "device": str(device),
        "split": {
            "seed": seed,
            "val_split": settings["val_split"],
            "test_split": settings["test_split"],
            "max_samples": settings["max_samples"],
            "n_test": len(test_indices),
            "check": split_check,
        },
        **summarise(y_true, y_pred, classes),
    }
    paths = save_reports(metrics, out_dir)
    print(
        f"Test accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"(n={metrics['n_test']}). Outputs saved to {Path(out_dir)}: "
        + ", ".join(p.name for p in paths.values())
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the held-out test split.")
    parser.add_argument("--backbone", default="baseline", choices=BACKBONES)
    parser.add_argument("--weights", required=True, help="state_dict from train.py, e.g. baseline_best.pt")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="/kaggle/working/outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metrics", default=None,
        help="training metrics_{backbone}.json to cross-check the split (default: next to --weights)",
    )
    parser.add_argument("--img-size", type=int, default=None, help="default: the size used in training")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="subset size (default: whatever the training run used, else all images)",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (0 = main process)")
    return parser


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    evaluate(
        backbone=args.backbone,
        weights_path=args.weights,
        data_root=args.data_root,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        metrics_path=args.metrics,
        img_size=args.img_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
