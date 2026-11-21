"""Training loop -- meant to run on Kaggle (GPU), invoked from a notebook or:

    python -m leafguard.train --backbone baseline --epochs 10 --seed 42 \
        --data-root /kaggle/input/plantdisease/PlantVillage \
        --out-dir /kaggle/working/outputs

Saves, per backbone: `{backbone}_best.pt` (weights from the epoch with the
best val accuracy), `metrics_{backbone}.json` (per-epoch history, classes,
and the seed + split sizes + test indices needed to rebuild the exact test
set later) and `training_curves_{backbone}.png`. Copy them back into this
repo: .pt -> models/ (gitignored, kept locally), .png -> reports/figures/,
.json -> reports/.

The test split is built but never touched here: checkpoints are chosen on
val only, so test accuracy stays an honest estimate for Day 4 evaluation.
"""

import argparse
import json
import random
import time
from pathlib import Path


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + all GPUs).

    This makes the data split, weight init, shuffling and augmentation
    repeatable. GPU kernels (cuDNN) can still differ slightly run to run.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    backbone: str,
    data_root: str,
    out_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> dict:
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    from leafguard.data import get_dataloaders
    from leafguard.model import build_transfer_model, BaselineCNN

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, seed: {seed}")

    # test_loader is deliberately unused: only its indices are recorded, for Day 4.
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_root, batch_size=batch_size, val_split=val_split, test_split=test_split, seed=seed
    )
    num_classes = len(classes)
    n_train, n_val, n_test = (len(dl.dataset) for dl in (train_loader, val_loader, test_loader))
    print(f"Found {num_classes} classes: {classes}")
    print(f"Split (seed={seed}): train={n_train} val={n_val} test={n_test} (test held out)")

    if backbone == "baseline":
        model = BaselineCNN(num_classes)
    else:
        model = build_transfer_model(backbone, num_classes, freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time_s": []}
    best_val_acc = 0.0
    best_epoch = None

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_s"].append(round(epoch_time, 1))

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({epoch_time:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), out_path / f"{backbone}_best.pt")

    metrics = {
        "backbone": backbone,
        "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed},
        "history": history,
        "classes": classes,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        # Everything Day 4 needs to rebuild the exact same held-out test set:
        # split_indices(n_total, val_split, test_split, seed) reproduces it, and
        # test_indices is stored too as a direct cross-check.
        "split": {
            "seed": seed,
            "val_split": val_split,
            "test_split": test_split,
            "n_total": n_train + n_val + n_test,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "test_indices": list(test_loader.dataset.indices),
        },
    }
    with open(out_path / f"metrics_{backbone}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    epoch_nums = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epoch_nums, history["train_loss"], label="train")
    axes[0].plot(epoch_nums, history["val_loss"], label="val")
    axes[0].set_title(f"{backbone} -- loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epoch_nums, history["train_acc"], label="train")
    axes[1].plot(epoch_nums, history["val_acc"], label="val")
    axes[1].set_title(f"{backbone} -- accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path / f"training_curves_{backbone}.png", dpi=150)
    plt.close(fig)

    print(f"Best val_acc={best_val_acc:.4f} (epoch {best_epoch}). Outputs saved to {out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18", choices=["baseline", "resnet18", "efficientnet_b0"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="/kaggle/working/outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        backbone=args.backbone,
        data_root=args.data_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
