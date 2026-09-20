"""Training loop -- meant to run on Kaggle (GPU), invoked from a notebook or:

    python -m leafguard.train --backbone resnet18 --epochs 10 \
        --data-root /kaggle/input/plantdisease/PlantVillage \
        --out-dir /kaggle/working/outputs

Saves: best model weights, a metrics.json (train/val loss & accuracy per
epoch), and a training_curves.png. Copy everything in --out-dir back into
this repo's models/ and reports/figures/ before committing.
"""

import argparse
import json
import time
from pathlib import Path


def train(
    backbone: str,
    data_root: str,
    out_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    from leafguard.data import get_dataloaders
    from leafguard.model import build_transfer_model, BaselineCNN

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(data_root, batch_size=batch_size)
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")

    if backbone == "baseline":
        model = BaselineCNN(num_classes)
    else:
        model = build_transfer_model(backbone, num_classes, freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

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

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({time.time() - t0:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path / f"{backbone}_best.pt")

    with open(out_path / "metrics.json", "w") as f:
        json.dump({"history": history, "classes": classes, "best_val_acc": best_val_acc}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path / "training_curves.png", dpi=150)

    print(f"Best val_acc={best_val_acc:.4f}. Outputs saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18", choices=["baseline", "resnet18", "efficientnet_b0"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="/kaggle/working/outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        backbone=args.backbone,
        data_root=args.data_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
