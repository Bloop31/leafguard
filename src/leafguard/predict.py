"""Lightweight inference helpers, safe to run on a laptop CPU.

`predict_image` / `predict_topk` are meant to run outside Kaggle (e.g. for a
small demo/Streamlit app later) -- they still need torch installed, but run
fine on CPU for single-image inference, and never download pretrained weights
(the checkpoint supplies every weight). From the command line:

    python -m leafguard.predict --weights models/baseline_best.pt leaf1.jpg leaf2.jpg

Class names, backbone and image size are read from the training
``metrics_{backbone}.json`` next to the weights (or ``--metrics``).
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

BACKBONES = ["baseline", "resnet18", "efficientnet_b0"]


def load_model(weights_path: str, backbone: str, num_classes: int, device=None):
    """Build the architecture for ``backbone`` and load a train.py state_dict into it (eval mode)."""
    import torch

    from leafguard.model import build_transfer_model, BaselineCNN

    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}, expected one of {BACKBONES}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backbone == "baseline":
        model = BaselineCNN(num_classes)
    else:
        # pretrained=False: the checkpoint replaces every weight, so don't download ImageNet ones.
        model = build_transfer_model(backbone, num_classes, freeze_backbone=True, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model.to(device).eval()


def predict_topk(
    image_path: str,
    model_path: str,
    classes: List[str],
    backbone: str = "resnet18",
    img_size: int = 224,
    k: int = 3,
    model=None,
) -> List[Tuple[str, float]]:
    """The k most likely classes for one image, as (class, probability) pairs, most likely first.

    Pass an already-loaded ``model`` (from ``load_model``) to skip reloading
    the checkpoint for every image.
    """
    import torch
    from PIL import Image

    from leafguard.data import build_transforms

    if k <= 0:
        raise ValueError("k must be positive")
    if model is None:
        model = load_model(model_path, backbone, len(classes))
    device = next(model.parameters()).device
    _, eval_tfms = build_transforms(img_size)  # same preprocessing as val/test

    img = Image.open(image_path).convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    if probs.numel() != len(classes):
        raise ValueError(f"model outputs {probs.numel()} classes but {len(classes)} class names were given")
    top = torch.topk(probs, min(k, len(classes)))
    return [(classes[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]


def predict_image(
    image_path: str,
    model_path: str,
    classes: List[str],
    backbone: str = "resnet18",
    img_size: int = 224,
) -> Tuple[str, float]:
    """Run inference on a single image and return (predicted_class, confidence)."""
    return predict_topk(image_path, model_path, classes, backbone=backbone, img_size=img_size, k=1)[0]


def load_classes_from_metrics(metrics_path: str) -> List[str]:
    """Convenience: read the `classes` list saved alongside training metrics."""
    with open(metrics_path) as f:
        data = json.load(f)
    return data["classes"]


def find_metrics_file(weights_path: str, backbone: Optional[str] = None) -> Path:
    """The training metrics JSON for a checkpoint: ``metrics_{backbone}.json`` next to it.

    Without ``backbone`` it is guessed from train.py's ``{backbone}_best.pt``
    naming, falling back to the only ``metrics_*.json`` in the folder.
    """
    weights = Path(weights_path)
    if backbone is None and weights.name.endswith("_best.pt"):
        backbone = weights.name[: -len("_best.pt")]
    if backbone is not None:
        candidate = weights.with_name(f"metrics_{backbone}.json")
        if candidate.is_file():
            return candidate
    found = sorted(weights.parent.glob("metrics_*.json"))
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        f"Could not find the training metrics JSON for {weights} -- pass it with --metrics"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict the leaf-disease class of one or more images.")
    parser.add_argument("images", nargs="+", help="image file(s) to classify")
    parser.add_argument("--weights", required=True, help="state_dict from train.py, e.g. baseline_best.pt")
    parser.add_argument(
        "--metrics", default=None,
        help="training metrics_{backbone}.json with the class names (default: next to --weights)",
    )
    parser.add_argument("--backbone", default=None, choices=BACKBONES, help="default: from the metrics file")
    parser.add_argument("--img-size", type=int, default=None, help="default: from the metrics file, else 224")
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main(argv: Optional[List[str]] = None) -> List[dict]:
    args = build_parser().parse_args(argv)
    metrics_file = Path(args.metrics) if args.metrics else find_metrics_file(args.weights, args.backbone)
    with open(metrics_file) as f:
        train_metrics = json.load(f)
    classes = train_metrics["classes"]
    backbone = args.backbone or train_metrics.get("backbone", "resnet18")
    img_size = args.img_size or train_metrics.get("config", {}).get("img_size", 224)

    model = load_model(args.weights, backbone, len(classes))
    print(f"{backbone} from {Path(args.weights).name}: {len(classes)} classes, {img_size}px")

    results = []
    for image in args.images:
        top = predict_topk(
            image, args.weights, classes, backbone=backbone, img_size=img_size, k=args.top_k, model=model
        )
        print(f"{image}: {top[0][0]} ({top[0][1]:.1%})")
        for cls, p in top[1:]:
            print(f"    {cls} ({p:.1%})")
        results.append({"image": image, "top_k": top})
    return results


if __name__ == "__main__":
    main()
