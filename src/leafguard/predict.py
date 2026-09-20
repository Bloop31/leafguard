"""Lightweight inference helpers.

`predict_image` is the only function meant to run outside Kaggle (e.g. for
a small demo/Streamlit app later) -- it still needs torch installed, but
runs fine on CPU for single-image inference.
"""

from typing import List, Tuple


def predict_image(
    image_path: str,
    model_path: str,
    classes: List[str],
    backbone: str = "resnet18",
) -> Tuple[str, float]:
    """Run inference on a single image and return (predicted_class, confidence)."""
    import torch
    from PIL import Image
    from torchvision import transforms

    from leafguard.model import build_transfer_model, BaselineCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if backbone == "baseline":
        model = BaselineCNN(len(classes))
    else:
        model = build_transfer_model(backbone, len(classes), freeze_backbone=True)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    x = tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)

    return classes[idx.item()], conf.item()


def load_classes_from_metrics(metrics_path: str) -> List[str]:
    """Convenience: read the `classes` list saved alongside training metrics."""
    import json

    with open(metrics_path) as f:
        data = json.load(f)
    return data["classes"]
