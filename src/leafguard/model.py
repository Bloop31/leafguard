"""Model definitions: a from-scratch baseline CNN, and a transfer-learning
head builder for EfficientNetB0 / ResNet18.

Heavy imports (torch, torchvision) are done at module level here since this
file is only ever imported on Kaggle when actually training/predicting;
tests import it inside a try/except so CI stays green without torch.
"""

try:
    import torch.nn as nn
    from torchvision import models

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where torch is absent
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch/torchvision are required for leafguard.model but are not "
            "installed. Run this on Kaggle, or `pip install torch torchvision`."
        )


class BaselineCNN:
    """Factory for a small from-scratch CNN -- Day 1 baseline.

    Kept intentionally simple: 3 conv blocks + a classifier head. The point
    of Day 1 is to have *something* trained end-to-end to compare later
    transfer-learning models against.
    """

    def __new__(cls, num_classes: int):
        _require_torch()

        class _Net(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)

        return _Net(num_classes)


def build_transfer_model(backbone: str, num_classes: int, freeze_backbone: bool = True):
    """Build an EfficientNetB0 or ResNet18 with a fresh classifier head.

    Args:
        backbone: "efficientnet_b0" or "resnet18"
        num_classes: number of leaf-disease classes
        freeze_backbone: if True, only the new classifier head is trained
            (fast, good for the first transfer-learning experiment). Set to
            False later for fine-tuning the whole network.
    """
    _require_torch()

    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. Use 'resnet18' or 'efficientnet_b0'.")

    return net
