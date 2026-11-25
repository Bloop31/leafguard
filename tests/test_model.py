"""Model-building checks on tiny random tensors -- no real or pretrained weights needed.

Needs torch + torchvision, so it runs in the CI torch-tests job and skips in
the lightweight one. tests/conftest.py makes any ImageNet download fail, so
these also prove ``pretrained=False`` stays offline.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from leafguard.model import BaselineCNN, build_transfer_model  # noqa: E402

NUM_CLASSES = 4


def test_baseline_forward_shape_at_any_image_size():
    model = BaselineCNN(NUM_CLASSES).eval()

    for size in (32, 224):
        with torch.no_grad():
            assert model(torch.randn(2, 3, size, size)).shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0"])
def test_transfer_model_without_pretrained_weights_builds_offline(backbone):
    model = build_transfer_model(backbone, NUM_CLASSES, pretrained=False).eval()

    with torch.no_grad():
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0"])
def test_frozen_backbone_only_trains_the_new_head(backbone):
    model = build_transfer_model(backbone, NUM_CLASSES, freeze_backbone=True, pretrained=False)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    head = "fc." if backbone == "resnet18" else "classifier.1."

    assert trainable and all(name.startswith(head) for name in trainable)

    unfrozen = build_transfer_model(backbone, NUM_CLASSES, freeze_backbone=False, pretrained=False)
    assert all(p.requires_grad for p in unfrozen.parameters())


def test_pretrained_default_still_asks_for_imagenet_weights():
    # Training keeps the old behaviour: the default tries to fetch ImageNet weights
    # (blocked here by conftest, which is how we can tell).
    with pytest.raises(RuntimeError, match="must not need the network"):
        build_transfer_model("resnet18", NUM_CLASSES)


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_transfer_model("vgg16", NUM_CLASSES, pretrained=False)
