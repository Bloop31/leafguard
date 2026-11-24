"""Checks that the conftest guard really blocks pretrained-weight downloads.

Needs torch + torchvision, so it only runs in the CI torch-tests job (or on
Kaggle) and skips cleanly in the lightweight job.
"""

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def test_torch_hub_download_is_blocked(tmp_path):
    with pytest.raises(RuntimeError, match="must not need the network"):
        torch.hub.load_state_dict_from_url("https://example.com/w.pth", model_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="must not need the network"):
        torch.hub.download_url_to_file("https://example.com/w.pth", str(tmp_path / "w.pth"))


def test_torchvision_pretrained_weights_are_blocked():
    from torchvision import models

    with pytest.raises(RuntimeError, match="must not need the network"):
        models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


def test_models_without_weights_build_offline():
    from torchvision import models

    net = models.resnet18(weights=None)
    assert net.fc.out_features == 1000
