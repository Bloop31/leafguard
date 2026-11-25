"""Shared pytest setup.

Tests must never need the network. When torch is installed (the CI
"torch-tests" job, or Kaggle), every download path torch/torchvision use for
pretrained weights is replaced with one that fails the test, so a model built
with ImageNet weights by mistake shows up as a clear error instead of a
silent ~20-100 MB download. Without torch this file does nothing.
"""

import importlib.util

import pytest


def _no_downloads(*args, **kwargs):
    url = args[0] if args else kwargs.get("url", "<unknown>")
    raise RuntimeError(
        f"Test tried to download {url}. Tests must not need the network: build transfer "
        "models with pretrained weights turned off."
    )


@pytest.fixture(autouse=True)
def block_weight_downloads(monkeypatch):
    if importlib.util.find_spec("torch") is None:
        yield
        return

    import torch.hub

    # load_state_dict_from_url (used by torchvision's WeightsEnum.get_state_dict)
    # looks download_url_to_file up on torch.hub at call time, so patching the
    # module attribute catches every pretrained-weights download.
    monkeypatch.setattr(torch.hub, "download_url_to_file", _no_downloads)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _no_downloads)
    # torchvision binds its own reference at import time; patching it too also
    # rejects weights that happen to be cached locally already (e.g. on Kaggle).
    if importlib.util.find_spec("torchvision") is not None:
        import torchvision.models._api as tv_api

        monkeypatch.setattr(tv_api, "load_state_dict_from_url", _no_downloads, raising=False)
    yield
