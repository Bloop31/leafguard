# LeafGuard

![CI](https://github.com/Bloop31/leafguard/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)


A deep learning model that detects crop leaf diseases from a single photo — built to help identify plant health issues early, before they spread across a field.

## Overview

LeafGuard classifies leaf images into healthy vs. diseased categories (with disease type) using the [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) dataset. The project starts with a CNN trained from scratch as a baseline, then moves to transfer learning with pretrained backbones (ResNet18 / EfficientNetB0) to push accuracy higher, and eventually adds Grad-CAM visual explanations so predictions aren't a black box — you can see *which part of the leaf* the model is reacting to.


## Project goals

- Build an accurate, well-evaluated leaf disease classifier
- Compare a from-scratch CNN against transfer-learning approaches
- Handle real-world issues: class imbalance, overfitting, robustness to lighting/angle
- Make predictions explainable with Grad-CAM
- Ship a small demo where you can upload a photo and get a prediction


## Current status

**Early stage.** Repo scaffolding, data pipeline, and baseline model code are in place. Exploratory data analysis on the PlantVillage dataset is done. The whole train → evaluate → predict pipeline can be checked on a laptop CPU in seconds (see below); real training results from Kaggle are not in the repo yet.

## Setup

```bash
git clone https://github.com/Bloop31/leafguard.git
cd leafguard
pip install -r requirements.txt
```

Real training needs a GPU — I run experiments on Kaggle Notebooks (PlantVillage dataset attached, GPU accelerator on) and pull results back into this repo.

## Quick CPU check (no GPU, no dataset, no downloads)

Install CPU-only PyTorch (~200 MB, optional — only needed for this check and the torch tests):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then run the smoke check. With no arguments it makes a tiny synthetic dataset, trains for one epoch, evaluates on the held-out test split and predicts one image, in about 10 seconds:

```bash
python scripts/cpu_smoke.py                                    # synthetic images
python scripts/cpu_smoke.py --data-root path/to/PlantVillage   # or your real data
```

The same steps by hand (`PYTHONPATH=src`, or run from a notebook that adds `src/` to `sys.path`):

```bash
python -m leafguard.train --backbone baseline --fast-dev-run --data-root path/to/PlantVillage --out-dir outputs/dev
python -m leafguard.evaluate --backbone baseline --weights outputs/dev/baseline_best.pt --data-root path/to/PlantVillage --out-dir outputs/dev
python -m leafguard.predict --weights outputs/dev/baseline_best.pt leaf.jpg
```

- `--fast-dev-run` = 1 epoch, a seeded 64-image subset, no ImageNet weight download, no dataloader workers. It checks the plumbing; its accuracy means nothing.
- `--max-samples N` trains on an N-image subset with every other setting unchanged; `--no-pretrained` starts ResNet18/EfficientNetB0 from random weights; `--img-size` and `--num-workers` are exposed too.
- `evaluate` and `predict` read the image size, subset and class names from the `metrics_{backbone}.json` that training writes next to the weights, so they need no extra flags. Loading a checkpoint never downloads pretrained weights.

## Tech stack

Python, PyTorch, torchvision, scikit-learn, matplotlib, pytest.

CI (GitHub Actions, every push) runs two jobs: flake8 lint + the torch-free tests, and the full test suite again with CPU-only torch/torchvision installed, so the model, transform and dataloader tests run too — including an end-to-end train → evaluate → predict run on synthetic images (`tests/test_pipeline.py`). No test downloads pretrained weights.
