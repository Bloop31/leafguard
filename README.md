# LeafGuard

![CI](https://github.com/Bloop31/leafguard/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)


A deep learning model that detects crop leaf diseases from a single photo — built to help identify plant health issues early, before they spread across a field.

## Overview

LeafGuard classifies leaf images into healthy vs. diseased categories (with disease type) using the [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) dataset. The project starts with a CNN trained from scratch as a baseline, then moves to transfer learning with pretrained backbones (ResNet18 / EfficientNetB0) to push accuracy higher, and eventually adds Grad-CAM visual explanations so predictions aren't a black box — you can see *which part of the leaf* the model is reacting to.

Training runs on Kaggle's free GPU notebooks; this repo holds the reusable code, experiment tracking, and results.

## Project goals

- Build an accurate, well-evaluated leaf disease classifier
- Compare a from-scratch CNN against transfer-learning approaches
- Handle real-world issues: class imbalance, overfitting, robustness to lighting/angle
- Make predictions explainable with Grad-CAM
- Ship a small demo where you can upload a photo and get a prediction


## Current status

**Early stage.** Repo scaffolding, data pipeline, and baseline model code are in place. Exploratory data analysis on the PlantVillage dataset is next.

## Setup

```bash
git clone https://github.com/Bloop31/leafguard.git
cd leafguard
pip install -r requirements.txt
```

Training requires a GPU — I run experiments on Kaggle Notebooks (PlantVillage dataset attached, GPU accelerator on) and pull results back into this repo.

## Tech stack

Python, PyTorch, torchvision, scikit-learn, matplotlib, pytest — CI runs lint + tests on every push via GitHub Actions.
