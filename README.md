# LeafGuard 

Deep learning project to detect crop leaf diseases from images, built as a **daily-commit GitHub streak project**: one issue → one branch → one PR → one merge, every day, for 30 days.

Training runs on **Kaggle Notebooks (free GPU)** — nothing heavy runs on your laptop. Your laptop's job is only to `git pull`, review the diff, and `git merge`.

## What this is

A crop leaf disease classifier (healthy vs. diseased, multi-class) using transfer learning (EfficientNet/ResNet) on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease), with the following progression:

1. Data exploration + baseline CNN
2. Transfer learning (EfficientNetB0 / ResNet18)
3. Data augmentation + regularization experiments
4. Model comparison & hyperparameter tuning
5. Explainability (Grad-CAM) — show *why* the model flagged a leaf as diseased
6. Evaluation report (confusion matrix, per-class F1, misclassified samples)
7. Lightweight inference script + optional Streamlit demo
8. Documentation, model card, polish

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full day-by-day plan and [`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md) for exactly what to do each day.

## Repo layout

```
leafguard/
├── src/leafguard/       # importable package: data loading, model, train, predict
├── notebooks/           # Kaggle-run notebooks (one per milestone)
├── data/                # raw/processed data (git-ignored, small samples only)
├── models/              # saved model weights (git-ignored beyond a small demo model)
├── reports/figures/     # exported plots, confusion matrices, Grad-CAM images
├── tests/               # lightweight smoke tests (run free on GitHub Actions)
├── docs/                # ROADMAP.md, GIT_WORKFLOW.md
└── .github/workflows/   # CI: lint + smoke tests on every push/PR
```

## Quickstart (Kaggle — where training happens)

1. Create a new Kaggle Notebook, add the **PlantVillage** dataset, turn on **GPU T4 x2** under Accelerator.
2. Upload/paste the day's notebook from `notebooks/`.
3. Run it. It saves metrics/plots/weights into `outputs/` on Kaggle.
4. Download the changed files and add them to your local `leafguard/` clone (see the daily workflow doc).
5. Commit, push a branch, open a PR, merge.

## Quickstart (laptop — where you just merge)

```bash
git clone https://github.com/<you>/leafguard.git
cd leafguard
pip install -r requirements.txt   # optional, only needed if you want to run the tiny smoke tests locally
```

Everyday after that, see [`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md) — it's copy-paste commands.

## Status

Day 0 — repo scaffold created. Day 1 issue is open. 🚀
