# LeafGuard — 30-Day Roadmap

Each day = one GitHub issue = one branch = one PR = one merge = one commit day on your profile. Days are grouped into 6 weekly milestones so the project actually builds toward something real, not just filler commits.

How to use this: open an issue for "Day N" (use the *Daily task* issue template), do the work, open a PR from a branch named `day-N-<slug>`, merge it. Ask Claude for that day's code before you start — paste the day's row below into the chat and say "give me the code for this".

## Week 1 — Data + Baseline

| Day | Task | Output |
|---|---|---|
| 1 | Repo scaffold (done ✅) + explore PlantVillage dataset on Kaggle: class counts, sample images, image size distribution | `notebooks/day01_eda.ipynb`, `reports/figures/class_distribution.png` |
| 2 | Write `leafguard.data` dataloaders (already scaffolded) + verify on Kaggle with a tiny batch | Confirm `get_dataloaders()` works on real data |
| 3 | Train `BaselineCNN` from scratch for 10 epochs on Kaggle GPU | `models/baseline_best.pt`, `reports/figures/training_curves_baseline.png` |
| 4 | Evaluate baseline: confusion matrix, per-class precision/recall/F1 | `reports/baseline_eval.md` + confusion matrix figure |
| 5 | Write up Week 1 findings: is the baseline overfitting? Which classes confuse it? | `docs/notes/week1.md` |
| 6 | Add proper unit tests for `data.py` (class balance check, transform shapes) using synthetic tiny images | `tests/test_data.py` |
| 7 | Buffer day / catch-up + clean up Week 1 code, update README status | — |

## Week 2 — Transfer Learning

| Day | Task | Output |
|---|---|---|
| 8 | Train ResNet18 (frozen backbone, new head) on Kaggle | `models/resnet18_best.pt` |
| 9 | Train EfficientNetB0 (frozen backbone) on Kaggle | `models/efficientnet_b0_best.pt` |
| 10 | Compare baseline vs ResNet18 vs EfficientNetB0 — table + bar chart | `reports/model_comparison.md` |
| 11 | Fine-tune the better backbone (unfreeze last few layers), retrain | `models/<backbone>_finetuned.pt` |
| 12 | Learning-rate scheduling experiment (step decay vs cosine) | `reports/lr_schedule_experiment.md` |
| 13 | Add `tests/test_model.py` — shape/forward-pass tests using tiny random tensors, no real weights needed | `tests/test_model.py` |
| 14 | Buffer day / write Week 2 notes | `docs/notes/week2.md` |

## Week 3 — Robustness & Data Augmentation

| Day | Task | Output |
|---|---|---|
| 15 | Augmentation ablation: no-aug vs current vs heavier aug (CutMix/MixUp) | `reports/augmentation_ablation.md` |
| 16 | Class imbalance handling: weighted loss or oversampling for minority classes | Updated `train.py` with `--class-weights` flag |
| 17 | Test-time robustness: evaluate on deliberately blurred/rotated/darkened images | `reports/robustness_check.md` |
| 18 | Error analysis: pull the 20 most-confident *wrong* predictions, inspect them visually | `reports/figures/misclassified_grid.png` |
| 19 | Try a second dataset split strategy (stratified k-fold, at least 3 folds) for a more reliable accuracy estimate | `reports/kfold_results.md` |
| 20 | Buffer day / consolidate best model so far as `models/best_overall.pt` | — |
| 21 | Buffer day / Week 3 notes | `docs/notes/week3.md` |

## Week 4 — Explainability

| Day | Task | Output |
|---|---|---|
| 22 | Implement Grad-CAM for the chosen backbone | `src/leafguard/explain.py` |
| 23 | Generate Grad-CAM heatmaps for 10 correctly-classified images per class | `reports/figures/gradcam_correct/` |
| 24 | Generate Grad-CAM heatmaps for misclassified images — does the model look at the leaf or the background? | `reports/figures/gradcam_wrong/` |
| 25 | Write an explainability findings doc: does the model actually learn disease lesions, or shortcuts (pot color, lighting)? | `docs/notes/explainability.md` |
| 26 | Add a `--gradcam` flag to `predict.py` so a single image prediction also saves its heatmap | Updated `predict.py` |
| 27 | Buffer day / polish explainability code | — |
| 28 | Buffer day / Week 4 notes | `docs/notes/week4.md` |

## Week 5 — Packaging & Demo

| Day | Task | Output |
|---|---|---|
| 29 | Build a minimal Streamlit (or Gradio) demo: upload a leaf photo, get prediction + confidence + Grad-CAM overlay | `app/streamlit_app.py` |
| 30 | Final polish: model card (`docs/MODEL_CARD.md`), update README with results table, screenshot of the demo, retrospective note | Updated `README.md`, `docs/MODEL_CARD.md` |

*(Weeks 6+ are optional stretch goals — see below — for once the streak habit is solid and you want to keep going.)*

## Stretch goals (optional, after Day 30)
- Deploy the Streamlit demo to Streamlit Community Cloud (free) so it's a live link on your resume.
- Try a second dataset (e.g. a different crop) to test generalization.
- Add ONNX export + benchmark inference latency on CPU.
- Write a short blog-style `docs/WRITEUP.md` summarizing the whole project — good for LinkedIn/resume.

## Status
- [x] Day 0: Repo scaffold, CI, baseline code structure
- [ ] Day 1 — 30: see table above
