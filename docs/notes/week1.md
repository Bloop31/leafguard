# Week 1 Notes

## Day 1 — EDA

Findings from `notebooks/day01_eda.ipynb` (run on Kaggle):

- Number of classes:
- Total images:
- Class imbalance ratio:
- Typical image size:
- Anything surprising in the sample grid:

_(fill in after running the notebook on Kaggle)_

## Day 3 — Baseline training

Results from `notebooks/day03_train_baseline.ipynb` (BaselineCNN from scratch, Kaggle GPU, seed 42,
70/15/15 train/val/test split — test set held out for Day 4):

- Best val acc:
- Epochs:
- Overfitting? (train vs val gap):
- Time per epoch:

_(fill in after running the notebook on Kaggle)_

## Day 4 — Baseline evaluation

Results from `notebooks/day04_eval_baseline.ipynb` (`baseline_best.pt` from Day 3, scored once on the held-out
test split: seed 42, 15% of the data, never used for training or checkpoint selection). Full numbers in
`reports/baseline_eval.md` / `reports/eval_baseline.json`, figure in `reports/figures/confusion_matrix_baseline.png`.

- Test accuracy:
- Macro F1:
- Test accuracy vs Day 3 best val acc:
- Worst 3 classes (by F1):
  1.
  2.
  3.
- Most-confused pairs (true -> predicted, count):
  1.
  2.
  3.

_(fill in after running the notebook on Kaggle)_
