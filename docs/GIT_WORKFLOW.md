# Daily Git Workflow

You do this every day. It's the same 6 steps, copy-paste. Total laptop time: ~5 minutes.

## The loop

### 1. Open today's issue
On GitHub → Issues → New issue → pick **Daily task** template → fill in "Day N: ...". Copy the task from `docs/ROADMAP.md`.

### 2. Get today's code from Claude
Open a chat with Claude and paste something like:

> "Today is Day 3 of LeafGuard. Here's the task: `Train BaselineCNN from scratch for 10 epochs on Kaggle GPU`. Give me the notebook code."

Claude writes the notebook cells (or updates a file in `src/leafguard/`). Copy what it gives you.

### 3. Create your branch
```bash
git checkout main
git pull origin main
git checkout -b day-3-baseline-training
```

### 4. Add the code
- If it's a **notebook**: paste Claude's cells into a new file `notebooks/day03_baseline_training.ipynb` (or open a Kaggle Notebook, paste it there, run it on GPU, then download the `.ipynb` back into this folder).
- If it's a **source file change** (e.g. an update to `src/leafguard/train.py`): paste it directly, overwriting the file.
- If Kaggle produced outputs (`metrics.json`, `.pt` weights, `.png` plots): download them from `/kaggle/working/outputs/` and drop them into `models/` and `reports/figures/` respectively.

### 5. Commit and push
```bash
git add .
git commit -m "Day 3: train baseline CNN on PlantVillage"
git push -u origin day-3-baseline-training
```

### 6. Open the PR and merge it
```bash
gh pr create --fill    # or open the PR on github.com — it'll prompt you
gh pr merge --squash --delete-branch
```
(No `gh` CLI? Just do it on github.com: "Compare & pull request" → "Merge pull request" → "Delete branch".)

Then check the box for today in `docs/ROADMAP.md` on `main` (a tiny follow-up commit is fine, or fold it into the same PR before merging).

## Running on Kaggle (the actual training)

1. https://www.kaggle.com/ → **New Notebook**.
2. Right panel → **Add Input** → search `plantdisease` (emmarex/plantdisease) → Add.
3. Right panel → **Accelerator** → **GPU T4 x2**.
4. Session options → **Internet: On** (needed the first time to download pretrained weights).
5. First cell of every notebook:
   ```python
   import sys
   sys.path.append("/kaggle/input/leafguard-src")  # if you've uploaded src/ as a dataset, OR
   # simplest for early days: just paste leafguard/src/leafguard/*.py contents directly into cells
   ```
   Easiest early on: skip packaging as a dataset, just paste the contents of `data.py` / `model.py` / `train.py` into cells directly, in that order, then call the functions. Once this feels repetitive (~Week 2), upload `src/` as a private Kaggle Dataset and `sys.path.append` it instead.
6. Run all cells. Training logs print epoch-by-epoch. Outputs land in `/kaggle/working/outputs/`.
7. Notebook menu → **Save Version** → "Save & Run All" so it's reproducible, then download the notebook (File → Download → .ipynb) and the `outputs/` folder (right panel → Output → download).

## If a day has no Kaggle work (docs, tests, refactors)
Skip straight to step 3 — no Kaggle needed, just edit files locally or ask Claude for the file content, paste it in, commit as usual.

## Tips for the streak
- It's fine for a "day" to be small (even just an issue + a doc update) — consistency matters more than size for the contribution graph and most achievements.
- Keep commits scoped to one issue/PR so the history stays readable.
- If you miss a day, don't try to fake it — just pick up the next issue. A realistic project history is more valuable on your GitHub profile than a padded one.
