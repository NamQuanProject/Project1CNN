# Multi-Label MNIST-Digits CNN

Multi-label image classification: unlike standard MNIST (one digit per image), each
image here contains **multiple digits**, and the model must predict the full set of
digit classes present (a 10-dim multi-hot vector, one entry per digit `0`-`9`).

See [`task.txt`](task.txt) for the full assignment description and
[`cnn/cnn.ipynb`](cnn/cnn.ipynb) for the original notebook (dataset walkthrough,
metric explanations, baseline model, and the lab exercise instructions in its final
cell).

## Project layout

```
task.txt                  Assignment description
requirements.txt          Python dependencies
cnn/
  cnn.ipynb                Original notebook: data walkthrough, baseline model, exercise instructions
data/
  train.pt, val.pt, test.pt   Dataset splits (images + multi-hot labels + metadata)
src/
  data.py                Load .pt splits into Keras-friendly numpy arrays
  metrics.py              exact_match_accuracy, per_position_accuracy
  model.py                 build_cnn_model (baseline) and build_improved_cnn_model
  train.py                 Headless training entrypoint (CLI)
  evaluate.py               Re-evaluate a saved checkpoint on the test set
  compare.py                 Diff two runs' test_metrics.json into a comparison table
outputs/                  Generated per-run artifacts (git-ignored)
```

`data/` and `src/` are kept at the repo root, separate from `cnn/` (which holds only
the original notebook) — the pipeline scripts don't depend on the notebook or vice
versa. `cnn/cnn.ipynb` loads data via `../data/*.pt` since it lives one level down in
`cnn/`; `src/train.py` and `src/evaluate.py` default to `<repo root>/data` and
`<repo root>/outputs` regardless of the current working directory.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset (`data/*.pt`) is expected to already be in place; it is git-ignored
due to size (see [`.gitignore`](.gitignore)) so make sure it's present before training.

## Running the pipeline

Train the baseline model (matches `cnn.ipynb` exactly):

```bash
python src/train.py --model baseline
```

Train the improved model (BatchNorm + extra conv block + augmentation — a starting
point for the "propose improvements" part of the assignment, tune as needed):

```bash
python src/train.py --model improved --augment --run-name improved_v1
```

Each run writes to `outputs/<run-name>/`:
- `model_summary.txt` — layer-by-layer architecture
- `training_curves.png` — loss / binary-accuracy curves
- `history.json` — full per-epoch training history
- `best_model.keras`, `final_model.keras` — saved checkpoints
- `test_metrics.json` — test-set loss, binary_accuracy, precision, recall,
  exact_match_accuracy, and per-digit (per-position) accuracy

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--patience`, `--seed`,
`--data-dir`, `--output-dir`. Run `python src/train.py --help` for the full list.

### Re-evaluating a saved checkpoint

```bash
python src/evaluate.py --model-path outputs/baseline_<timestamp>/final_model.keras
```

Writes `test_metrics.json` and a `sample_predictions.png` grid to
`outputs/eval/` by default.

### Comparing baseline vs. improved

```bash
python src/compare.py \
  --baseline outputs/baseline_<timestamp>/test_metrics.json \
  --improved outputs/improved_v1/test_metrics.json \
  --output outputs/comparison.md
```

Produces the overall-metrics and per-digit-accuracy comparison table needed for the
assignment's "Results" section.

## Evaluation metrics

- **Binary accuracy** — per-label correctness across all 10 outputs (can be
  misleading since most labels are 0).
- **Precision / Recall** — correctness / completeness of predicted digit sets.
- **Exact-match accuracy** — strictest metric: correct only if all 10 label
  positions match. This is the primary metric the assignment asks you to improve.
- **Per-position accuracy** — accuracy of each individual digit's presence/absence
  prediction, useful for error analysis (e.g. which digits are most often confused).

Prediction threshold is fixed at `0.5` (`src/metrics.py:PREDICTION_THRESHOLD`) —
per the notebook, students are not meant to change this.
