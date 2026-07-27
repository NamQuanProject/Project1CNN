# Multi-Label MNIST-Digits CNN

Multi-label image classification: unlike standard MNIST (one digit per image), each
image here contains **multiple digits**, and the model must predict the full set of
digit classes present (a 10-dim multi-hot vector, one entry per digit `0`-`9`).

See [`task.txt`](task.txt) for the full assignment description and
[`cnn/cnn.ipynb`](cnn/cnn.ipynb) for the instructor-provided starter notebook
(dataset walkthrough, metric explanations, TensorFlow/Keras baseline model, and the
lab exercise instructions in its final cell).

The `src/` pipeline below is an independent **PyTorch** reimplementation of the same
task — same architecture, data, and metrics as the notebook's baseline, but as a
headless, script-based pipeline instead of a TF/Keras notebook. It does not depend on
`cnn.ipynb`, and `cnn.ipynb` does not depend on it.

## Project layout

```
task.txt                  Assignment description
requirements.txt          Python dependencies
cnn/
  cnn.ipynb                Instructor-provided notebook (TensorFlow/Keras baseline + exercise instructions)
data/
  train.pt, val.pt, test.pt   Dataset splits (images + multi-hot labels + metadata)
src/                      PyTorch pipeline (independent of cnn.ipynb)
  data.py                MultiLabelDigitsDataset, load_splits
  metrics.py              exact_match_accuracy, per_position_accuracy, binary_accuracy, precision_recall
  model.py                 BaselineCNN and ImprovedCNN (nn.Module)
  train.py                 Headless training entrypoint (CLI)
  evaluate.py               Re-evaluate a saved checkpoint on the test set
  compare.py                 Diff two runs' test_metrics.json into a comparison table
outputs/                  Generated per-run artifacts (git-ignored)
```

`data/` and `src/` are kept at the repo root, separate from `cnn/` (which holds only
the original notebook). `cnn/cnn.ipynb` loads data via `../data/*.pt` since it lives
one level down in `cnn/`; `src/train.py` and `src/evaluate.py` default to
`<repo root>/data` and `<repo root>/outputs` regardless of the current working
directory.

### PyTorch pipeline details

- Models (`src/model.py`) return raw **logits**, not sigmoid probabilities — training
  uses `nn.BCEWithLogitsLoss` for numerical stability, and `torch.sigmoid()` is
  applied only at evaluation/inference time.
- `BaselineCNN` has the identical layer structure (and identical parameter count —
  2,192,650) to the notebook's Keras baseline: 3x (conv + ReLU + maxpool) with
  32/64/128 filters, then dense(256) + dropout(0.3) + dense(10).
- `--device` auto-selects CUDA, then Apple Silicon MPS, then CPU (override with
  `--device cpu|cuda|mps`).
- Early stopping and `ReduceLROnPlateau` are reimplemented manually (PyTorch has no
  Keras-style callbacks): both watch `val_loss` with the same `--patience`, and the
  best-val-loss weights are restored before the final save/evaluation, mirroring the
  notebook's `EarlyStopping(restore_best_weights=True)`.
- `--augment` applies light random rotation/translation/scale (`torchvision.transforms`)
  to the training split only.

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
- `model_summary.txt` — layer-by-layer architecture and parameter count
- `training_curves.png` — loss / binary-accuracy curves
- `history.json` — full per-epoch training history
- `best_model.pt`, `final_model.pt` — saved checkpoints (`{"model_name", "state_dict"}`)
- `test_metrics.json` — test-set loss, binary_accuracy, precision, recall,
  exact_match_accuracy, and per-digit (per-position) accuracy

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--patience`, `--seed`, `--device`,
`--data-dir`, `--output-dir`. Run `python src/train.py --help` for the full list.

### Re-evaluating a saved checkpoint

```bash
python src/evaluate.py --model-path outputs/baseline_<timestamp>/final_model.pt
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
