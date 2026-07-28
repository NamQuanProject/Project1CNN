# Multi-Label MNIST-Digits CNN

Multi-label image classification: unlike standard MNIST (one digit per image), each
image here contains **multiple digits**, and the model must predict the full set of
digit classes present (a 10-dim multi-hot vector, one entry per digit `0`-`9`).

See [`task.txt`](task.txt) for the full assignment description and
[`cnn/cnn.ipynb`](cnn/cnn.ipynb) for the instructor-provided starter notebook
(dataset walkthrough, metric explanations, TensorFlow/Keras baseline model, and the
lab exercise instructions in its final cell).

The `src/` pipeline below is an independent **PyTorch** implementation focused
entirely on one model: a ResNet-style residual CNN (`ResNetCNN` in `src/model.py`).
It does not depend on `cnn.ipynb`, and `cnn.ipynb` does not depend on it.

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
  model.py                 ResNetCNN (nn.Module) and its building blocks
  train.py                 Headless training entrypoint (CLI)
  evaluate.py               Re-evaluate a saved checkpoint on the test set
  compare.py                 Diff two runs' test_metrics.json into a comparison table
  visualize_errors.py        Save an annotated PNG for every exact-match mismatch
outputs/                  Generated per-run artifacts (git-ignored)
visualizations/           Generated error-analysis images (git-ignored)
```

`data/` and `src/` are kept at the repo root, separate from `cnn/` (which holds only
the original notebook). `cnn/cnn.ipynb` loads data via `../data/*.pt` since it lives
one level down in `cnn/`; `src/train.py` and `src/evaluate.py` default to
`<repo root>/data` and `<repo root>/outputs` regardless of the current working
directory.

## The model: `ResNetCNN`

`src/model.py` defines one architecture, built from a few small pieces:

- **`ECA`** — Efficient Channel Attention (Wang et al., CVPR 2020,
  [arxiv.org/abs/1910.03151](https://arxiv.org/abs/1910.03151)). Two earlier channel
  gates were tried and superseded here: plain Squeeze-Excitation (a
  channels→hidden→channels MLP bottleneck), then CBAM (channel + spatial
  attention) — CBAM empirically underperformed SE on this dataset/training budget.
  ECA replaces both: the ECA paper argues SE's dimensionality-reduction bottleneck
  actually *hurts* channel-attention quality (not just a compute/quality
  trade-off), so ECA instead runs a single lightweight 1D conv directly over the
  pooled per-channel descriptor — cheaper than SE (no bottleneck, ~3-5 params per
  gate vs. thousands) and, per the paper, more effective.
- **`DropPath`** — stochastic depth (Huang et al., 2016): randomly drops the entire
  residual branch for some training samples, an ensembling-style regularizer.
- **`ResidualBlock`** — `(Conv → BN → SiLU) × 2` with a skip connection, plus ECA
  and DropPath applied to the residual branch before the skip-add. Uses SiLU
  (`x * sigmoid(x)`) instead of ReLU: smooth and non-zero everywhere (no "dead
  unit" gradient collapse the way ReLU can have), which several modern CNN designs
  (EfficientNet and others) found to outperform ReLU slightly at negligible cost.

**`ResNetCNN`** assembles these into (block topology originally matched a reference
implementation — a classmate's `assignment1_cnn.py` — that reached ~80% exact-match
accuracy on this dataset; since widened/deepened):

- **Stem**: a single 3×3 conv+BN+SiLU, no downsampling (stays at 64×64), outputting
  64 channels.
- **12 residual blocks in 3 stages of 4**, channels 64 → 128 → 256: stage 1 (64ch,
  64×64) → stage 2 (128ch, 32×32) → stage 3 (256ch, 16×16). Only the first block of
  each of stage 2/3 downsamples — same "don't over-downsample" philosophy as
  before (a "downsample every block" design would end at a much smaller map,
  hurting separation of several small, overlapping digits), just with more blocks
  per stage and more channels per stage. The 3 same-resolution blocks per stage add
  nonlinear depth before handing off to the next (downsampling + widening) stage.
- Drop probability for stochastic depth increases with block depth (0 → 0.1 across
  the 12 blocks, deeper blocks being more overfitting-prone) — regularization to
  offset the substantially larger capacity of this configuration.
- **Head**: SpatialDropout (`nn.Dropout2d`, 0.15) → GlobalAveragePooling →
  Dropout(0.35) → `Linear(256, 10)`. No `Flatten`/`Dense(256-unit-MLP)`: avoids a
  large dense classifier, historically the main source of overfitting for this
  dataset size (~50K training images).
- Conv weights use explicit Kaiming-normal ("he_normal") init (`nonlinearity="relu"`
  for the gain calculation, the standard stand-in for SiLU/Swish since PyTorch has
  no dedicated entry — both behave near-linearly for positive inputs).
- **~5.9M total parameters.** A ResNet-50 backbone was also tried on this task and
  badly overfit at 23.5M parameters for a ~50K-image dataset — this configuration
  sits at roughly a quarter of that, deliberately paired with MixUp (below) as
  extra regularization to offset the larger capacity than the ~988K/0.88-exact-match
  configuration this was widened from.
- Returns raw **logits**, not sigmoid probabilities — pair with one of the losses
  below, and apply `torch.sigmoid()` only at evaluation/inference time.

```bash
python src/train.py --augment --run-name resnet_v2
```

**Possible next step**: migrating to a full ConvNeXt-style block (depthwise 7×7
convs, LayerNorm, inverted-bottleneck MLP with GELU, layer scale) is a materially
different architecture family from the ResNet lineage above, not an incremental
change — worth its own separate pass if you want to go there next.

## Full parameter reference (`python src/train.py`)

Every flag defaults to `None`/`auto` unless noted, which resolves through
`MODEL_HPARAM_DEFAULTS["resnet"]` in `src/model.py` (shown as **default** below) —
pass any flag explicitly to override. Run `python src/train.py --help` for the same
list from argparse itself.

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--data-dir` | path | `<repo root>/data` | Directory containing `train.pt`/`val.pt`/`test.pt`. |
| `--output-dir` | path | `<repo root>/outputs` | Where run folders get written. |
| `--run-name` | str | `resnet_<timestamp>` | Subfolder name under `--output-dir`. |
| `--epochs` | int | `100` | Max training epochs (early stopping may end it sooner). |
| `--batch-size` | int | `128` | Training/val/test batch size. |
| `--lr` | float | `3e-4` | Learning rate. |
| `--optimizer` | `adam`\|`adamw` | `adamw` | Optimizer. Weight decay only applies to `adamw`. |
| `--weight-decay` | float | `1e-4` | L2 weight decay — applied only to conv/linear weights, never BatchNorm scale/shift or biases (`build_param_groups()` splits them out; decaying those hurts normalization for no benefit). |
| `--scheduler` | `plateau`\|`cosine_warmup`\|`cosine` | `cosine` | LR schedule. `cosine` = plain `CosineAnnealingLR` (`T_max=epochs`, `eta_min=--min-lr`), stepped once per epoch, no warmup. `plateau` = `ReduceLROnPlateau` on val loss. `cosine_warmup` = linear warmup then cosine decay, stepped every batch. |
| `--warmup-epochs` | int | `0` | Warmup length, only used by `--scheduler cosine_warmup`. |
| `--lr-patience` | int | `3` | Epochs of no val-loss improvement before `plateau` halves the LR. Only relevant for `--scheduler plateau`. Independent of `--patience` (see below). |
| `--min-lr` | float | `1e-6` | Floor LR — `eta_min` for `cosine`, or the floor for `plateau`. |
| `--patience` | int | `30` | Epochs of no improvement (on `--monitor-metric`) before early stopping. Generous by default since the default `cosine` schedule is fixed-length and benefits from running to completion rather than being cut short. |
| `--monitor-metric` | `loss`\|`binary_accuracy`\|`precision`\|`recall`\|`exact_match_accuracy` | `exact_match_accuracy` | Validation metric used for early-stopping and best-checkpoint selection. Independent of the scheduler. |
| `--monitor-mode` | `min`\|`max` | `max` (`min` if monitoring `loss`) | Direction of "improvement" for `--monitor-metric`. |
| `--grad-clip-norm` | float | `0.0` (disabled) | Gradient-clipping max-norm. |
| `--loss-type` | `bce`\|`focal`\|`asl` | `asl` | Training/eval loss function — see below. |
| `--label-smoothing` | float | `0.05` | Softens hard 0/1 targets toward 0.5 by this amount, training loss only (never metrics or val/test loss). |
| `--mixup-alpha` | float | `0.2` | MixUp (Zhang et al., 2018) `Beta(alpha,alpha)` interpolation strength for images+labels during training (`0` disables). Blended images/labels are used for the loss only — the running training-accuracy diagnostic and all val/test metrics use the original, unmixed labels. |
| `--focal-gamma` | float | `2.0` | Focusing exponent, only used when `--loss-type focal`. |
| `--asl-gamma-neg` | float | `4.0` | Asymmetric Loss negative-class focusing exponent, only used when `--loss-type asl`. |
| `--asl-gamma-pos` | float | `1.0` | Asymmetric Loss positive-class focusing exponent, only used when `--loss-type asl`. |
| `--asl-clip` | float | `0.05` | Asymmetric Loss probability-shifting margin for easy negatives, only used when `--loss-type asl`. |
| `--asl-weight` | float | `1.0` (pure ASL) | Blends ASL with plain BCE: `asl_weight*ASL + (1-asl_weight)*BCE`, only used when `--loss-type asl`. |
| `--ema` / `--no-ema` | flag | `--ema` (on) | Exponential moving average of model weights. Validation and the final saved/evaluated model use the EMA shadow weights, not raw last-step weights. |
| `--ema-decay` | float | `0.9995` | EMA decay rate. |
| `--augment` | flag | off | Light training-time augmentation: translate (±4%) + zoom (±8%) + contrast (±12%) jitter. Deliberately **no rotation or flip** — those can turn a `6` into a `9` (or vice versa) and corrupt the label. |
| `--seed` | int | `42` | Random seed (Python/NumPy/PyTorch). |
| `--device` | `auto`\|`cpu`\|`cuda`\|`mps` | `auto` | `auto` picks CUDA, then Apple Silicon MPS, then CPU. |
| `--num-workers` | int | auto | `DataLoader` worker processes. Auto-picks `min(8, cpu_count())` on CUDA, else `0`. |
| `--amp` | `auto`\|`on`\|`off` | `auto` | Mixed-precision autocast. `auto`/`on` only take effect on CUDA — prefers `bfloat16` (no `GradScaler` needed) when the GPU supports it (Ampere+, e.g. A100), else `float16` with a `GradScaler`. |
| `--compile` | `auto`\|`on`\|`off` | `auto` | Wraps the model in `torch.compile()`. `auto`/`on` only take effect on CUDA; falls back to eager mode with a warning if compilation fails. |

**Loss functions** (`--loss-type`):
- **`asl` (default)** — Asymmetric Loss (Ben-Baruch/Ridnik et al., ICCV 2021),
  widely regarded as one of the strongest losses for multi-label classification. Each
  image here has only ~6-8 of 10 possible digits present, so negatives outnumber
  positives per sample; plain BCE lets the easy majority of negatives flatten the
  gradient. ASL counters this with (1) a stronger focusing exponent on negatives than
  positives (`--asl-gamma-neg`/`--asl-gamma-pos`, asymmetric unlike symmetric focal
  loss) and (2) probability-shifting (`--asl-clip`) that discards already-easy,
  confidently-correct negatives from the loss entirely, concentrating gradient on the
  genuinely hard/ambiguous ones.
- **`focal`** — symmetric focal loss (Lin et al., 2017): down-weights already-easy
  predictions relative to plain BCE, simpler than ASL.
- **`bce`** — plain `nn.BCEWithLogitsLoss`.

Two independent training-mechanics decisions worth calling out explicitly:
- **`--monitor-metric`/`--monitor-mode`** (early stopping + best-checkpoint
  selection) and **`--scheduler plateau`** (LR reduction) are deliberately
  decoupled: the scheduler always watches val loss, while monitoring/checkpointing
  defaults to watching `exact_match_accuracy` directly — the metric that actually
  matters for this task, which can keep improving even while val loss is flat.
- **`--patience`** (early stopping) and **`--lr-patience`** (LR reduction) are
  separate knobs with different default values (`10` vs `3`), matching the reference
  recipe's dual-callback setup (`EarlyStopping` on `val_exact_match_accuracy`,
  `ReduceLROnPlateau` on `val_loss`) rather than one shared patience value.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset (`data/*.pt`) is expected to already be in place; it is git-ignored
due to size (see [`.gitignore`](.gitignore)) so make sure it's present before training.

## Running the pipeline

```bash
python src/train.py --augment --run-name resnet_v1
```

Each run writes to `outputs/<run-name>/`:
- `model_summary.txt` — layer-by-layer architecture and parameter count
- `training_curves.png` — loss / binary-accuracy curves
- `history.json` — full per-epoch training history
- `best_model.pt`, `final_model.pt` — saved checkpoints (`{"model_name", "state_dict", "ema_state_dict"}`)
- `test_metrics.json` — test-set loss, binary_accuracy, precision, recall,
  exact_match_accuracy, and per-digit (per-position) accuracy

### Re-evaluating a saved checkpoint

```bash
python src/evaluate.py --model-path outputs/resnet_v1/final_model.pt
```

Writes `test_metrics.json` and a `sample_predictions.png` grid to `outputs/eval/` by
default. Uses the checkpoint's EMA weights automatically when present; pass
`--no-ema` to evaluate the raw weights instead.

### Visualizing exact-match errors

```bash
python src/visualize_errors.py --model-path outputs/resnet_v1/final_model.pt
```

Runs the checkpoint over the full test set and, for every image whose prediction
isn't an *exact* match, saves one annotated PNG (`error_<index>.png`) to
`visualizations/<run-name>/` — `<run-name>` is inferred from the checkpoint's parent
folder (override with `--output-dir`). Each image's title shows the true digits, the
predicted digits, and precisely which digits were **missing** (present but not
predicted — false negatives) or **extra** (predicted but not present — false
positives), so you can see at a glance not just *that* a prediction was wrong but
*how*. A `summary.json` (total images, mismatch count/rate) is written alongside
them. Use `--max-images N` to cap how many get saved on a model with a lot of
mismatches; `--no-ema` and `--amp`/`--device` behave the same as `evaluate.py`.

### Comparing two runs

```bash
python src/compare.py \
  --baseline outputs/resnet_v1/test_metrics.json \
  --improved outputs/resnet_v2/test_metrics.json \
  --output outputs/comparison.md
```

`--baseline`/`--improved` are just labels for "before"/"after" — point them at any
two runs' `test_metrics.json`. Produces the overall-metrics and per-digit-accuracy
comparison table needed for the assignment's "Results" section.

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
