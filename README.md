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
  model.py                 BaselineCNN, ImprovedCNN, ResNetCNN, ResNet50CNN, UNetCNN (nn.Module)
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
- `--augment` applies light augmentation (`torchvision.transforms`) to the training
  split only: rotation+translation+scale for `baseline`/`improved`, or
  translation+zoom+contrast (no rotation/flip) for `resnet`.
- `--lr`, `--backbone-lr`, `--optimizer`, `--weight-decay`, `--batch-size`,
  `--scheduler`, `--warmup-epochs`, `--grad-clip-norm`, `--patience`, `--ema`,
  `--ema-decay` all default to `None`/`auto`, which resolves to each model's
  recommended setting (`MODEL_HPARAM_DEFAULTS` in `src/model.py`) — pass any of them
  explicitly to override.

### Improvement 2: `resnet`

`ResNetCNN` (`src/model.py`) replaces the plain conv stack with 3 residual blocks
(Conv → BN → ReLU ×2 + skip connection, filters 32 → 64 → 128, each downsampling by
2), and replaces the Flatten→Dense(256)→Dropout head — the source of the baseline's
~2.1M dense-layer parameters and its main overfitting driver — with
SpatialDropout(0.15) → GlobalAveragePooling → Dropout(0.35) → Linear(10) (309K params
total). When `--model resnet` is selected, `train.py` defaults to `AdamW(lr=3e-4,
weight_decay=1e-4)` and, with `--augment`, applies only translate/zoom/contrast
jitter — no rotation or flip, since those can turn a `6` into a `9` (or vice versa)
and corrupt the label.

```bash
python src/train.py --model resnet --augment --run-name resnet_v1
```

### Improvement 3: `resnet50` (real ResNet-50 + optimized GPU training recipe)

`ResNet50CNN` (`src/model.py`) wraps `torchvision.models.resnet50` instead of a
hand-rolled residual net, adapted for 64×64 grayscale input:

- **Small-image stem**: `conv1` is 3×3 stride-1 (not the ImageNet 7×7 stride-2) and
  the initial maxpool is removed. The stock ImageNet stem downsamples 4× before the
  residual stages even start; on a 64×64 canvas that collapses to a 2×2 feature map
  and destroys the fine detail needed to separate overlapping digits. This keeps an
  8×8×2048 map into the final stage instead — the standard adaptation used for
  CIFAR-/small-image ResNets.
- **`fc`** is replaced with `Linear(2048, 10)`, returning logits (paired with
  `BCEWithLogitsLoss`, like every other model here).
- **ImageNet-pretrained weights** (`--pretrained`, on by default) load into
  `layer1`–`layer4`; the stem and `fc` are always freshly initialized since their
  shapes changed. If the pretrained-weights download fails (e.g. no internet on a
  compute node), it prints a warning and falls back to random init rather than
  crashing — pass `--no-pretrained` to skip the download entirely.

```bash
python src/train.py --model resnet50 --augment --run-name resnet50_v1
```

**The training recipe** (`src/train.py`) applies several widely-used techniques for
training CNNs efficiently and to higher accuracy, auto-enabled on CUDA and
specifically tuned as the `resnet50` defaults in `MODEL_HPARAM_DEFAULTS`:

- **Differential learning rates**: the pretrained backbone (`layer1`-`4`) trains at
  `--backbone-lr` (default `1e-4`), 10× lower than the freshly-initialized stem/fc
  (`--lr`, default `1e-3`) — large early updates from a random head would otherwise
  blow away the useful pretrained features.
- **No weight decay on norm/bias params**: `build_param_groups()` in `src/model.py`
  splits parameters so BatchNorm scale/shift and all biases skip weight decay
  (`--weight-decay`, default `0.05` on the remaining conv/linear weights) — decaying
  those hurts normalization behavior for no regularization benefit.
- **Cosine LR schedule with linear warmup** (`--scheduler cosine_warmup`, the
  `resnet50` default; `--warmup-epochs`, default `5`) instead of `ReduceLROnPlateau`
  — a fixed, well-behaved schedule for a training run of a known length. Because it
  assumes running to completion, early stopping is effectively disabled for this
  model (`patience` defaults very high) rather than cutting the schedule short.
- **Mixed precision** (`--amp`, default `auto` = on whenever CUDA is available):
  prefers `bfloat16` autocast (no `GradScaler`/loss-scaling needed, no underflow
  risk) when the GPU supports it — e.g. Ampere and newer, including A100 — else
  falls back to `float16` with a `GradScaler`.
- **`torch.compile`** (`--compile`, default `auto` = on whenever CUDA is available),
  wrapped in a try/except so a compilation failure (e.g. missing Triton on some
  clusters) just prints a warning and continues eagerly instead of crashing.
- **Channels-last memory format**, applied to the model and every input batch on
  CUDA — standard Tensor Core layout optimization for conv-heavy nets.
- **Gradient clipping** (`--grad-clip-norm`, default `1.0`) for stability at the
  larger effective batch size / learning rate.
- **EMA of model weights** (`--ema`, on by default for `resnet50`; `--ema-decay`,
  default `0.9998`): validation and the final saved model use the exponential
  moving average of weights rather than the raw last-step weights, which typically
  generalizes a bit better. Checkpoints store both (`state_dict` and
  `ema_state_dict`); `evaluate.py` prefers the EMA weights automatically when
  present (`--no-ema` to use the raw weights instead).
- **Larger batch size** (`--batch-size`, default `256` for `resnet50` vs. `128` for
  the other models) and **auto-scaled `DataLoader` workers**
  (`min(8, os.cpu_count())` with `pin_memory`/`persistent_workers` on CUDA) to keep
  the GPU fed.
- `cudnn.benchmark = True` and `torch.set_float32_matmul_precision("high")` (TF32
  matmuls) are enabled globally whenever training on CUDA.

All of the above are plain CLI flags — every default can be overridden, e.g.
`python src/train.py --model resnet50 --no-pretrained --amp off --compile off` to
train from scratch in full precision without `torch.compile`.

> **Note:** in practice, `resnet50`'s 23.5M parameters badly overfit this 50k-image
> dataset despite the regularization above. `resnet` (309K params) and `unet` (below,
> ~483K params) are the better-fitted options for a dataset this size — reach for
> `resnet50` only if you have a strong reason to believe more capacity will help
> (e.g. after confirming the smaller models have plateaued and aren't overfitting).

### Improvement 4: `unet`

`UNetCNN` (`src/model.py`) is a small, standard U-Net encoder-decoder —
`DoubleConv` (Conv→BN→ReLU ×2) blocks, 3 downsampling stages (channels
`base`→`base*2`→`base*4`, `base=16` by default) into a `base*8` bottleneck at 8×8,
then 3 upsampling stages (`ConvTranspose2d` + concat matching encoder skip +
`DoubleConv`) back to full 64×64 resolution — repurposed for classification instead
of its usual per-pixel segmentation:

- **Head**: `GlobalAveragePooling → Dropout(0.3) → Linear(10)` on the final
  64×64×`base` decoder output, instead of U-Net's usual 1×1-conv-per-pixel
  segmentation head.
- **Motivation**: the skip connections keep full-resolution detail available late in
  the network (unlike a plain encoder, which only ever sees a heavily downsampled
  8×8 view by the time it reaches the classifier) — this can help separate small,
  overlapping digits that a plain encoder would blur away.
- **Size**: ~483K params — deliberately close to `resnet`'s budget (309K), not
  `resnet50`'s (23.5M), given what happened above. Same `AdamW(lr=3e-4,
  weight_decay=1e-4)` recipe and rotation/flip-free augmentation as `resnet`.

```bash
python src/train.py --model unet --augment --run-name unet_v1
```

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
- `best_model.pt`, `final_model.pt` — saved checkpoints
  (`{"model_name", "state_dict", "ema_state_dict", "pretrained"}`)
- `test_metrics.json` — test-set loss, binary_accuracy, precision, recall,
  exact_match_accuracy, and per-digit (per-position) accuracy

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--patience`, `--seed`, `--device`.
See [Improvement 3](#improvement-3-resnet50-real-resnet-50--optimized-gpu-training-recipe)
above for the additional `resnet50`-specific flags (`--backbone-lr`, `--scheduler`,
`--amp`, `--compile`, `--ema`, etc). Run `python src/train.py --help` for the full list.

### Re-evaluating a saved checkpoint

```bash
python src/evaluate.py --model-path outputs/baseline_<timestamp>/final_model.pt
```

Writes `test_metrics.json` and a `sample_predictions.png` grid to
`outputs/eval/` by default. If the checkpoint has EMA weights (`resnet50`), it uses
those automatically; pass `--no-ema` to evaluate the raw weights instead.

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
