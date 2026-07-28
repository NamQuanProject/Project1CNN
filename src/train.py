"""Train the ResNet-style multi-label digit CNN (src/model.py:ResNetCNN)
using PyTorch: mixed precision (bf16/fp16 autocast), channels-last memory
format, torch.compile, no weight decay on norm/bias params, a choice of LR
schedules, several loss functions (BCE / focal / Asymmetric Loss), gradient
clipping, and EMA of weights.

Usage (from repo root or from src/, either works):
    python src/train.py --augment
    python src/train.py --augment --run-name resnet_v1
"""

import argparse
import contextlib
import copy
import functools
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data import load_splits
from src.metrics import (
    PREDICTION_THRESHOLD,
    binary_accuracy,
    exact_match_accuracy,
    per_position_accuracy,
    precision_recall,
)
from src.model import MODEL_HPARAM_DEFAULTS, build_model, build_param_groups

MODEL_NAME = "resnet"

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference="auto"):
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_augmentation():
    """Light training-time augmentation. No rotation/flip: that can turn a
    6 into a 9 (or vice versa) and corrupt the label.
    """
    return transforms.Compose(
        [
            # Keras RandomTranslation(0.04, 0.04) / RandomZoom(-0.08, 0.08).
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.92, 1.08)),
            # Keras RandomContrast(0.12).
            transforms.ColorJitter(contrast=0.12),
        ]
    )


def resolve_hparam(value, model_name, key, fallback):
    """CLI value wins if given (not None); otherwise use the model's
    recommended default, falling back to `fallback` if the model doesn't
    specify one.
    """
    if value is not None:
        return value
    return MODEL_HPARAM_DEFAULTS.get(model_name, {}).get(key, fallback)


# "higher is better" for every metric except loss.
MONITOR_MODE_BY_METRIC = {"loss": "min"}


def is_better(current, best, mode):
    return current > best if mode == "max" else current < best


def smooth_labels(labels, label_smoothing):
    """Soften hard 0/1 multi-label targets toward 0.5, applied only to the
    *training* loss (never to metrics, or to val/test loss) -- standard
    label smoothing, adapted for BCE since nn.BCEWithLogitsLoss has no
    built-in label_smoothing option (unlike CrossEntropyLoss). Useful here
    since overlapping/occluded digits make some labels genuinely ambiguous;
    softening discourages the model from being overconfident about them.
    """
    if label_smoothing <= 0.0:
        return labels
    return labels * (1 - label_smoothing) + 0.5 * label_smoothing


def mixup_batch(images, labels, alpha):
    """MixUp (Zhang et al., 2018): blend two random training images and
    linearly interpolate their multi-hot label vectors by the same factor.
    nn.BCEWithLogitsLoss (and the ASL/focal losses here) handle the
    resulting soft (non-0/1) targets natively. A no-op when alpha <= 0.

    Note: after this, the *original* `labels` no longer describes the
    (now-blended) `images` exactly -- train.py's running training-accuracy
    diagnostic still compares against the original labels for simplicity,
    so it reads a bit noisy/pessimistic while mixup is active. This doesn't
    affect validation/test metrics, which never use mixup.
    """
    if alpha is None or alpha <= 0.0:
        return images, labels
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[perm]
    mixed_labels = lam * labels + (1 - lam) * labels[perm]
    return mixed_images, mixed_labels


def asymmetric_loss_with_logits(logits, targets, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
    """Asymmetric Loss (ASL) for multi-label classification (Ben-Baruch,
    Ridnik, et al., ICCV 2021, "Asymmetric Loss For Multi-Label
    Classification") -- widely regarded as one of the strongest losses for
    multi-label tasks. Unlike plain BCE or symmetric focal loss, it treats
    positives and negatives asymmetrically: each image here has only ~6-8 of
    10 possible digits present, so negatives outnumber positives per sample,
    and the easy majority of negatives can otherwise dominate/flatten the
    gradient. ASL addresses this with two mechanisms:
    - **Asymmetric focusing**: a stronger focusing exponent on negatives
      (`gamma_neg`, default 4) than positives (`gamma_pos`, default 1),
      down-weighting easy negatives more aggressively than focal loss does
      symmetrically.
    - **Probability shifting/clipping** (`clip`, default 0.05): negatives
      predicted confidently correct beyond this margin are shifted to
      contribute ~zero loss, entirely discarding the easiest negatives so
      gradient concentrates on the hard, ambiguous ones (e.g. a distractor
      digit that's partially occluded by an overlapping digit).

    Reference implementation: github.com/Alibaba-MIIL/ASL. The focusing
    weight is computed under `torch.no_grad()` (i.e. treated as a constant
    multiplier, not differentiated through) matching the reference's default
    `disable_torch_grad_focal_loss=True` -- this is the standard, stable way
    to use ASL and is what the paper's reported results use.
    """
    xs_pos = torch.sigmoid(logits)
    xs_neg = 1 - xs_pos

    if clip is not None and clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1)

    los_pos = targets * torch.log(xs_pos.clamp(min=eps))
    los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=eps))
    loss = los_pos + los_neg

    if gamma_neg > 0 or gamma_pos > 0:
        with torch.no_grad():
            pt = xs_pos * targets + xs_neg * (1 - targets)
            one_sided_gamma = gamma_pos * targets + gamma_neg * (1 - targets)
            one_sided_w = torch.pow((1 - pt).clamp(min=0), one_sided_gamma)
        loss = loss * one_sided_w

    return -loss.mean()


def focal_loss_with_logits(logits, targets, gamma):
    """Focal loss (Lin et al., 2017), binary/multi-label form: down-weights
    already-confident (easy) predictions and up-weights uncertain (hard)
    ones relative to plain BCE.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t).clamp(min=0.0) ** gamma
    return (focal_weight * bce).mean()


def build_criterion(loss_type, focal_gamma=0.0, asl_gamma_neg=4.0, asl_gamma_pos=1.0, asl_clip=0.05, asl_weight=1.0):
    """Build the training/eval loss function selected by --loss-type.

    "asl" supports being a *combined* (weighted) loss: `asl_weight` (0-1)
    blends ASL with plain BCE -- `asl_weight * ASL + (1 - asl_weight) * BCE`
    -- defaulting to 1.0 (pure ASL, the standard way the paper uses it), but
    lower values let you experiment with an explicit ASL+BCE combination.
    """
    if loss_type == "asl":
        def combined_asl(logits, targets):
            asl = asymmetric_loss_with_logits(
                logits, targets, gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip
            )
            if asl_weight >= 1.0:
                return asl
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            return asl_weight * asl + (1 - asl_weight) * bce

        return combined_asl
    if loss_type == "focal" and focal_gamma and focal_gamma > 0:
        return functools.partial(focal_loss_with_logits, gamma=focal_gamma)
    return nn.BCEWithLogitsLoss()


def resolve_amp_dtype(device, amp_mode):
    """Resolve --amp {auto,on,off} to an autocast dtype, or None to disable.

    Only CUDA is supported here. Prefers bf16 (no GradScaler needed, no risk
    of fp16 underflow/NaN) when the GPU supports it (Ampere+, e.g. A100),
    else falls back to fp16 with a GradScaler.
    """
    if amp_mode == "off":
        return None
    if device.type != "cuda":
        if amp_mode == "on":
            print("AMP requested but device is not CUDA; autocast is only enabled for CUDA here. Skipping.")
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def resolve_compile(device, compile_mode):
    if compile_mode == "off":
        return False
    if device.type != "cuda":
        if compile_mode == "on":
            print("torch.compile requested but device is not CUDA; skipping (only enabled for CUDA here).")
        return False
    return True


class ModelEMA:
    """Exponential moving average of model weights.

    Standard technique in modern high-accuracy training recipes (e.g. the
    "ResNet strikes back" procedures, timm): averaging weights over the
    trailing window of training steps gives a smoother, usually
    better-generalizing final model than the raw last-step weights.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b)


def cosine_warmup_lambda(step, warmup_steps, total_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ResNet-style multi-label digit CNN (PyTorch).")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Defaults to the recommended batch size (128)."
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Defaults to the recommended lr (see MODEL_HPARAM_DEFAULTS)."
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default=None,
        help="Defaults to the recommended optimizer (adamw).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Defaults to the recommended weight decay. Never applied to norm/bias params.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["plateau", "cosine_warmup", "cosine"],
        default=None,
        help="LR schedule. 'plateau' = ReduceLROnPlateau on val loss. 'cosine_warmup' = "
        "linear warmup then cosine decay, stepped every batch. 'cosine' = plain "
        "CosineAnnealingLR (T_max=epochs, eta_min=--min-lr), stepped once per epoch, "
        "no warmup. Defaults per-model.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Linear warmup length for the cosine_warmup scheduler. Defaults to 0.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Gradient clipping max-norm (0 disables). Defaults to 0 (disabled).",
    )
    parser.add_argument("--patience", type=int, default=None, help="Early-stopping patience in epochs.")
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=None,
        help="Patience (epochs) for the 'plateau' scheduler's LR reduction. Defaults to the "
        "recommended value, or --patience if not set (the old coupled behavior).",
    )
    parser.add_argument(
        "--min-lr", type=float, default=None, help="Floor LR for the 'plateau' scheduler. Defaults to 1e-6."
    )
    parser.add_argument(
        "--monitor-metric",
        choices=["loss", "binary_accuracy", "precision", "recall", "exact_match_accuracy"],
        default=None,
        help="Validation metric used for early-stopping and best-checkpoint selection. Defaults "
        "to exact_match_accuracy.",
    )
    parser.add_argument(
        "--monitor-mode",
        choices=["min", "max"],
        default=None,
        help="Defaults to 'min' for --monitor-metric loss, 'max' for every other metric.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help="Softens 0/1 BCE targets toward 0.5 by this amount (training loss only). Defaults to 0.05.",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=None,
        help="MixUp Beta(alpha,alpha) interpolation strength for images+labels during "
        "training (0 disables). Defaults per-model.",
    )
    parser.add_argument(
        "--loss-type",
        choices=["bce", "focal", "asl"],
        default=None,
        help="Training/eval loss function. Defaults to 'asl'.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=None,
        help="Focusing exponent, only used when --loss-type focal. Defaults to 2.0.",
    )
    parser.add_argument(
        "--asl-gamma-neg",
        type=float,
        default=None,
        help="Asymmetric Loss negative-class focusing exponent, only used when --loss-type asl. "
        "Defaults to 4.0 (the paper's recommendation).",
    )
    parser.add_argument(
        "--asl-gamma-pos",
        type=float,
        default=None,
        help="Asymmetric Loss positive-class focusing exponent, only used when --loss-type asl. "
        "Defaults to 1.0 (the paper's recommendation).",
    )
    parser.add_argument(
        "--asl-clip",
        type=float,
        default=None,
        help="Asymmetric Loss probability-shifting margin for easy negatives, only used when "
        "--loss-type asl. Defaults to 0.05 (the paper's recommendation).",
    )
    parser.add_argument(
        "--asl-weight",
        type=float,
        default=None,
        help="Blends ASL with plain BCE: asl_weight*ASL + (1-asl_weight)*BCE, only used when "
        "--loss-type asl. Defaults to 1.0 (pure ASL).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes. Defaults to an auto-picked value based on device/CPU count.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--amp",
        choices=["auto", "on", "off"],
        default="auto",
        help="Mixed-precision autocast. 'auto' enables it whenever the device is CUDA.",
    )
    parser.add_argument(
        "--compile",
        choices=["auto", "on", "off"],
        default="auto",
        help="torch.compile the model. 'auto' enables it whenever the device is CUDA.",
    )

    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--ema", dest="ema", action="store_true", default=None)
    ema_group.add_argument("--no-ema", dest="ema", action="store_false", help="Disable EMA of model weights.")
    parser.add_argument(
        "--ema-decay", type=float, default=None, help="Defaults to the recommended EMA decay (0.9995)."
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply light augmentation to training data (translate+zoom+contrast, no rotation/flip).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Subfolder name under output-dir. Defaults to resnet_<timestamp>.",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype=None, channels_last=False):
    """Run a full pass over `loader`, returning (metrics_dict, y_true, y_pred)."""
    model.eval()
    total_loss = 0.0
    n = 0
    all_targets = []
    all_probs = []
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
    )
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        with autocast_ctx:
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_targets.append(labels.cpu())
        all_probs.append(torch.sigmoid(logits.float()).cpu())

    y_true = torch.cat(all_targets)
    y_pred = torch.cat(all_probs)
    precision, recall = precision_recall(y_true, y_pred, threshold=PREDICTION_THRESHOLD)
    metrics = {
        "loss": total_loss / n,
        "binary_accuracy": binary_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD),
        "precision": precision,
        "recall": recall,
        "exact_match_accuracy": exact_match_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD),
    }
    return metrics, y_true, y_pred


def train_one_epoch(
    model,
    raw_model,
    loader,
    criterion,
    optimizer,
    device,
    amp_dtype,
    scaler,
    grad_clip_norm,
    channels_last,
    ema,
    step_scheduler,
    label_smoothing=0.0,
    mixup_alpha=0.0,
):
    model.train()
    total_loss = 0.0
    n = 0
    all_targets = []
    all_probs = []
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
    )
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)

        mixed_images, mixed_labels = mixup_batch(images, labels, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = model(mixed_images)
            loss = criterion(logits, smooth_labels(mixed_labels, label_smoothing))

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if step_scheduler is not None:
            step_scheduler.step()
        if ema is not None:
            ema.update(raw_model)

        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_targets.append(labels.detach().cpu())
        all_probs.append(torch.sigmoid(logits.detach().float()).cpu())

    y_true = torch.cat(all_targets)
    y_pred = torch.cat(all_probs)
    train_binary_acc = binary_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD)
    return total_loss / n, train_binary_acc


def main():
    args = parse_args()
    fix_random_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    run_name = args.run_name or f"{MODEL_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    lr = resolve_hparam(args.lr, MODEL_NAME, "lr", 1e-3)
    optimizer_name = resolve_hparam(args.optimizer, MODEL_NAME, "optimizer", "adam")
    weight_decay = resolve_hparam(args.weight_decay, MODEL_NAME, "weight_decay", 0.0)
    batch_size = resolve_hparam(args.batch_size, MODEL_NAME, "batch_size", 128)
    scheduler_type = resolve_hparam(args.scheduler, MODEL_NAME, "scheduler", "plateau")
    warmup_epochs = resolve_hparam(args.warmup_epochs, MODEL_NAME, "warmup_epochs", 0)
    grad_clip_norm = resolve_hparam(args.grad_clip_norm, MODEL_NAME, "grad_clip_norm", 0.0)
    patience = resolve_hparam(args.patience, MODEL_NAME, "patience", 10)
    # lr_patience/min_lr are for ReduceLROnPlateau specifically; fall back to
    # the early-stopping `patience` (old coupled behavior) if not specified.
    lr_patience = resolve_hparam(args.lr_patience, MODEL_NAME, "lr_patience", patience)
    min_lr = resolve_hparam(args.min_lr, MODEL_NAME, "min_lr", 1e-5)
    monitor_metric = resolve_hparam(args.monitor_metric, MODEL_NAME, "monitor_metric", "loss")
    monitor_mode = args.monitor_mode or resolve_hparam(
        None, MODEL_NAME, "monitor_mode", MONITOR_MODE_BY_METRIC.get(monitor_metric, "max")
    )
    label_smoothing = resolve_hparam(args.label_smoothing, MODEL_NAME, "label_smoothing", 0.0)
    mixup_alpha = resolve_hparam(args.mixup_alpha, MODEL_NAME, "mixup_alpha", 0.0)
    loss_type = resolve_hparam(args.loss_type, MODEL_NAME, "loss_type", "bce")
    focal_gamma = resolve_hparam(args.focal_gamma, MODEL_NAME, "focal_gamma", 2.0)
    asl_gamma_neg = resolve_hparam(args.asl_gamma_neg, MODEL_NAME, "asl_gamma_neg", 4.0)
    asl_gamma_pos = resolve_hparam(args.asl_gamma_pos, MODEL_NAME, "asl_gamma_pos", 1.0)
    asl_clip = resolve_hparam(args.asl_clip, MODEL_NAME, "asl_clip", 0.05)
    asl_weight = resolve_hparam(args.asl_weight, MODEL_NAME, "asl_weight", 1.0)
    ema_enabled = resolve_hparam(args.ema, MODEL_NAME, "ema", False)
    ema_decay = resolve_hparam(args.ema_decay, MODEL_NAME, "ema_decay", 0.999)

    print(
        f"Hyperparameters: lr={lr}, optimizer={optimizer_name}, "
        f"weight_decay={weight_decay}, batch_size={batch_size}, scheduler={scheduler_type}, "
        f"warmup_epochs={warmup_epochs}, grad_clip_norm={grad_clip_norm}, patience={patience}, "
        f"lr_patience={lr_patience}, min_lr={min_lr}, monitor={monitor_metric} ({monitor_mode}), "
        f"label_smoothing={label_smoothing}, mixup_alpha={mixup_alpha}, loss_type={loss_type}"
        + (
            f" (asl_gamma_neg={asl_gamma_neg}, asl_gamma_pos={asl_gamma_pos}, asl_clip={asl_clip}, "
            f"asl_weight={asl_weight})"
            if loss_type == "asl"
            else f" (focal_gamma={focal_gamma})"
            if loss_type == "focal"
            else ""
        )
        + f", ema={ema_enabled} (decay={ema_decay})"
    )

    amp_dtype = resolve_amp_dtype(device, args.amp)
    use_compile = resolve_compile(device, args.compile)
    channels_last = device.type == "cuda"
    print(f"amp_dtype={amp_dtype}, compile={use_compile}, channels_last={channels_last}")

    if args.num_workers is not None:
        num_workers = args.num_workers
    elif device.type == "cuda":
        num_workers = min(8, os.cpu_count() or 1)
    else:
        num_workers = 0
    pin_memory = device.type == "cuda"

    print(f"Loading data from {args.data_dir} ...")
    transform_train = build_augmentation() if args.augment else None
    splits = load_splits(args.data_dir, transform_train=transform_train)
    train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]
    print("Train:", len(train_ds))
    print("Val:  ", len(val_ds))
    print("Test: ", len(test_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        # Drop the ragged last batch so every training step has the same
        # shape -- with torch.compile / cudnn.benchmark enabled, a
        # differently-sized last batch forces an extra recompile/autotune
        # pass every single epoch. val/test keep every sample (no drop).
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    raw_model = build_model(MODEL_NAME).to(device)
    if channels_last:
        raw_model = raw_model.to(memory_format=torch.channels_last)

    num_params = sum(p.numel() for p in raw_model.parameters())
    with open(os.path.join(run_dir, "model_summary.txt"), "w") as f:
        f.write(str(raw_model) + "\n\n")
        f.write(f"Total parameters: {num_params:,}\n")
    print(raw_model)
    print(f"Total parameters: {num_params:,}")

    model = raw_model
    if use_compile:
        try:
            model = torch.compile(raw_model)
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(f"WARNING: torch.compile failed ({exc}); continuing without it.")
            model = raw_model

    ema = ModelEMA(raw_model, decay=ema_decay) if ema_enabled else None

    criterion = build_criterion(
        loss_type,
        focal_gamma=focal_gamma,
        asl_gamma_neg=asl_gamma_neg,
        asl_gamma_pos=asl_gamma_pos,
        asl_clip=asl_clip,
        asl_weight=asl_weight,
    )
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    param_groups = build_param_groups(raw_model, lr=lr, weight_decay=weight_decay)
    optimizer = optimizer_cls(param_groups, lr=lr)

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16)) if device.type == "cuda" else None

    plateau_scheduler = None
    step_scheduler = None
    epoch_scheduler = None
    if scheduler_type == "cosine_warmup":
        steps_per_epoch = max(1, len(train_loader))
        total_steps = args.epochs * steps_per_epoch
        warmup_steps = min(warmup_epochs * steps_per_epoch, max(1, total_steps - 1))
        step_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=functools.partial(cosine_warmup_lambda, warmup_steps=warmup_steps, total_steps=total_steps),
        )
    elif scheduler_type == "cosine":
        # Plain cosine decay over the full run, no warmup, stepped once per
        # epoch -- a fixed, well-behaved schedule for a training run of a
        # known length (unlike ReduceLROnPlateau, doesn't depend on val_loss
        # improving to make progress).
        epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=min_lr
        )
    else:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            # Always watches val_loss regardless of --monitor-metric: LR
            # plateau-reduction and early-stopping/checkpoint-selection are
            # deliberately separate concerns (see MODEL_HPARAM_DEFAULTS["resnet"]).
            optimizer,
            mode="min",
            factor=0.5,
            patience=lr_patience,
            min_lr=min_lr,
        )

    history = {
        "loss": [],
        "val_loss": [],
        "binary_accuracy": [],
        "val_binary_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_exact_match_accuracy": [],
        "lr": [],
    }

    best_monitor_value = float("-inf") if monitor_mode == "max" else float("inf")
    best_state = None
    best_ema_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_binary_acc = train_one_epoch(
            model,
            raw_model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp_dtype,
            scaler,
            grad_clip_norm,
            channels_last,
            ema,
            step_scheduler,
            label_smoothing,
            mixup_alpha,
        )

        eval_model = ema.shadow if ema is not None else model
        val_metrics, _, _ = evaluate(eval_model, val_loader, criterion, device, amp_dtype, channels_last)
        if plateau_scheduler is not None:
            plateau_scheduler.step(val_metrics["loss"])
        if epoch_scheduler is not None:
            epoch_scheduler.step()

        history["loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["binary_accuracy"].append(train_binary_acc)
        history["val_binary_accuracy"].append(val_metrics["binary_accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_exact_match_accuracy"].append(val_metrics["exact_match_accuracy"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"loss: {train_loss:.4f} - val_loss: {val_metrics['loss']:.4f} - "
            f"binary_acc: {train_binary_acc:.4f} - val_binary_acc: {val_metrics['binary_accuracy']:.4f} - "
            f"val_precision: {val_metrics['precision']:.4f} - val_recall: {val_metrics['recall']:.4f} - "
            f"val_exact_match_acc: {val_metrics['exact_match_accuracy']:.4f} - "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        current_monitor_value = val_metrics[monitor_metric]
        if is_better(current_monitor_value, best_monitor_value, monitor_mode):
            best_monitor_value = current_monitor_value
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            best_ema_state = (
                {k: v.detach().cpu().clone() for k, v in ema.shadow.state_dict().items()} if ema is not None else None
            )
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": MODEL_NAME,
                    "state_dict": best_state,
                    "ema_state_dict": best_ema_state,
                },
                os.path.join(run_dir, "best_model.pt"),
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val_{monitor_metric} improvement for {patience} epochs)."
                )
                break

    if best_state is not None:
        raw_model.load_state_dict(best_state)
        if ema is not None and best_ema_state is not None:
            ema.shadow.load_state_dict(best_ema_state)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["binary_accuracy"], label="train_binary_acc")
    plt.plot(history["val_binary_accuracy"], label="val_binary_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Accuracy")
    plt.title("Training and validation binary accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150)
    plt.close()

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    final_ema_state = (
        {k: v.detach().cpu().clone() for k, v in ema.shadow.state_dict().items()} if ema is not None else None
    )
    torch.save(
        {
            "model_name": MODEL_NAME,
            "state_dict": raw_model.state_dict(),
            "ema_state_dict": final_ema_state,
        },
        os.path.join(run_dir, "final_model.pt"),
    )

    print("\nEvaluating on test set...")
    eval_model = ema.shadow if ema is not None else raw_model
    test_metrics, y_true, y_pred = evaluate(eval_model, test_loader, criterion, device, amp_dtype, channels_last)
    pos_acc = per_position_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD)

    print("\nTest metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("Per-position accuracy (digits 0-9):")
    for digit, acc in enumerate(pos_acc):
        print(f"  digit {digit}: {acc:.4f}")

    summary = {
        "model": MODEL_NAME,
        "epochs_ran": len(history["loss"]),
        "config": vars(args),
        "resolved_hparams": {
            "lr": lr,
            "optimizer": optimizer_name,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "scheduler": scheduler_type,
            "warmup_epochs": warmup_epochs,
            "grad_clip_norm": grad_clip_norm,
            "patience": patience,
            "lr_patience": lr_patience,
            "min_lr": min_lr,
            "monitor_metric": monitor_metric,
            "monitor_mode": monitor_mode,
            "label_smoothing": label_smoothing,
            "mixup_alpha": mixup_alpha,
            "loss_type": loss_type,
            "focal_gamma": focal_gamma,
            "asl_gamma_neg": asl_gamma_neg,
            "asl_gamma_pos": asl_gamma_pos,
            "asl_clip": asl_clip,
            "asl_weight": asl_weight,
            "ema": ema_enabled,
            "ema_decay": ema_decay,
        },
        "test_metrics": test_metrics,
        "per_position_accuracy": {str(d): float(a) for d, a in enumerate(pos_acc)},
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRun artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
