"""Continual training phase: uncertainty-guided batch selection + a careful
(low, decaying) learning rate, continuing from an already-trained
checkpoint. Reference: "Batch Selection for Multi-Label Classification
Guided by Uncertainty and Dynamic Label Correlations" (arxiv 2412.16521).

This does NOT modify the main training recipe (src/train.py) -- it's a
separate, optional continuation phase you run on top of a checkpoint that's
already finished normal training (e.g. outputs/resnet_v2/final_model.pt),
intended to squeeze out a bit more by focusing extra gradient updates on
the genuinely hard/uncertain/unstable-prediction training examples instead
of uniform random shuffling.

How it works (paper-inspired; PDF text extraction failed for this paper, so
the equations below come from a secondary summary, not the literal paper --
symbols that weren't fully recoverable are reconstructed as documented
inline, aiming to reproduce the *described behavior* faithfully):

1. Warm-up (`--warmup-epochs`, default 5): train normally (uniform random
   shuffling) while building up a rolling window of each training
   example's per-label predictions -- no uncertainty signal exists yet.
2. After warm-up, each epoch:
   a. Per-label uncertainty u_ij combines (i) how much the prediction has
      been *changing* over the last T epochs (instability, Eq 4) and (ii)
      the *entropy* of the current prediction (closeness to 0.5, Eq 1) --
      combined via Eq 5.
   b. A label-correlation matrix C is estimated via pairwise mutual
      information between labels' uncertainty distributions across all
      training examples (Eq 6), then used to "smear" each label's
      uncertainty across labels it tends to be jointly uncertain with
      (Eq 7), summed per instance into a scalar weight (Eq 8).
   c. Per-instance weights become a sampling distribution (Eq 10),
      probability-weighted (not deterministic top-k) so hard examples are
      *more likely* to be drawn, not exclusively drawn.
   d. A "selection pressure" (Eq 11) exponentially decays from strongly
      uncertainty-biased right after warm-up to uniform by the final
      epoch -- avoids permanently ignoring "easy" examples.
3. Learning rate: this phase starts from an already-converged checkpoint,
   so it deliberately does NOT reuse the original (larger) training LR --
   default is 1/10th of the model's normal LR, with a short linear warmup
   then cosine decay to a very low floor, to avoid a large-LR shock
   knocking the converged weights off their optimum.

Usage:
    python src/continual_train.py --model-path outputs/resnet_v2/final_model.pt --epochs 20
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.data import load_splits
from src.metrics import PREDICTION_THRESHOLD, per_position_accuracy
from src.model import build_model, build_param_groups
from src.train import (
    ModelEMA,
    build_augmentation,
    build_criterion,
    evaluate,
    fix_random_seed,
    get_device,
    is_better,
    mixup_batch,
    resolve_amp_dtype,
    resolve_compile,
    resolve_hparam,
    smooth_labels,
)

MODEL_NAME = "resnet"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


class IndexedDataset(Dataset):
    """Wraps a dataset to also return each sample's index, so predictions
    can be written back into the right slot of the uncertainty history.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        return image, label, idx


# ---------------------------------------------------------------------------
# Uncertainty math (arxiv 2412.16521, Eq 1 / 4 / 5 / 6 / 7 / 8 / 10 / 11).
# ---------------------------------------------------------------------------


def binary_entropy_bits(p, eps=1e-6):
    """Eq 1: per-label entropy (bits) of the current prediction -- highest
    (=1) when p=0.5 (maximally uncertain), lowest (=0) when p is near 0/1.

    eps=1e-6 (not smaller): float32 has ~1.19e-7 precision near 1.0, so a
    smaller eps would make `1 - eps` round back to exactly 1.0, leaving p
    unclamped and producing log2(0) = -inf -> nan for p exactly 0 or 1.
    """
    p = p.clamp(eps, 1 - eps)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


def history_diff_uncertainty(history):
    """Eq 4: mean absolute difference between consecutive predictions
    within the sliding window -- high when the model's prediction for this
    (sample, label) has been unstable/flip-flopping across recent epochs.
    `history`: [N, T, num_classes] (oldest -> newest along dim 1).
    """
    diffs = (history[:, 1:, :] - history[:, :-1, :]).abs()
    return diffs.mean(dim=1)


def combined_uncertainty(entropy, diff, lam1=0.5):
    """Eq 5: convex combination of instability (diff) and entropy."""
    return lam1 * diff + (1 - lam1) * entropy


def label_correlation_matrix(U, num_bins=10, eps=1e-8):
    """Eq 6: q x q matrix of pairwise mutual information between labels'
    per-instance uncertainty distributions (each label's uncertainty
    values across all N instances, discretized into `num_bins` bins).

    Diagonal set to 1.0 -- the paper's summary didn't specify a diagonal
    convention; this keeps Eq 7 (U @ C) from zeroing out a label's own
    uncertainty contribution to itself.
    """
    n, q = U.shape
    bin_idx = torch.clamp((U * num_bins).long(), 0, num_bins - 1)
    C = torch.eye(q, dtype=torch.float32)
    for a in range(q):
        for b in range(a + 1, q):
            joint_idx = bin_idx[:, a] * num_bins + bin_idx[:, b]
            counts = torch.bincount(joint_idx, minlength=num_bins * num_bins).float()
            joint = (counts / n).reshape(num_bins, num_bins)
            pa = joint.sum(dim=1)
            pb = joint.sum(dim=0)
            outer = pa.unsqueeze(1) * pb.unsqueeze(0)
            mask = (joint > eps) & (outer > eps)
            mi = (joint[mask] * torch.log(joint[mask] / outer[mask])).sum().item()
            C[a, b] = mi
            C[b, a] = mi
    return C


def instance_weights(U, C):
    """Eq 7 (U_bar = U . C) + Eq 8 (sum across labels -> scalar per instance)."""
    adjusted = U @ C
    return adjusted.sum(dim=1)


def selection_probabilities(weights, selection_pressure):
    """Eq 10: turn per-instance weights into a sampling distribution over
    the pool, biased toward high-weight (high-uncertainty) instances by
    `selection_pressure` (>=1; ~1.0 = uniform).

    RECONSTRUCTION NOTE: the paper's Eq 10 defines this via a rank-quantile
    index Q(z) = floor((1-z)/Delta), Delta=1/n, but the exact definitions of
    z and n weren't recoverable from the available summary (direct PDF text
    extraction failed). This implementation uses z = each instance's
    normalized rank among `weights` (0 = lowest uncertainty, 1 = highest)
    and n = pool size, which reproduces the paper's *described* behavior:
    instances rank-ordered by uncertainty, with `selection_pressure`
    controlling how strongly the distribution favors the high-uncertainty
    end (pressure -> 1 = uniform, matching Eq 11's decay target). Verified
    below (see the module's self-test) that pressure=1 gives ~uniform
    probabilities and pressure>>1 concentrates mass on high-weight instances.
    """
    n = weights.shape[0]
    ranks = torch.argsort(torch.argsort(weights)).float()
    z = ranks / max(1, n - 1)
    delta = 1.0 / n
    q = torch.floor((1 - z) / delta)
    log_base = -math.log(max(selection_pressure, 1.0 + 1e-8)) / n
    log_unnorm = q * log_base
    log_unnorm = log_unnorm - log_unnorm.max()
    probs = torch.exp(log_unnorm)
    return probs / probs.sum()


def selection_pressure_schedule(epoch, warmup_epochs, total_epochs, s0=100.0):
    """Eq 11: exponential decay of selection pressure from s0 (right after
    warm-up) to 1.0 (uniform, at the final epoch).
    """
    remaining = max(1, total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / remaining))
    return s0 ** (1 - progress)


class PredictionHistory:
    """Rolling window of the last `window` sigmoid predictions for every
    (training example, label) pair -- feeds Eq 4's instability term.
    Initialized to 0.5 (maximally uncertain / uninformative) for samples
    not yet seen, which naturally gives them high early sampling priority.
    """

    def __init__(self, num_samples, num_classes, window):
        self.window = window
        self.buffer = torch.full((num_samples, window, num_classes), 0.5)

    @torch.no_grad()
    def update(self, indices, preds):
        indices = indices.cpu()
        preds = preds.detach().cpu()
        self.buffer[indices] = torch.roll(self.buffer[indices], shifts=-1, dims=1)
        self.buffer[indices, -1, :] = preds

    def current(self):
        return self.buffer[:, -1, :]

    def full_history(self):
        return self.buffer


def compute_sampling_distribution(history, lam1, num_bins, selection_pressure):
    current = history.current()
    hist = history.full_history()
    e = binary_entropy_bits(current)
    d = history_diff_uncertainty(hist)
    U = combined_uncertainty(e, d, lam1)
    C = label_correlation_matrix(U, num_bins)
    w = instance_weights(U, C)
    probs = selection_probabilities(w, selection_pressure)
    return probs, w, U, C


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------


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
    history,
    label_smoothing,
    mixup_alpha,
):
    model.train()
    total_loss = 0.0
    n = 0
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
    )
    for images, labels, idx in tqdm(loader, desc="train", leave=False):
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

        if ema is not None:
            ema.update(raw_model)

        # Record predictions on the *original* (unmixed) images for the
        # uncertainty history -- if mixup blended the images actually
        # trained on, its logits don't correspond to any single real
        # sample, so we only pay for an extra forward pass when mixup is
        # actually active.
        if mixup_alpha and mixup_alpha > 0:
            with torch.no_grad():
                preds = torch.sigmoid(model(images).float())
        else:
            preds = torch.sigmoid(logits.detach().float())
        history.update(idx, preds)

        total_loss += loss.item() * images.size(0)
        n += images.size(0)

    return total_loss / n


def parse_args():
    parser = argparse.ArgumentParser(
        description="Uncertainty-guided continual training phase (arxiv 2412.16521-inspired)."
    )
    parser.add_argument("--model-path", required=True, help="Checkpoint to continue training from.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None, help="Defaults to continual_<timestamp>.")
    parser.add_argument("--epochs", type=int, default=20, help="Total continual-training epochs (incl. warm-up).")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Epochs of uniform random sampling before uncertainty-guided selection begins (paper's gamma).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Defaults to the model's recommended batch size."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Peak LR for this phase. Defaults to 1/10th of the model's normal training LR -- "
        "deliberately conservative since we're continuing from a converged checkpoint, not "
        "training from scratch.",
    )
    parser.add_argument(
        "--lr-warmup-epochs", type=int, default=2, help="Linear LR warmup before cosine decay begins."
    )
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Floor LR for the cosine decay.")
    parser.add_argument(
        "--weight-decay", type=float, default=None, help="Defaults to the model's recommended weight decay."
    )
    parser.add_argument(
        "--window", type=int, default=5, help="T: sliding window size for the instability term (Eq 4)."
    )
    parser.add_argument(
        "--lam1", type=float, default=0.5, help="lambda_1: trade-off between instability and entropy (Eq 5)."
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Bins for discretizing uncertainty when estimating label mutual information (Eq 6).",
    )
    parser.add_argument(
        "--selection-pressure-s0",
        type=float,
        default=100.0,
        help="Initial selection pressure right after warm-up, decaying to 1.0 by the final epoch (Eq 11).",
    )
    parser.add_argument("--loss-type", choices=["bce", "focal", "asl"], default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Off by default: uncertainty history needs clean-image predictions, so mixup costs an "
        "extra forward pass per batch here for no benefit to the selection signal unless you "
        "specifically also want it as a loss regularizer.",
    )
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--asl-gamma-neg", type=float, default=None)
    parser.add_argument("--asl-gamma-pos", type=float, default=None)
    parser.add_argument("--asl-clip", type=float, default=None)
    parser.add_argument("--asl-weight", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument(
        "--monitor-metric",
        choices=["loss", "binary_accuracy", "precision", "recall", "exact_match_accuracy"],
        default="exact_match_accuracy",
    )
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs.")

    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--ema", dest="ema", action="store_true", default=None)
    ema_group.add_argument("--no-ema", dest="ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument(
        "--no-init-ema",
        dest="use_ema_init",
        action="store_false",
        default=True,
        help="Start from the checkpoint's raw weights instead of its EMA weights.",
    )

    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--compile", choices=["auto", "on", "off"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    fix_random_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    run_name = args.run_name or f"continual_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    base_lr = resolve_hparam(None, MODEL_NAME, "lr", 1e-3)
    lr = args.lr if args.lr is not None else base_lr / 10.0
    weight_decay = resolve_hparam(args.weight_decay, MODEL_NAME, "weight_decay", 0.0)
    batch_size = resolve_hparam(args.batch_size, MODEL_NAME, "batch_size", 128)
    loss_type = resolve_hparam(args.loss_type, MODEL_NAME, "loss_type", "bce")
    label_smoothing = resolve_hparam(args.label_smoothing, MODEL_NAME, "label_smoothing", 0.0)
    focal_gamma = resolve_hparam(args.focal_gamma, MODEL_NAME, "focal_gamma", 2.0)
    asl_gamma_neg = resolve_hparam(args.asl_gamma_neg, MODEL_NAME, "asl_gamma_neg", 4.0)
    asl_gamma_pos = resolve_hparam(args.asl_gamma_pos, MODEL_NAME, "asl_gamma_pos", 1.0)
    asl_clip = resolve_hparam(args.asl_clip, MODEL_NAME, "asl_clip", 0.05)
    asl_weight = resolve_hparam(args.asl_weight, MODEL_NAME, "asl_weight", 1.0)
    grad_clip_norm = resolve_hparam(args.grad_clip_norm, MODEL_NAME, "grad_clip_norm", 0.0)
    ema_enabled = resolve_hparam(args.ema, MODEL_NAME, "ema", False)
    ema_decay = resolve_hparam(args.ema_decay, MODEL_NAME, "ema_decay", 0.999)

    print(
        f"Continual-training hyperparameters: lr={lr} (base_lr/10 unless --lr given), "
        f"weight_decay={weight_decay}, batch_size={batch_size}, loss_type={loss_type}, "
        f"label_smoothing={label_smoothing}, warmup_epochs={args.warmup_epochs}, "
        f"window={args.window}, lam1={args.lam1}, num_bins={args.num_bins}, "
        f"selection_pressure_s0={args.selection_pressure_s0}, ema={ema_enabled}"
    )

    amp_dtype = resolve_amp_dtype(device, args.amp)
    use_compile = resolve_compile(device, args.compile)
    channels_last = device.type == "cuda"

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
    train_ds = IndexedDataset(splits["train"])
    val_ds, test_ds = splits["val"], splits["test"]
    n_train = len(train_ds)
    num_classes = train_ds.base.labels.shape[1]
    print("Train:", n_train, " Val:", len(val_ds), " Test:", len(test_ds))

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    checkpoint = torch.load(args.model_path, map_location=device)
    raw_model = build_model(checkpoint["model_name"]).to(device)
    ema_state = checkpoint.get("ema_state_dict")
    if args.use_ema_init and ema_state is not None:
        print("Continuing from the checkpoint's EMA weights.")
        raw_model.load_state_dict(ema_state)
    else:
        print("Continuing from the checkpoint's raw weights.")
        raw_model.load_state_dict(checkpoint["state_dict"])
    if channels_last:
        raw_model = raw_model.to(memory_format=torch.channels_last)

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
    param_groups = build_param_groups(raw_model, lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16)) if device.type == "cuda" else None

    # Careful LR handling: short linear warmup (avoid a large-LR shock to
    # the converged checkpoint), then cosine decay to a very low floor.
    lr_warmup_epochs = min(args.lr_warmup_epochs, args.epochs)

    def lr_lambda(epoch):
        if lr_warmup_epochs > 0 and epoch < lr_warmup_epochs:
            return (epoch + 1) / lr_warmup_epochs
        progress = (epoch - lr_warmup_epochs) / max(1, args.epochs - lr_warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        floor_ratio = args.min_lr / lr if lr > 0 else 0.0
        return floor_ratio + (1 - floor_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    history = PredictionHistory(n_train, num_classes, args.window)

    history_log = {
        "loss": [],
        "val_loss": [],
        "val_binary_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_exact_match_accuracy": [],
        "lr": [],
        "selection_pressure": [],
    }

    monitor_mode = "min" if args.monitor_metric == "loss" else "max"
    best_monitor_value = float("-inf") if monitor_mode == "max" else float("inf")
    best_state = None
    best_ema_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        is_warmup = epoch <= args.warmup_epochs
        if is_warmup:
            loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=n_train > batch_size,
            )
            selection_pressure = None
        else:
            pressure = selection_pressure_schedule(
                epoch, args.warmup_epochs, args.epochs, args.selection_pressure_s0
            )
            probs, _weights, _U, _C = compute_sampling_distribution(history, args.lam1, args.num_bins, pressure)
            sampler = WeightedRandomSampler(probs, num_samples=n_train, replacement=True)
            loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=n_train > batch_size,
            )
            selection_pressure = pressure

        train_loss = train_one_epoch(
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
            history,
            label_smoothing,
            args.mixup_alpha,
        )
        scheduler.step()

        eval_model = ema.shadow if ema is not None else model
        val_metrics, _, _ = evaluate(eval_model, val_loader, criterion, device, amp_dtype, channels_last)

        history_log["loss"].append(train_loss)
        history_log["val_loss"].append(val_metrics["loss"])
        history_log["val_binary_accuracy"].append(val_metrics["binary_accuracy"])
        history_log["val_precision"].append(val_metrics["precision"])
        history_log["val_recall"].append(val_metrics["recall"])
        history_log["val_exact_match_accuracy"].append(val_metrics["exact_match_accuracy"])
        history_log["lr"].append(optimizer.param_groups[0]["lr"])
        history_log["selection_pressure"].append(selection_pressure)

        phase = "warmup" if is_warmup else "selected"
        pressure_str = f" - selection_pressure: {selection_pressure:.2f}" if selection_pressure is not None else ""
        print(
            f"Epoch {epoch}/{args.epochs} [{phase}] - loss: {train_loss:.4f} - "
            f"val_loss: {val_metrics['loss']:.4f} - val_exact_match_acc: {val_metrics['exact_match_accuracy']:.4f} - "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}{pressure_str}"
        )

        current_value = val_metrics[args.monitor_metric]
        if is_better(current_value, best_monitor_value, monitor_mode):
            best_monitor_value = current_value
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            best_ema_state = (
                {k: v.detach().cpu().clone() for k, v in ema.shadow.state_dict().items()} if ema is not None else None
            )
            epochs_without_improvement = 0
            torch.save(
                {"model_name": MODEL_NAME, "state_dict": best_state, "ema_state_dict": best_ema_state},
                os.path.join(run_dir, "best_model.pt"),
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val_{args.monitor_metric} improvement for {args.patience} epochs)."
                )
                break

    if best_state is not None:
        raw_model.load_state_dict(best_state)
        if ema is not None and best_ema_state is not None:
            ema.shadow.load_state_dict(best_ema_state)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_log["loss"], label="train_loss")
    plt.plot(history_log["val_loss"], label="val_loss")
    plt.axvline(args.warmup_epochs - 0.5, color="gray", linestyle="--", label="warm-up ends")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Continual training loss")
    plt.subplot(1, 2, 2)
    plt.plot(history_log["val_exact_match_accuracy"], label="val_exact_match_accuracy")
    plt.axvline(args.warmup_epochs - 0.5, color="gray", linestyle="--", label="warm-up ends")
    plt.xlabel("Epoch")
    plt.ylabel("Exact-match accuracy")
    plt.legend()
    plt.title("Continual training val exact-match")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "continual_training_curves.png"), dpi=150)
    plt.close()

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history_log, f, indent=2)

    final_ema_state = (
        {k: v.detach().cpu().clone() for k, v in ema.shadow.state_dict().items()} if ema is not None else None
    )
    torch.save(
        {"model_name": MODEL_NAME, "state_dict": raw_model.state_dict(), "ema_state_dict": final_ema_state},
        os.path.join(run_dir, "final_model.pt"),
    )

    print("\nEvaluating on test set...")
    eval_model = ema.shadow if ema is not None else raw_model
    test_metrics, y_true, y_pred = evaluate(eval_model, test_loader, criterion, device, amp_dtype, channels_last)
    pos_acc = per_position_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD)

    print("\nTest metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    summary = {
        "source_checkpoint": args.model_path,
        "epochs_ran": len(history_log["loss"]),
        "config": vars(args),
        "resolved_hparams": {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "loss_type": loss_type,
            "label_smoothing": label_smoothing,
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
