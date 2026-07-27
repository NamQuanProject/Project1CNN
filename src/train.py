"""Train the multi-label digit CNN using PyTorch, with an optimized recipe
available for the resnet50 model: mixed precision (bf16/fp16 autocast),
channels-last memory format, torch.compile, differential learning rates
(pretrained backbone vs. new head), no weight decay on norm/bias params,
cosine LR schedule with warmup, gradient clipping, and EMA of weights.

Usage (from repo root or from src/, either works):
    python src/train.py --model baseline
    python src/train.py --model resnet50 --augment --run-name resnet50_v1
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
from src.model import MODEL_BUILDERS, MODEL_HPARAM_DEFAULTS, build_model, build_param_groups

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


def build_augmentation(model_name):
    """Light training-time augmentation. No rotation/flip for "resnet" or
    "resnet50": that can turn a 6 into a 9 (or vice versa) and corrupt the
    label.
    """
    if model_name in ("resnet", "resnet50"):
        return transforms.Compose(
            [
                # Keras RandomTranslation(0.04, 0.04) / RandomZoom(-0.08, 0.08).
                transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.92, 1.08)),
                # Keras RandomContrast(0.12).
                transforms.ColorJitter(contrast=0.12),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
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
    parser = argparse.ArgumentParser(description="Train the multi-label digit CNN (PyTorch).")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), default="baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Defaults to the model's recommended batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Defaults to the model's recommended lr (see MODEL_HPARAM_DEFAULTS)."
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="LR for pretrained backbone layers (resnet50 only). Defaults to the model's recommendation.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default=None,
        help="Defaults to the model's recommended optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Defaults to the model's recommended weight decay. Never applied to norm/bias params.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["plateau", "cosine_warmup"],
        default=None,
        help="Defaults to the model's recommendation ('plateau' for most models, "
        "'cosine_warmup' for resnet50).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Linear warmup length for the cosine_warmup scheduler. Defaults to the model's recommendation.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Gradient clipping max-norm (0 disables). Defaults to the model's recommendation.",
    )
    parser.add_argument("--patience", type=int, default=None, help="Early-stopping patience in epochs.")
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

    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    pretrained_group.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="resnet50 only: skip loading ImageNet-pretrained backbone weights.",
    )

    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--ema", dest="ema", action="store_true", default=None)
    ema_group.add_argument("--no-ema", dest="ema", action="store_false", help="Disable EMA of model weights.")
    parser.add_argument(
        "--ema-decay", type=float, default=None, help="Defaults to the model's recommended EMA decay."
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply light augmentation to training data (rotation+translate+scale, "
        "or for --model resnet/resnet50: translate+zoom+contrast only, no rotation/flip).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Subfolder name under output-dir. Defaults to <model>_<timestamp>.",
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

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = model(images)
            loss = criterion(logits, labels)

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

    run_name = args.run_name or f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    lr = resolve_hparam(args.lr, args.model, "lr", 1e-3)
    backbone_lr = resolve_hparam(args.backbone_lr, args.model, "backbone_lr", lr)
    optimizer_name = resolve_hparam(args.optimizer, args.model, "optimizer", "adam")
    weight_decay = resolve_hparam(args.weight_decay, args.model, "weight_decay", 0.0)
    batch_size = resolve_hparam(args.batch_size, args.model, "batch_size", 128)
    scheduler_type = resolve_hparam(args.scheduler, args.model, "scheduler", "plateau")
    warmup_epochs = resolve_hparam(args.warmup_epochs, args.model, "warmup_epochs", 0)
    grad_clip_norm = resolve_hparam(args.grad_clip_norm, args.model, "grad_clip_norm", 0.0)
    patience = resolve_hparam(args.patience, args.model, "patience", 5)
    pretrained = resolve_hparam(args.pretrained, args.model, "pretrained", False)
    ema_enabled = resolve_hparam(args.ema, args.model, "ema", False)
    ema_decay = resolve_hparam(args.ema_decay, args.model, "ema_decay", 0.999)

    print(
        f"Hyperparameters: lr={lr}, backbone_lr={backbone_lr}, optimizer={optimizer_name}, "
        f"weight_decay={weight_decay}, batch_size={batch_size}, scheduler={scheduler_type}, "
        f"warmup_epochs={warmup_epochs}, grad_clip_norm={grad_clip_norm}, patience={patience}, "
        f"pretrained={pretrained}, ema={ema_enabled} (decay={ema_decay})"
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
    transform_train = build_augmentation(args.model) if args.augment else None
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

    raw_model = build_model(args.model, pretrained=pretrained).to(device)
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

    criterion = nn.BCEWithLogitsLoss()
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    param_groups = build_param_groups(
        raw_model,
        lr=lr,
        weight_decay=weight_decay,
        backbone_lr=backbone_lr if hasattr(raw_model, "head_and_backbone_named_parameters") else None,
    )
    optimizer = optimizer_cls(param_groups, lr=lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16)) if device.type == "cuda" else None

    plateau_scheduler = None
    step_scheduler = None
    if scheduler_type == "cosine_warmup":
        steps_per_epoch = max(1, len(train_loader))
        total_steps = args.epochs * steps_per_epoch
        warmup_steps = min(warmup_epochs * steps_per_epoch, max(1, total_steps - 1))
        step_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=functools.partial(cosine_warmup_lambda, warmup_steps=warmup_steps, total_steps=total_steps),
        )
    else:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience, min_lr=1e-5
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

    best_val_loss = float("inf")
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
        )

        eval_model = ema.shadow if ema is not None else model
        val_metrics, _, _ = evaluate(eval_model, val_loader, criterion, device, amp_dtype, channels_last)
        if plateau_scheduler is not None:
            plateau_scheduler.step(val_metrics["loss"])

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

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            best_ema_state = (
                {k: v.detach().cpu().clone() for k, v in ema.shadow.state_dict().items()} if ema is not None else None
            )
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": best_state,
                    "ema_state_dict": best_ema_state,
                    "pretrained": pretrained,
                },
                os.path.join(run_dir, "best_model.pt"),
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no val_loss improvement for {patience} epochs).")
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
            "model_name": args.model,
            "state_dict": raw_model.state_dict(),
            "ema_state_dict": final_ema_state,
            "pretrained": pretrained,
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
        "model": args.model,
        "epochs_ran": len(history["loss"]),
        "config": vars(args),
        "resolved_hparams": {
            "lr": lr,
            "backbone_lr": backbone_lr,
            "optimizer": optimizer_name,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "scheduler": scheduler_type,
            "warmup_epochs": warmup_epochs,
            "grad_clip_norm": grad_clip_norm,
            "patience": patience,
            "pretrained": pretrained,
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
