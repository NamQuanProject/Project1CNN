"""Test-time augmentation (TTA) evaluation: average predictions over several
augmented views of each test image instead of a single forward pass, and
compare against a plain (no-TTA) baseline.

Mirrors the SAME augmentation space used at training time (src/train.py:
build_augmentation -- translate +-4%, zoom +-8%, contrast +-12%,
deliberately no rotation or flip, since those can turn a 6 into a 9 or vice
versa and corrupt the label), but as a fixed, deterministic set of views
(not random sampling) so results are reproducible run to run.

Usage:
    python src/evaluate_augmentation.py --model-path outputs/resnet_v2/final_model.pt
"""

import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import load_splits
from src.metrics import (
    PREDICTION_THRESHOLD,
    binary_accuracy,
    exact_match_accuracy,
    per_position_accuracy,
    precision_recall,
)
from src.model import build_model
from src.train import get_device, resolve_amp_dtype

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "eval_tta")


def shift_image_batch(images, dx, dy):
    """Shift a batch of images [B, C, H, W] by (dx, dy) pixels, zero-padding
    the revealed border. Pure tensor slicing -- deterministic, no
    torchvision-version-dependent affine API involved.
    """
    if dx == 0 and dy == 0:
        return images
    _, _, h, w = images.shape
    shifted = torch.zeros_like(images)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    shifted[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = images[:, :, src_y0:src_y1, src_x0:src_x1]
    return shifted


def zoom_image_batch(images, scale):
    """Zoom a batch of images by `scale` (>1 = zoom in + center-crop, <1 =
    zoom out + center-pad), keeping the output the same H x W. Uses only
    torch.nn.functional.interpolate (stable core PyTorch, avoids any
    torchvision-functional-transform version dependency).
    """
    if scale == 1.0:
        return images
    b, c, h, w = images.shape
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
    if scale >= 1.0:
        top, left = (new_h - h) // 2, (new_w - w) // 2
        return resized[:, :, top : top + h, left : left + w]
    out = torch.zeros((b, c, h, w), dtype=images.dtype, device=images.device)
    top, left = (h - new_h) // 2, (w - new_w) // 2
    out[:, :, top : top + new_h, left : left + new_w] = resized
    return out


def contrast_image_batch(images, factor):
    """Adjust contrast by scaling each image's deviation from its own mean
    intensity, clamped back to [0, 1] -- matches the training-time
    ColorJitter(contrast=...) semantics.
    """
    if factor == 1.0:
        return images
    mean = images.mean(dim=(2, 3), keepdim=True)
    return ((images - mean) * factor + mean).clamp(0.0, 1.0)


def build_tta_views(use_shift=True, use_zoom=True, use_contrast=True):
    """Build the list of (name, transform_fn) TTA views. Deliberately no
    rotation/flip -- see module docstring. Fixed points spanning the
    training-time augmentation ranges (translate +-4% ~= +-2-3px, zoom
    +-8%, contrast +-12%), rather than random samples.
    """
    views = [("identity", lambda x: x)]
    if use_shift:
        for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            views.append((f"shift_{dx}_{dy}", lambda x, dx=dx, dy=dy: shift_image_batch(x, dx, dy)))
    if use_zoom:
        for scale in [0.92, 1.08]:
            views.append((f"zoom_{scale}", lambda x, scale=scale: zoom_image_batch(x, scale)))
    if use_contrast:
        for factor in [0.88, 1.12]:
            views.append((f"contrast_{factor}", lambda x, factor=factor: contrast_image_batch(x, factor)))
    return views


@torch.no_grad()
def tta_predict(model, images, views, device, amp_dtype, channels_last):
    """Average sigmoid probabilities over `views` of each image."""
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
    )
    probs_sum = None
    for _name, transform in views:
        view = transform(images)
        if channels_last:
            view = view.to(memory_format=torch.channels_last)
        with autocast_ctx:
            logits = model(view)
        probs = torch.sigmoid(logits.float())
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(views)


def _metrics_from_probs(y_true, y_pred, total_loss, n):
    precision, recall = precision_recall(y_true, y_pred, threshold=PREDICTION_THRESHOLD)
    return {
        "loss": total_loss / n,
        "binary_accuracy": binary_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD),
        "precision": precision,
        "recall": recall,
        "exact_match_accuracy": exact_match_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD),
    }


@torch.no_grad()
def evaluate_with_tta(model, loader, views, device, amp_dtype, channels_last):
    model.eval()
    all_targets, all_probs = [], []
    total_loss, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        probs = tta_predict(model, images, views, device, amp_dtype, channels_last)
        loss = F.binary_cross_entropy(probs.clamp(1e-7, 1 - 1e-7), labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_targets.append(labels.cpu())
        all_probs.append(probs.cpu())
    y_true, y_pred = torch.cat(all_targets), torch.cat(all_probs)
    return _metrics_from_probs(y_true, y_pred, total_loss, n), y_true, y_pred


@torch.no_grad()
def evaluate_plain(model, loader, device, amp_dtype, channels_last):
    """Single-forward-pass baseline (no TTA), for side-by-side comparison."""
    model.eval()
    all_targets, all_probs = [], []
    total_loss, n = 0.0, 0
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
        probs = torch.sigmoid(logits.float())
        loss = F.binary_cross_entropy(probs.clamp(1e-7, 1 - 1e-7), labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_targets.append(labels.cpu())
        all_probs.append(probs.cpu())
    y_true, y_pred = torch.cat(all_targets), torch.cat(all_probs)
    return _metrics_from_probs(y_true, y_pred, total_loss, n), y_true, y_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with test-time augmentation (TTA).")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        default=True,
        help="If the checkpoint has EMA weights, use the raw (non-EMA) weights instead.",
    )
    parser.add_argument(
        "--no-shift", dest="use_shift", action="store_false", default=True, help="Disable shift TTA views."
    )
    parser.add_argument(
        "--no-zoom", dest="use_zoom", action="store_false", default=True, help="Disable zoom TTA views."
    )
    parser.add_argument(
        "--no-contrast", dest="use_contrast", action="store_false", default=True, help="Disable contrast TTA views."
    )
    parser.add_argument(
        "--no-compare",
        dest="compare",
        action="store_false",
        default=True,
        help="Skip the plain (no-TTA) baseline pass -- only run TTA.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.amp)
    channels_last = device.type == "cuda"

    checkpoint = torch.load(args.model_path, map_location=device)
    model = build_model(checkpoint["model_name"]).to(device)
    ema_state = checkpoint.get("ema_state_dict")
    if args.use_ema and ema_state is not None:
        print("Using EMA weights from checkpoint.")
        model.load_state_dict(ema_state)
    else:
        model.load_state_dict(checkpoint["state_dict"])
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    splits = load_splits(args.data_dir)
    test_ds = splits["test"]
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    views = build_tta_views(use_shift=args.use_shift, use_zoom=args.use_zoom, use_contrast=args.use_contrast)
    print(f"TTA views ({len(views)}): {[name for name, _ in views]}")

    summary = {"model_path": args.model_path, "tta_views": [name for name, _ in views]}

    if args.compare:
        print("\nEvaluating WITHOUT test-time augmentation (single forward pass)...")
        plain_metrics, _, _ = evaluate_plain(model, test_loader, device, amp_dtype, channels_last)
        print("Plain test metrics:")
        for name, value in plain_metrics.items():
            print(f"  {name}: {value:.4f}")
        summary["plain_metrics"] = plain_metrics

    print(f"\nEvaluating WITH test-time augmentation ({len(views)} views, averaged)...")
    tta_metrics, y_true, y_pred = evaluate_with_tta(model, test_loader, views, device, amp_dtype, channels_last)
    pos_acc = per_position_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD)

    print("TTA test metrics:")
    for name, value in tta_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("Per-position accuracy (digits 0-9), TTA:")
    for digit, acc in enumerate(pos_acc):
        print(f"  digit {digit}: {acc:.4f}")

    summary["tta_metrics"] = tta_metrics
    summary["per_position_accuracy_tta"] = {str(d): float(a) for d, a in enumerate(pos_acc)}

    if args.compare:
        delta = tta_metrics["exact_match_accuracy"] - summary["plain_metrics"]["exact_match_accuracy"]
        print(f"\nexact_match_accuracy delta (TTA - plain): {delta:+.4f}")
        summary["exact_match_accuracy_delta"] = delta

    with open(os.path.join(args.output_dir, "tta_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
