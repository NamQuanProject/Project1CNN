"""Standalone evaluation of a saved PyTorch checkpoint against the test set.

Usage:
    python src/evaluate.py --model-path outputs/baseline_.../final_model.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import load_splits, multihot_to_digits
from src.metrics import PREDICTION_THRESHOLD, per_position_accuracy
from src.model import build_model
from src.train import evaluate as run_evaluate
from src.train import get_device, resolve_amp_dtype

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved multi-label digit CNN checkpoint.")
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
    parser.add_argument("--num-samples-viz", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.amp)
    channels_last = device.type == "cuda"

    checkpoint = torch.load(args.model_path, map_location=device)
    # Always pretrained=False: we're about to overwrite the weights with the
    # checkpoint's trained state_dict, so downloading ImageNet weights first
    # would be wasted work (and fail on offline eval machines).
    model = build_model(checkpoint["model_name"], pretrained=False).to(device)

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

    criterion = nn.BCEWithLogitsLoss()
    test_metrics, y_true, y_pred = run_evaluate(model, test_loader, criterion, device, amp_dtype, channels_last)
    pos_acc = per_position_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD)

    print("Test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("Per-position accuracy (digits 0-9):")
    for digit, acc in enumerate(pos_acc):
        print(f"  digit {digit}: {acc:.4f}")

    summary = {
        "model_path": args.model_path,
        "test_metrics": test_metrics,
        "per_position_accuracy": {str(d): float(a) for d, a in enumerate(pos_acc)},
    }
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    model.eval()
    n = min(args.num_samples_viz, len(test_ds))
    indices = np.random.choice(len(test_ds), size=n, replace=False)

    images = torch.stack([test_ds[i][0] for i in indices]).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(images)).cpu().numpy()

    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i, idx in enumerate(indices):
        true_digits = multihot_to_digits(test_ds.labels[idx])
        pred_digits = multihot_to_digits(probs[i], threshold=PREDICTION_THRESHOLD)
        center_label = test_ds.meta["center_labels"][idx].item()

        plt.subplot(rows, cols, i + 1)
        plt.imshow(test_ds.images[idx].numpy(), cmap="gray")
        plt.title(f"pred={pred_digits}\ntrue={true_digits}, center={center_label}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "sample_predictions.png"), dpi=150)
    plt.close()

    print(f"\nEvaluation artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
