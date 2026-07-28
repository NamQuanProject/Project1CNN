"""Save every test-set image whose prediction doesn't exactly match its true
label to a visualization folder, one annotated PNG per mismatch (image plus
true digits, predicted digits, and exactly which digits were missed/extra) --
for visual error analysis.

Usage:
    python src/visualize_errors.py --model-path outputs/resnet_v2/final_model.pt
"""

import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data import load_splits, multihot_to_digits
from src.metrics import PREDICTION_THRESHOLD
from src.model import build_model
from src.train import get_device, resolve_amp_dtype

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEFAULT_VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visualizations")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save annotated PNGs of every test-set exact-match mismatch."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to visualizations/<run-name>/, where <run-name> is the checkpoint's parent folder name.",
    )
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
        "--max-images",
        type=int,
        default=None,
        help="Cap the number of mismatch images saved (default: save all mismatches).",
    )
    return parser.parse_args()


def infer_run_name(model_path):
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(model_path)))
    return run_dir or "run"


def save_mismatch_image(test_ds, idx, probs, true_label, output_dir):
    true_digits = multihot_to_digits(true_label)
    pred_digits = multihot_to_digits(probs, threshold=PREDICTION_THRESHOLD)
    missing = sorted(set(true_digits) - set(pred_digits))  # false negatives: present but not predicted
    extra = sorted(set(pred_digits) - set(true_digits))  # false positives: predicted but not present
    center_label = test_ds.meta["center_labels"][idx].item()

    plt.figure(figsize=(4, 4.6))
    plt.imshow(test_ds.images[idx].numpy(), cmap="gray")
    plt.axis("off")
    plt.title(
        f"true={true_digits}\npred={pred_digits}\n"
        f"missing={missing}  extra={extra}\ncenter={center_label}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"error_{idx:05d}.png"), dpi=150)
    plt.close()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.amp)
    channels_last = device.type == "cuda"

    output_dir = args.output_dir or os.path.join(DEFAULT_VIZ_DIR, infer_run_name(args.model_path))
    os.makedirs(output_dir, exist_ok=True)

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
    model.eval()

    splits = load_splits(args.data_dir)
    test_ds = splits["test"]
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
    )

    n_total = 0
    n_mismatch = 0
    n_saved = 0
    idx_offset = 0
    for images, labels in loader:
        images_dev = images.to(device)
        if channels_last:
            images_dev = images_dev.to(memory_format=torch.channels_last)
        with autocast_ctx:
            logits = model(images_dev)
        probs = torch.sigmoid(logits.float()).cpu()
        preds_binary = (probs >= PREDICTION_THRESHOLD).float()
        matches = torch.all(preds_binary == labels, dim=1)

        for i in range(images.size(0)):
            global_idx = idx_offset + i
            n_total += 1
            if not matches[i]:
                n_mismatch += 1
                if args.max_images is None or n_saved < args.max_images:
                    save_mismatch_image(test_ds, global_idx, probs[i], labels[i], output_dir)
                    n_saved += 1
        idx_offset += images.size(0)

    exact_match_accuracy = 1 - n_mismatch / n_total
    print(
        f"Test set: {n_total} images, {n_mismatch} exact-match mismatches "
        f"({n_mismatch / n_total:.2%}), exact_match_accuracy={exact_match_accuracy:.4f}"
    )
    print(f"Saved {n_saved} annotated mismatch images to {output_dir}")

    summary = {
        "model_path": args.model_path,
        "total_images": n_total,
        "mismatches": n_mismatch,
        "exact_match_accuracy": exact_match_accuracy,
        "images_saved": n_saved,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
