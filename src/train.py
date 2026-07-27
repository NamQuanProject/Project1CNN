"""Train the multi-label digit CNN (baseline or improved) using PyTorch.

Usage (from repo root or from src/, either works):
    python src/train.py --model baseline
    python src/train.py --model improved --augment --epochs 100
"""

import argparse
import json
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
from src.model import MODEL_BUILDERS

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
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the multi-label digit CNN (PyTorch).")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), default="baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply light rotation/translation/scale augmentation to training data.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Subfolder name under output-dir. Defaults to <model>_<timestamp>.",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run a full pass over `loader`, returning (metrics_dict, y_true, y_pred)."""
    model.eval()
    total_loss = 0.0
    n = 0
    all_targets = []
    all_probs = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_targets.append(labels.cpu())
        all_probs.append(torch.sigmoid(logits).cpu())

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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    all_targets = []
    all_probs = []
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_targets.append(labels.detach().cpu())
        all_probs.append(torch.sigmoid(logits).detach().cpu())

    y_true = torch.cat(all_targets)
    y_pred = torch.cat(all_probs)
    train_binary_acc = binary_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD)
    return total_loss / n, train_binary_acc


def main():
    args = parse_args()
    fix_random_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    run_name = args.run_name or f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Loading data from {args.data_dir} ...")
    transform_train = build_augmentation() if args.augment else None
    splits = load_splits(args.data_dir, transform_train=transform_train)
    train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]
    print("Train:", len(train_ds))
    print("Val:  ", len(val_ds))
    print("Test: ", len(test_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = MODEL_BUILDERS[args.model]().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    with open(os.path.join(run_dir, "model_summary.txt"), "w") as f:
        f.write(str(model) + "\n\n")
        f.write(f"Total parameters: {num_params:,}\n")
    print(model)
    print(f"Total parameters: {num_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience, min_lr=1e-5
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
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_binary_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

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
            f"val_exact_match_acc: {val_metrics['exact_match_accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            torch.save(
                {"model_name": args.model, "state_dict": best_state},
                os.path.join(run_dir, "best_model.pt"),
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val_loss improvement for {args.patience} epochs)."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

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

    torch.save(
        {"model_name": args.model, "state_dict": model.state_dict()},
        os.path.join(run_dir, "final_model.pt"),
    )

    print("\nEvaluating on test set...")
    test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)
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
        "test_metrics": test_metrics,
        "per_position_accuracy": {str(d): float(a) for d, a in enumerate(pos_acc)},
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRun artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
