"""Standalone evaluation of a saved model checkpoint against the test set.

Usage:
    python cnn/src/evaluate.py --model-path cnn/outputs/baseline_.../final_model.keras
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
import tensorflow as tf

from src.data import load_splits, multihot_to_digits
from src.metrics import PREDICTION_THRESHOLD, exact_match_accuracy, per_position_accuracy

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved multi-label digit CNN checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples-viz", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = tf.keras.models.load_model(
        args.model_path, custom_objects={"exact_match_accuracy": exact_match_accuracy}
    )

    splits = load_splits(args.data_dir)
    x_test, y_test, meta_test = splits["test"]

    results = model.evaluate(x_test, y_test, verbose=1)
    test_metrics = dict(zip(model.metrics_names, results))

    probs = model.predict(x_test, verbose=0)
    pos_acc = per_position_accuracy(y_test, probs, threshold=PREDICTION_THRESHOLD)

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

    n = min(args.num_samples_viz, len(x_test))
    indices = np.random.choice(len(x_test), size=n, replace=False)
    sample_probs = model.predict(x_test[indices], verbose=0)

    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i, idx in enumerate(indices):
        true_digits = multihot_to_digits(y_test[idx])
        pred_digits = multihot_to_digits(sample_probs[i], threshold=PREDICTION_THRESHOLD)
        center_label = meta_test["center_labels"][idx]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(x_test[idx].squeeze(), cmap="gray")
        plt.title(f"pred={pred_digits}\ntrue={true_digits}, center={center_label}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "sample_predictions.png"), dpi=150)
    plt.close()

    print(f"\nEvaluation artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
