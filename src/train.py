"""Train the multi-label digit CNN (baseline or improved) headlessly.

Usage (from repo root or from cnn/, either works):
    python cnn/src/train.py --model baseline
    python cnn/src/train.py --model improved --augment --epochs 100
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
import tensorflow as tf
import torch

from src.data import load_splits
from src.metrics import PREDICTION_THRESHOLD, exact_match_accuracy, per_position_accuracy
from src.model import MODEL_BUILDERS

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)


def build_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomZoom(0.05),
        ],
        name="augmentation",
    )


def make_train_dataset(images, labels, batch_size, augment_layer, seed):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda x, y: (augment_layer(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the multi-label digit CNN.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), default="baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply light rotation/translation/zoom augmentation to training data.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Subfolder name under output-dir. Defaults to <model>_<timestamp>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fix_random_seed(args.seed)

    run_name = args.run_name or f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Loading data from {args.data_dir} ...")
    splits = load_splits(args.data_dir)
    x_train, y_train, _ = splits["train"]
    x_val, y_val, _ = splits["val"]
    x_test, y_test, _ = splits["test"]
    print("Train:", x_train.shape, y_train.shape)
    print("Val:  ", x_val.shape, y_val.shape)
    print("Test: ", x_test.shape, y_test.shape)

    input_shape = x_train.shape[1:]
    model = MODEL_BUILDERS[args.model](input_shape=input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            exact_match_accuracy,
        ],
    )
    model.summary()
    with open(os.path.join(run_dir, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=args.patience, min_lr=1e-5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(run_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    if args.augment:
        augment_layer = build_augmentation()
        train_ds = make_train_dataset(x_train, y_train, args.batch_size, augment_layer, args.seed)
        history = model.fit(
            train_ds,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["binary_accuracy"], label="train_binary_acc")
    plt.plot(history.history["val_binary_accuracy"], label="val_binary_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Accuracy")
    plt.title("Training and validation binary accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150)
    plt.close()

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    model.save(os.path.join(run_dir, "final_model.keras"))

    print("\nEvaluating on test set...")
    results = model.evaluate(x_test, y_test, verbose=1)
    test_metrics = dict(zip(model.metrics_names, results))

    probs = model.predict(x_test, verbose=0)
    pos_acc = per_position_accuracy(y_test, probs, threshold=PREDICTION_THRESHOLD)

    print("\nTest metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("Per-position accuracy (digits 0-9):")
    for digit, acc in enumerate(pos_acc):
        print(f"  digit {digit}: {acc:.4f}")

    summary = {
        "model": args.model,
        "epochs_ran": len(history.history["loss"]),
        "config": vars(args),
        "test_metrics": test_metrics,
        "per_position_accuracy": {str(d): float(a) for d, a in enumerate(pos_acc)},
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRun artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
