"""Data loading for the multi-label MNIST-digits dataset (.pt files)."""

import os

import numpy as np
import torch


def load_data_pt(path):
    """
    Load one saved .pt split and convert it to Keras-friendly arrays.

    Returns
    -------
    images:
        NumPy array of shape [N, 64, 64, 1], values in [0, 1].

    labels:
        NumPy array of shape [N, 10].
        These are multi-hot labels used for training.

    meta:
        Dict of extra metadata, not required for model training.
    """
    data = torch.load(path, map_location="cpu")

    images = data["images"].numpy().astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)

    labels = data["labels"].numpy().astype("float32")

    meta = {
        "center_labels": data["center_labels"].numpy().astype("int64"),
        "count_labels": data["count_labels"].numpy().astype("int64"),
        "num_digits": data["num_digits"].numpy().astype("int64"),
        "all_digit_labels": data["all_digit_labels"].numpy().astype("int64"),
        "all_positions": data["all_positions"].numpy().astype("int64"),
        "label_type": data.get("label_type", None),
    }

    return images, labels, meta


def load_splits(data_dir):
    """Load train/val/test splits from `data_dir`.

    Expects train.pt, val.pt, test.pt to exist in `data_dir`.
    """
    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")
    test_path = os.path.join(data_dir, "test.pt")

    missing = [p for p in (train_path, val_path, test_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing dataset file(s): {missing}")

    x_train, y_train, meta_train = load_data_pt(train_path)
    x_val, y_val, meta_val = load_data_pt(val_path)
    x_test, y_test, meta_test = load_data_pt(test_path)

    return {
        "train": (x_train, y_train, meta_train),
        "val": (x_val, y_val, meta_val),
        "test": (x_test, y_test, meta_test),
    }


def multihot_to_digits(multihot, threshold=0.5):
    """Convert a multi-hot vector into a list of digit labels, e.g. [1, 4, 9]."""
    return np.where(multihot >= threshold)[0].tolist()
