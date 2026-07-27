"""Data loading for the multi-label MNIST-digits dataset (.pt files), PyTorch."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiLabelDigitsDataset(Dataset):
    """Wraps one saved .pt split as a PyTorch Dataset.

    images: float32 tensor [1, 64, 64], normalized to [0, 1].
    labels: float32 tensor [10], multi-hot digit-presence vector.
    """

    def __init__(self, path, transform=None):
        data = torch.load(path, map_location="cpu")

        self.images = data["images"].float() / 255.0  # [N, 64, 64]
        self.labels = data["labels"].float()  # [N, 10]
        self.transform = transform

        self.meta = {
            "center_labels": data["center_labels"].long(),
            "count_labels": data["count_labels"].long(),
            "num_digits": data["num_digits"].long(),
            "all_digit_labels": data["all_digit_labels"].long(),
            "all_positions": data["all_positions"].long(),
            "label_type": data.get("label_type", None),
        }

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx].unsqueeze(0)  # [1, 64, 64]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def load_splits(data_dir, transform_train=None):
    """Load train/val/test splits as MultiLabelDigitsDataset objects.

    `transform_train` (optional) is applied only to the training split
    (e.g. data augmentation); val/test are left untransformed.
    """
    paths = {
        "train": os.path.join(data_dir, "train.pt"),
        "val": os.path.join(data_dir, "val.pt"),
        "test": os.path.join(data_dir, "test.pt"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing dataset file(s): {missing}")

    return {
        "train": MultiLabelDigitsDataset(paths["train"], transform=transform_train),
        "val": MultiLabelDigitsDataset(paths["val"]),
        "test": MultiLabelDigitsDataset(paths["test"]),
    }


def multihot_to_digits(multihot, threshold=0.5):
    """Convert a multi-hot vector (tensor or array) into a list of digit labels."""
    if torch.is_tensor(multihot):
        multihot = multihot.detach().cpu().numpy()
    return np.where(multihot >= threshold)[0].tolist()
