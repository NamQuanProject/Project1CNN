"""Evaluation metrics for multi-label digit classification (PyTorch)."""

import torch

PREDICTION_THRESHOLD = 0.5


def exact_match_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD):
    """
    Exact-match accuracy for multi-label classification.

    A sample is counted correct only if every one of the 10 label
    positions matches the thresholded prediction.
    """
    y_pred_binary = (y_pred >= threshold).float()
    matches = torch.all(y_true == y_pred_binary, dim=1)
    return matches.float().mean().item()


def per_position_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD):
    """
    Accuracy of each of the 10 label positions independently.

    Returns a NumPy array of shape [10].
    """
    y_pred_binary = (y_pred >= threshold).float()
    correct = (y_true == y_pred_binary).float()
    return correct.mean(dim=0).cpu().numpy()


def binary_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD):
    """Per-label accuracy across all 10 outputs, averaged over the whole batch."""
    y_pred_binary = (y_pred >= threshold).float()
    return (y_true == y_pred_binary).float().mean().item()


def precision_recall(y_true, y_pred, threshold=PREDICTION_THRESHOLD, eps=1e-7):
    """Micro-averaged precision/recall over all label positions and samples."""
    y_pred_binary = (y_pred >= threshold).float()
    tp = (y_pred_binary * y_true).sum().item()
    fp = (y_pred_binary * (1 - y_true)).sum().item()
    fn = ((1 - y_pred_binary) * y_true).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return precision, recall
