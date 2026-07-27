"""Evaluation metrics for multi-label digit classification."""

import tensorflow as tf

PREDICTION_THRESHOLD = 0.5


def exact_match_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD):
    """
    Exact-match accuracy for multi-label classification.

    A sample is counted correct only if every one of the 10 label
    positions matches the thresholded prediction.
    """
    y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)
    matches = tf.reduce_all(tf.equal(y_true, y_pred_binary), axis=1)
    return tf.reduce_mean(tf.cast(matches, tf.float32))


def per_position_accuracy(y_true, y_pred, threshold=PREDICTION_THRESHOLD):
    """
    Accuracy of each of the 10 label positions independently.

    Returns a NumPy array of shape [10].
    """
    y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)
    correct = tf.equal(y_true, y_pred_binary)
    return tf.reduce_mean(tf.cast(correct, tf.float32), axis=0).numpy()
