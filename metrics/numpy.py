import numpy as np


def average_precision(labels, desired_label):
    """ labels: shape (N,)"""
    correct = labels == desired_label

    if not np.any(correct):
        return .0

    indices = np.where(correct)
    correct_count = np.cumsum(correct)
    item_numbers = np.arange(1, len(labels) + 1)
    precision = correct_count / item_numbers

    return np.mean(precision[indices])


def mean_average_precision(labels, desired_label):
    """ labels: shape (batch, N)"""
    avg_precisions = np.zeros_like(desired_label, dtype=np.float)
    for i, (_labels, target) in enumerate(zip(labels, desired_label)):
        avg_precisions[i] = average_precision(_labels, target)

    return np.mean(avg_precisions)
