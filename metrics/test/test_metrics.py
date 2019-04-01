import numpy as np
import pytest
from metrics.numpy import average_precision, mean_average_precision


def test_avg_precision():
    labels = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 1])
    assert average_precision(labels, 1) == pytest.approx(0.78, 0.01)

    labels = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 1])
    assert average_precision(labels, 1) == pytest.approx(0.52, 0.01)

    labels = np.array([3, 1, 3, 5, 1, 3, 1, 7, 3, 3])
    assert average_precision(labels, 3) == pytest.approx(0.62, 0.01)

    labels = np.array([3, 1, 3, 5, 1, 3, 1, 7, 3, 3])
    assert average_precision(labels, 1) == pytest.approx(0.44, 0.01)

    labels = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 1])
    assert average_precision(labels, 2) == pytest.approx(0.0, 0.01)

    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert average_precision(labels, 1) == pytest.approx(1.0, 0.01)


def test_map():
    labels = np.array([
        [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
    ])
    target_labels = np.array([1, 1])
    score = mean_average_precision(labels, target_labels)
    assert score == pytest.approx(0.65, 0.01)


    labels = np.array([
        [3, 1, 3, 5, 1, 3, 1, 7, 3, 3],
        [3, 1, 3, 5, 1, 3, 1, 7, 3, 3]
    ])
    target_labels = np.array([3, 1])
    score = mean_average_precision(labels, target_labels)
    assert score == pytest.approx(0.53, 0.01)
