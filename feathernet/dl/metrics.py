import numpy as np


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    predicted_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    correct = np.sum(predicted_labels == true_labels)
    total = len(true_labels)
    return correct / total
