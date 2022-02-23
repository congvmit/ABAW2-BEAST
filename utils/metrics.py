from sklearn.metrics import f1_score
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
import os
import pandas as pd

def print_metrics(y_true, y_pred):
    raise 
    LABEL_MAPPING = {"moderate - 0": 0, "not depression - 1": 1, "severe - 2": 2}
    col, _ = os.get_terminal_size()
    col = int(col * 0.85)
    print("=" * col)
    print(
        classification_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=range(len(LABEL_MAPPING)),
            target_names=LABEL_MAPPING.keys(),
        )
    )
    print("-" * col)
    print("Confusion Matrix: Row (True) - Col (Pred)")
    print(
        pd.DataFrame(
            confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                labels=range(len(LABEL_MAPPING)),
                normalize="true",
            ),
            columns=LABEL_MAPPING.keys(),
        )
    )


def EXP_metric(input, target):
    """Compute F1 Score for AU"""
    return f1_score(
        input, target, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8], average="macro"
    )


def AU_metric(input, target):
    """Compute F1 Score for AU"""
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(target[:, i], input[:, i])
        f1s.append(f1)
    return np.mean(f1s)


def CCC_score(x, y):
    """Compute CCC Score for AU"""
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2 * rho * x_s * y_s / (x_s**2 + y_s**2 + (x_m - y_m) ** 2)
    return ccc


def VA_metric(input, target):
    ccc = [CCC_score(input[:, 0], target[:, 0]), CCC_score(input[:, 1], target[:, 1])]
    return np.mean(ccc)


