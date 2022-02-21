from sklearn.metrics import f1_score
import numpy as np


def exp_f1_score(input, target):
    return f1_score(
        input, target, pos_label=[0, 1, 2, 3, 4, 5, 6, 7, 8], average="macro"
    )


def averaged_f1_score(input, target):
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(target[:, i], input[:, i])
        f1s.append(f1)
    return np.mean(f1s), f1s
