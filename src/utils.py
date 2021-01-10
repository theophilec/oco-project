import numpy as np


def softmax(x):
    return np.exp(x) / np.exp(x).sum()
