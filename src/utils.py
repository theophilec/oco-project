from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


class Logger:
    def __init__(self, algo_tag: str):
        self.tag = algo_tag
        self.iterations = []
        self.loss = []
        self.train_error = []
        self.test_error = []

    def log(self, iteration: int, loss: float, train_err: float, test_err: float):
        self.iterations.append(iteration)
        self.loss.append(loss)
        self.train_error.append(train_err)
        self.test_error.append(test_err)


def plot_results(loggers: List[Logger]):
    # Create plots and set axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))
    ax1.set_title("test error")
    ax2.set_title("train error")
    ax3.set_title("training loss")
    for ax in [ax1, ax2]:
        # log scale for error plots
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # log results from our algorithms
    for logger in loggers:
        ax1.plot(logger.iterations, logger.test_error, label=logger.tag)
        ax2.plot(logger.iterations, logger.train_error, label=logger.tag)
        ax3.plot(logger.iterations, logger.loss, label=logger.tag)

    # show the plot
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()


def accuracy(a: np.array, b: np.array, x: np.array):
    predict = np.sign(a.dot(x))
    correct = np.sum(b == predict)
    return correct / len(b)


def error(a: np.array, b: np.array, x: np.array):
    return 1 - accuracy(a, b, x)


def hinge_loss(a: np.array, b: np.array, x: np.array, alpha: float):
    """

    Args:
        a: (n, d) input vectors
        b: (n,) labels
        x: (d,) weights of the classifier
        alpha: L2 regularization parameter

    Returns:
        Average empirical hinge loss over the n samples

    """
    n, d = a.shape
    assert (n,) == b.shape and (d,) == x.shape

    margin = 1 - b * a.dot(x)  # (n,)
    return np.max(margin, 0).mean() + 0.5 * alpha * x.dot(x)


def hinge_loss_grad(a: np.array, b: np.array, x: np.array, alpha: float):
    """

    Args:
        a: (n, d) input vectors
        b: (n,) labels
        x: (d,) weights of the classifier
        alpha: L2 regularization parameter

    Returns:
        Gradient of the empirical hinge loss over the n samples

    """
    n, d = a.shape
    assert (n,) == b.shape and (d,) == x.shape

    mask = b * a.dot(x) < 1  # (n,)
    # sum of the hinge loss gradients of given samples: grad_l_i = (b_i * x.dot(a_i) < 1) * b_i * a_i
    sum_grad_l_i = (mask * b).dot(a)  # (d,)
    grad = -(1 / n) * sum_grad_l_i + alpha * x  # (d,)
    return grad
