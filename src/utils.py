from datetime import datetime
from typing import List, Union

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
        self.eta_t = []

    def log(self, iteration: int, loss: float, train_err: float, test_err: float, eta_t=-1.):
        self.iterations.append(iteration)
        self.loss.append(loss)
        self.train_error.append(train_err)
        self.test_error.append(test_err)
        if eta_t > 0.:
            self.eta_t.append(eta_t)


class AvgLogger:
    def __init__(self, loggers: List[Logger]):
        # check that all loggers have the same number of iterations
        assert max([len(log.iterations) for log in loggers]) == min([len(log.iterations) for log in loggers])

        self.tag = loggers[0].tag
        self.iterations = loggers[0].iterations
        self.eta_t = np.array([log.eta_t for log in loggers]).T.mean(axis=1)

        self.loss = np.array([log.loss for log in loggers]).T.mean(axis=1)
        self.train_error = np.array([log.train_error for log in loggers]).T.mean(axis=1)
        self.test_error = np.array([log.test_error for log in loggers]).T.mean(axis=1)

        self.train_error_std = np.array([log.train_error for log in loggers]).T.std(axis=1)
        self.test_error_std = np.array([log.test_error for log in loggers]).T.std(axis=1)

        self.train_error_p95 = np.percentile(np.array([log.train_error for log in loggers]).T, 95, axis=1)
        self.test_error_p95 = np.percentile(np.array([log.test_error for log in loggers]).T, 95, axis=1)
        self.train_error_p5 = np.percentile(np.array([log.train_error for log in loggers]).T, 5, axis=1)
        self.test_error_p5 = np.percentile(np.array([log.test_error for log in loggers]).T, 5, axis=1)


def plot_results(loggers: List[Union[Logger, AvgLogger]], add_to_title=''):
    # Create plots and set axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))
    ax1.set_title(rf"test error{add_to_title}")
    ax2.set_title(rf"train error{add_to_title}")
    # ax3.set_title("training loss")
    ax3.set_title(rf"step size $\eta_t${add_to_title}")
    for ax in [ax1, ax2, ax3]:
        # log scale for error plots
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # log results from our algorithms
    for logger in loggers:
        plot_std = isinstance(logger, AvgLogger)
        x = logger.iterations
        # test error
        ax1.plot(x, logger.test_error, label=logger.tag, marker='x')
        if plot_std:
            ax1.fill_between(x, logger.test_error_p5, logger.test_error_p95, alpha=0.33)
        # train error
        ax2.plot(x, logger.train_error, label=logger.tag, marker='x')
        if plot_std:
            ax2.fill_between(x, logger.train_error_p5, logger.train_error_p95, alpha=0.33)
        # step size
        if len(logger.eta_t) > 0:
            ax3.plot(x, logger.eta_t, label=logger.tag, marker='x')

    # show the plot
    for ax in [ax1, ax2, ax3]:
        ax.legend()
    plt.savefig(f'../figures/{datetime.now().strftime("%d_%H%M%S")}.png', dpi=300)
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
