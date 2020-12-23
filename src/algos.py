from pathlib import Path
from typing import List
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from data_utils import load_processed_data
from projection import l1_ball_proj


np.random.seed(0)

class Logger:
    def __init__(self, algo_tag: str):
        self.tag = algo_tag
        self.loss = []
        self.train_error = []
        self.test_error = []

    def log(self, loss: float, train_err: float, test_err: float):
        self.loss.append(loss)
        self.train_error.append(train_err)
        self.test_error.append(test_err)


def plot_results(loggers: List[Logger]):
    # Create plots and set axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))
    ax1.set_title('test error')
    ax2.set_title('train error')
    ax3.set_title('training loss')
    for ax in [ax1, ax2]:
        # log scale for error plots
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # log results from our algorithms
    for logger in loggers:
        T = len(logger.test_error)
        ax1.plot(np.arange(1, T + 1), logger.test_error, label=logger.tag)
        ax2.plot(np.arange(1, T + 1), logger.train_error, label=logger.tag)
        ax3.plot(np.arange(1, T + 1), logger.loss, label=logger.tag)

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
    grad = - (1 / n) * sum_grad_l_i + alpha * x  # (d,)
    return grad


def train_gradient_descent(a: np.array, b: np.array, a_test: np.array, b_test: np.array, T: int, alpha: float):
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # the weights of our SVM classifier
    x = np.zeros(d)

    logger = Logger(algo_tag=rf'$GD - \alpha={alpha}$')
    for t in range(1, T + 1):
        # log our results (before training, to match plots from the class)
        logger.log(loss=hinge_loss(a, b, x, alpha),
                   train_err=error(a, b, x),
                   test_err=error(a_test, b_test, x))

        if alpha == 0:
            # our problem is simply convex (as the hinge loss is a convex function)
            eta_t = 1 / np.sqrt(t)
        else:
            # thanks to the regularization, our problem is alpha strongly convex
            # eta_t = 2 / (alpha * (t + 1))
            eta_t = 1 / t

        grad = hinge_loss_grad(a, b, x, alpha)
        x = x - eta_t * grad

    return x, logger

def train_proj_gradient_descent(a: np.array, b: np.array, a_test: np.array, b_test: np.array, T: int, alpha: float, radius: float):
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # the weights of our SVM classifier
    x = np.zeros(d)

    logger = Logger(algo_tag=rf'$PGD - \alpha={alpha} - z={radius}$')
    for t in range(1, T + 1):
        # log our results (before training, to match plots from the class)
        logger.log(loss=hinge_loss(a, b, x, alpha),
                   train_err=error(a, b, x),
                   test_err=error(a_test, b_test, x))

        if alpha == 0:
            # our problem is simply convex (as the hinge loss is a convex function)
            eta_t = 1 / np.sqrt(t)
        else:
            # thanks to the regularization, our problem is alpha strongly convex
            # eta_t = 2 / (alpha * (t + 1))
            eta_t = 1 / t

        grad = hinge_loss_grad(a, b, x, alpha)


        x, d_0, theta = l1_ball_proj(x - eta_t * grad, radius)
        # TODO: we could log d_0 as well?


    return x, logger


def train_proj_online_gradient_descent(a: np.array, b: np.array, a_test: np.array, b_test: np.array, T: int, alpha: float, radius: float):
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # the weights of our SVM classifier
    # x is the averaged weights (online to batch conversion)
    x = np.zeros(d)
    # y is weight (online version)
    y = np.zeros(d)

    logger = Logger(algo_tag=rf'P-OGD - $\alpha={alpha} - z={radius}$')
    for t in tqdm(range(1, T + 1)):
        # pick random sample
        i = np.random.randint(n)
        a_, b_ = a[i][np.newaxis, :], np.array([b[i]])

        # log our results (before training, to match plots from the class)
        logger.log(loss=hinge_loss(a, b, x, alpha),
                   train_err=error(a, b, x),
                   test_err=error(a_test, b_test, x))

        if alpha == 0:
            # our problem is simply convex (as the hinge loss is a convex function)
            eta_t = 1 / np.sqrt(t)
        else:
            # thanks to the regularization, our problem is alpha strongly convex
            # eta_t = 2 / (alpha * t)
            eta_t = 1 / t

        grad = hinge_loss_grad(a_, b_, y, alpha)

        y = y - eta_t * grad
        y, d_0, theta = l1_ball_proj(y, radius)
        # TODO: we could log d_0 as well?

        # averaging
        x = (x * (t - 1) + y) / t


    return x, logger


def train_all():
    dir_data = Path(__file__).resolve().parents[1].joinpath('data/')
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    # projected gradient descent
    x, logger_p_ogd = train_proj_online_gradient_descent(
        a=x_train, b=y_train, a_test=x_test, b_test=y_test, T=1000, alpha=0.33, radius=20.
    )
    # projected gradient descent
    x, logger_p_gd = train_proj_gradient_descent(
        a=x_train, b=y_train, a_test=x_test, b_test=y_test, T=100, alpha=0.33, radius=20.
    )
    # gradient descent
    x, logger_gd = train_gradient_descent(
        a=x_train, b=y_train, a_test=x_test, b_test=y_test, T=100, alpha=0.33
    )

    # plot results
    plot_results([logger_p_ogd, logger_gd, logger_p_gd])


if __name__ == '__main__':
    train_all()
