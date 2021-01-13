from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_utils import load_processed_data
from projection import l1_ball_proj, l1_ball_proj_weighted
from utils import softmax, Logger, plot_results, error, hinge_loss, hinge_loss_grad

np.random.seed(10)




def train_gd(
    a: np.array, b: np.array, a_test: np.array, b_test: np.array, T: int, alpha: float
):
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # the weights of our SVM classifier
    x = np.zeros(d)

    logger = Logger(algo_tag=rf"GD - $\alpha={alpha}$")
    for t in tqdm(range(1, T + 1)):
        # log our results (before training, to match plots from the class)
        k = max(int(np.log10(t)), 0)
        if t % int(10 ** k) == 1 or t < 10:
            logger.log(
                iteration=t,
                loss=hinge_loss(a, b, x, alpha),
                train_err=error(a, b, x),
                test_err=error(a_test, b_test, x),
            )

        if alpha == 0:
            # our problem is simply convex (as the hinge loss is a convex function)
            eta_t = 1 / np.sqrt(t)
        else:
            # thanks to the regularization, our problem is alpha strongly convex
            # eta_t = 2 / (alpha * (t + 1))
            eta_t = 1 / (alpha * t)

        grad = hinge_loss_grad(a, b, x, alpha)
        x = x - eta_t * grad

    return x, logger


def train_gd_proj(
    a: np.array,
    b: np.array,
    a_test: np.array,
    b_test: np.array,
    T: int,
    alpha: float,
    radius: float,
):
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # the weights of our SVM classifier
    x = np.zeros(d)

    logger = Logger(algo_tag=rf"GDproj - $\alpha={alpha} - z={radius}$")
    for t in tqdm(range(1, T + 1)):
        # log our results (before training, to match plots from the class)
        k = max(int(np.log10(t)), 0)
        if t % int(10 ** k) == 1 or t < 10:
            logger.log(
                iteration=t,
                loss=hinge_loss(a, b, x, alpha),
                train_err=error(a, b, x),
                test_err=error(a_test, b_test, x),
            )

        if alpha == 0:
            # our problem is simply convex (as the hinge loss is a convex function)
            eta_t = 1 / np.sqrt(t)
        else:
            # thanks to the regularization, our problem is alpha strongly convex
            # eta_t = 2 / (alpha * (t + 1))
            eta_t = 1 / (alpha * t)

        grad = hinge_loss_grad(a, b, x, alpha)

        x, d_0, theta = l1_ball_proj(x - eta_t * grad, radius)
        # TODO: we could log d_0 as well?

    return x, logger

def train_sgd(
    a: np.array,
    b: np.array,
    a_test: np.array,
    b_test: np.array,
    T: int,
    alpha: float,
):
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # x_avg is the averaged weights (online to batch conversion)
    # x is weight (online version)
    x_avg = np.zeros(d)
    x = np.zeros(d)

    logger = Logger(algo_tag=rf"SGD - $\alpha={alpha}$")

    I = np.random.randint(0, n, T)
    for t in tqdm(range(1, T + 1)):
        # pick random sample
        i = I[t-1]
        a_, b_ = a[i][np.newaxis, :], np.array([b[i]])

        # log our results (before training, to match plots from the class)
        k = max(int(np.log10(t)), 0)
        if t % int(10 ** k) == 1 or t < 10:
            logger.log(
                iteration=t,
                loss=hinge_loss(a, b, x_avg, alpha),
                train_err=error(a, b, x_avg),
                test_err=error(a_test, b_test, x_avg),
            )

        if alpha == 0:
            # our problem is convex (as the hinge loss is a convex function)
            eta_t = 1 / np.sqrt(t)
        else:
            # eta_t = 2 / (alpha * t)
            eta_t = 1 / (alpha * t)

        grad = hinge_loss_grad(a_, b_, x, alpha)

        x = x - eta_t * grad

        # averaging
        x_avg = (x_avg * (t - 1) + x) / t

    return x_avg, logger


def train_sgd_proj(
    a: np.array,
    b: np.array,
    a_test: np.array,
    b_test: np.array,
    T: int,
    alpha: float,
    radius: float,
):
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # x_avg is the averaged weights (online to batch conversion)
    # x is weight (online version)
    x_avg = np.zeros(d)
    x = np.zeros(d)

    logger = Logger(algo_tag=rf"SGDproj - $\alpha={alpha} - z={radius}$")

    I = np.random.randint(0, n, T)
    for t in tqdm(range(1, T + 1)):
        # pick random sample
        i = I[t-1]
        a_, b_ = a[i][np.newaxis, :], np.array([b[i]])

        # log our results (before training, to match plots from the class)
        k = max(int(np.log10(t)), 0)
        if t % int(10 ** k) == 1 or t < 10:
            logger.log(
                iteration=t,
                loss=hinge_loss(a, b, x_avg, alpha),
                train_err=error(a, b, x_avg),
                test_err=error(a_test, b_test, x_avg),
            )

        if alpha == 0:
            # our problem is convex (as the hinge loss is a convex function)
            eta_t = 1 / np.sqrt(t)
        else:
            # eta_t = 2 / (alpha * t)
            eta_t = 1 / (alpha * t)

        grad = hinge_loss_grad(a_, b_, x, alpha)

        x = x - eta_t * grad
        x, d_0, theta = l1_ball_proj(x, radius)

        # averaging
        x_avg = (x_avg * (t - 1) + x) / t

    return x_avg, logger


def train_smd(
    a: np.array, b: np.array, a_test: np.array, b_test: np.array, T: int, radius: float
):
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # the weights of our SVM classifier
    # x is the averaged weights (online to batch conversion)
    x_avg = np.zeros(d)
    x = np.zeros(d)
    # y is weight (online version)
    y = np.zeros(d)

    logger = Logger(algo_tag=rf"SMD Proj - $z={radius}$")
    I = np.random.randint(0, n, T)
    for t in tqdm(range(1, T + 1)):
        # pick random sample
        i = I[t-1]
        a_, b_ = a[i][np.newaxis, :], np.array([b[i]])

        # log our results (before training, to match plots from the class)
        k = max(int(np.log10(t)), 0)
        if t % int(10 ** k) == 1 or t < 10:
            logger.log(
                iteration=t,
                loss=hinge_loss(a, b, x_avg, 0),
                train_err=error(a, b, x_avg),
                test_err=error(a_test, b_test, x_avg),
            )

        eta_t = 1 / np.sqrt(t)

        grad = hinge_loss_grad(a_, b_, x, 0)

        y = y - eta_t * grad
        x, d_0, theta = l1_ball_proj(y, radius)

        # averaging
        x_avg = (x_avg * (t - 1) + x) / t

    return x, logger


def train_seg_pm(
    a: np.array, b: np.array, a_test: np.array, b_test: np.array, T: int, radius: float
):
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # the weights of our SVM classifier
    # x is the averaged weights (online to batch conversion)
    x_avg = np.zeros(d)
    x = np.zeros(d)
    theta = np.zeros(2 * d)
    w = np.zeros(2 * d)

    logger = Logger(algo_tag=rf"Seg +- proj - $z={radius}$")
    I = np.random.randint(0, n, T)
    for t in tqdm(range(1, T + 1)):
        # pick random sample
        i = I[t-1]
        a_, b_ = a[i][np.newaxis, :], np.array([b[i]])

        # log our results (before training, to match plots from the class)
        k = max(int(np.log10(t)), 0)
        if t % int(10 ** k) == 1 or t < 10:
            logger.log(
                iteration=t,
                loss=hinge_loss(a, b, x_avg, 0),
                train_err=error(a, b, x_avg),
                test_err=error(a_test, b_test, x_avg),
            )

        eta_t = 1 / np.sqrt(t)

        grad = hinge_loss_grad(a_, b_, x, 0)

        theta[:d] = theta[:d] - eta_t * grad
        theta[d:] = theta[d:] + eta_t * grad

        w = softmax(theta)

        x = radius * (w[:d] - w[d:])

        # averaging
        x_avg = (x_avg * (t - 1) + x) / t

    return x, logger


def train_adagrad(
    a: np.array, b: np.array, a_test: np.array, b_test: np.array, T: int, radius: float
):
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # the weights of our SVM classifier
    # x is the averaged weights (online to batch conversion)
    x_avg = np.zeros(d)
    x = np.zeros(d)
    y = np.zeros(d)
    DELTA = 1e-5
    S = np.ones(d) * DELTA

    logger = Logger(algo_tag=rf"Adagrad - $z={radius}$")
    I = np.random.randint(0, n, T)
    for t in tqdm(range(1, T + 1)):
        # pick random sample
        i = I[t-1]
        a_, b_ = a[i][np.newaxis, :], np.array([b[i]])

        # log our results (before training, to match plots from the class)
        k = max(int(np.log10(t)), 0)
        if t % int(10 ** k) == 1 or t < 10:
            logger.log(
                iteration=t,
                loss=hinge_loss(a, b, x_avg, 0),
                train_err=error(a, b, x_avg),
                test_err=error(a_test, b_test, x_avg),
            )

        grad = hinge_loss_grad(a_, b_, x, 0)
        S += grad ** 2

        D = np.diag(np.sqrt(S))
        D_inv = np.diag(1 / np.sqrt(S))

        y = x - D_inv.dot(grad)
        x, d_0, theta = l1_ball_proj_weighted(y, radius, np.diag(D))

        # averaging
        x_avg = (x_avg * (t - 1) + x) / t

    return x, logger


def train_all():
    dir_data = Path(__file__).resolve().parents[1].joinpath("data/")
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []
    # for alpha in [0.01, 0.1, 0.5, 1.0]:
    alpha = 0.1
    _, logger = train_gd(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=1000,
        #radius=100,
        alpha=alpha
    )
    results.append(logger)
    plot_results(results)
    quit()
    results.append(logger)
    _, logger = train_sgd_proj(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=100,
        radius=100,
        alpha=alpha
    )
    results.append(logger)
    plot_results(results)
    quit()
    results.append(logger)
    _, logger = train_seg_pm(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=1000,
        radius=50,
    )
    _, logger = train_smd(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=1000,
        radius=100,
    )
    results.append(logger)
    _, logger = train_smd(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=1000,
        radius=50,
    )
    results.append(logger)
    plot_results(results)


if __name__ == "__main__":
    train_all()
