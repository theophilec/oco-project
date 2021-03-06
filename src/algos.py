import os
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns
import random

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from data_utils import load_processed_data
from projection import l1_ball_proj, l1_ball_proj_weighted
from utils import (
    AvgLogger,
    Logger,
    error,
    hinge_loss,
    hinge_loss_grad,
    hinge_loss_grad_partial,
    plot_results_,
    softmax, plot_results,
)

root_dir = Path(__file__).resolve().parents[1]
dir_data = root_dir.joinpath('data')
try:
    os.mkdir(root_dir.joinpath('src/joblib_memmap'))  # to store shared parameters (hogwild)
    os.mkdir(root_dir.joinpath('figures'))  # to save figures
except FileExistsError:
    pass


# ----- OCO algorithms from the lectures

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
        return_avg=True,
        seed=0,
        use_logger=True,
):
    np.random.seed(seed)
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # x_avg is the averaged weights (online to batch conversion)
    # x is weight (online version)
    x_avg = np.zeros(d)
    x = np.zeros(d)

    if use_logger:
        logger = Logger(algo_tag=rf"SGD - {'x_avg' if return_avg else 'x_T'}")

    t0 = perf_counter_ns()

    I = np.random.randint(0, n, T)
    for t in tqdm(range(1, T + 1)):
        # pick random sample
        i = I[t - 1]
        a_, b_ = a[i][np.newaxis, :], np.array([b[i]])

        if alpha == 0:
            # our problem is convex (as the hinge loss is a convex function)
            eta_t = 1 / np.sqrt(t)
        else:
            eta_t = 1 / (alpha * t)

        # log our results (before training, to match plots from the class)
        k = max(int(np.log10(t)), 0)
        if (t % int(10 ** k) == 1 or t < 10) and use_logger:
            logger.log(
                iteration=t,
                loss=hinge_loss(a, b, x_avg, alpha),
                train_err=error(a, b, x_avg),
                test_err=error(a_test, b_test, x_avg),
                eta_t=eta_t,
                time_elapsed=(perf_counter_ns() - t0) / 1e9,
            )

        grad = hinge_loss_grad(a_, b_, x, alpha)

        x = x - eta_t * grad

        # averaging
        if return_avg:
            x_avg = (x_avg * (t - 1) + x) / t
        else:
            x_avg = x

    dt = (perf_counter_ns() - t0) / 1e9  # execution time in sec
    if use_logger:
        return x_avg, logger
    else:
        return dt, T, error(a_test, b_test, x)


def train_sgd_proj(
        a: np.array,
        b: np.array,
        a_test: np.array,
        b_test: np.array,
        T: int,
        alpha: float,
        radius: float,
        seed=0,
):
    np.random.seed(seed)
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
        i = I[t - 1]
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
        a: np.array,
        b: np.array,
        a_test: np.array,
        b_test: np.array,
        T: int,
        radius: float,
        seed=0,
):
    np.random.seed(seed)
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
        i = I[t - 1]
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
        a: np.array,
        b: np.array,
        a_test: np.array,
        b_test: np.array,
        T: int,
        radius: float,
        seed=0,
):
    np.random.seed(seed)
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
        i = I[t - 1]
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


def train_sreg_pm(
        a: np.array,
        b: np.array,
        a_test: np.array,
        b_test: np.array,
        T: int,
        radius: float,
        seed=0,
):
    np.random.seed(seed)
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    x_avg = np.zeros(d)
    x = np.zeros(d)
    w = np.ones(2 * d) / (2 * d)
    directions = np.arange(d)
    I = np.random.randint(0, n, T)
    logger = Logger(algo_tag=rf"SREG $\pm$ proj - $z={radius}$")
    for t in tqdm(range(1, T + 1)):
        direction = random.choices(directions, weights=0.5 * (w[:d] + w[d:]))[0]
        # pick random sample
        i = I[t - 1]
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

        grad = hinge_loss_grad_partial(a_, b_, x, 0, direction)

        if grad > 0:
            if abs(w[direction]) > 1e-9:
                w[direction] = np.exp(-eta_t * grad / w[direction]) * w[direction]
        else:
            if abs(w[direction + d]) > 1e-9:
                w[direction + d] = (
                        np.exp(eta_t * grad / w[direction + d]) * w[direction + d]
                )

        w = w / np.sum(w)

        x = radius * (w[:d] - w[d:])

        # averaging
        x_avg = (x_avg * (t - 1) + x) / t

    return x, logger


def train_adagrad(
        a: np.array,
        b: np.array,
        a_test: np.array,
        b_test: np.array,
        T: int,
        radius: float,
        seed=0,
):
    np.random.seed(seed)
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
        i = I[t - 1]
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


def train_ons(
        a: np.array,
        b: np.array,
        a_test: np.array,
        b_test: np.array,
        T: int,
        gamma: float,
        alpha: float,
        radius: float,
        seed=0,
):
    np.random.seed(seed)
    # add a column of ones to the input data, to avoid having to define an explicit bias in our weights
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # the weights of our SVM classifier
    # x is the averaged weights (online to batch conversion)
    x_avg = np.zeros(d)
    x = np.zeros(d)
    y = np.zeros(d)
    A = 1 / gamma ** 2 * np.eye(d)
    A_inv = gamma ** 2 * np.eye(d)

    logger = Logger(
        algo_tag=rf"ONS - $\alpha = {alpha} - \gamma = {gamma} - z={radius}$"
    )
    I = np.random.randint(0, n, T)
    for t in tqdm(range(1, T + 1)):
        # pick random sample
        i = I[t - 1]
        a_, b_ = a[i][np.newaxis, :], np.array([b[i]])

        # log our results (before training, to match plots from the class)
        k = max(int(np.log10(t)), 0)
        if t % int(10 ** k) == 0 or t < 10:
            logger.log(
                iteration=t,
                loss=hinge_loss(a, b, x_avg, alpha),
                train_err=error(a, b, x_avg),
                test_err=error(a_test, b_test, x_avg),
            )

        grad = hinge_loss_grad(a_, b_, x, alpha)
        gg = np.outer(grad, grad)
        assert gg.shape == (d, d)
        A += gg
        num = A_inv.dot(gg).dot(A_inv)
        denum = 1 + grad.dot(A_inv).dot(grad)
        A_inv -= num / denum

        y = x - 1 / gamma * A_inv.dot(grad)
        x, d_0, theta = l1_ball_proj_weighted(y, radius, np.diag(A))

        # averaging
        x_avg = (x_avg * (t - 1) + x) / t

    return x, logger


# ----- Hogwild algorithm

def train_epoch_hogwild(x, a, b, I_p, eta, alpha):
    for i in I_p:
        a_i, b_i = a[i][np.newaxis, :], np.array([b[i]])
        grad = hinge_loss_grad(a_i, b_i, x, alpha)
        for index in np.where(abs(grad) > 0.01)[0]:
            x[index] -= eta * grad[index]
    return x


def train_hogwild(
        a: np.array,
        b: np.array,
        a_test: np.array,
        b_test: np.array,
        T: int,
        alpha: float,
        K: int,
        beta: float,
        theta: float,
        n_processes: int,
        sequential: bool,
        seed: int,
        use_logger=True,
):
    np.random.seed(seed)
    a = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    a_test = np.concatenate([a_test, np.ones((len(a_test), 1))], axis=1)
    n, d = a.shape

    # create x using a shared memory, so that all processes can write to it
    x_memmap = os.path.join(folder, f'x_{datetime.now().strftime("%H%M%S")}')
    x = np.memmap(x_memmap, dtype=a.dtype, shape=d, mode="w+")

    print(f"Training hogwild with seed {seed}")
    if use_logger:
        logger = Logger(
            algo_tag=rf"Hogwild {'seq' if sequential else f'n_jobs={n_processes}'}- $K={K}$"
        )

    t0 = perf_counter_ns()
    t = 1
    eta_t = theta / alpha
    while t <= T:
        if use_logger:
            logger.log(
                iteration=t,
                loss=hinge_loss(a, b, x, alpha),
                train_err=error(a, b, x),
                test_err=error(a_test, b_test, x),
                eta_t=eta_t,
                time_elapsed=(perf_counter_ns() - t0) / 1e9,
            )

        # don't do more steps than necessary
        if t + K * n_processes < T:
            steps_per_processor = K
        else:
            steps_per_processor = int((T - t) / n_processes)
        indices = [
            np.random.randint(0, n, steps_per_processor) for p in range(n_processes)
        ]
        if sequential:
            # mimic Hogwild without multiprocessing (similar to SGD)
            for I_p in indices:
                train_epoch_hogwild(x, a, b, I_p, eta_t, alpha)
        else:
            Parallel(n_jobs=n_processes, verbose=0)(
                delayed(train_epoch_hogwild)(x, a, b, I_p, eta_t, alpha)
                for I_p in indices
            )

        # increase the number of steps and decrease the learning rate
        K = int(K / beta)
        eta_t = beta * eta_t  # original learning rate from the paper
        t += steps_per_processor

    dt = (perf_counter_ns() - t0) / 1e9  # execution time in sec
    if use_logger:
        logger.log(
            iteration=t,
            loss=hinge_loss(a, b, x, alpha),
            train_err=error(a, b, x),
            test_err=error(a_test, b_test, x),
            eta_t=eta_t,
            time_elapsed=dt,
        )
        return x, logger
    else:
        return dt, t, error(a_test, b_test, x)


def plot_hogwild():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    # --- 2. plot hogwild for various values of K
    n_runs = 3
    n_workers = 8
    T = 1000000
    alpha = 0.33
    beta = 0.37
    theta = 0.2
    results = [AvgLogger([
        train_hogwild(a=x_train, b=y_train, a_test=x_test, b_test=y_test, T=T, alpha=alpha, beta=beta,
                      K=K, theta=theta, n_processes=n_workers, sequential=False, seed=s)[1]
        for s in range(n_runs)
    ]) for K in [3, 10, 50]]

    # --- 1. plot comparison between SGD and hogwild, fixed K
    # n_runs = 3
    # n_workers = 8
    # T = 1000000
    # alpha = 0.33
    # beta = 0.37
    # theta = 0.2
    # results = [AvgLogger([
    #     train_hogwild(a=x_train, b=y_train, a_test=x_test, b_test=y_test, T=T, alpha=alpha, beta=beta,
    #                   K=K, theta=theta, n_processes=n_workers, sequential=False, seed=s)[1]
    #     for s in range(n_runs)
    # ]) for K in [3]]
    # results.append(AvgLogger([
    #     train_hogwild(a=x_train, b=y_train, a_test=x_test, b_test=y_test, T=T, alpha=alpha, beta=beta,
    #                   K=3, theta=theta, n_processes=n_workers, sequential=True, seed=s)[1]
    #     for s in range(n_runs)
    # ]))
    # results.append(AvgLogger([
    #     train_sgd(a=x_train, b=y_train, a_test=x_test, b_test=y_test, T=T, alpha=alpha, return_avg=True, seed=s)[1]
    #     for s in range(n_runs)
    # ]))

    plot_results(results, add_to_title=rf" ($\alpha={alpha}, \beta={beta}, \theta={theta}$, n_runs={n_runs})")


# ----- Plotting the results (to reproduce figures from the report)

def gd_gd_proj_z():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []
    for radius in [0.5, 1.0, 10, 100, 500]:
        alpha = 0.33
        # gamma = 0.1
        # radius = 100
        T = 1000
        _, logger = train_gd(
            a=x_train,
            b=y_train,
            a_test=x_test,
            b_test=y_test,
            T=T,
            # gamma = gamma,
            # radius=radius,
            alpha=alpha,
        )
        results.append(logger)
        _, logger = train_gd_proj(
            a=x_train,
            b=y_train,
            a_test=x_test,
            b_test=y_test,
            T=T,
            # gamma = gamma,
            radius=radius,
            alpha=alpha,
        )
        results.append(logger)
    plot_results_(results)


def gd_sgd():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []
    alpha = 0.1
    # gamma = 0.1
    radius = 100
    T = 1000
    _, logger = train_gd(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        alpha=alpha,
    )
    results.append(logger)
    _, logger = train_gd_proj(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        radius=radius,
        alpha=alpha,
    )
    results.append(logger)
    _, logger = train_sgd(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=10 * T,
        alpha=alpha,
    )
    results.append(logger)
    _, logger = train_sgd_proj(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=10 * T,
        radius=radius,
        alpha=alpha,
    )
    results.append(logger)

    plot_results_(results)
    quit()


def sgd_smd():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []
    alpha = 0.1
    # gamma = 0.1
    radius = 100
    T = 10000
    _, logger = train_sgd(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        alpha=alpha,
    )
    results.append(logger)
    _, logger = train_sgd_proj(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        radius=radius,
        alpha=alpha,
    )
    results.append(logger)
    _, logger = train_smd(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        radius=radius,
        # alpha=alpha,
    )
    results.append(logger)
    plot_results_(results)
    quit()


def sgd_seg():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []
    alpha = 0.1
    # gamma = 0.1
    radius = 100
    T = 10000
    _, logger = train_sgd_proj(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        radius=radius,
        alpha=alpha,
    )
    results.append(logger)
    _, logger = train_seg_pm(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        radius=radius,
        # alpha=alpha,
    )
    results.append(logger)
    plot_results_(results)
    quit()


def sgd_ons():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []
    alpha = 0.1
    gamma = 0.1
    radius = 100
    T = 10000
    _, logger = train_sgd_proj(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        radius=radius,
        alpha=alpha,
    )
    results.append(logger)
    _, logger = train_ons(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        gamma=gamma,
        radius=radius,
        alpha=alpha,
    )
    results.append(logger)
    plot_results_(results)
    quit()


def train_all():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []
    alpha = 0.1
    gamma = 0.1
    radius = 100
    T = 10000
    _, logger = train_sgd(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        alpha=alpha,
    )
    results.append(logger)
    _, logger = train_sgd_proj(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        radius=radius,
        alpha=alpha,
    )
    results.append(logger)
    _, logger = train_smd(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        radius=radius,
        # alpha=alpha,
    )
    _, logger = train_seg_pm(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        radius=radius,
        # alpha=alpha,
    )
    results.append(logger)
    _, logger = train_sreg_pm(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        # gamma=gamma,
        radius=radius,
        # alpha=alpha,
    )
    results.append(logger)
    _, logger = train_ons(
        a=x_train,
        b=y_train,
        a_test=x_test,
        b_test=y_test,
        T=T,
        gamma=gamma,
        radius=radius,
        alpha=alpha,
    )
    results.append(logger)
    plot_results_(results)


def sgd_sgdproj_var():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []

    n_runs = 10
    T = 10000
    alpha = 0.1
    results.append(
        AvgLogger(
            [
                train_sgd(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    alpha=alpha,
                    return_avg=True,
                    seed=s,
                )[1]
                for s in range(n_runs)
            ]
        )
    )
    results.append(
        AvgLogger(
            [
                train_sgd_proj(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    alpha=alpha,
                    radius=100,
                    seed=s,
                )[1]
                for s in range(n_runs)
            ]
        )
    )

    plot_results_(results, add_to_title=rf" - $\alpha={alpha}$, n_runs={n_runs}")


def sgdproj_smd_var():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []

    n_runs = 10
    T = 10000
    alpha = 0.1
    radius = 100
    results.append(
        AvgLogger(
            [
                train_smd(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    radius=radius,
                    seed=s,
                )[1]
                for s in range(n_runs)
            ]
        )
    )
    results.append(
        AvgLogger(
            [
                train_sgd_proj(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    alpha=alpha,
                    radius=100,
                    seed=s,
                )[1]
                for s in range(n_runs)
            ]
        )
    )

    plot_results_(results, add_to_title=rf" - $\alpha={alpha}$, n_runs={n_runs}")


def sgdproj_segpm_var():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []

    n_runs = 5
    T = 10000
    alpha = 0.1
    radius = 100
    results.append(
        AvgLogger(
            [
                train_seg_pm(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    radius=radius,
                    seed=s,
                )[1]
                for s in range(n_runs)
            ]
        )
    )
    results.append(
        AvgLogger(
            [
                train_sgd_proj(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    alpha=alpha,
                    radius=100,
                    seed=s,
                )[1]
                for s in range(n_runs)
            ]
        )
    )

    plot_results_(results, add_to_title=rf" - $\alpha={alpha}$, n_runs={n_runs}")


def sgd_adagrad_var():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []

    n_runs = 5
    T = 10000
    alpha = 0.1
    radius = 100
    results.append(
        AvgLogger(
            [
                train_adagrad(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    radius=radius,
                    seed=s,
                )[1]
                for s in range(n_runs)
            ]
        )
    )
    results.append(
        AvgLogger(
            [
                train_sgd_proj(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    alpha=alpha,
                    radius=100,
                    seed=s,
                )[1]
                for s in range(n_runs)
            ]
        )
    )

    plot_results_(results, add_to_title=rf" - $\alpha={alpha}$, n_runs={n_runs}")


def seg_sreg_var():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []

    n_runs = 5
    T = 100000
    alpha = 0.1
    radius = 100
    gamma = 0.1
    results.append(
        AvgLogger(
            [
                train_sreg_pm(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    radius=radius,
                    seed=s,
                )[1]
                for s in range(n_runs)
            ]
        )
    )
    results.append(
        AvgLogger(
            [
                train_sgd_proj(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    alpha=alpha,
                    radius=100,
                    seed=s,
                )[1]
                for s in range(n_runs)
            ]
        )
    )

    plot_results_(results, add_to_title=rf" - $\alpha={alpha}$, n_runs={n_runs}")


def sgd_ons_var():
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    results = []

    n_runs = 5
    T = 1000
    alpha = 0.1
    radius = 100
    gamma = 0.1
    results.append(
        AvgLogger(
            [
                train_ons(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    gamma=gamma,
                    radius=radius,
                    seed=s,
                    alpha=alpha,
                )[1]
                for s in range(n_runs)
            ]
        )
    )
    results.append(
        AvgLogger(
            [
                train_sgd_proj(
                    a=x_train,
                    b=y_train,
                    a_test=x_test,
                    b_test=y_test,
                    T=T,
                    alpha=alpha,
                    radius=100,
                    seed=s,
                )[1]
                for s in range(n_runs)
            ]
        )
    )

    plot_results_(results, add_to_title=rf" - $\alpha={alpha}$, n_runs={n_runs}")


if __name__ == "__main__":
    train_all()
