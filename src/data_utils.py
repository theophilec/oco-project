from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_raw_data(dir_data: Path, print_descriptive_stats: bool):
    """
    Load MNIST dataset from CSV and return as np.array.

    Args:
        dir_data:
        print_descriptive_stats:

    Returns:
        (x_train, y_train, x_test, y_test), the normalized (to (0, 1)) train/test MNIST data

    """
    print("loading data...")
    df_train = pd.read_csv(dir_data.joinpath("raw/mnist_train.csv"), header=None)
    df_test = pd.read_csv(dir_data.joinpath("raw/mnist_test.csv"), header=None)
    x_train = df_train.values[:, 1:]
    y_train = df_train.values[:, 0]
    x_test = df_test.values[:, 1:]
    y_test = df_test.values[:, 0]
    print(
        f"data loaded: train data {x_train.shape, y_train.shape}, test data {x_test.shape, y_test.shape}"
    )

    # normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # we are only interested in predicting the digit 0
    y_train = 2 * (y_train == 0) - 1
    y_test = 2 * (y_test == 0) - 1

    if print_descriptive_stats:
        _, counts_train = np.unique(y_train, return_counts=True)
        _, counts_test = np.unique(y_test, return_counts=True)
        freq_0_train = counts_train[1] / counts_train.sum()
        freq_0_test = counts_test[1] / counts_test.sum()
        print(
            f"Frequency of digit 0\n"
            f"in train set : {counts_train[1]}/{counts_train.sum()} = {freq_0_train:.3f}\n"
            f"in test set : {counts_test[1]}/{counts_test.sum()} = {freq_0_test:.3f}"
        )
        mnist_sparsity_analysis(a=x_train)

    return x_train, y_train, x_test, y_test


def load_processed_data(dir_data: Path):
    """
    Useful to load the (normalized) MNIST train/test data faster, if it has already been processed.

    Args:
        dir_data:

    Returns:
        (x_train, y_train, x_test, y_test)

    """
    print("loading processed data...")
    x_train = np.load(str(dir_data.joinpath("processed/x_train.npy")))
    y_train = np.load(str(dir_data.joinpath("processed/y_train.npy")))
    x_test = np.load(str(dir_data.joinpath("processed/x_test.npy")))
    y_test = np.load(str(dir_data.joinpath("processed/y_test.npy")))
    print("done !")
    return x_train, y_train, x_test, y_test


def save_data(dir_data: Path, x_train, y_train, x_test, y_test):
    """
    Save normalized data (train/test inputs and labels) to numpy arrays to the disk.
    The .npy files are larger than the original .csv data, but they can be loaded faster.

    Args:
        dir_data:
        x_train:
        y_train:
        x_test:
        y_test:

    Returns:

    """
    np.save(str(dir_data.joinpath("processed/x_train")), x_train)
    np.save(str(dir_data.joinpath("processed/x_test")), x_test)
    np.save(str(dir_data.joinpath("processed/y_train")), y_train)
    np.save(str(dir_data.joinpath("processed/y_test")), y_test)


def mnist_sparsity_analysis(a: np.array):
    """

    Args:
        a: Input data of shape (N, d) with d=784

    Returns:

    """
    n, d = a.shape
    # for each example, number of non-zero features
    features_count = np.sum(a > 0, axis=1)
    omega = np.max(features_count)

    # for each feature, frequency of non-zero in the training data
    feature_freq = np.sum(a > 0, axis=0) / n
    delta = np.max(feature_freq)

    ex_with_features_in_common = []
    for a_i in a:
        e = a_i > 0  # (d,)
        e_hat = a > 0  # (N, d)
        # for example a_i, number of examples which have at least one feature in common
        f_common = np.logical_and(e, e_hat).any(axis=1).sum()
        ex_with_features_in_common.append(f_common)
        if f_common == n:  # max number
            break
    rho = max(ex_with_features_in_common) / n

    print(f"n={n}, d={d}, omega = {omega}, delta={delta}, rho={rho}")

    plt.imshow(feature_freq.reshape((28, 28)))
    plt.colorbar()
    plt.show()


def main():
    dir_data = Path(__file__).resolve().parents[1].joinpath("data/")
    x_train, y_train, x_test, y_test = load_raw_data(
        dir_data, print_descriptive_stats=True
    )
    save_data(dir_data, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == "__main__":
    main()
