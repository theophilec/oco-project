import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from algos import train_hogwild, train_sgd
from data_utils import load_processed_data

if __name__ == "__main__":
    dir_data = Path(__file__).resolve().parents[1].joinpath("data/")
    save_folder = dir_data.joinpath("../csv_results")
    try:
        os.mkdir(save_folder)
    except FileExistsError:
        pass
    x_train, y_train, x_test, y_test = load_processed_data(dir_data)

    n_runs = 1
    T_config = 2017521
    alpha = 0.33
    beta = 0.37
    theta = 0.2
    K = 3
    results = {
        "algo": [],
        "time": [],
        "test_error": [],
        "T": [],
        "n_workers": [],
        "K": [],
    }

    for algo in ["sgd"]:
        if algo == "hogwild":
            try_workers = [1, 4, 8]
        else:
            try_workers = [8]

        for n_workers in try_workers:
            for seed in range(n_runs):
                if algo == "hogwild":
                    dt, T, test_error = train_hogwild(
                        a=x_train,
                        b=y_train,
                        a_test=x_test,
                        b_test=y_test,
                        T=T_config,
                        alpha=alpha,
                        beta=beta,
                        K=K,
                        theta=theta,
                        n_processes=n_workers,
                        sequential=False,
                        seed=seed,
                        use_logger=False,
                    )
                elif algo == "hogwild_seq":
                    dt, T, test_error = train_hogwild(
                        a=x_train,
                        b=y_train,
                        a_test=x_test,
                        b_test=y_test,
                        T=T_config,
                        alpha=alpha,
                        beta=beta,
                        K=K,
                        theta=theta,
                        n_processes=1,
                        sequential=True,
                        seed=seed,
                        use_logger=False,
                    )
                elif algo == "sgd":
                    dt, T, test_error = train_sgd(
                        a=x_train,
                        b=y_train,
                        a_test=x_test,
                        b_test=y_test,
                        T=T_config,
                        alpha=alpha,
                        return_avg=True,
                        seed=seed,
                        use_logger=False,
                    )

                results["algo"].append(algo)
                results["time"].append(dt)
                results["test_error"].append(test_error)
                results["T"].append(T)
                results["n_workers"].append(n_workers)
                results["K"].append(K)

    df = pd.DataFrame(results)
    df.to_csv(
        save_folder.joinpath(f'{datetime.now().strftime("%H%M%S")}.csv'), index=False
    )
