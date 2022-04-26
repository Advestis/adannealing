# Useful refeences, number refer to my note
# calculate temperature for current epoch
# [1] http: // what - when - how.com / artificial - intelligence / a - comparison - of - cooling - schedules -
# for -simulated - annealing - artificial - intelligence /
# [2] http://www.scielo.org.mx/pdf/cys/v21n3/1405-5546-cys-21-03-00493.pdf
# [3] https://www.researchgate.net/publication/227061666_Computing_the_Initial_Temperature_of_Simulated_Annealing/link/
# 543f88a20cf2e76f02246e49/download
# [4] https://nathanrooy.github.io/posts/2020-05-14/simulated-annealing-with-python/

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from adutils import setup_logger
import numpy as np

setup_logger()
from pathlib import Path
import logging
import argparse
from adlearn.engine import Engine
from time import time
import matplotlib.pyplot as plt
import pandas as pd


engine = Engine(kind="multiproc")

from adannealing import Annealer
from profiling.financial import (
    load_financial_configurations,
    LossPortfolioMeanVar,
)

Annealer.set_cpu_limit(1)

logger = logging.getLogger(__name__)

(
    path_save_images,
    date,
    common_fee,
    overall_risk_coeff,
    overall_sparse_coeff,
    overall_norm_coeff,
    sparsity,
    desired_norm,
    continous_window,
    n_iterations,
    step_size,
    alpha,
    all_prices,
) = load_financial_configurations("profiling/run_configs.json")

def run(number_isins, verbose=True):

    logger.info("")
    logger.info(f"Starting annealing profiler with {number_isins} isins...")
    limits = tuple([(-1, 1) for _ in range(number_isins)])

    selected_prices = all_prices.dropna(how="any", axis=1)

    chosen_isins = selected_prices.columns[:number_isins]
    # selected_prices is dense
    selected_prices = selected_prices[chosen_isins]
    selected_returns = selected_prices.pct_change()

    selected_cov = selected_returns.cov()

    # test loss evaluation at some dates
    # startin equi-w
    weights_day_before = pd.DataFrame(
        data=np.full(shape=(len(chosen_isins), 1), fill_value=(1.0 / number_isins)), index=[chosen_isins]
    )

    fees = pd.DataFrame(data=np.full(shape=(number_isins, 1), fill_value=common_fee), index=[chosen_isins])
    portfolio_opt_constraints = LossPortfolioMeanVar(
        wt_1_np=weights_day_before.to_numpy(),
        r_np=selected_returns.loc[date].to_numpy().reshape((number_isins, 1)),
        lambda_risk=overall_risk_coeff,
        lambda_sparse=overall_sparse_coeff,
        lambda_norm=overall_norm_coeff,
        fees=fees,
        cov_risk=selected_cov.to_numpy(),
        sparsity_target=sparsity,
        constraints=limits,
        sum_w_target=desired_norm,
        continous_window=continous_window,
        n=len(chosen_isins),
    )

    bounds_min = np.full(shape=(1, number_isins), fill_value=-1.0)
    bounds_max = np.full(shape=(1, number_isins), fill_value=+1.0)
    bounds = np.concatenate([bounds_min, bounds_max]).T

    # Using custom start temp.
    t0 = time()
    hpath = Path(path_save_images) / f"history_{number_isins}"
    if not hpath.is_dir():
        hpath.mkdir()
    ann = Annealer(
        loss=portfolio_opt_constraints,
        weights_step_size=step_size,
        bounds=bounds,
        alpha=alpha,
        iterations=n_iterations,
        verbose=verbose,
        history_path=str(hpath),
        logger_level="INFO",
        # TODO: test more this experimental feature
        optimal_step_size=True,  # experimental
    )
    numerical_solution, val_at_best, _, hist, final_hist, _ = ann.fit(
        alpha=alpha, stopping_limit=0.001, npoints=1, stop_at_first_found=True
    )
    tf = time() - t0
    ann.plot(hpath, step_size=10, weights_names=chosen_isins, do_3d=True)

    logger.info(f"date : {date}")
    logger.info(f"Numerical loss : {val_at_best}")
    logger.info(f"Annealing time: {tf} s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AdAnnealing Profiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-n", "--nisins", type=int, default=5, help="Number of isins to use")
    parser.add_argument("-p", "--plot", action="store_true", help="Do plot if nisins <= 5")
    parser.add_argument("-s", "--start", type=int, default=5, help="Initial number of isins to use with 'profile'")
    parser.add_argument("-S", "--step", type=int, default=1, help="Steps in number of isins to use with 'profile'")
    parser.add_argument("-e", "--end", type=int, default=40, help="Final number of isins to use with 'profile'")
    parser.add_argument("-P", "--profile", action="store_true", help="Do profiling")
    parser.add_argument("-m", "--multiproc", action="store_true", help="Do profiling in parallel")

    args = parser.parse_args()

    if args.profile:
        if args.end != -1:
            isins = list(range(args.start, args.end + 1, args.step))
        else:
            isins = list(range(args.start, len(all_prices.columns), args.step))
        if args.multiproc:
            errors_norms_times = engine(run, isins, do_plot=False, verbose=False)
        else:
            errors_norms_times = [run(i, False) for i in isins]
        isins, errors, times = zip(*errors_norms_times)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        axes[1].set_xlabel("# Isins", fontsize=15)
        axes[0].set_ylabel("Errors (%)", fontsize=15)
        axes[1].set_ylabel("Annealing time (s)", fontsize=15)
        axes[0].grid(True, ls="--", lw=0.2, alpha=0.5)
        axes[1].grid(True, ls="--", lw=0.2, alpha=0.5)
        axes[0].scatter(isins, errors)
        axes[1].scatter(isins, times)
        fig.savefig(str(Path(path_save_images) / f"profile_{args.start}_{args.end}_{args.step}.pdf"))
    else:
        run(args.nisins, args.plot)
