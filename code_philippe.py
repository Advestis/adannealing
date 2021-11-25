# Useful refeences, number refer to my note
# calculate temperature for current epoch
# [1] http: // what - when - how.com / artificial - intelligence / a - comparison - of - cooling - schedules -
# for -simulated - annealing - artificial - intelligence /
# [2] http://www.scielo.org.mx/pdf/cys/v21n3/1405-5546-cys-21-03-00493.pdf
# [3] https://www.researchgate.net/publication/227061666_Computing_the_Initial_Temperature_of_Simulated_Annealing/link/
# 543f88a20cf2e76f02246e49/download
# [4] https://nathanrooy.github.io/posts/2020-05-14/simulated-annealing-with-python/

import adutils
adutils.init("logger")
from adutils import setup_logger
setup_logger()
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import argparse
import matplotlib.pyplot as plt

from adannealing import Annealer, plot

logger = logging.getLogger(__name__)

all_prices = pd.read_parquet("PRICEACTION_adj_eur_isin_bloom.parquet")

# parameters run ++++++++++++++++++++++++++++++++++++++++++++++++
# Assumptions in the code:
#   - common fees
#   - weights_t-1 = equi
path_save_images = "profiler"
date = "2021-03-03"
common_fee = 0.0
overall_risk_coeff = 1e2
n_iterations = int(1e5)
step_size = 1e-2
# alpha in geometrical cooling
alpha = 0.999

date_start = "2021-01-01"
date_end = "2021-10-01"
# parameters run ++++++++++++++++++++++++++++++++++++++++++++++++

all_prices = all_prices[all_prices.index >= pd.Timestamp(date_start)]
all_prices = all_prices[all_prices.index <= pd.Timestamp(date_end)]


def loss_portfolio_score_based():
    pass


def loss_portfolio_mean_var(
    wt_np: np.array,
    wt_1_np: np.array,
    r_np: np.array,
    risk_coeff: float,
    eps_np: np.array,
    cov_np: np.array,
    n: int,
    by_component: bool = False,
) -> float:

    common_shape = (n, 1)
    assert wt_np.shape == common_shape
    assert wt_1_np.shape == common_shape
    assert r_np.shape == common_shape
    assert eps_np.shape == common_shape
    assert cov_np.shape == (n, n)

    return_term = r_np.T.dot(wt_np)
    risk_term = 0.5 * wt_np.T.dot(cov_np.dot(wt_np))
    fees_term = np.abs(wt_np - wt_1_np).T.dot(eps_np)

    if by_component:
        logger.info(f" [LOSS] return term : {return_term}")
        logger.info(f" [LOSS] risk term : {risk_term}")
        logger.info(f" [LOSS] fees term : {fees_term}")

    loss = -return_term + risk_term * risk_coeff + fees_term

    return loss[0][0]


def box(x, point_low, point_high, sharpness, height, speed):

    if point_low >= point_high:
        raise RuntimeError("Box functon: lower boundary is larger-equal to the upper boundary.")

    l_ = -1.0
    r = +1.0
    val = wall(l_, x, point_low, sharpness, height, speed)
    val = val + wall(r, x, point_high, sharpness, height, speed)

    return val


def wall(lr, x, point, sharpness, height, speed):
    val = (lr * speed * (x - point) + height) * sigmoid(lr * (x - point) * sharpness)
    return val


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def analy_optim_mean_var(r_np: np.array, risk_coeff: float, cov_np: np.array, n: int) -> np.ndarray:

    common_shape = (n, 1)
    assert r_np.shape == common_shape
    assert cov_np.shape == (n, n)

    optimum_w = np.linalg.inv(cov_np).dot(r_np) / risk_coeff

    return optimum_w


def run(number_isins, do_plot, verbose=True):

    # TODO: - implement the new loss with scores, including constraints: ery important also mathemtically as
    # TODO:   I do not have anymore the quadratic term
    # TODO: - Optimise via TF
    
    selected_prices = all_prices.dropna(how="any", axis=1)

    chosen_isins = selected_prices.columns[:number_isins]
    # selected_prices is dense
    selected_prices = selected_prices[chosen_isins]
    selected_returns = selected_prices.pct_change()

    selected_cov = selected_returns.cov()

    fees = pd.DataFrame(data=np.full(shape=(len(chosen_isins), 1), fill_value=common_fee), index=[chosen_isins])
    # test loss evaluation at some dates
    # startin equi-w
    weights_day_before = pd.DataFrame(
        data=np.full(shape=(len(chosen_isins), 1), fill_value=(1.0 / number_isins)), index=[chosen_isins]
    )
    # just a 50% variation of 2 isins
    weights_today = weights_day_before.copy()
    weights_day_before.iloc[0] = weights_day_before.iloc[0] * (1 - 0.5)
    weights_day_before.iloc[1] = weights_day_before.iloc[1] * (1 + 0.5)
    # noinspection PyArgumentList
    assert np.isclose(weights_day_before.sum()[0], 1)

    loss_date = loss_portfolio_mean_var(
        wt_np=weights_today.to_numpy(),
        wt_1_np=weights_day_before.to_numpy(),
        r_np=selected_returns.loc[date].to_numpy().reshape((number_isins, 1)),
        risk_coeff=overall_risk_coeff,
        eps_np=fees.to_numpy(),
        cov_np=selected_cov.to_numpy(),
        n=len(chosen_isins),
        by_component=False,
    )

    analy_opt = analy_optim_mean_var(
        r_np=selected_returns.loc[date].to_numpy().reshape((number_isins, 1)),
        risk_coeff=overall_risk_coeff,
        cov_np=selected_cov.to_numpy(),
        n=len(chosen_isins),
    )

    # loss value at optimum
    fees = pd.DataFrame(data=np.full(shape=(number_isins, 1), fill_value=common_fee), index=[chosen_isins])
    loss_at_min = loss_portfolio_mean_var(
        wt_np=analy_opt,
        wt_1_np=weights_day_before.to_numpy(),
        r_np=selected_returns.loc[date].to_numpy().reshape((number_isins, 1)),
        risk_coeff=overall_risk_coeff,
        eps_np=fees.to_numpy(),
        cov_np=selected_cov.to_numpy(),
        n=len(chosen_isins),
        by_component=True,
    )

    def objective(w):
        return loss_portfolio_mean_var(
            wt_np=w,
            wt_1_np=weights_day_before.to_numpy(),
            r_np=selected_returns.loc[date].to_numpy().reshape((number_isins, 1)),
            risk_coeff=overall_risk_coeff,
            eps_np=fees.to_numpy(),
            cov_np=selected_cov.to_numpy(),
            n=len(chosen_isins),
            by_component=False,
        )
    # check the function is working correctly
    assert objective(analy_opt) == loss_at_min

    # weights boundaries
    bounds_min = np.full(shape=(1, number_isins), fill_value=-1.0)
    bounds_max = np.full(shape=(1, number_isins), fill_value=+1.0)
    bounds = np.concatenate([bounds_min, bounds_max]).T

    # Using custom start temp.
    ann = Annealer(
        loss=objective,
        weights_step_size=step_size,
        bounds=bounds,
        alpha=alpha,
        iterations=n_iterations,
        verbose=verbose
    )
    numerical_solution, val_at_best, _, hist = ann.fit(
        alpha=alpha,
        stopping_limit=0.001
    )
    fig_hist = plot(hist, step_size=10)
    fig_hist.savefig(f"profiler/history_{number_isins}.pdf")
    
    error = ((analy_opt.T[0] - numerical_solution) ** 2).sum() / number_isins

    logger.info(f"date : {date}. Current loss: {loss_date}")
    logger.info(f"Numerical weights : {numerical_solution}")
    logger.info(f"Numerical loss : {val_at_best}")
    logger.info(f"Analytical optimum : {analy_opt.T[0]}")
    logger.info(f"Loss at analytical optimum : {loss_at_min}")
    logger.info(f"Error : {100 * error}%")

    if number_isins < 6 and do_plot:
        # doing a surface plot of the loss

        specs = [[None, {"rowspan": 1, "colspan": 1}] for _ in range(number_isins + 1)]
        specs[0] = [{"rowspan": number_isins + 1, "colspan": 1, "type": "surface"}, {"rowspan": 1, "colspan": 1}]

        fig = make_subplots(
            rows=number_isins + 1,
            cols=2,
            specs=specs,
            print_grid=True,
        )

        weights = np.array([w for w in hist.weights.values])

        # TODO: to change following line if you look to different params rather than 1st and 2nd
        x_explored = weights[:, 0]
        y_explored = weights[:, 1]
        z_explored = hist.loss.copy()

        wx = np.linspace(np.min(x_explored), np.max(x_explored), 100)
        wy = np.linspace(np.min(y_explored), np.max(y_explored), 100)
        # TODO: to change following line if you look to different params rather than 1st and 2nd

        def objective_2d(np_array_2):
            return objective(np.concatenate([np_array_2, analy_opt[2:]]))

        domain = pd.DataFrame(data=np.zeros((len(wx), len(wy))), index=wx, columns=wy)
        for w_x in domain.index:
            for w_y in domain.columns:
                w_x_y = np.array([[w_x], [w_y]])
                domain.loc[w_x, w_y] = objective_2d(w_x_y)

        fig.add_trace(
            go.Surface(
                z=domain.values, y=domain.index, x=domain.columns, colorscale="Blues", showscale=False, opacity=0.5
            ),
            row=1,
            col=1,
        )
        fig.update_layout(
            title="Loss Portfolio Optimisation 2 Isins",
        )

        if number_isins > 2:
            for i, (w_x, w_y) in enumerate(zip(x_explored, y_explored)):
                z_explored[i] = objective_2d(np.array([[w_x], [w_y]]))

        fig.add_scatter3d(
            # for some reason, need to transpose
            x=y_explored,
            y=x_explored,
            z=z_explored,
            mode="markers",
            marker=dict(
                size=1.2,
                color=hist.temp,
                symbol=list(map(lambda val: "x" if val else "circle", hist.accepted)),
                showscale=True,
                colorbar=dict(x=0.45),
            ),
            row=1,
            col=1,
        )
        fig.add_scatter3d(
            # for some reason, need to transpose
            x=[numerical_solution[1]],
            y=[numerical_solution[0]],
            z=[val_at_best],
            mode="markers",
            marker=dict(
                size=3,
                color="red",
                symbol="circle",
            ),
            row=1,
            col=1,
        )

        temp_accepted = hist.data.loc[hist.data.accepted.values]["temp"]
        for parameter_i in range(number_isins):
            this_param_accepted = weights[:, parameter_i][hist.data.accepted.values]

            name = "w_" + str(parameter_i)

            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(this_param_accepted)),
                    y=this_param_accepted,
                    name=name,
                    mode="lines+markers",
                    line=dict(color="black", width=0.5),
                    marker=dict(
                        size=2,
                        color=temp_accepted,
                    ),
                ),
                row=parameter_i + 1,
                col=2,
            )

            if parameter_i == 0:
                xref = "x"
                yref = "y"
            else:
                xref = "x" + str(parameter_i + 1)
                yref = "y" + str(parameter_i + 1)

            if parameter_i == number_isins:
                ref_line = val_at_best
            else:
                ref_line = analy_opt[parameter_i][0]

            fig.add_shape(
                type="line",
                x0=0,
                y0=ref_line,
                x1=len(this_param_accepted),
                y1=ref_line,
                xref=xref,
                yref=yref,
                line=dict(
                    color="black",
                    width=1,
                ),
            )

        this_param_accepted = hist.data.loss[hist.data.accepted.values]

        name = "loss"

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(this_param_accepted)),
                y=this_param_accepted,
                name=name,
                mode="lines+markers",
                line=dict(color="black", width=0.5),
                marker=dict(
                    size=2,
                    color=temp_accepted,
                ),
            ),
            row=number_isins,
            col=2,
        )

        xref = "x" + str(number_isins)
        yref = "y" + str(number_isins)

        ref_line = val_at_best

        fig.add_shape(
            type="line",
            x0=0,
            y0=ref_line,
            x1=len(this_param_accepted),
            y1=ref_line,
            xref=xref,
            yref=yref,
            line=dict(
                color="black",
                width=1,
            ),
        )

        fig.write_html(Path(path_save_images) / f"nsins_{number_isins}.html")

    return error


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AdAnnealing Profiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-n", "--nisins", type=int, default=5, help="Number of isins to use")
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Do plot if nisins <= 5"
    )
    parser.add_argument(
        "-s", "--start", type=int, default=5, help="initial number of isins to use with 'profile'"
    )
    parser.add_argument(
        "-e", "--end", type=int, default=40, help="final number of isins to use with 'profile'"
    )
    parser.add_argument(
        "-P", "--profile", action="store_true", help="Do profiling"
    )

    args = parser.parse_args()

    if args.profile:
        isins = list(range(args.start, args.end))
        errors = [run(i, False, False) for i in isins]
        plt.scatter(isins, errors, s=2)
        ax = plt.gca()
        ax.set_xlabel("# Isins")
        ax.set_ylabel("error")
        plt.gcf().savefig("profile.pdf")
    else:
        run(args.nisins, args.plot)
