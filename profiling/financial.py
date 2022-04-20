import numpy as np
import logging
import json
import pandas as pd

logger = logging.getLogger(__name__)


def load_financial_configurations(path):
    with open(path) as f:
        configs = json.load(f)
    all_prices = pd.read_parquet(configs["prices_path"])

    # parameters run ++++++++++++++++++++++++++++++++++++++++++++++++
    # Assumptions in the code:
    #   - common fees
    #   - weights_t-1 = equi
    path_save_images = configs["path_save_images"]
    date = configs["date"]
    common_fee = configs["common_fee"]
    overall_risk_coeff = configs["overall_risk_coeff"]
    n_iterations = configs["n_iterations"]
    step_size = configs["step_size"]
    alpha = configs["alpha"]

    date_start = configs["date_start"]
    date_end = configs["date_end"]
    # parameters run ++++++++++++++++++++++++++++++++++++++++++++++++

    all_prices = all_prices[all_prices.index >= pd.Timestamp(date_start)]
    all_prices = all_prices[all_prices.index <= pd.Timestamp(date_end)]
    return path_save_images, date, common_fee, overall_risk_coeff, n_iterations, step_size, alpha, all_prices


def loss_portfolio_score_based():
    pass


FIRST_CALL = True
PENALTY = None


def loss_portfolio_mean_var(
    wt_np: np.array,
    wt_1_np: np.array,
    r_np: np.array,
    risk_coeff: float,
    sparse_coeff: float,
    norm_coeff: float,
    eps_np: np.array,
    cov_np: np.array,
    sparsity: float,
    limits: np.array,
    desired_norm: float,
    n: int,
    by_component: bool = False,
) -> float:

    global FIRST_CALL
    global NORMED_BOX

    if FIRST_CALL:
        common_shape = (n, 1)
        assert wt_np.shape == common_shape
        assert wt_1_np.shape == common_shape
        assert r_np.shape == common_shape
        assert eps_np.shape == common_shape
        assert cov_np.shape == (n, n)
        assert limits.shape == (n, 2)
        PENALTY = lambda w: np.prod([normed_box(w, point_low, point_high) for (point_low, point_high) in limits])

    return_term = r_np.T.dot(wt_np)
    risk_term = 0.5 * wt_np.T.dot(cov_np.dot(wt_np))
    fees_term = np.abs(wt_np - wt_1_np).T.dot(eps_np)
    sparse_term = np.linalg.norm(wt_np) ** 2 - np.linalg.norm(wt_np, ord=1) ** 2 * sparsity
    penalty = PENALTY(wt_np)
    norm = np.array(np.linalg.norm(wt_np, ord=1) - desired_norm)

    if by_component:
        logger.info(f" [LOSS] return term : {return_term}")
        logger.info(f" [LOSS] risk term : {risk_term}")
        logger.info(f" [LOSS] fees term : {fees_term}")
        logger.info(f" [LOSS] sparsity term : {sparse_term}")
        logger.info(f" [LOSS] penalty term : {penalty}")
        logger.info(f" [LOSS] norm term : {norm}")

    loss = -return_term + risk_term * risk_coeff + fees_term + sparse_coeff * sparse_term + penalty + norm * norm_coeff

    return loss[0][0]


def normed_box(x, point_low, point_high):
    if point_low is not None and point_high is not None:
        eps = np.abs(point_low - point_high) / 100.0
        return box(x, point_low, point_high, 5.0 / eps, 1, 0)

    elif point_low is None and point_high is None:
        return 0.0


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


def analy_optim_mean_var(
    r_np: np.array, risk_coeff: float, cov_np: np.array, n: int, cut: float = None, return_cond: bool = False
):

    common_shape = (n, 1)
    assert r_np.shape == common_shape
    assert cov_np.shape == (n, n)

    cond = np.linalg.cond(cov_np)
    logger.info(f"Condition number: {cond}")
    if cut is not None:
        optimum_w = np.linalg.pinv(cov_np, cut).dot(r_np) / risk_coeff
    else:
        optimum_w = np.linalg.inv(cov_np).dot(r_np) / risk_coeff

    if not return_cond:
        return optimum_w
    else:
        return optimum_w, cond
