import numpy as np
import logging
import json
import pandas as pd
import warnings

logger = logging.getLogger(__name__)


class LossPortfolioMeanVar:
    def __init__(
        self,
        wt_1_np: np.array,
        r_np: np.array,
        lambda_risk: float,
        lambda_sparse: float,
        lambda_norm: float,
        fees: np.array,
        cov_risk: np.array,
        sparsity_target: float,
        constraints: tuple,
        sum_w_target: float,
        continous_window: bool,
        n: int,
    ):

        self.wt_1_np = wt_1_np
        self.r_np = r_np
        self.lambda_risk = lambda_risk
        self.lambda_sparse = lambda_sparse
        self.lambda_norm = lambda_norm
        self.fees = fees
        self.cov_risk = cov_risk
        self.sparsity_target = sparsity_target
        self.constraints = constraints
        self.sum_w_target = sum_w_target
        self.continous_window = continous_window
        self.n = n
        self.common_shape = (n, 1)
        self.first_call = False
        self.final_call = False
        self.init_loss_status = None
        self.final_loss_status = None

        self.__setup()

    def __setup(self):
        assert self.wt_1_np.shape == self.common_shape
        assert self.r_np.shape == self.common_shape
        assert self.fees.shape == self.common_shape
        assert self.cov_risk.shape == (self.n, self.n)
        if self.constraints is not None:
            assert len(self.constraints) == self.n

        if self.constraints is not None:
            # TODO : set height in penalty in a dynamic way
            # TODO : I would like the annealer to spend more time in the feasable region
            # TODO : not just towards the end of the run (at low temperatures)
            not_none = np.where([lim != (None, None) for lim in self.constraints])[0]
            if len(not_none) > 0:
                if self.continous_window:
                    self.penalty = lambda w: np.sum(
                        [
                            continuous_constraint(w[i], point_low, point_high)
                            for i, (point_low, point_high) in enumerate(
                                self.constraints
                            )
                            if i in not_none
                        ]
                    )
                else:
                    self.penalty = lambda w: np.sum(
                        [
                            non_continuous_constraint(w[i], point_low, point_high)
                            for i, (point_low, point_high) in enumerate(
                                self.constraints
                            )
                            if i in not_none
                        ]
                    )
            else:
                self.constraints = None
                self.penalty = lambda w: 0.0

        else:
            self.penalty = lambda w: 0.0

    def on_fit_end(self, best_fit):
        
        self.final_call = True
        self.final_loss_status = self.__call__(best_fit)

        if isinstance(self.init_loss_status, list):
            logger.info("Code has run with several annealers. List of different intialian points:")
            for i, initial_status in enumerate(self.final_loss_status):
                logger.info(f"Initial Status Annealer {i}")
                logger.info(initial_status)

        elif isinstance(self.init_loss_status, dict):
            logger.info("Initial Status components:")
            logger.info(self.init_loss_status)

        else:
            raise RuntimeError('Unknown initial status loss.')

        logger.info("Final loss components:")
        logger.info(self.final_loss_status)


        if self.lambda_norm > 0:
            try:
                assert np.isclose(np.sum(best_fit), self.sum_w_target, rtol=1e-2)
            except AssertionError:
                logger.info(
                    "The solution DOES NOT respect the constraint on the sum of the components."
                )
            else:
                logger.info(
                    "The solution DOES respect the constraint on the sum of the components."
                )

        if self.constraints is not None:
            try:
                assert all(
                    map(
                        lambda a: a[0] > a[1][0] and a[0] < a[1][1],
                        zip(best_fit, self.constraints),
                    )
                )
            except AssertionError:
                logger.info("The solution DOES NOT respect the constraints.")
            else:
                logger.info("The solution DOES respect the constraints.")

        if self.lambda_sparse:
            sparse_term = (
                np.linalg.norm(best_fit) ** 2
                - np.linalg.norm(best_fit, ord=1) ** 2 * self.sparsity_target
            )
            try:
                assert np.isclose(sparse_term, 0.0, atol=1e-3)
            except AssertionError:
                logger.info(
                    "The solution DOES NOT meet the requested level of sparsity."
                )
            else:
                logger.info("The solution DOES meet the requested level of sparsity.")
                
    def on_fit_start(self, initial_point):

        if isinstance(initial_point, np.ndarray):
            self.first_call = True
            self.init_loss_status = self.__call__(initial_point)

        elif isinstance(initial_point, tuple):
            self.init_loss_status = []
            for init in initial_point:
                self.first_call = True
                self.init_loss_status.append(self.__call__(init))

        else:
            raise RuntimeError('Unknown itialisation type')

        self.first_call = False

    def __call__(self, wt_np):
        try:
            assert wt_np.shape == self.common_shape
        except AssertionError:
            try:
                wt_np = wt_np.T
                assert wt_np.shape == self.common_shape
            except AssertionError:
                raise RuntimeError('Input shape is wrong even if transposed.')

        return_term = self.r_np.T.dot(wt_np)
        risk_term = 0.5 * wt_np.T.dot(self.cov_risk.dot(wt_np))
        fees_term = np.abs(wt_np - self.wt_1_np).T.dot(self.fees)
        sparse_term = (
            np.linalg.norm(wt_np) ** 2
            - np.linalg.norm(wt_np, ord=1) ** 2 * self.sparsity_target
        )
        penalty = self.penalty(wt_np)
        norm = np.abs(np.linalg.norm(wt_np, ord=1) - self.sum_w_target)

        return_term = -return_term
        risk_term = risk_term * self.lambda_risk
        sparse_term = self.lambda_sparse * sparse_term
        norm = norm * self.lambda_norm

        logger.debug("\n")
        logger.debug(f" [LOSS] return term : {return_term}")
        logger.debug(f" [LOSS] risk term : {return_term}")
        logger.debug(f" [LOSS] fees term : {fees_term}")
        logger.debug(f" [LOSS] sparsity term : {sparse_term}")
        logger.debug(f" [LOSS] penalty term : {penalty}")
        logger.debug(f" [LOSS] norm term : {norm}")

        if self.first_call or self.final_call:
            return return_term, risk_term, fees_term, sparse_term, penalty, norm

        else:
            loss = return_term + risk_term + fees_term + sparse_term + penalty + norm
            return loss[0][0]


def load_financial_configurations(path):
    with open(path) as f:
        configs = json.load(f)
    all_prices = pd.read_parquet(configs["prices_path"])

    # parameters run ++++++++++++++++++++++++++++++++++++++++++++++++
    path_save_images = configs["path_save_images"]
    date = configs["date"]
    common_fee = configs["common_fee"]
    overall_risk_coeff = configs["overall_risk_coeff"]
    overall_sparse_coeff = configs["overall_sparse_coeff"]
    overall_norm_coeff = configs["overall_norm_coeff"]
    n_iterations = configs["n_iterations"]
    step_size = configs["step_size"]
    alpha = configs["alpha"]
    sparsity = configs["sparsity"]
    desired_norm = configs["desired_norm"]
    continous_window = bool(eval(configs["continous_window"]))

    date_start = configs["date_start"]
    date_end = configs["date_end"]
    # parameters run ++++++++++++++++++++++++++++++++++++++++++++++++

    all_prices = all_prices[all_prices.index >= pd.Timestamp(date_start)]
    all_prices = all_prices[all_prices.index <= pd.Timestamp(date_end)]
    return (
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
    )


def continuous_constraint(x, point_low, point_high):

    val = 0.0
    if point_low is not None and point_high is not None:
        eps = np.abs(point_low - point_high) / 100.0
        sharpness = 5.0 / eps

    else:
        sharpness = 500.0

    if point_low is not None:
        val += wall(-1.0, x, point_low, sharpness, 1.0, 1.0)

    if point_high is not None:
        val += wall(+1.0, x, point_high, sharpness, 1.0, 1.0)

    return val


def non_continuous_constraint(x, point_low, point_high):

    val = 0.0

    if point_low is not None and x < point_low:
        val += (point_low - x) * 10.0

    if point_high is not None and x > point_high:
        val += (x - point_high) * 10.0

    return val


def wall(lr, x, point, sharpness, height, speed):
    val = (lr * speed * (x - point) + height) * sigmoid(lr * (x - point) * sharpness)
    return val


def sigmoid(x):

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            val = 1.0 / (1.0 + np.exp(-x))
        except RuntimeWarning:
            val = 0.0

    return val


def analy_optim_mean_var(
    r_np: np.array,
    risk_coeff: float,
    cov_np: np.array,
    n: int,
    cut: float = None,
    return_cond: bool = False,
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
