import numpy as np
from typing import Callable, Union
import os
import logging
import inspect

import pandas as pd

logger = logging.getLogger(__name__)


def to_array(value: Union[int, float, list, np.ndarray, pd.Series, pd.DataFrame], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        if isinstance(value, (float, int)):
            value = np.array([value])
        elif isinstance(value, (list, set, tuple)):
            value = np.array(value)
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.values
        else:
            raise TypeError(f"'{name}' must be castable into a numpy array, got a {type(value)}.")
    if any(np.isnan(value.flatten())):
        raise ValueError(f"'{name}' can not contain NANs")
    return value.astype(float)


class Annealer:

    __AVAILABLE_CORES = os.cpu_count()
    __LIMIT_CORES = __AVAILABLE_CORES - 1
    __PARALLEL = True
    if __LIMIT_CORES == 0:
        __PARALLEL = False

    @classmethod
    def set_parallel(cls, value: bool):
        if not isinstance(value, bool):
            raise ValueError(f"Argument of set_parellel should be a boolean, got {type(value)} instead")
        if value is True and cls.__LIMIT_CORES == 0:
            logger.warning("Can not use parallel annealing : only one core is available on this machine.")
            cls.__PARALLEL = False
            return
        cls.__PARALLEL = value

    @classmethod
    def limit_cores(cls, value: int):
        if not isinstance(value, int):
            raise ValueError(f"Number of cores should be an integer, got {type(value)} instead")
        if value >= cls.__AVAILABLE_CORES:
            logger.warning(
                f"Number of core used for parallel annealing can not be geater than "
                f"{cls.__AVAILABLE_CORES - 1} on this machine. Using this value as a limit."
            )
            cls.__LIMIT_CORES = cls.__AVAILABLE_CORES - 1
            return
        cls.__LIMIT_CORES = value

    def __init__(
        self,
        loss: Callable,
        weights_step_size: Union[float, np.ndarray],
        bounds: np.ndarray = None,
        init_states: np.ndarray = None,
        temp_0: float = None,
        temp_min: float = 0,
        alpha: float = 0.85,
        iterations: int = 1000,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        loss: Callable
            The loss function to minimize
        weights_step_size: Union[float, np.ndarray]
            Size of the variation to apply to each weights at each epoch. If a float is given, the same size is used for
            every weights. If a np.ndarray is given, it must have 'dimensions' entries, each entry will be the step size
            of one weight.
        bounds: np.ndarray
            Optional. The limit of the weights. Must be a 2-D array of size ('dimensions', 2). If not specified, will
            not use any bounds. Note that if bounds is not specified, then init_states must be, and vice-versa.
        init_states: np.ndarray
            Optional. Initial values of the weights. Will use a random value if not specified. If specified, its size
            defines the number of dimensions. Note that if init_states is not specified, then bounds must be,
            and vice-versa.
        temp_0: float
            Initial temperature. If not specified, will use get_t_max to get it.
        temp_min: float
            Final temperature. Default is 0
        alpha: float
            Leraning rate, default is 0.85
        iterations: int
            Number of iterations to make (default value = 1000)
        verbose: bool (default is False)

        The number of iterations will be equal to int((temp_0 - temp_min) / temp_step_size).
        If temp_step_size is not specified, then the number of iterations is equal to 200. (0.5% at each step).
        """

        self.dimensions = None
        if not isinstance(loss, Callable):
            raise TypeError(f"The loss function must be callable, got a {type(loss)} instead")
        if len(inspect.signature(loss).parameters) != 1:
            raise ValueError("The loss function must accept exactly one parameter")
        self.loss = loss

        if weights_step_size is None:
            raise TypeError("'weights_step_size' can not be None")
        if alpha is None:
            raise TypeError("'alpha' can not be None")
        if temp_min is None:
            raise TypeError("'temp_min' can not be None")
        if iterations is None:
            raise TypeError("'iterations' can not be None")

        if bounds is None and init_states is None:
            raise ValueError("At least one of 'init_states' and 'bounds' must be specified")

        if bounds is not None:
            bounds = to_array(bounds, "bounds")
            if bounds.ndim != 2 or bounds.shape[1] != 2:
                raise ValueError(f"'bounds' dimension should be (any, 2), got {bounds.shape}")
            self.dimensions = bounds.shape[0]
        self.bounds = bounds

        if init_states is not None:
            if isinstance(init_states, int):
                init_states = float(init_states)
            if not isinstance(init_states, float):
                init_states = to_array(init_states, "init_states")
                if init_states.ndim != 1:
                    raise ValueError("'init_states' must be a 1-D numpy array")
            else:
                if np.isnan(init_states):
                    raise ValueError("'init_states' can not be NAN")
                init_states = np.array([init_states])
            if self.dimensions is None:
                self.dimensions = len(init_states)
            elif self.dimensions != len(init_states):
                raise ValueError(f"Dimension of 'bounds' is {self.dimensions} but 'init_states' has {len(init_states)}"
                                 f" elements.")
            self.init_states = init_states
        else:
            self.init_states = (
                    self.bounds[:, 0]
                    + np.random.uniform(size=(1, len(self.bounds)))
                    * (self.bounds[:, 1] - self.bounds[:, 0])
            ).flatten()

        if isinstance(weights_step_size, int):
            weights_step_size = float(weights_step_size)
        if not isinstance(weights_step_size, float):
            weights_step_size = to_array(weights_step_size, "weights_step_size")
            if weights_step_size.shape != (self.dimensions,):
                raise ValueError(
                    f"Shape of 'weights_step_size' should be ({self.dimensions},), but it is {weights_step_size.shape}."
                )
        else:
            if np.isnan(weights_step_size):
                raise ValueError("weights_step_size can not be NAN")
            weights_step_size = np.array([weights_step_size for _ in range(self.dimensions)])
        self.weights_step_size = weights_step_size

        if temp_0 is not None:
            if isinstance(temp_0, int):
                temp_0 = float(temp_0)
            if not isinstance(temp_0, float):
                raise TypeError(f"'temp_0' must be a float, got {type(temp_0)} instead.")
            if np.isnan(temp_0):
                raise ValueError("'temp_0' can ont be NAN")
        self.temp_0 = temp_0

        if isinstance(temp_min, int):
            temp_min = float(temp_min)
        if not isinstance(temp_min, float):
            raise TypeError(f"'temp_min' must be a float, got {type(temp_min)} instead.")
        if np.isnan(temp_min):
            raise ValueError("'temp_min' can ont be NAN")
        self.temp_min = temp_min

        if isinstance(alpha, int):
            alpha = float(alpha)
        if not isinstance(alpha, float):
            raise TypeError(f"'alpha' must be a float, got {type(alpha)} instead.")
        if np.isnan(alpha):
            raise ValueError("'alpha' can ont be NAN")
        if not (0 < alpha <= 1):
            raise ValueError("'alpha' must be between 0 excluded and 1.")
        self.alpha = alpha

        if not isinstance(verbose, bool):
            raise TypeError(f"'verbose' must be a boolean, got {type(verbose)} instead.")
        self.verbose = verbose
        
        if self.temp_0 is None:
            self.temp_0 = self.get_temp_max()

        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Number of iterations must be an integer greater than 0")
        self.iterations = iterations

    def info(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def debug(self, msg: str):
        if self.verbose:
            logger.debug(msg)

    # TODO docstring (mrizzato)
    def get_temp_max(self):

        self.info("Looking for starting temperature")

        tmax_guesses = [10 ** (float(log10)) for log10 in np.linspace(-3, 2, 40)]

        def func(t):
            tmp_annealer = Annealer(
                loss=self.loss,
                weights_step_size=self.weights_step_size,
                bounds=self.bounds,
                init_states=self.init_states,
                temp_0=t,
            )
            return tmp_annealer.fit()[2]

        acc_ratios = [func(tmg) for tmg in tmax_guesses]
        ibest = np.argmin(np.abs(np.array(acc_ratios) - 0.8))

        try:
            np.isclose(acc_ratios[ibest][0], 0.8, rtol=1e-2, atol=1e-02)
        except Exception:
            raise RuntimeError("No temperature found with an acceptance ratio close to 0.8.")

        return tmax_guesses[ibest]

    def fit(self, alpha=0.85, tmin=None, t0=None, iterations=None) -> tuple:

        if alpha is None:
            alpha = self.alpha
        if tmin is None:
            tmin = self.temp_min
        if t0 is None:
            t0 = self.temp_0
        if iterations is None:
            iterations = self.iterations

        if alpha is None or not isinstance(alpha, float) or not (0 < alpha <= 1):
            raise ValueError("'alpha' must be a float between 0 excluded and 1")
        if tmin is None or tmin < 0:
            raise ValueError("'tmin' must be a float greater than or equal to 0")
        if t0 is None or t0 <= tmin:
            raise ValueError("'t0' must be a float greater than tmin")
        if iterations is None or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Number of iterations must be an integer greater than 0")
    
        b = tmin * (1 - alpha)
    
        self.info(f"Starting temp : {t0}")
        curr = self.init_states.copy()
        curr_eval = self.loss(curr.T)
    
        history = []
    
        # run the algorithm
        points_accepted = 0
        for i_ in range(self.iterations):
    
            # take a step
            unit_v = np.random.uniform(size=(1, self.dimensions))
            unit_v = unit_v / np.linalg.norm(unit_v)
            assert np.isclose(np.linalg.norm(unit_v), 1.0)
            cov = np.zeros((curr.shape[0], curr.shape[0]), float)
            np.fill_diagonal(cov, self.weights_step_size)
            candidate = np.random.multivariate_normal(mean=curr, cov=cov).reshape(curr.shape)
    
            # evaluate candidate point
            candidate_eval = self.loss(candidate.T)
    
            # check for new best solution
            if candidate_eval < curr_eval:
                points_accepted = points_accepted + 1
                # store new best point
                history.append([i_, candidate, np.NaN, candidate_eval, t0])
                curr, curr_eval = candidate, candidate_eval
                # report progress
                self.info(f"Accepted : {i_} f({curr}) = {curr_eval}")
    
            else:
                diff = candidate_eval - curr_eval
                metropolis = np.exp(-diff / t0)
                # check if we should keep the new point
    
                if np.random.uniform() < metropolis:
                    points_accepted = points_accepted + 1
                    # store the new current point
                    curr, curr_eval = candidate, candidate_eval
                    self.info(f"Accepted : {i_} f({curr}) = {curr_eval}")
    
                    history.append([i_, candidate, np.NaN, candidate_eval, t0])
    
                else:
                    # rejected point
                    self.info(f"Rejected :{i_} f({candidate}) = {candidate_eval}")
                    history.append([i_, np.NaN, candidate, candidate_eval, t0])
    
            t0 = t0 * alpha + b
            if i_ > 1:
                acc_ratio = float(points_accepted) / float(i_ + 1)
                self.info(f"Temperature : {t0} | acc. ratio so far : {acc_ratio}")

        acc_ratio = float(points_accepted) / float(self.iterations)
    
        self.info(f"Final temp : {t0}")
        self.info(f"Acc. ratio : {acc_ratio}")
    
        return curr, curr_eval, acc_ratio, history
