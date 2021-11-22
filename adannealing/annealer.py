import numpy as np
from typing import Callable, Union
import os
import logging

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
    if any(np.isnan(value)):
        raise ValueError(f"'{name}' can not contain NANs")
    return value


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
        dimensions: int = None,
        init_states: np.ndarray = None,
        temp_step_size: float = None,
        temp_0: float = None,
        temp_min: float = 0,
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
        dimensions: int
            The number of weights. Optional if 'init_states' is specified. One of 'dimensions' and 'init_states' MUST be
            specified.
        init_states: np.ndarray
            Optional. Initial values of the weights. Will use a random value if not specified. If specified, its size
            defines the number of dimensions. Note that if init_states is not specified, then bounds must be,
            and vice-versa.
        temp_step_size: float
            Size of the variation of temperature to apply to each epoch. If not specified, will use 0.5% of the
            difference between temp_0 and temp_min.
        temp_0: float
            Initial temperature. If not specified, will use get_t_max to get it.
        temp_min: float
            Final temperature. Default is 0
        verbose: bool (default is False)

        The number of iterations will be equal to int((temp_min - temp_0) / temp_step_size).
        If temp_step_size is not specified, then the number of iterations is equal to 200. (0.5% at each step).
        """

        if not isinstance(loss, Callable):
            raise TypeError(f"Loss must be callable, got a {type(loss)} instead")
        self.loss = loss

        if dimensions is None and init_states is None:
            raise ValueError("At least one of 'dimensions' and 'init_states' must be specified")

        if init_states is not None:
            if isinstance(init_states, int):
                init_states = float(init_states)
            if not isinstance(init_states, float):
                init_states = to_array(init_states, "init_states")
                if len(init_states.shape) != 1:
                    raise ValueError("'init_states' must be a 1-D numpy array")
            else:
                if np.isnan(init_states):
                    raise ValueError("init_states can not be NAN")
                init_states = np.array([init_states])

        if dimensions is not None and init_states is not None:
            if len(init_states) != dimensions:
                raise ValueError(f"Specified {dimensions} dimensions but init_states has {len(init_states)} elements.")

        if dimensions is not None:
            self.dimensions = dimensions
        else:
            self.dimensions = len(init_states)
        self.init_states = init_states

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

        if bounds is not None:
            bounds = to_array(bounds, "bounds")
            if bounds.shape != (self.dimensions, 2):
                raise ValueError(f"Shape of 'bounds' should be ({self.dimensions}, 2), but it is {bounds.shape}.")
        self.bounds = bounds

        if self.bounds is None and self.init_states is None:
            raise ValueError("At least one of 'init_states' and 'bounds' must be specified")

        if temp_step_size is not None:
            if isinstance(temp_step_size, int):
                temp_step_size = float(temp_step_size)
            if not isinstance(temp_step_size, float):
                raise TypeError(f"'temp_step_size' must be a float, got {type(temp_step_size)} instead.")
            if np.isnan(temp_step_size):
                raise ValueError("temp_step_size can ont be NAN")
        self.temp_step_size = temp_step_size

        if temp_0 is not None:
            if isinstance(temp_0, int):
                temp_0 = float(temp_0)
            if not isinstance(temp_0, float):
                raise TypeError(f"'temp_0' must be a float, got {type(temp_0)} instead.")
            if np.isnan(temp_0):
                raise ValueError("temp_0 can ont be NAN")
        self.temp_0 = temp_0

        if temp_min is None:
            raise TypeError("'temp_min' can not be None")
        if isinstance(temp_min, int):
            temp_min = float(temp_min)
        if not isinstance(temp_min, float):
            raise TypeError(f"'temp_min' must be a float, got {type(temp_min)} instead.")
        if np.isnan(temp_min):
            raise ValueError("temp_min can ont be NAN")
        self.temp_min = temp_min

        if not isinstance(verbose, bool):
            raise TypeError(f"'verbose' must be a boolean, got {type(verbose)} instead.")
        self.verbose = verbose
        
        if self.temp_0 is None:
            self.temp_0 = self.get_temp_max()

    def info(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def debug(self, msg: str):
        if self.verbose:
            logger.debug(msg)

    # TODO docstring (mrizzato)
    def get_temp_max(self):

        self.info("  [TMAX] Looking for starting temperature")

        tmax_guesses = [10 ** (float(log10)) for log10 in np.linspace(-3, 2, 40)]

        # engine = get_engine(kind="serial")

        def func(t):
            tmp_annealer = Annealer(
                loss=self.loss,
                weights_step_size=self.weights_step_size,
                bounds=self.bounds,
                dimensions=self.dimensions,
                init_states=self.init_states,
                temp_0=t,
            )
            return tmp_annealer.fit()[2]

        # acc_ratios = engine(func, Tmax_guesses)
        acc_ratios = [func(tmg) for tmg in tmax_guesses]

        msg = "\n\t".join(*list(zip(tmax_guesses, np.array(list(acc_ratios)).T.ravel())))

        self.info(f"  [TMAX] tmp vs acc_ratio : {msg}")

        ibest = np.argmin(np.abs(np.array(acc_ratios) - 0.8))

        try:
            np.isclose(acc_ratios[ibest][0], 0.8, rtol=1e-2, atol=1e-02)
        except Exception:
            raise RuntimeError("No temperature found with an acceptance ratio close to 0.8.")

        return tmax_guesses[ibest]

    def fit(self, alpha=0.85, n_iterations=1000) -> tuple:
    
        b = self.temp_min * (1 - alpha)
    
        print(" [SIM. ANN] Starting temp : ", self.temp_0)
    
        if self.init_states is None:
            curr = self.bounds[:, 0] + np.random.uniform(
                size=(1, len(self.bounds))
            ) * (self.bounds[:, 1] - self.bounds[:, 0])
        else:
            curr = self.init_states.copy()
        curr_eval = self.loss(curr.T)
    
        history = []
    
        # run the algorithm
        points_accepted = 0
        for i_ in range(n_iterations):
    
            # take a step
            unit_v = np.random.uniform(size=(1, self.dimensions))
            unit_v = unit_v / np.linalg.norm(unit_v)
            assert np.isclose(np.linalg.norm(unit_v), 1.0)
            cov = np.zeros((curr.shape[1], curr.shape[1]), float)
            np.fill_diagonal(cov, self.weights_step_size)
            candidate = np.random.multivariate_normal(mean=curr.ravel(), cov=cov).reshape(curr.shape)
    
            # evaluate candidate point
            candidate_eval = self.loss(candidate.T)
    
            # check for new best solution
            if candidate_eval < curr_eval:
                points_accepted = points_accepted + 1
                # store new best point
                history.append([i_, candidate, np.NaN, candidate_eval, self.temp_0])
                curr, curr_eval = candidate, candidate_eval
                # report progress
                self.info(f" [SIM. ANN] Accepted : {i_} f({curr}) = {curr_eval}")
    
            else:
                diff = candidate_eval - curr_eval
                metropolis = np.exp(-diff / self.temp_0)
                # check if we should keep the new point
    
                if np.random.uniform() < metropolis:
                    points_accepted = points_accepted + 1
                    # store the new current point
                    curr, curr_eval = candidate, candidate_eval
                    self.info(f" [SIM. ANN] Accepted : {i_} f({curr}) = {curr_eval}")
    
                    history.append([i_, candidate, np.NaN, candidate_eval, self.temp_0])
    
                else:
                    # rejected point
                    self.info(f" [SIM. ANN] Rejected :{i_} f({candidate}) = {candidate_eval}")
                    history.append([i_, np.NaN, candidate, candidate_eval, self.temp_0])
    
            self.temp_0 = self.temp_0 * alpha + b
            if i_ > 1:
                acc_ratio = float(points_accepted) / float(i_ + 1)
                self.info(f" [SIM. ANN] Temperature : {self.temp_0} | acc. ratio so far : {acc_ratio}")

        acc_ratio = float(points_accepted) / float(n_iterations)
    
        print(" [SIM. ANN] Final temp : ", self.temp_0)
        print(" [SIM. ANN] Acc. ratio : ", acc_ratio)
    
        return curr, curr_eval, acc_ratio, history
