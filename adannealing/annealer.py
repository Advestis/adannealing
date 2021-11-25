import numpy as np
from typing import Callable, Union
import os
import logging
import inspect

import pandas as pd

from .plotting import Sampler, SamplePoint

logger = logging.getLogger(__name__)


def make_counter(iterable):
    if isinstance(iterable, int):
        nitems = iterable
    else:
        nitems = len(iterable)
    dt = int(nitems / 10)
    if nitems < 10:
        dt = 1
    indexes_to_print = {
        i: f"{i}/{nitems}, {round(100 * i / nitems, 2)}%" for i in list(range(dt, nitems, dt))
    }
    return indexes_to_print


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
            Optional. The limit of the weights, used to determine initial state.
            Must be a 2-D array of size ('dimensions', 2). Note that if bounds are not specified,
            then init_states must be, and vice-versa.
        init_states: np.ndarray
            Optional. Initial values of the weights. Will use random values using 'bounds' if not specified.
            If specified, its size defines the number of dimensions. Note that if init_states are not specified,
            then bounds must be, and vice-versa.
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

        if bounds is not None and init_states is not None:
            logger.warning("Specified bounds and init_states. Bounds are then ignored.")

        if init_states is None:
            bounds = to_array(bounds, "bounds")
            if bounds.ndim != 2 or bounds.shape[1] != 2:
                raise ValueError(f"'bounds' dimension should be (any, 2), got {bounds.shape}")
            self.dimensions = bounds.shape[0]
            for coordinate in range(self.dimensions):
                if bounds[coordinate][0] > bounds[coordinate][1]:
                    raise ValueError("Bounds are not valid : some lower limits are greater then their upper limits:\n"
                                     f"{bounds}")
            self.init_states = (
                    bounds[:, 0]
                    + np.random.uniform(size=(1, len(bounds)))
                    * (bounds[:, 1] - bounds[:, 0])
            )
        else:
            if isinstance(init_states, int):
                init_states = float(init_states)
            if not isinstance(init_states, float):
                init_states = to_array(init_states, "init_states")
                if init_states.ndim != 1 and not (init_states.ndim == 2 and init_states.shape[0] == 1):
                    raise ValueError("'init_states' must be a 1-D numpy array or a line vector")
            else:
                if np.isnan(init_states):
                    raise ValueError("'init_states' can not be NAN")
                init_states = np.array([init_states])
            if init_states.ndim == 1:
                init_states = init_states.reshape(1, init_states.shape[0])
            self.dimensions = init_states.shape[1]

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

    def get_temp_max(self, ar_limit_low=0.79, ar_limit_up=0.81, max_attempt=100):

        self.info(f"Looking for starting temperature...")

        if ar_limit_up < ar_limit_low:
            raise ValueError("Acceptance ratio limit up must be greater than Acceptance ratio limit low")
        if not isinstance(max_attempt, int):
            raise TypeError("'max_attempts' must be an integer")
        if max_attempt <= 0:
            raise ValueError("'max_attempts' must be greater than 0")
        
        if ar_limit_up >= 0.95:
            raise ValueError("Acceptance ratio limit up can not be equal to or greater than 0.95")

        acc_ratio = 0
        attempts = 0
        t1_i = 1e-5
        t2 = 10
        t1 = t1_i
        acc_ratio_2, t0, acc_ratio_0 = None, None, None
        ar_limit_mean = (ar_limit_up + ar_limit_low) / 2.
        ann = Annealer(
                loss=self.loss,
                weights_step_size=self.weights_step_size,
                init_states=self.init_states,
                temp_0=1,
                temp_min=0,
                alpha=1,
                iterations=100,
                verbose=False
            )
        acc_ratio_1 = ann.fit(temp_0=t1, iterations=10000)[2]

        if ar_limit_low < acc_ratio_1 < ar_limit_up:
            # Lucky strike : t0 is already good !
            acc_ratio_0 = acc_ratio_1
            t0 = t1
        else:
            # Unlucky strike : t1 gives an acc_ratio greater than the upper limit.
            while acc_ratio_1 > ar_limit_up:
                if attempts > max_attempt:
                    raise ValueError(f"Could not find a temperature giving an acceptance ratio between {ar_limit_low} "
                                     f"and {ar_limit_up} in less than {max_attempt} attempts")
                t1 = t1 / 10
                acc_ratio_1 = ann.fit(temp_0=t1, iterations=10000)[2]
                attempts += 1

            attempts = 0
            while not ar_limit_low < acc_ratio < ar_limit_up:
                if attempts > max_attempt:
                    raise ValueError(f"Could not find a temperature giving an acceptance ratio between {ar_limit_low} "
                                     f"and {ar_limit_up} in less than {max_attempt} attempts")
                acc_ratio_2 = ann.fit(temp_0=t2)[2]
                self.info(f"Attempt {attempts}")
                self.info(f"t1: {t1}, Acc. ratio : {acc_ratio_1} (fixed)")
                self.info(f"t2: {t2}, Acc. ratio : {acc_ratio_2}")

                if ar_limit_low < acc_ratio_2 < ar_limit_up:
                    acc_ratio_0 = acc_ratio_2
                    t0 = t2
                    break

                if acc_ratio_2 > 0.95:
                    t2 = (t2 - t1) / 10
                    attempts += 1
                    continue

                slope = (acc_ratio_2 - acc_ratio_1) / (t2 - t1)
                if slope < 0:
                    self.debug("Got a negative slope when trying to find starting temperature. Impossible : "
                               "acceptance ratio should be strictly increasing with temperature")
                    attempts += 1
                    continue
                if slope == 0:
                    self.debug("Got a null slope when trying to find starting temperature. Impossible : "
                               "acceptance ratio should be strictly increasing with temperature")
                    attempts += 1
                    continue
                t2 = max([0, (ar_limit_mean - acc_ratio_1) / slope - t1])
                if t2 <= 0:
                    t2 = 2e-16
                attempts += 1

        if t0 is None:
            t0 = t2
            acc_ratio_0 = acc_ratio_2
            logger.warning(f"Could not find a suitable starting temperature. Will try annealing anyway with t0={t0} "
                           f"(acc. ratio = {acc_ratio_0})")
        else:
            self.info(f"Found starting temperature t0 = {t0} (acc. ratio = {acc_ratio_0})")
        return t0

    # TODO (pcotte) : implement more cooling schedule
    def fit(self, alpha=None, temp_min=None, temp_0=None, iterations=None, stopping_limit=None, history_path=None):

        if alpha is None:
            alpha = self.alpha
        if temp_min is None:
            temp_min = self.temp_min
        if temp_0 is None:
            temp_0 = self.temp_0
        if iterations is None:
            iterations = self.iterations

        if stopping_limit is not None and not 0 < stopping_limit < 1:
            raise ValueError("'limit' should be between 0 and 1")

        if alpha is None or not isinstance(alpha, float) or not (0 < alpha <= 1):
            raise ValueError("'alpha' must be a float between 0 excluded and 1")
        if temp_min is None or temp_min < 0:
            raise ValueError("'tmin' must be a float greater than or equal to 0")
        if temp_0 is None or temp_0 <= temp_min:
            raise ValueError(f"'t0' must be a float greater than tmin, got {temp_0} <= {temp_min}")
        if iterations is None or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Number of iterations must be an integer greater than 0")
    
        b = temp_min * (1 - alpha)
    
        self.info(f"Starting temp : {temp_0}")
        curr = self.init_states.copy()
        curr_loss = self.loss(curr.T)

        check_loss_every = int(self.iterations / 100)
        if check_loss_every <= 5:
            stopping_limit = None
    
        history = Sampler()

        to_print = make_counter(self.iterations)
    
        points_accepted = 0
        acc_ratio = None
        loss_for_finishing = None
        n_finishing = 0
        n_finishing_max = 100
        finishing_history = Sampler()
        finishing = False
        prev_loss = None
        for i_ in range(self.iterations):
    
            # take a step
            unit_v = np.random.uniform(size=(1, self.dimensions))
            unit_v = unit_v / np.linalg.norm(unit_v)
            assert np.isclose(np.linalg.norm(unit_v), 1.0)
            cov = np.zeros((curr.shape[1], curr.shape[1]), float)
            np.fill_diagonal(cov, self.weights_step_size)
            candidate = np.random.multivariate_normal(mean=curr.ravel(), cov=cov).reshape(curr.shape)
    
            candidate_loss = self.loss(candidate.T)

            accepted = candidate_loss < curr_loss
            if accepted:
                points_accepted = points_accepted + 1
                prev_loss = curr_loss
                curr, curr_loss = candidate, candidate_loss
                self.debug(f"Accepted : {i_} f({curr}) = {curr_loss}")
            else:
                diff = candidate_loss - curr_loss
                metropolis = np.exp(-diff / temp_0)
                if np.random.uniform() < metropolis:
                    accepted = True
                    points_accepted = points_accepted + 1
                    prev_loss = curr_loss
                    curr, curr_loss = candidate, candidate_loss
                    self.debug(f"Accepted : {i_} f({curr}) = {curr_loss}")
                else:
                    # prev_loss = None
                    # loss_for_finishing = None
                    # finishing = False
                    # n_finishing = 0
                    # finishing_history.clean()
                    self.debug(f"Rejected :{i_} f({candidate}) = {candidate_loss}")

            acc_ratio = float(points_accepted) / float(i_ + 1)
            sample = SamplePoint(
                weights=candidate[0],
                iteration=i_,
                accepted=accepted,
                loss=candidate_loss,
                temp=temp_0,
                acc_ratio=acc_ratio,
            )
            history.append(sample)

            temp_0 = temp_0 * alpha + b
            if i_ in to_print and to_print[i_] is not None:
                self.info(f"step {to_print[i_]}, Temperature : {temp_0} | acc. ratio so far : {acc_ratio}")

            # Checking stopping criterion
            if stopping_limit is not None and prev_loss is not None:
                if not finishing:
                    loss_for_finishing = prev_loss
                    ratio = abs(curr_loss / loss_for_finishing - 1)
                else:
                    ratio = abs(curr_loss / loss_for_finishing - 1)
                if ratio < stopping_limit:
                    finishing = True
                    finishing_history.append(sample)
                    n_finishing += 1
                    if n_finishing > n_finishing_max:
                        self.info(f"Variation of loss is small enough after step {i_}, stopping")
                        curr, curr_loss, acc_ratio = finish(finishing_history)
                        curr = curr.reshape(1, curr.shape[0])
                        break
                else:
                    loss_for_finishing = None
                    finishing = False
                    n_finishing = 0
                    finishing_history.clean()

        self.info(f"Final temp : {temp_0}")
        self.info(f"Acc. ratio : {acc_ratio}")

        if history_path is not None:
            history.data.to_csv(history_path)
    
        return curr, curr_loss, acc_ratio, history


def finish(sampler: Sampler):
    data = sampler.data
    mask = data["loss"].drop_duplicates(keep="last").index
    data = data.loc[mask]
    # noinspection PyUnresolvedReferences
    data = data.loc[(data["loss"] == data["loss"].min()).values]
    return data["weights"].iloc[-1], data["loss"].iloc[-1], data["acc_ratio"].iloc[-1]
