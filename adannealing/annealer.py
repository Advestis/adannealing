import math
import os
from time import sleep, time
import plotly.graph_objects as go
from typing import Dict, Any, Collection, List, Union, Tuple, Optional
import inspect
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from more_itertools import distinct_combinations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import logging
import matplotlib.markers as mmarkers
from plotly.subplots import make_subplots
import itertools

import pandas as pd

from .plotting import Sampler, SamplePoint

logger = logging.getLogger(__name__)


class AbstractLoss:
    """Abstract class from which any loss given to `adannealing.annealer.Annealer` must derive.

    The __init__ function must be entierly defined by the user. If the object has the attribute "constraints", it will
    be detected by the `adannealing.annealer.Annealer` as constraints that should be applied to the loss.
    """

    def __call__(self, w: np.array) -> float:
        """It will be called to evaluate the loss at a given set of weights.

        To be implemented in daughter class

        Parameters
        ----------
        w: np.array
            Weights at which to compute the loss
        """
        pass

    def on_fit_start(self, val: Union[np.array, Tuple[np.array]]):
        """
        This method is called by the fitter before optimisation.

        To be implemented in daughter class

        Parameters
        ----------
        val: Union[np.array, Tuple[np.array]]
            Either the starting weights of the optimiser (for single annealer) or the tuple containing different
            starting weights if more than one annealer are used.
        """
        pass

    def on_fit_end(
        self,
        val: Union[
            List[Tuple[np.ndarray, float, float, Sampler, Sampler, bool]],
            Tuple[np.ndarray, float, float, Sampler, Sampler, bool],
        ],
    ):
        """
        This method is called by the fitter after optimisation.

        To be implemented in daughter class

        Parameters
        ----------
        val: Union[
            List[Tuple[np.ndarray, float, float, Sampler, Sampler, bool]],
            Tuple[np.ndarray, float, float, Sampler, Sampler, bool],
        ]
            Either the result of the optimiser (for single annealer) or the list of results if more than one annealer
            reaches the end of fit.
        """
        pass


def clean_dir(path: Path) -> None:
    """
    Deletes a directory and all its content.
    Will throw a warning if 'path' is a file.
    If 'path' does not point to an existing directory, will retry each seconds for 1 minutes,
    showing warnings everytime. If at the end of the minutes the directory did not appear,
    returns without doing anything.

    The directory must contain only files, not other directories. If at least one item is not a file, the function
    raises IsADirectoryError without deleting anything.

    Parameters
    ----------
    path: Path

    Returns
    -------
    None
    """
    t = time()
    if path.is_file():
        logger.warning(f"Could not delete directory {path} : is a file.")
        return
    while not path.is_dir():
        if time() - t > 60:
            logger.warning(f"Could not delete directory {path} : directory not found.")
            return
        sleep(1)

    if not all([f.is_file() for f in path.glob("*")]):
        raise IsADirectoryError(f"At least one of the items in {path} is not a file.")
    [f.unlink() for f in path.glob("*")]
    path.rmdir()


def make_counter(iterable: Union[int, Collection[Any]]) -> Dict[int, str]:
    """From a given sized iterable or number of items, returns a dictionnary of int:str with keys being the index of the
    iterable (0, 1, ... , n) and values being 'i/n, xx, xx %' for each index matching 10%, 20%, ... 100% (rounded to the
    closest index), and None everywhere.

    Parameters
    ----------
    iterable: Union[int, Collection[Any]]

    Returns
    -------
    Dict[int, str]
    """
    if isinstance(iterable, int):
        nitems = iterable
    else:
        nitems = len(iterable)
    dt = int(nitems / 10)
    if nitems < 10:
        dt = 1
    indexes_to_print = {i: f"{i}/{nitems}, {round(100 * i / nitems, 2)}%" for i in list(range(dt, nitems, dt))}
    return indexes_to_print


def to_array(value: Union[int, float, list, np.ndarray, pd.Series, pd.DataFrame], name: str) -> np.ndarray:
    """
    Converts 'value' into a numpy array, does not reshape. Will raise an error if value's type is not one of :
     * int or bool
     * float
     * list
     * np.ndarray
     * pd.Series
     * pd.DataFrame

    'value' can not contain NaNs.
    Argument 'name' is used for better error messages only.

    Parameters
    ----------
    value: Union[int, float, list, np.ndarray, pd.Series, pd.DataFrame]
    name: str
        For better error messages

    Returns
    -------
    np.ndarray
    """
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

    """Class to do simulated annealing.

    It is recommended to do

    >>> import os
    >>> os.environ["OMP_NUM_THREADS"] = "1"
    >>> os.environ["OPENBLAS_NUM_THREADS"] = "1"
    >>> os.environ["MKL_NUM_THREADS"] = "1"
    >>> os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    >>> os.environ["NUMEXPR_NUM_THREADS"] = "1"

    Before using Annealer. The commands above deactivate the automatic multithreading of NumPy operations, which
    will acutally make you gain time. Indeed, the numpy operations used in Annealer are not very slow, but numerous, and
    attempting multithreading will result in a lot of overhead, eventually slowing down the process. It is better to use
    the feature allowing to do several runs in parallel in order to have one finding the minimum before the others.
    """

    __PARALLEL = False
    __CPU_LIMIT = os.cpu_count()
    __POSSIBLE_SCHEDULES = [
        "logarithmic",
        "linear",
        "geometric",
        "arithmetic-geometric",
    ]

    @classmethod
    def set_parallel(cls):
        """When called, tells the Annealer class that it should use multiprocessing when doing a fit with more than
        one initial states, otherwise that is done in serial. Throws a warning if the cpu limit is 1."""
        if cls.__CPU_LIMIT > 1:
            cls.__PARALLEL = True
        else:
            logger.warning("CPU limit is 1, can not set Annealer to parallel. ")

    @classmethod
    def set_cpu_limit(cls, limit: int):
        """Sets the max number of CPU to use when doing a fit with more than one initial states.
        Will throw a warning if the specified limit is inferior to the hardware capabilities. In that case, will use
        as many CPUs as possible.
        If the specified limit is greater than 1, sets the class to use multiprocessing (same as calling 'set_parallel')
        """
        if not isinstance(limit, int):
            raise TypeError(f"Number of CPUs must be an integer, got {type(limit)}")
        os_limit = os.cpu_count()
        if os_limit < limit:
            logger.warning(
                f"CPU limit can not be set to {limit} : hardware only has {os_limit} cores. Limiting cpu "
                f"limit to this value."
            )
            limit = os_limit
        cls.__CPU_LIMIT = limit
        if limit == 1:
            cls.__PARALLEL = False
        else:
            cls.__PARALLEL = True

    @classmethod
    def unset_parallel(cls):
        """Deactiates parallel runs when doing a fit with more than one initial state. Does not change the CPU limit."""
        cls.__PARALLEL = False

    # noinspection PyUnresolvedReferences
    @classmethod
    def _fit_many(
        cls,
        iterable: Collection["Annealer"],
        stop_at_first_found: bool,
        history_path: Union[None, str, Path],
    ) -> Union[
        List[Tuple[np.ndarray, float, float, Sampler, Sampler, bool]],
        Tuple[np.ndarray, float, float, Sampler, Sampler, bool],
    ]:
        """Will call 'Annealer.fit_one' on an iterable of Annealer objects. Will either loop on the iterable if the
        Annealer class was not specified to run in parallel, or use a ProcessPoolExecutor if it was.

        If 'stop_at_first_found' is True, will do:
        * If running in serial, will stop as soon a fit was successful (see self.fit_one for definition of 'successful')
        * If running in parallel, will cancel runs as soon as at least one run is successful. Among all successful
        finished runs, will keep the one with the smallest final loss.
        * In both cases, if 'history_path' is specified, will clean the output directories of the discarded runs.
        * Will only return one fit result
        * If no run was successful , will return the one with the smallest loss and throw a warning.

        Else, will return the list of all fit results.

        Parameters
        ----------
        iterable: Collection[Annealer]
        stop_at_first_found: bool
        history_path: Union[None, str, Path]

        Returns
        -------
        Union[
            List[Tuple[np.ndarray, float, float, Sampler, Sampler, bool]],
            Tuple[np.ndarray, float, float, Sampler, Sampler, bool],
        ]
        """
        if len(iterable) == 0:
            return []

        fit_method = (
            cls._fit_one_canonical if iterable[0].annealing_type == "canonical" else cls._fit_one_microcanonical
        )

        if not cls.__PARALLEL:
            if not stop_at_first_found:
                return [cls._fit_one_canonical(iterable[i]) for i in range(len(iterable))]
            results = []
            for i in range(len(iterable)):
                results.append(fit_method(iterable[i], history_path=None))
                if results[-1][-1]:
                    if history_path is not None:
                        results[-1][-3].data.to_csv(Path(history_path) / "history.csv")
                        results[-1][-2].data.to_csv(Path(history_path) / "result.csv")
                        for j in range(i + 1, len(iterable)):
                            path = Path(history_path) / str(j)
                            if path.is_dir():
                                [f.unlink() for f in path.glob("*")]
                                path.rmdir()
                    return results[-1]
        else:
            context = mp.get_context("spawn")
            bads = []
            found = False
            with ProcessPoolExecutor(max_workers=cls.__CPU_LIMIT, mp_context=context) as pool:

                if not stop_at_first_found:
                    return list(pool.map(fit_method, iterable))

                for it in iterable:
                    it.history_path = None
                results = [pool.submit(fit_method, it) for it in iterable]

                while True:
                    returns_index = [i for i in range(len(results)) if results[i].done()]
                    returns = [results[i].result() for i in returns_index]

                    if len(returns) > 0:
                        goods = [res for res in returns if returns[-1]]
                        if len(goods) > 0:
                            best_loss = min([res[1] for res in goods])
                            [run.cancel() for run in results]
                            results = [res for res in goods if res[1] == best_loss][0]
                            if history_path is not None:
                                bads = [j for j in range(len(iterable)) if j not in returns]
                                results[-3].data.to_csv(Path(history_path) / "history.csv")
                                results[-2].data.to_csv(Path(history_path) / "result.csv")
                            found = True
                            break
                    if len(returns) == len(iterable):
                        break

            if found:
                if len(bads) > 0:
                    with ThreadPoolExecutor(max_workers=cls.__CPU_LIMIT) as pool:
                        pool.map(clean_dir, [history_path / str(j) for j in bads])
                return results

            results = [res.result() for res in results]

        logger.warning("No run managed to reach the desired limit. Returning the run with lowest loss.")
        best_loss = min([res[1] for res in results])
        results = [res for res in results if res[1] == best_loss][0]
        if history_path is not None:
            results[-3].data.to_csv(Path(history_path) / "history.csv")
            results[-2].data.to_csv(Path(history_path) / "result.csv")

        # TODO : instability: depending on the number of initial annealers, the solution in the first position
        # TODO : of results is either a column or row vector
        return results

    def __init__(
        self,
        loss: AbstractLoss,
        weights_step_size: Union[float, tuple, list, set, np.ndarray],
        bounds: Optional[Union[tuple, list, set, np.ndarray]] = None,
        init_states: Optional[Union[tuple, list, set, np.ndarray]] = None,
        temp_0: Optional[float] = None,
        temp_min: float = 0,
        alpha: float = 0.85,
        iterations: int = 5000,
        verbose: bool = False,
        stopping_limit: Optional[float] = None,
        cooling_schedule: str = "arithmetic-geometric",
        annealing_type: str = "canonical",
        history_path: Optional[str] = None,
        logger_level=None,
        optimal_step_size=False,
    ):
        """
        Parameters
        ----------
        loss: AbstractLoss
            The loss function to minimize. First and only argument is the np.ndarrays of all weights.
        weights_step_size: Union[float, tuple, list, set, np.ndarray]
            Size of the variation to apply to each weights at each epoch. If a float is given, the same size is used for
            every weight. If a np.ndarray is given, it must have 'dimensions' entries, each entry will be the step size
            of one weight.
        init_states: Union[tuple, list, set, np.ndarray]
            Optional. Initial values of the weights. Will use random values using 'bounds' if not specified.
            If specified, its size defines the number of dimensions. Note that if init_states are not specified,
            then bounds must be, and vice-versa.
        temp_0: float
            Initial temperature. If not specified, will use get_t_max to get it. Useless if using microcanonical
            annealing
        temp_min: float
            Final temperature. Default is 0. Useless if using microcanonical annealing.
        alpha: float
            Leraning rate, default is 0.85. Useless if using microcanonical annealing.
        iterations: int
            Number of iterations to make (default value = 1000)
        verbose: bool (default is False)
        stopping_limit: float
            If specified, the algorithm will stop once the loss has stabilised around differences of less than
            'stopping_limit' (see fit_one method).
        cooling_schedule: str
            The cooling schedule to use. Useless if using microcanonical annealing. Can be :
            * 'logarithmic': T <- alpha / ln(1+T), asymptotically converges towards global minimum, but very slowly
            * 'linear':  T <- T - alpha
            * 'geometric': T <- T * alpha, like 'linear' converges quickly, but is not garanteed to find the best
               solution. Should be close to it however.
            * 'arithmetic-geometric' (default): T <- T * alpha + (T_min * (1 - alpha)), fast and precise
        annealing_type: str
            Can be 'canonical' or 'microcanonical'
        history_path: str
            If specified, fit will be stored here. Must be an existing directory.
        logger_level: str
            Logger level
        optimal_step_size: bool
            If True, weights_step_size is overwritten considering the typical scale fixed by the bounds of the annealer

        The number of iterations will be equal to int((temp_0 - temp_min) / temp_step_size).
        If temp_step_size is not specified, then the number of iterations is equal to 200. (0.5% at each step).
        """
        self.results = None
        if logger_level is not None:
            global logger
            logger.setLevel(logger_level)

        if not isinstance(verbose, bool):
            raise TypeError(f"'verbose' must be a boolean, got {type(verbose)} instead.")
        self.verbose = verbose

        if annealing_type != "canonical" and annealing_type != "microcanonical":
            raise ValueError(f"Unknown annealing type '{annealing_type}'. Can be 'canonical' or 'microcanonical'")
        self.annealing_type = annealing_type

        self.dimensions = None

        if not issubclass(loss.__class__, AbstractLoss):
            raise TypeError(f"The loss function must derive from AbstractLoss, got a {type(loss)} instead")
        if len(inspect.signature(loss).parameters) != 1:
            raise ValueError(f"The loss function must accept exactly 1 parameter")
        self.loss = loss

        if weights_step_size is None:
            raise TypeError("'weights_step_size' can not be None")
        if iterations is None:
            raise TypeError("'iterations' can not be None")

        # noinspection PyUnresolvedReferences
        if hasattr(loss, "constraints") and loss.constraints is not None:
            # parts of bounds will be overwritten
            # noinspection PyUnresolvedReferences
            limits_ = np.array(loss.constraints)
            if bounds is None:
                bounds = limits_
            else:
                # Do not use 'is not None' in order to have an iterable of booleans instead of only one boolean.
                mask = (limits_ != None).astype(int)
                bounds = np.ma.array(bounds, mask=mask).filled(fill_value=limits_)

        else:
            # else .... bounds are the same as in previous version
            pass

        if bounds is None and init_states is None:
            raise ValueError("At least one of 'init_states' and 'bounds' must be specified")

        if bounds is not None and init_states is not None:
            # noinspection PyUnboundLocalVariable
            logger.warning("Specified bounds and init_states. Bounds are then ignored.")

        if init_states is None:
            # Do not use 'is not None' in order to have an iterable of booleans instead of only one boolean.
            assert ~(bounds == None).any()
            bounds = to_array(bounds, "bounds")
            if bounds.ndim != 2 or bounds.shape[1] != 2:
                raise ValueError(f"'bounds' dimension should be (any, 2), got {bounds.shape}")
            self.dimensions = bounds.shape[0]
            for coordinate in range(self.dimensions):
                if bounds[coordinate][0] > bounds[coordinate][1]:
                    raise ValueError(
                        "Bounds are not valid : some lower limits are greater then their upper limits:\n" f"{bounds}"
                    )
            self.bounds = bounds
            self.init_states = None  # set later
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
            self.bounds = None
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

        # Experimental
        if optimal_step_size:
            self._info("optimal_step_size is True: this is experimental.")
            self.weights_step_size = np.abs(self.bounds[:, 0] - self.bounds[:, 1]) ** 2 / 100.

        if temp_0 is not None:
            if isinstance(temp_0, int):
                temp_0 = float(temp_0)
            if not isinstance(temp_0, float):
                raise TypeError(f"'temp_0' must be a float, got {type(temp_0)} instead.")
            if np.isnan(temp_0):
                raise ValueError("'temp_0' can not be NAN")
        self.temp_0 = temp_0

        if temp_min is not None:
            if isinstance(temp_min, int):
                temp_min = float(temp_min)
            if not isinstance(temp_min, float):
                raise TypeError(f"'temp_min' must be a float, got {type(temp_min)} instead.")
            if np.isnan(temp_min):
                raise ValueError("'temp_min' can ont be NAN")
        self.temp_min = temp_min

        if alpha is not None:
            if isinstance(alpha, int):
                alpha = float(alpha)
            if not isinstance(alpha, float):
                raise TypeError(f"'alpha' must be a float, got {type(alpha)} instead.")
            if np.isnan(alpha):
                raise ValueError("'alpha' can ont be NAN")
            if not (0 < alpha <= 1):
                raise ValueError("'alpha' must be between 0 excluded and 1.")
        self.alpha = alpha

        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError(f"Number of iterations must be an integer greater than 0, got {iterations}")
        self.iterations = iterations

        if stopping_limit is not None and (not isinstance(stopping_limit, float) or not 0 < stopping_limit < 1):
            raise ValueError(f"Stopping limit must be a float between 0 and 1, got {stopping_limit}")
        self.stopping_limit = stopping_limit

        if self.iterations < 5000 and self.stopping_limit is not None:
            logger.warning("Iteration should not be less than 5000. Using 5000 instead.")
            self.iterations = 5000

        if cooling_schedule is not None and not isinstance(cooling_schedule, str):
            raise ValueError(f"cooling_schedule must be a str, got {type(cooling_schedule)}")
        if cooling_schedule is not None and cooling_schedule not in Annealer.__POSSIBLE_SCHEDULES:
            raise NotADirectoryError(f"Unknown cooling schedule '{cooling_schedule}'")
        self.cooling_schedule = cooling_schedule

        if history_path is not None and not isinstance(history_path, str):
            raise ValueError(f"history_path must be a str, got {type(history_path)}")
        if history_path is not None and not Path(history_path).is_dir():
            raise NotADirectoryError(f"Output '{history_path}' is not a directory.")
        self.history_path = history_path

        if self.init_states is None:
            self.init_states = generate_init_states(bounds, 1, weights_step_size)

    def _info(self, msg: str):
        """logs 'msg' with INFO level if self.verbose is True"""
        if self.verbose:
            logger.info(msg)

    def _debug(self, msg: str):
        """logs 'msg' with DEBUG level if self.verbose is True"""
        if self.verbose:
            logger.debug(msg)

    def _get_temp_max(self, ar_limit_low=0.79, ar_limit_up=0.81, max_attempts=100, t1_i=1e-5, t2=10.0) -> float:
        """From self.loss, finds the starting temperature.

        Will try to find a temperature T giving a final acceptance ratio AR between 'ar_limit_low'% and 'ar_limit_up'%,
        by running several fits with fixed temperature (alpha=1) for each fit.
        Between two consecutive fits, the function computes the slope (AR_j+1 - AR_j) / (T_j+1 - T_j) and estimates the
        temperature of the next fit that would correspond to a value of (ar_limit_low + ar_limit_up)/2% using this
        slope.
        Stops once the acceptance ratio of the current fit is between 'ar_limit_low'% and 'ar_limit_up'%, or when
        'max_attempts' unsuccessful attempts were made.
        In that case, raises ValueError.

        Parameters
        ----------
        ar_limit_low: float
            Default value = 0.79
        ar_limit_up: float
            Default value = 0.81
        max_attempts: int
            Default value = 100
        t1_i: float
            Temperature of first fit attempt (Default value = 1e-5)
        t2: float
            Temperature of second fit attempt (Default value = 10)

        Returns
        -------
        float
        """

        self._info(f"Looking for starting temperature...")

        if ar_limit_up < ar_limit_low:
            raise ValueError("Acceptance ratio limit up must be greater than Acceptance ratio limit low")
        if not isinstance(max_attempts, int):
            raise TypeError("'max_attempts' must be an integer")
        if max_attempts <= 0:
            raise ValueError("'max_attempts' must be greater than 0")

        if ar_limit_up >= 0.95:
            raise ValueError("Acceptance ratio limit up can not be equal to or greater than 0.95")

        acc_ratio = 0
        attempts = 0
        t1 = t1_i
        acc_ratio_2 = None
        ar_limit_mean = (ar_limit_up + ar_limit_low) / 2.0
        ann = Annealer(
            loss=self.loss,
            weights_step_size=self.weights_step_size,
            init_states=self.init_states,
            temp_0=1,
            temp_min=0,
            alpha=1,
            iterations=100,
            verbose=False,
            stopping_limit=None,
        )
        _, _, acc_ratio_1, _, _, _ = ann.fit(
            temp_0=t1, iterations=1000, stopping_limit=None, verbose=False, searching_t=True
        )

        if ar_limit_low < acc_ratio_1 < ar_limit_up:
            # Lucky strike : t0 is already good !
            acc_ratio_2 = acc_ratio_1
            t2 = t1
        else:
            # Unlucky strike : t1 gives an acc_ratio greater than the upper limit.
            while acc_ratio_1 > ar_limit_up:
                if attempts > max_attempts:
                    raise ValueError(
                        f"Could not find a temperature giving an acceptance ratio between {ar_limit_low} "
                        f"and {ar_limit_up} in less than {max_attempts} attempts"
                    )
                t1 = t1 / 10
                _, _, acc_ratio_1, _, _, _ = ann.fit(
                    temp_0=t1, iterations=1000, stopping_limit=None, verbose=False, searching_t=True
                )
                attempts += 1

            attempts = 0
            while not ar_limit_low < acc_ratio < ar_limit_up:
                if attempts > max_attempts:
                    raise ValueError(
                        f"Could not find a temperature giving an acceptance ratio between {ar_limit_low} "
                        f"and {ar_limit_up} in less than {max_attempts} attempts"
                    )
                _, _, acc_ratio_2, _, _, _ = ann.fit(
                    temp_0=t2, iterations=1000, stopping_limit=None, verbose=False, searching_t=True
                )
                self._info(f"---------------------------------------------")
                self._info(f"Attempt {attempts}")
                self._info(f"t1: {t1}, Acc. ratio : {acc_ratio_1} (fixed)")
                self._info(f"t2: {t2}, Acc. ratio : {acc_ratio_2}\n")

                if ar_limit_low < acc_ratio_2 < ar_limit_up:
                    break

                if acc_ratio_2 > 0.95:
                    t2 = (t2 - t1) / 10
                    attempts += 1
                    continue

                slope = (acc_ratio_2 - acc_ratio_1) / (t2 - t1)
                if slope < 0:
                    self._debug(
                        "Got a negative slope when trying to find starting temperature. Impossible : "
                        "acceptance ratio should be strictly increasing with temperature"
                    )
                    attempts += 1
                    continue
                if slope == 0:
                    self._debug(
                        "Got a null slope when trying to find starting temperature. Impossible : "
                        "acceptance ratio should be strictly increasing with temperature"
                    )
                    attempts += 1
                    continue
                t2 = max([0, (ar_limit_mean - acc_ratio_1) / slope - t1])
                if t2 <= 0:
                    t2 = 2e-16
                attempts += 1

        self._info(f"Found starting temperature t0 = {round(t2, 3)} (acc. ratio = {round(acc_ratio_2, 3)})")
        return t2

    def fit(
        self,
        npoints: int = 1,
        alpha: float = None,
        temp_min: float = None,
        temp_0: float = None,
        iterations: int = None,
        stopping_limit: float = None,
        history_path: str = None,
        stop_at_first_found: bool = False,
        init_states: Union[tuple, list, set, np.ndarray] = None,
        cooling_schedule: Optional[str] = None,
        annealing_type: Optional[str] = None,
        verbose: bool = True,
        searching_t: bool = False,
    ) -> Union[
        List[Tuple[np.ndarray, float, float, Sampler, Sampler, bool]],
        Tuple[np.ndarray, float, float, Sampler, Sampler, bool],
    ]:
        """Try to find 'npoints' local minima of self.loss by running simulated annealing.

        'npoints' defines the number of starting initial states to use. If it is greater than 1 and if 'stop_at_first'
        if True, will return the first fit that is successful (see self.fit_one for definition of 'successful').

        All the other arguments are detailed in self.__init__ and/or self._fit_one.

        See Annealer._fit_many for meaning of returns.
        """

        if annealing_type is None:
            annealing_type = self.annealing_type
        if annealing_type != "canonical" and annealing_type != "microcanonical":
            raise ValueError(f"Unknown annealing type '{annealing_type}'. Can be 'canonical' or 'microcanonical'")

        if annealing_type == "microcanonical":
            raise NotImplementedError

        if alpha is None:
            alpha = self.alpha

        if temp_min is None:
            temp_min = self.temp_min

        if temp_0 is None:
            temp_0 = self.temp_0

        if iterations is None:
            iterations = self.iterations

        if stopping_limit is None:
            stopping_limit = self.stopping_limit

        if history_path is None:
            history_path = self.history_path

        if cooling_schedule is None:
            cooling_schedule = self.cooling_schedule

        if verbose and cooling_schedule == "canonical":
            self._info(f"Fitting with cooling schedule '{cooling_schedule}'")

        if init_states is not None:
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

        if npoints == 1:
            initialisation = self.init_states if init_states is None else init_states
            if not searching_t:
                self.loss.on_fit_start(initialisation)
            if annealing_type == "canonical":
                results = self._fit_one_canonical(
                    alpha,
                    temp_min,
                    temp_0,
                    iterations,
                    stopping_limit,
                    history_path,
                    initialisation,
                    cooling_schedule,
                )
            else:
                results = self._fit_one_micronanonical(iterations, stopping_limit, history_path, init_states)

        else:
            if self.bounds is None:
                raise ValueError(
                    "If you want the annealing to start more than one initial states, bounds must be specified."
                )
            if history_path is not None:
                history_path = Path(history_path)
            if init_states is None:
                init_states = generate_init_states(self.bounds, npoints, self.weights_step_size)
            elif len(init_states) != npoints:
                raise ValueError(
                    f"Asked to find {npoints} local minima, but specified only {len(init_states)} initial states."
                )
            if history_path is not None:
                [(history_path / str(i)).mkdir() for i in range(npoints) if not (history_path / str(i)).is_dir()]
            if not searching_t:
                self.loss.on_fit_start(tuple(init_states))
            annealers = [
                Annealer(
                    loss=self.loss,
                    weights_step_size=self.weights_step_size,
                    init_states=init_states[i],
                    bounds=self.bounds,
                    temp_0=temp_0,
                    temp_min=temp_min,
                    alpha=alpha,
                    iterations=iterations,
                    verbose=self.verbose,
                    stopping_limit=stopping_limit,
                    cooling_schedule=cooling_schedule,
                    annealing_type=annealing_type,
                    history_path=str(history_path / str(i)) if history_path is not None else None,
                )
                for i in range(npoints)
            ]
            # noinspection PyUnboundLocalVariable
            # TODO : the research for the start temperature can be done once for all annealers
            results = Annealer._fit_many(
                annealers,
                stop_at_first_found=stop_at_first_found,
                history_path=history_path,
            )

        self.results = results
        if not searching_t:
            self.loss.on_fit_end(results)
        return results

    def _get_next_temperature(
        self,
        temp: float,
        alpha: float,
        method: str,
        temp_min: Optional[float] = None,
        t: Optional[int] = None,
    ) -> float:
        if method == "logarithmic":
            if alpha <= 0:
                ValueError("If using logarithmic cooling schedule, alpha must be a positive number")
            if t is None:
                raise ValueError("Is using logarithmic cooling schedule, t must be specified")
            return self._logarithmic_cooling(t, alpha)
        elif method == "geometric":
            if not 0 <= alpha <= 1:
                ValueError("If using geometric cooling schedule, alpha must be a positive number lower than 1")
            return self._geometric_cooling(temp, alpha)
        elif method == "linear":
            if alpha <= 0:
                ValueError("If using linear cooling schedule, alpha must be a positive number")
            return self._linear_cooling(temp, alpha)
        elif method == "arithmetic-geometric":
            if temp_min is None:
                raise ValueError("Is using arithmetic-geometric cooling schedule, temp_min must be specified")
            if not 0 <= alpha <= 1:
                ValueError(
                    "If using arithmetic-geometric cooling schedule, alpha must be a positive number lower than 1"
                )
            return self._arithmetic_geometric_cooling(temp, alpha, temp_min)
        else:
            ValueError(f"Unknown cooling schedule '{method}'")

    @staticmethod
    def _logarithmic_cooling(temp, alpha):
        return alpha / (math.log(1 + temp))

    @staticmethod
    def _geometric_cooling(temp, alpha):
        return temp * alpha

    @staticmethod
    def _linear_cooling(temp, alpha):
        return max(temp - alpha, 0)

    @staticmethod
    def _arithmetic_geometric_cooling(temp, alpha, temp_min):
        return temp * alpha + (temp_min * (1 - alpha))

    def _take_step(self, curr):
        unit_v = np.random.uniform(size=(1, self.dimensions))
        unit_v = unit_v / np.linalg.norm(unit_v)
        assert np.isclose(np.linalg.norm(unit_v), 1.0)
        cov = np.zeros((curr.shape[0], curr.shape[0]), float)
        np.fill_diagonal(cov, self.weights_step_size)
        candidate = np.random.multivariate_normal(mean=curr.ravel(), cov=cov).reshape(curr.shape)

        candidate_loss = self.loss(candidate)
        if not isinstance(candidate_loss, (int, float)):
            raise ValueError(f"Return of loss function should be a number, got {type(candidate_loss)}")
        return candidate, candidate_loss

    def _fit_one_micronanonical(
        self,
        iterations: Optional[float] = None,
        stopping_limit: Optional[float] = None,
        history_path: Optional[Union[str, Path]] = None,
        init_states: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float, Sampler, Sampler, bool]:
        """Microcanonical annealing"""

        if iterations is None:
            iterations = self.iterations

        if stopping_limit is None:
            stopping_limit = self.stopping_limit

        if history_path is None:
            history_path = self.history_path

        if init_states is None:
            init_states = self.init_states
        init_states = init_states.reshape(-1, 1)

        if iterations < 5000 and stopping_limit is not None:
            logger.warning(
                "Outer loop iterations should not be less than 5000 if using a stopping limit." " Using 5000 instead."
            )
            iterations = 5000

        if stopping_limit is not None and not 0 < stopping_limit < 1:
            raise ValueError("'limit' should be between 0 and 1")

        if iterations is None or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Number of outer iterations must be an integer greater than 0")

        curr = init_states.copy()
        curr_loss = self.loss(curr)
        while hasattr(curr_loss, "__len__") and len(curr_loss) == 1:
            curr_loss = curr_loss[0]
        if not isinstance(curr_loss, (int, float)):
            raise ValueError(f"Return of loss function should be a number, got {type(curr_loss)}")

        history = Sampler()

        to_print = make_counter(iterations)

        points_accepted = 0
        acc_ratio = None
        loss_for_finishing = None
        n_finishing = 0
        n_finishing_max = int(iterations / 1000)
        finishing_history = Sampler()
        finishing = False
        finished = False
        prev_loss = None
        demon_loss = 0

        for i_ in range(iterations):
            candidate, candidate_loss = self._take_step(curr)

            diff = candidate_loss - curr_loss
            accepted = diff < 0 or diff <= demon_loss
            if accepted:
                points_accepted = points_accepted + 1
                prev_loss = curr_loss
                curr, curr_loss = candidate, candidate_loss
                demon_loss -= diff
                self._debug(f"Accepted : {i_} f({candidate}) = {candidate_loss}")
            else:
                self._debug(f"Rejected :{i_} f({candidate}) = {candidate_loss}")

            acc_ratio = float(points_accepted) / float(i_ + 1)
            sample = SamplePoint(
                weights=candidate.T[0],
                iteration=i_,
                accepted=accepted,
                loss=candidate_loss,
                demon_loss=demon_loss,
                acc_ratio=acc_ratio,
            )
            history.append(sample)

            if i_ in to_print and to_print[i_] is not None:
                self._info(
                    f"step {to_print[i_]}, Demon loss : {round(demon_loss, 3)} | acc. ratio : {round(acc_ratio, 3)}"
                    f" | loss : {round(candidate_loss, 3)}"
                )

            """
            Checks stopping conditions
            """
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
                        self._info(f"Variation of loss is small enough after step {i_}, stopping")
                        curr, curr_loss, acc_ratio, last_index = finish(finishing_history)
                        # noinspection PyProtectedMember
                        finishing_history = Sampler(finishing_history._data.iloc[[last_index]])
                        finished = True
                        break
                else:
                    loss_for_finishing = None
                    finishing = False
                    n_finishing = 0
                    finishing_history.clean()

        self._info(f"Final demon loss : {round(demon_loss, 3)}")
        self._info(f"Acc. ratio : {round(acc_ratio, 3)}")

        if not finished and not history.data.empty:
            # noinspection PyProtectedMember
            finishing_history = Sampler(history._data.iloc[[-1]])
        if history_path is not None:
            history.data.to_csv(Path(history_path) / "history.csv")
            finishing_history.data.to_csv(Path(history_path) / "result.csv")

        return curr, curr_loss, acc_ratio, history, finishing_history, finished

    def _fit_one_canonical(
        self,
        alpha: Optional[float] = None,
        temp_min: Optional[float] = None,
        temp_0: Optional[float] = None,
        iterations: Optional[float] = None,
        stopping_limit: Optional[float] = None,
        history_path: Optional[Union[str, Path]] = None,
        init_states: Optional[np.ndarray] = None,
        cooling_schedule: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float, Sampler, Sampler, bool]:
        """Canonical annealing

        Try to find one local minimum of self.loss by running simulated annealing.
        Starts at 'temp_0', finishes at 'temp_min', in 'iterations' steps.
        Each step decreases the temperature according to cooling_schedule.

        Stopping criterion:
        -------------------

        ratio = abs(loss(i) - loss(i-1) - 1) must be lower than 'stopping_limit'.

        CONSIDERS LOSSES OF ACCEPTED AND REJECTED POINTS.

        If so, remember loss(i-1) as loss_for_finishing and let the program run for n_finishing_max more iterations.
        At each of those iterations, check that ratio < stopping_limit. n_finishing_max is one thousandth of the
        number of iterations.

        If not, forget loss_for_finishing, forget how many iterations we did since ratio was first lower than
        stopping_limit, and continue decreasing temperature.

        If it is, continue until n_finishing_max is reached or until ratio >= stopping_limit.

        If n_finishing_max is reached, stop the algorithm. The returned weights are those matching the minimal value
        of the loss among the n_finishing_max previous losses.

        parameters
        ----------
        alpha: Optional[float]
        temp_min: Optional[float]
        temp_0: Optional[float]
        iterations: Optional[float]
        stopping_limit: Optional[float]
        history_path: Optional[Union[str, Path]]
        init_states: Optional[np.ndarray]
        cooling_schedule: Optional[str]

        All parameters are optional.
        Those which are not specified will default to the values specified in self.__init__.

        Returns
        -------
        Tuple[np.ndarray, float, float, Sampler, Sampler, bool]
        The weights of local minimum, its corresponding loss, the final acceptance ratio, the history of the fit in
        the form of a Sampler object, another Sampler object containing only the point corresponding to the local
        minimum, and True if the stopping limit was reached, else False.

        """

        if alpha is None:
            alpha = self.alpha

        if temp_min is None:
            temp_min = self.temp_min

        if alpha is None:
            raise TypeError("'alpha' can not be None")
        if temp_min is None:
            raise TypeError("'temp_min' can not be None")

        if temp_0 is None:
            if self.temp_0 is None:
                self.temp_0 = self._get_temp_max()
            temp_0 = self.temp_0

        if iterations is None:
            iterations = self.iterations

        if stopping_limit is None:
            stopping_limit = self.stopping_limit

        if history_path is None:
            history_path = self.history_path

        if init_states is None:
            init_states = self.init_states
        # loss callables work with vertical vectors, while Annealer class works with horizontal vectors
        init_states = init_states.reshape(-1, 1)

        if cooling_schedule is None:
            cooling_schedule = self.cooling_schedule

        if iterations < 5000 and stopping_limit is not None:
            logger.warning("Iteration should not be less than 5000 if using a stopping limit. Using 5000 instead.")
            iterations = 5000

        if stopping_limit is not None and not 0 < stopping_limit < 1:
            raise ValueError("'limit' should be between 0 and 1")

        if temp_min is None or temp_min < 0:
            raise ValueError("'tmin' must be a float greater than or equal to 0")
        if temp_0 is None or temp_0 <= temp_min:
            raise ValueError(f"'t0' must be a float greater than tmin, got {temp_0} <= {temp_min}")
        if iterations is None or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Number of iterations must be an integer greater than 0")

        self._info(f"Starting temp : {round(temp_0, 3)}")
        temp = temp_0
        curr = init_states.copy()
        curr_loss = self.loss(curr)
        while hasattr(curr_loss, "__len__") and len(curr_loss) == 1:
            curr_loss = curr_loss[0]
        if not isinstance(curr_loss, (int, float)):
            raise ValueError(f"Return of loss function should be a number, got {type(curr_loss)}")

        history = Sampler()

        to_print = make_counter(iterations)

        points_accepted = 0
        acc_ratio = None
        loss_for_finishing = None
        n_finishing = 0
        n_finishing_max = int(iterations / 1000)
        finishing_history = Sampler()
        finishing = False
        finished = False
        prev_loss = None

        for i_ in range(iterations):
            candidate, candidate_loss = self._take_step(curr)

            diff = candidate_loss - curr_loss
            accepted = diff < 0
            if accepted:
                points_accepted = points_accepted + 1
                prev_loss = curr_loss
                curr, curr_loss = candidate, candidate_loss
                self._debug(f"Accepted : {i_} f({candidate}) = {candidate_loss}")
            else:
                metropolis = np.exp(-diff / temp)
                if np.random.uniform() < metropolis:
                    accepted = True
                    points_accepted = points_accepted + 1
                    prev_loss = curr_loss
                    curr, curr_loss = candidate, candidate_loss
                    self._debug(f"Accepted : {i_} f({candidate}) = {candidate_loss}")
                else:
                    self._debug(f"Rejected :{i_} f({candidate}) = {candidate_loss}")

            acc_ratio = float(points_accepted) / float(i_ + 1)
            sample = SamplePoint(
                weights=candidate.T[0],
                iteration=i_,
                accepted=accepted,
                loss=candidate_loss,
                temp=temp,
                acc_ratio=acc_ratio,
            )
            history.append(sample)

            temp = self._get_next_temperature(temp, alpha, cooling_schedule, temp_min, i_ + 1)

            if i_ in to_print and to_print[i_] is not None:
                self._info(
                    f"step {to_print[i_]}, Temperature : {round(temp, 3)} | acc. ratio : {round(acc_ratio, 3)}"
                    f" | loss = {round(candidate_loss, 3)}"
                )

            """
            Checks stopping conditions
            """
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
                        self._info(f"Variation of loss is small enough after step {i_}, stopping")
                        curr, curr_loss, acc_ratio, last_index = finish(finishing_history)
                        # noinspection PyProtectedMember
                        finishing_history = Sampler(finishing_history._data.iloc[[last_index]])
                        finished = True
                        break
                else:
                    loss_for_finishing = None
                    finishing = False
                    n_finishing = 0
                    finishing_history.clean()

        self._info(f"Final temp : {round(temp, 3)}")
        self._info(f"Acc. ratio : {round(acc_ratio, 3)}")

        if not finished and not history.data.empty:
            # noinspection PyProtectedMember
            finishing_history = Sampler(history._data.iloc[[-1]])
        if history_path is not None:
            history.data.to_csv(Path(history_path) / "history.csv")
            finishing_history.data.to_csv(Path(history_path) / "result.csv")

        return curr, curr_loss, acc_ratio, history, finishing_history, finished

    def plot(
        self,
        sampler_path: Union[str, Path, Tuple[Sampler, Sampler]],
        axisfontsize: int = 15,
        step_size: int = 1,
        nweights: int = 10,
        weights_names: Optional[list] = None,
        do_3d: bool = False,
    ) -> Union[Tuple[plt.Figure, list], None]:
        """From a directory containing 'result.csv' and 'history.csv', produces plots.
        Will produce the file "annealing.pdf" in 'sampler_path' and return the corresponding Figure object.
        If subfolders themselves containing 'result.csv' and 'history.csv' are present, will plot will call itself on
        them too.
        In the plot, will show the first 'nweights'.

        'sampler_path' can also be a tuple of two Sampler objects, the first should then be the full history of the fit
         and the second the point of the local minimum.

        Parameters
        ----------
            sampler_path: Union[str, Path, Tuple[Sampler, Sampler]]
                Either the path to the directory containing the annealing result, or two Sampler objects
            axisfontsize: int
                default value = 15
            step_size: int
                plots every 'step_size' iterations instead of all of them (default value = 1)
            nweights: int
                Number of weights to display in plots (default value = 10)
            weights_names: Optional[list]
                List of names of weights, for axis labels. If None, weights are named using their position index.
            do_3d: bool
                Either do or not do 3-dim plot of the annealer evolution


        Returns
        -------
        Union[Tuple[plt.Figure, go.Figure, list], None]
            Returns None if the given directory does not contain history.csv or result.csv. Otherwise returns the
             created figure and the weights and loss of the local minimum.
        """

        points = []

        if isinstance(sampler_path, (str, Path)):
            if isinstance(sampler_path, str):
                sampler_path = Path(sampler_path)

            for directory in sampler_path.glob("*"):
                if not directory.is_dir():
                    continue
                try:
                    int(directory.stem)
                except ValueError:
                    continue
                points.append(
                    self.plot(
                        directory,
                        axisfontsize,
                        step_size,
                        nweights,
                        weights_names,
                        do_3d,
                    )[-1]
                )

            logger.info(f"Plotting annealing results from {sampler_path}...")

            sampler = sampler_path / "history.csv"
            final_sampler = sampler_path / "result.csv"
            if not sampler.is_file():
                logger.warning(f"No file 'history.csv' found in '{sampler_path}'")
                return None
            if not final_sampler.is_file():
                logger.warning(f"No file 'result.csv' found in '{sampler_path}'")
                return None
            sampler = Sampler(pd.read_csv(sampler, index_col=0))
            final_sampler = Sampler(pd.read_csv(final_sampler, index_col=0))

        else:
            sampler, final_sampler = sampler_path

        weights = sampler.weights
        if nweights is None:
            nweights = 0
        else:
            if nweights == "all":
                nweights = len(weights.columns)
            else:
                nweights = min(nweights, len(weights.columns))

        weights = weights.iloc[::step_size, :nweights].values
        losses = sampler.losses.iloc[::step_size].values
        iterations = sampler.iterations.values[::step_size]
        acc_ratios = sampler.acc_ratios.iloc[::step_size].values
        temps = sampler.parameters.values[::step_size]
        accepted = sampler.accepted.values[::step_size]

        final_weights = final_sampler.weights.iloc[0, :nweights].values
        final_loss = final_sampler.losses.values

        grid = GridSpec(
            3 + nweights,
            6,
            left=0.05,
            right=0.9,
            bottom=0.03,
            top=0.97,
            hspace=0.3,
            wspace=0.5,
        )
        fig = plt.figure(figsize=(22, 3 * (nweights + 3)))

        first_ax = fig.add_subplot(grid[0, :])
        second_ax = fig.add_subplot(grid[1, :])
        third_ax = fig.add_subplot(grid[2, :])
        first_ax.set_xlabel("Iterations", fontsize=axisfontsize)
        first_ax.set_ylabel("Temp", fontsize=axisfontsize)
        second_ax.set_xlabel("Iterations", fontsize=axisfontsize)
        second_ax.set_ylabel("Acc. ratio", fontsize=axisfontsize)
        third_ax.set_xlabel("Iterations", fontsize=axisfontsize)
        third_ax.set_ylabel("Loss", fontsize=axisfontsize)
        first_ax.grid(True, ls="--", lw=0.2, alpha=0.5)
        second_ax.grid(True, ls="--", lw=0.2, alpha=0.5)
        third_ax.grid(True, ls="--", lw=0.2, alpha=0.5)
        cmap = plt.get_cmap("inferno")

        conditioned_marker = ["o" if a else "x" for a in accepted]
        mscatter(
            iterations,
            temps,
            ax=first_ax,
            m=conditioned_marker,
            c=temps,
            cmap=cmap,
            norm=LogNorm(),
            s=7,
        )
        mscatter(
            iterations,
            acc_ratios,
            ax=second_ax,
            m=conditioned_marker,
            c=temps,
            cmap=cmap,
            norm=LogNorm(),
            s=7,
        )
        im = mscatter(
            iterations,
            losses,
            ax=third_ax,
            m=conditioned_marker,
            c=temps,
            cmap=cmap,
            norm=LogNorm(),
            s=7,
        )
        third_ax.plot([iterations[0], iterations[-1]], [final_loss[-1], final_loss[-1]], c="black")
        third_ax.text(iterations[0], final_loss[-1], s=f"{round(final_loss[-1], 3)}", c="black")
        fig.subplots_adjust(right=0.8)

        add_colorbar(fig, im, first_ax, axisfontsize)
        add_colorbar(fig, im, second_ax, axisfontsize)
        add_colorbar(fig, im, third_ax, axisfontsize)

        for iplot in range(0, nweights):
            ax1 = fig.add_subplot(grid[iplot + 3, 0:5])
            ax2 = fig.add_subplot(grid[iplot + 3, 5])
            ax1.grid(True, ls="--", lw=0.2, alpha=0.5)
            ax2.grid(True, ls="--", lw=0.2, alpha=0.5)
            ax1.set_xlabel("Iterations")
            ax1.set_ylabel(f"Weights {iplot if weights_names is None else weights_names[iplot]}")
            ax2.set_ylabel("Loss")
            ax2.set_xlabel(f"Weight {iplot if weights_names is None else weights_names[iplot]}")

            mscatter(
                iterations,
                weights[:, iplot],
                ax=ax1,
                m=conditioned_marker,
                s=7,
                c=temps,
                cmap=cmap,
                norm=LogNorm(),
            )
            ax1.plot(
                [iterations[0], iterations[-1]],
                [final_weights[iplot], final_weights[iplot]],
                c="black",
            )
            ax1.text(
                iterations[0],
                final_weights[iplot],
                s=f"{round(final_weights[iplot], 3)}",
                c="black",
            )
            mscatter(
                weights[:, iplot],
                losses,
                ax=ax2,
                m=conditioned_marker,
                s=7,
                c=temps,
                cmap=cmap,
                norm=LogNorm(),
            )

            if len(points) > 0:
                for point in points:
                    ax2.scatter(point[0][iplot], point[1], s=10, c="blue")
            ax2.scatter(final_weights[iplot], final_loss, s=10, c="red")

            add_colorbar(fig, im, ax2, axisfontsize)

        fig.savefig(str(sampler_path / "annealing.pdf"))

        if do_3d:

            i_couples = list(itertools.combinations(range(nweights), 2))

            def objective_2d(i_, j_, wi, wj):
                ann_solution = self.results[0].reshape(-1, 1).copy()
                ann_solution[i_] = wi
                ann_solution[j_] = wj
                return self.loss(ann_solution)

            # TODO : it should work, but it does not. Understand why
            # objective_couples = [
            #     lambda wi, wj: objective_2d(ii, jj, wi, wj) for (ii, jj) in i_couples
            # ]

            # TODO : parallelise
            for i, (col, row) in enumerate(i_couples):

                specs = [[{"type": "surface"}]]

                fig_3d = make_subplots(rows=1, cols=1, specs=specs)

                fig_3d.update_yaxes(title_text=weights_names[row], row=1, col=1)
                fig_3d.update_xaxes(title_text=weights_names[col], row=1, col=1)

                explored_w_y = weights[:, row]
                explored_w_x = weights[:, col]

                w_y = np.linspace(np.min(explored_w_y), np.max(explored_w_y), 100)
                w_x = np.linspace(np.min(explored_w_x), np.max(explored_w_x), 100)

                domain = pd.DataFrame(data=np.zeros((100, 100)), index=w_y, columns=w_x)
                for wy in domain.index:
                    for wx in domain.columns:
                        domain.loc[wy, wx] = objective_2d(col, row, wx, wy)

                fig_3d.add_trace(
                    go.Surface(
                        z=domain.values,
                        y=domain.index,
                        x=domain.columns,
                        colorscale="Blues",
                        showscale=False,
                        opacity=0.5,
                    ),
                    row=1,
                    col=1,
                )

                z_explored = np.zeros_like(temps)
                for k in range(len(temps)):
                    z_explored[k] = objective_2d(col, row, explored_w_x[k], explored_w_y[k])

                tickvals = np.arange(np.floor(np.log10(np.min(temps))), np.ceil(np.log10(np.max(temps))))
                ticktext = [str(10 ** val) for val in tickvals]

                fig_3d.add_scatter3d(
                    # for some reason, need to transpose
                    x=explored_w_x,
                    y=explored_w_y,
                    z=z_explored,
                    mode="markers",
                    marker=dict(
                        size=1.2,
                        color=np.log10(temps),
                        symbol=list(map(lambda val: "x" if val else "circle", accepted)),
                        showscale=True,
                        colorbar=dict(tickvals=tickvals, ticktext=ticktext),
                        colorscale='inferno'
                    ),
                    row=1,
                    col=1,
                )

                # TODO : add title to colorbar
                fig_3d.add_scatter3d(
                    # for some reason, need to transpose
                    x=[self.results[0].reshape(-1, 1)[col][0]],
                    y=[self.results[0].reshape(-1, 1)[row][0]],
                    z=[self.results[1]],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color="red",
                        symbol="circle",
                    ),
                    row=1,
                    col=1,
                )

                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title=weights_names[col],
                        yaxis_title=weights_names[row],
                    )
                )

                fig_3d.write_html(
                    str(sampler_path / f"3d_visualisation_{weights_names[col]}_{weights_names[row]}.html")
                )

        return fig, [final_weights, final_loss]


def finish(sampler: Sampler) -> Tuple[np.ndarray, float, float, int]:
    """From a given history in the form of a Sampler object, finds the point with lowest loss and returns the
    corresponding weights, loss, acceptance ratio and position in the index of the Sampler."""
    data = sampler.data
    mask = data["loss"].drop_duplicates(keep="last").index
    data = data.loc[mask]
    # noinspection PyUnresolvedReferences
    data = data.loc[(data["loss"] == data["loss"].min()).values]
    return (
        data["weights"].iloc[-1],
        data["loss"].iloc[-1],
        data["acc_ratio"].iloc[-1],
        data.index[-1],
    )


def generate_init_states(
    bounds: np.ndarray, npoints: int, step_size: Union[int, np.ndarray]
) -> Union[Tuple[np.ndarray, np.ndarray], List[np.ndarray], np.ndarray]:
    """From given bounds, generates npoints initial states.

    'bounds' must be of the form
    [
        [low_bound, high_bound],
        [low_bound, high_bound],
        [low_bound, high_bound],
        ...,
        [low_bound, high_bound]
    ]
    with one tuple [low_bound, high_bound] per dimension of the problem.

    If npoints is lower or equal to 0, raises ValueError.
    If npoints is one, will generate a random point in the allowed space.
    If npoints is 2, will generate one point at the lowest bounds and 1 point at the highest bounds.
    If npoints <= 2**ndims, will generate one point per vertices defined by 'bounds' until npoints are generated.
    If npoints > 2**ndims, will generate one point at each vertex defined by 'bounds' and npoints - 2**ndims
    random points in the allowed space.
    """
    if npoints <= 0:
        raise ValueError("npoints must be greater than 0.")
    if npoints == 1:
        return bounds[:, 0] + step_size + np.random.uniform(size=(1, len(bounds))) * (bounds[:, 1] - bounds[:, 0])
    if npoints == 2:
        return bounds[:, 0] + step_size, bounds[:, 1] - step_size
    dim = bounds.shape[0]
    zeros = np.zeros(shape=(dim,))
    ones = np.ones(shape=(dim,))
    l1 = np.concatenate((zeros, ones))
    l2 = np.concatenate((ones, zeros))
    init_states_indexes = np.array(
        list(distinct_combinations(l1, dim)) + list(distinct_combinations(l2, dim))[1:-1],
        dtype=np.uint8,
    )
    if npoints < init_states_indexes.shape[0]:
        to_keep = np.random.permutation(np.arange(0, init_states_indexes.shape[0]))
        to_keep = to_keep[:npoints]
        init_states_indexes = init_states_indexes[to_keep]

    init_states = []
    for init_state_index in init_states_indexes:
        init_states.append([])
        for i in range(len(init_state_index)):
            index = init_state_index[i]
            point = bounds[i, index]
            if index == 0:
                point += step_size if isinstance(step_size, float) else step_size[i]
            else:
                point -= step_size if isinstance(step_size, float) else step_size[i]
            init_states[-1].append(point)

    if npoints > init_states_indexes.shape[0]:
        for _ in range(npoints - len(init_states)):
            init_states.append(
                (bounds[:, 0] + step_size + np.random.uniform(size=(1, len(bounds))) * (bounds[:, 1] - bounds[:, 0]))[
                    0
                ].tolist()
            )
    return np.array(init_states)


def mscatter(x, y, ax=None, m=None, **kw):
    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def add_colorbar(fig, im, ax, axisfontsize):

    cbar_ax = fig.add_axes(
        [
            ax.get_position().xmax + 0.01,
            ax.get_position().ymin,
            0.025,
            ax.get_position().ymax - ax.get_position().ymin,
        ]
    )
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.yaxis.labelpad = 15
    cbar.set_label("Temperature", rotation=270, fontsize=axisfontsize)
