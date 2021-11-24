import pytest
import numpy as np

from adannealing import Annealer


def wrong_loss(x, y) -> float:
    return (x ** 2 + y ** 2) ** 2 - (x ** 2 + y ** 2)


def loss_func(w) -> float:
    x = w[0]
    return (x - 5) * (x - 2) * (x - 1) * x


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "loss,"
    "weights_step_size,"
    "bounds,"
    "init_states,"
    "temp_0,"
    "temp_min,"
    "alpha,"
    "iterations,"
    "verbose,"
    "expected_error_type,"
    "expected_error_message",
    [
        (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            TypeError,
            "The loss function must be callable",
        ),
        (
            wrong_loss,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ValueError,
            "The loss function must accept exactly one parameter",
        ),
        (
            loss_func,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            TypeError,
            "'weights_step_size' can not be None",
        ),
        (
            loss_func,
            np.array([1]),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            TypeError,
            "'alpha' can not be None",
        ),
        (
            loss_func,
            np.array([1]),
            None,
            None,
            None,
            None,
            0.85,
            None,
            None,
            TypeError,
            "'temp_min' can not be None",
        ),
        (
            loss_func,
            np.array([1]),
            None,
            None,
            None,
            0,
            0.85,
            None,
            None,
            TypeError,
            "'iterations' can not be None",
        ),
        (
            loss_func,
            np.array([1]),
            None,
            None,
            None,
            0,
            0.85,
            1000,
            True,
            ValueError,
            "At least one of 'init_states' and 'bounds' must be specified",
        ),
        (
            loss_func,
            np.array([1, 1]),
            np.array([-10, 10]),
            None,
            None,
            0,
            0.85,
            1000,
            None,
            ValueError,
            "'bounds' dimension should be (any, 2), got ",
        ),
        (
            loss_func,
            np.array([1, 1]),
            np.array([[-10, 10, 0]]),
            None,
            None,
            0,
            0.85,
            1000,
            None,
            ValueError,
            "'bounds' dimension should be (any, 2), got ",
        ),
        (
            loss_func,
            np.array([1, 1]),
            np.array([[-10, 10]]),
            None,
            None,
            0,
            0.85,
            1000,
            None,
            ValueError,
            "Shape of 'weights_step_size' should be (1,)",
        ),
        (
            loss_func,
            1,
            np.array([(-10, 10), (-10, 10)]),
            None,
            20,
            0,
            0.85,
            1000,
            True,
            None,
            "",
        ),
        (
            loss_func,
            (1, 1),
            np.array([(-10, 10), (-10, 10)]),
            None,
            20,
            0,
            0.85,
            1000,
            True,
            None,
            "",
        ),
        (
            loss_func,
            np.array((1, 1)),
            np.array([(-10, 10), (-10, 10)]),
            None,
            20,
            0,
            0.85,
            1000,
            True,
            None,
            "",
        ),
        (
            loss_func,
            [1, 1],
            np.array([(-10, 10), (-10, 10)]),
            None,
            20,
            0,
            0.85,
            1000,
            True,
            None,
            "",
        ),
        (
            loss_func,
            [1, np.nan],
            np.array([(-10, 10), (-10, 10)]),
            None,
            20,
            0,
            0.85,
            1000,
            True,
            ValueError,
            "can not contain NANs",
        ),
        (
            loss_func,
            np.nan,
            np.array([(-10, 10), (-10, 10)]),
            None,
            20,
            0,
            0.85,
            1000,
            True,
            ValueError,
            "can not be NAN",
        ),
        (
            loss_func,
            1,
            np.array([(-10, 10), (-10, 10)]),
            np.array([0.2]),
            20,
            0,
            0.85,
            1000,
            True,
            ValueError,
            "Dimension of 'bounds' is ",
        ),
        (
            loss_func,
            1,
            np.array([(-10, 10), (-10, 10)]),
            np.nan,
            20,
            0,
            0.85,
            1000,
            True,
            ValueError,
            "'init_states' can not be NAN",
        ),
        (
            loss_func,
            1,
            np.array([(-10, 10), (-10, 10)]),
            np.array([(-10, 10), (-10, 10)]),
            20,
            0,
            0.85,
            1000,
            True,
            ValueError,
            "'init_states' must be a 1-D numpy array",
        ),
        (
            loss_func,
            1,
            np.array([(-10, 10), (-10, 10)]),
            [0, 0],
            20,
            0,
            0.85,
            1000,
            True,
            None,
            "",
        ),
        (
            loss_func,
            1,
            np.array([(-10, 10), (-10, 10)]),
            (0, 0),
            20,
            0,
            0.85,
            1000,
            True,
            None,
            "",
        ),
        (
            loss_func,
            1,
            np.array([(-10, 10), (-10, 10)]),
            np.array([0, 0]),
            20,
            0,
            0.85,
            1000,
            True,
            None,
            "",
        ),
     ]
)
def test_init(
        loss,
        weights_step_size,
        bounds,
        init_states,
        temp_0,
        temp_min,
        alpha,
        iterations,
        verbose,
        expected_error_type,
        expected_error_message,
):
    if expected_error_type is not None:
        with pytest.raises(expected_error_type) as e:
            _ = Annealer(
                loss,
                weights_step_size,
                bounds,
                init_states,
                temp_0,
                temp_min,
                alpha,
                iterations,
                verbose,
            )
        assert expected_error_message in str(e.value)
    else:
        ann = Annealer(
            loss,
            weights_step_size,
            bounds,
            init_states,
            temp_0,
            temp_min,
            alpha,
            iterations,
            verbose,
        )
        assert isinstance(ann.weights_step_size, np.ndarray)
        assert ann.weights_step_size.dtype == float
        assert isinstance(ann.bounds, np.ndarray)
        assert ann.bounds.dtype == float
        assert isinstance(ann.init_states, np.ndarray)
        assert ann.init_states.dtype == float
        assert isinstance(ann.temp_0, float)
        assert isinstance(ann.temp_min, float)
        assert isinstance(ann.alpha, float)
        assert isinstance(ann.verbose, bool)
        assert isinstance(ann.dimensions, int)
        assert isinstance(ann.iterations, int)


# def test_fit():
#     ann = Annealer(
#         loss=loss_func,
#         weights_step_size=0.1,
#         bounds=np.array([[0, 6]]),
#         init_states=np.array([0]),
#         verbose=True
#     )
#     ann.fit()
