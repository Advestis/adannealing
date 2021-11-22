import pytest
import numpy as np

from adannealing import Annealer


def wrong_loss(x, y) -> float:
    return (x ** 2 + y ** 2) ** 2 - (x ** 2 + y ** 2)


def loss_func(w) -> float:
    x = w[0]
    y = w[1]
    return (x ** 2 + y ** 2) ** 2 - (x ** 2 + y ** 2)


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "loss,weights_step_size,bounds,dimensions,init_states,temp_step_size,temp_0,temp_min,verbose,"
    "expected_error_type,expected_error_message,iterations",
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
            "Loss must be callable",
            None
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
            None
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
            "weights_step_size can not be None",
            None
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
            ValueError,
            "At least one of 'dimensions' and 'init_states' must be specified",
            None
        ),
        (
            loss_func,
            np.array([1]),
            None,
            2,
            None,
            None,
            None,
            None,
            None,
            ValueError,
            "Shape of 'weights_step_size' should be",
            None
        ),
        (
            loss_func,
            np.array([1, 1]),
            None,
            2,
            None,
            None,
            None,
            None,
            None,
            ValueError,
            "At least one of",
            None
        ),
        (
            loss_func,
            np.array([1, 1]),
            np.array([-10, 10]),
            2,
            None,
            None,
            None,
            None,
            None,
            ValueError,
            "Shape of 'bounds' should be",
            None
        ),
        (
            loss_func,
            np.array([1, 1]),
            np.array([(-10, 10), (-10, 10)]),
            2,
            None,
            None,
            None,
            None,
            None,
            TypeError,
            "'temp_min' can not be None",
            None
        ),
        (
            loss_func,
            np.array([1]),
            np.array([(-10, 10), (-10, 10)]),
            2,
            None,
            None,
            None,
            None,
            None,
            ValueError,
            "Shape of 'weights_step_size'",
            None
        ),
        (
            loss_func,
            1,
            np.array([(-10, 10), (-10, 10)]),
            2,
            None,
            None,
            20,
            0,
            True,
            None,
            "",
            None
        ),
        (
            loss_func,
            (1, 1),
            np.array([(-10, 10), (-10, 10)]),
            2,
            None,
            None,
            20,
            0,
            True,
            None,
            "",
            None
        ),
        (
            loss_func,
            np.array((1, 1)),
            np.array([(-10, 10), (-10, 10)]),
            2,
            None,
            None,
            20,
            0,
            True,
            None,
            "",
            None
        ),
        (
            loss_func,
            [1, 1],
            np.array([(-10, 10), (-10, 10)]),
            2,
            None,
            None,
            20,
            0,
            True,
            None,
            "",
            None
        ),
        (
            loss_func,
            [1, np.nan],
            np.array([(-10, 10), (-10, 10)]),
            2,
            None,
            None,
            20,
            0,
            True,
            ValueError,
            "can not contain NANs",
            None
        ),
        (
            loss_func,
            np.nan,
            np.array([(-10, 10), (-10, 10)]),
            2,
            None,
            None,
            20,
            0,
            True,
            ValueError,
            "can not be NAN",
            None
        ),
        (
            loss_func,
            [1, 1],
            np.array([(-10, 10), (-10, 10)]),
            2,
            2,
            None,
            20,
            0,
            True,
            ValueError,
            "dimensions but init_states has",
            None
        ),
        (
            loss_func,
            [1, 1],
            np.array([(-10, 10), (-10, 10)]),
            2,
            [0, 0],
            None,
            20,
            0,
            True,
            None,
            "",
            None
        ),
        (
            loss_func,
            [1, 1],
            np.array([(-10, 10), (-10, 10)]),
            2,
            (0, 0),
            None,
            20,
            0,
            True,
            None,
            "",
            None
        ),
        (
            loss_func,
            [1, 1],
            np.array([(-10, 10), (-10, 10)]),
            2,
            np.array([0, 0]),
            None,
            20,
            0,
            True,
            None,
            "",
            None
        ),
        (
            loss_func,
            [1, 1],
            np.array([(-10, 10), (-10, 10)]),
            2,
            np.array([0, 0]),
            200,
            20,
            0,
            True,
            ValueError,
            "larger than ",
            None
        ),
        (
            loss_func,
            [1, 1],
            np.array([(-10, 10), (-10, 10)]),
            2,
            np.array([0, 0]),
            20,
            200,
            0,
            True,
            None,
            "",
            10
        )
     ]
)
def test_init(
        loss,
        weights_step_size,
        bounds,
        dimensions,
        init_states,
        temp_step_size,
        temp_0,
        temp_min,
        verbose,
        expected_error_type,
        expected_error_message,
        iterations
):
    if expected_error_type is not None:
        with pytest.raises(expected_error_type) as e:
            _ = Annealer(
                loss,
                weights_step_size,
                bounds,
                dimensions,
                init_states,
                temp_step_size,
                temp_0,
                temp_min,
                verbose
            )
        assert expected_error_message in str(e.value)
    else:
        ann = Annealer(
            loss,
            weights_step_size,
            bounds,
            dimensions,
            init_states,
            temp_step_size,
            temp_0,
            temp_min,
            verbose
        )
        assert isinstance(ann.weights_step_size, np.ndarray)
        assert ann.weights_step_size.dtype == float
        assert isinstance(ann.bounds, np.ndarray)
        assert ann.bounds.dtype == float
        assert isinstance(ann.init_states, np.ndarray)
        assert ann.init_states.dtype == float
        assert isinstance(ann.temp_step_size, float)
        assert isinstance(ann.temp_0, float)
        assert isinstance(ann.temp_min, float)
        assert isinstance(ann.verbose, bool)
        assert isinstance(ann.dimensions, int)
        assert isinstance(ann.iterations, int)
        if iterations is not None:
            assert ann.iterations == iterations
