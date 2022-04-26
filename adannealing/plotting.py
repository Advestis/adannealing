from typing import Union
import pandas as pd
import numpy as np
import ast
import logging

logger = logging.getLogger(__name__)


class Sampler:
    """A class used by Annealer to keep track of its progress."""

    @staticmethod
    def _manage_data(data: pd.DataFrame) -> pd.DataFrame:
        def _handle_weights(weights: str) -> list:
            while "  " in weights:
                weights = weights.replace("  ", " ")
            return ast.literal_eval(weights.replace(" ", ",").replace("[,", "["))

        expected_columns = ["weights", "iteration", "acc_ratio", "accepted", "loss", "parameter"]
        for col in expected_columns:
            if col not in data:
                raise IndexError(f"Missing columns '{col}'")
        if data.empty:
            return data
        if type(data["weights"].iloc[0]) == str:
            data.loc[:, "weights"] = data.loc[:, "weights"].apply(_handle_weights)
        return data

    def __init__(self, data: pd.DataFrame = None):
        if data is not None:
            self._data = Sampler._manage_data(data)
            self.points = None
        else:
            self._data = pd.DataFrame()
            self.points = []

    def append(self, value):
        if self.points is None:
            raise ValueError("Sampler was initialised with an outside history : can not add more points.")
        self.points.append(value)

    def clean(self):
        self._data = pd.DataFrame()
        self.points = []

    def __len__(self):
        if self.points is None:
            return len(self._data.index)
        else:
            return len(self.points)

    def _process(self):
        if self.points is None:
            raise ValueError("Sampler was initialised with an outside history : nothing to process.")
        self._data = pd.DataFrame(
            [[p.weights, p.iteration, p.acc_ratio, p.accepted, p.loss, p.parameter] for p in self.points],
            columns=["weights", "iteration", "acc_ratio", "accepted", "loss", "parameter"],
        )

    @property
    def weights(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return pd.DataFrame(index=self._data.index, data=np.array([w for w in self._data.loc[:, "weights"].values]))

    @property
    def acc_ratios(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data.loc[:, "acc_ratio"]

    @property
    def accepted(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data.loc[:, "accepted"]

    @property
    def losses(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data.loc[:, "loss"]

    @property
    def parameters(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data.loc[:, "parameter"]

    @property
    def iterations(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data.index

    @property
    def data(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data


class SamplePoint:
    """A class used by Annealer to keep track of its progress."""

    def __init__(
        self,
        weights,
        iteration,
        acc_ratio,
        accepted,
        loss,
        temp=None,
        demon_loss=None,
        sampler: Union[None, Sampler] = None,
    ):
        if sampler is not None and not isinstance(sampler, Sampler):
            raise TypeError(f"Sampler must be of type 'Sampler', got {type(sampler)}")
        self.weights = weights
        self.iteration = iteration
        self.acc_ratio = acc_ratio
        self.accepted = accepted
        self.loss = loss
        self.temp = temp
        self.demon_loss = demon_loss

        if self.demon_loss is None and self.temp is None:
            raise ValueError("One and only one of temp and demon_loss must be specified")
        if self.demon_loss is not None and self.temp is not None:
            raise ValueError("One and only one of temp and demon_loss must be specified")

        if sampler is not None:
            sampler.append(self)

    @property
    def parameter(self):

        if self.demon_loss is None and self.temp is None:
            raise ValueError("One and only one of temp and demon_loss must be specified")
        if self.demon_loss is not None and self.temp is not None:
            raise ValueError("One and only one of temp and demon_loss must be specified")

        if self.temp is not None:
            return self.temp
        else:
            return self.demon_loss
