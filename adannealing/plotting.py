from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
from matplotlib.colors import LogNorm


def colorline(x, y, z, cmap=plt.get_cmap("terrain"), linewidth=3, alpha=1.0, ax=None):
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=linewidth, alpha=alpha)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


class Sampler:
    def __init__(self, data: pd.DataFrame = None):
        if data is not None:
            self._data = data
            self.points = None
        else:
            self._data = pd.DataFrame()
            self.points = []

    def append(self, value):
        if self.points is None:
            raise ValueError("Sampler was initialised with an outside history : can not add more points.")
        self.points.append(value)

    def __len__(self):
        if self.points is None:
            return len(self._data.index)
        else:
            return len(self.points)

    def _process(self):
        if self.points is None:
            raise ValueError("Sampler was initialised with an outside history : nothing to process.")
        self._data = pd.DataFrame(
            [[p.weights, p.iteration, p.acc_ratio, p.accepted, p.loss, p.temp] for p in self.points],
            columns=["weights", "iteration", "acc_ratio", "accepted", "loss", "temp"],
        )

    @property
    def weights(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data.loc[:, "weights"]

    @property
    def iteration(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data.loc[:, "iteration"]

    @property
    def acc_ratio(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data.loc[:, "acc_ratio"]

    @property
    def accepted(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data.loc[:, "accepted"]

    @property
    def loss(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data.loc[:, "loss"]

    @property
    def temp(self):
        if self._data.empty or len(self._data.index) != len(self):
            self._process()
        return self._data.loc[:, "temp"]

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

    def __init__(self, weights, iteration, acc_ratio, accepted, loss, temp, sampler: Union[None, Sampler] = None):
        if sampler is not None and not isinstance(sampler, Sampler):
            raise TypeError(f"Sampler must be of type 'Sampler', got {type(sampler)}")
        self.weights = weights
        self.iteration = iteration
        self.acc_ratio = acc_ratio
        self.accepted = accepted
        self.loss = loss
        self.temp = temp
        if sampler is not None:
            sampler.append(self)


def plot(sampler: Sampler, axisfontsizes=15, step_size=1):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[1].set_xlabel("Iterations", fontsize=axisfontsizes)
    axes[0].set_ylabel("Acc. ratio", fontsize=axisfontsizes)
    axes[1].set_ylabel("Loss", fontsize=axisfontsizes)
    axes[0].grid(True, ls="--", lw=0.2, alpha=0.5)
    axes[1].grid(True, ls="--", lw=0.2, alpha=0.5)
    cmap = plt.get_cmap("inferno")
    iterations = sampler.iterations.values[::step_size]
    temps = sampler.temp.values[::step_size]
    axes[0].scatter(iterations, sampler.acc_ratio.values[::step_size], c=temps, cmap=cmap, s=2, norm=LogNorm())
    im = axes[1].scatter(iterations, sampler.loss.values[::step_size], c=temps, cmap=cmap, s=2, norm=LogNorm())
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.11, 0.03, 0.77])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.yaxis.labelpad = 15
    cbar.set_label('Temperature', rotation=270, fontsize=axisfontsizes)
    return fig
