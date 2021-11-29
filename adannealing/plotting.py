from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec


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
            [[p.weights, p.iteration, p.acc_ratio, p.accepted, p.loss, p.temp] for p in self.points],
            columns=["weights", "iteration", "acc_ratio", "accepted", "loss", "temp"],
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
    def temps(self):
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


def plot(
    sampler_path: Union[str, tuple],
    axisfontsizes=15,
    step_size=1,
    nweights: int = 10,
):
    if isinstance(sampler_path, str):
        sampler_path = Path(sampler_path)
        sampler = sampler_path / "history.csv"
        final_sampler = sampler_path / "result.csv"
        if not sampler.is_file():
            raise FileNotFoundError(f"No file 'history.csv' found in '{sampler_path}'")
        if not final_sampler.is_file():
            raise FileNotFoundError(f"No file 'result.csv' found in '{sampler_path}'")
        sampler = Sampler(pd.read_csv(sampler, index_col=0))
        final_sampler = Sampler(pd.read_csv(final_sampler, index_col=0))
    else:
        sampler, final_sampler = sampler_path

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[1].set_xlabel("Iterations", fontsize=axisfontsizes)
    axes[0].set_ylabel("Acc. ratio", fontsize=axisfontsizes)
    axes[1].set_ylabel("Loss", fontsize=axisfontsizes)
    axes[0].grid(True, ls="--", lw=0.2, alpha=0.5)
    axes[1].grid(True, ls="--", lw=0.2, alpha=0.5)
    cmap = plt.get_cmap("inferno")
    iterations = sampler.iterations.values[::step_size]
    temps = sampler.temps.values[::step_size]
    axes[0].scatter(iterations, sampler.acc_ratios.values[::step_size], c=temps, cmap=cmap, norm=LogNorm())
    im = axes[1].scatter(iterations, sampler.losses.values[::step_size], c=temps, cmap=cmap, norm=LogNorm())
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.11, 0.03, 0.77])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.yaxis.labelpad = 15
    cbar.set_label("Temperature", rotation=270, fontsize=axisfontsizes)

    if nweights == 0 or nweights is None:
        return fig
    weights = sampler.weights
    losses = sampler.losses
    final_weights = final_sampler.weights
    final_loss = final_sampler.losses
    nweights = min(nweights, len(weights.columns))

    weights = weights.iloc[::step_size, :nweights]
    losses = losses.iloc[::step_size]
    final_weights = final_weights.iloc[0, :nweights]

    grid = GridSpec(nweights, 6, left=0.05, right=0.95, bottom=0.03, top=0.97, hspace=0.3, wspace=0.5)
    fig2 = plt.figure(figsize=(22, 3 * nweights))

    for iplot in range(0, nweights):
        ax1 = fig2.add_subplot(grid[iplot, 0:5])
        ax2 = fig2.add_subplot(grid[iplot, 5])
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel(f"Weights {iplot}")
        ax2.set_ylabel("Loss")
        ax2.set_xlabel(f"Weight {iplot}")

        ax1.scatter(
            weights.index,
            weights.iloc[:, iplot],
            s=7,
            c=temps,
            cmap=cmap,
            norm=LogNorm(),
        )
        ax1.plot(
            [weights.index[0], weights.index[-1]], [final_weights.iloc[iplot], final_weights.iloc[iplot]], c="black"
        )
        ax1.text(weights.index[0], final_weights.iloc[iplot], s=f"{round(final_weights.iloc[iplot], 3)}", c="black")
        ax2.scatter(weights.iloc[:, iplot], losses, s=7, c=temps, cmap=cmap, norm=LogNorm())
        ax2.scatter(final_weights.iloc[iplot], final_loss, s=10, c="red")
    return fig, fig2
