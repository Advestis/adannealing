from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import colors as cs
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


mss = list([m for m, v in Line2D.markers.items() if v != "nothing" and isinstance(m, str)])
lmss = len(mss)
colors = [c for c in list(cs.CSS4_COLORS.keys()) if "white" not in c]
colors.remove("aliceblue")
colors.remove("lavender")
colors.remove("honeydew")
colors.remove("lemonchiffon")
colors.remove("linen")
colors.remove("mistyrose")
colors.remove("palegoldenrod")
colors.remove("aqua")
colors.remove("cyan")
colors.remove("lavenderblush")
colors.remove("lightyellow")
colors.remove("moccasin")
colors.remove("aquamarine")
colors.remove("lawngreen")
colors.remove("azure")
colors.remove("cornsilk")
colors.remove("lightgoldenrodyellow")
colors.remove("beige")
colors.remove("bisque")
colors.remove("oldlace")
colors.remove("peachpuff")
lc = len(colors)


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


def plot(sampler_path: Union[str, tuple], axisfontsizes=15, step_size=1, nweights: int = 10):
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
    temps = sampler.temp.values[::step_size]
    axes[0].scatter(iterations, sampler.acc_ratio.values[::step_size], c=temps, cmap=cmap, norm=LogNorm())
    im = axes[1].scatter(iterations, sampler.loss.values[::step_size], c=temps, cmap=cmap, norm=LogNorm())
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.11, 0.03, 0.77])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.yaxis.labelpad = 15
    cbar.set_label("Temperature", rotation=270, fontsize=axisfontsizes)

    if nweights == 0 or nweights is None:
        return fig
    weights = sampler.weights
    final_weights = final_sampler.weights
    nweights = min(nweights, len(weights.columns))

    weights = weights.iloc[::step_size, :nweights]
    final_weights = final_weights.iloc[0, :nweights]

    fig2, axes2 = plt.subplots(nweights, 1, figsize=(15, 2 * nweights))
    for iplot in range(0, nweights):
        axes2[iplot].set_xlabel("Iterations")
        axes2[iplot].set_ylabel("Weights")

        axes2[iplot].scatter(
            weights.index,
            weights.iloc[:, iplot],
            s=7,
            c=temps,
            cmap=cmap,
            norm=LogNorm(),
        )
        axes2[iplot].plot(
            [weights.index[0], weights.index[-1]], [final_weights.iloc[iplot], final_weights.iloc[iplot]], c="black"
        )
        axes2[iplot].text(
            weights.index[0], final_weights.iloc[iplot], s=f"{round(final_weights.iloc[iplot], 3)}", c="black"
        )
    return fig, fig2


def pick_ms(n, nmax):
    return pick(n, nmax, mss, lmss)


def pick_color(n, nmax):
    return pick(n, nmax, colors, lc)


def pick(n, nmax, iterable, literable=None):
    if literable is None:
        literable = len(iterable)
    if n >= literable:
        while n >= literable:
            n = n - literable
        return iterable[n]
    item = int(literable / nmax) * n
    if item == nmax:
        item = 0
    return iterable[item]
