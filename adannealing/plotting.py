import math
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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


def plot(sampler: Sampler, axisfontsizes=15, step_size=1, nweights: int = 10):
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
    nweights = min(nweights, len(weights.columns))

    weights = weights.iloc[::step_size, :nweights]
    weights = weights - weights.mean()
    xlims = (weights.index[0], weights.index[-1])
    ylims = (min(weights.values.flatten()), max(weights.values.flatten()))
    xticks = np.linspace(xlims[0], xlims[1], 10)
    yticks = np.linspace(ylims[0], ylims[1], 10)
    nxticks, nyticks = len(xticks), len(yticks)

    xstep, ystep = 15 / (100 * nweights), 7 / (10 * nweights)
    theta = math.atan(ystep / xstep)
    fig2, axes2 = plt.subplots(1, nweights, figsize=(15, 7))
    pos1 = axes2[0].get_position()
    axes2[-1].set_xlabel("Iterations")
    axes2[-1].set_ylabel("Centered weights")
    for iplot in range(0, nweights):
        i = nweights - iplot - 1

        shift = math.atan(math.sqrt((xstep * i) ** 2 + (ystep * i) ** 2))
        xshift = math.atan(xstep * i)
        print(xshift, xstep * i)
        axes2[iplot].set_position(
            [
                pos1.x0 + math.atan(xstep * i),
                pos1.y0 + math.atan(ystep * i),
                pos1.width * nweights - math.atan(xstep * i),
                pos1.height - math.atan(ystep * i),
            ]
        )

        axes2[iplot].spines["right"].set_visible(False)
        axes2[iplot].spines["top"].set_visible(False)
        axes2[iplot].plot(
            weights.index,
            weights.iloc[:, iplot],
            c=pick_color(iplot, nweights),
            lw=2,
            alpha=0.75,
            # marker=pick_ms(iplot, nweights),
        )
        axes2[iplot].set_xlim(xlims)
        axes2[iplot].set_ylim(ylims)
        axes2[iplot].set_xticks(xticks)
        axes2[iplot].set_yticks(yticks)
        if iplot != nweights - 1:
            axes2[iplot].set_xticklabels([""] * nxticks)
            axes2[iplot].set_yticklabels([""] * nyticks)

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
