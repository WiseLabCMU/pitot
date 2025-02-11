"""Hyperparameter Ablations."""

import numpy as np
from beartype.typing import Any
from matplotlib import pyplot as plt

import plot


def _plot_ablations(
    axs, fmt: str, experiments: list[Any], selected: Any = 32,
    key: str = "mape", idx: int = -1, label: str = ""
) -> None:
    data = {x: dict(np.load(fmt.format(x))) for x in experiments}
    keys = ['_'.join([x, key]) for x in ["mf", "if2", "if3", "if4"]]

    for ax, k in zip(axs, keys):
        for x in experiments:
            y = data[x][k]
            if idx >= 0:
                y = y[:, :, idx]
            y[y > 1000] = np.nan
            plot.plot_errorbar(
                ax, np.arange(9), y, label=label.format(x),
                linestyle="-" if x == selected else ":")
        plot.format_xsplits(ax)


fig, axs = plt.subplots(4, 4, figsize=(12, 11))


_plot_ablations(
    axs[0], "summary/learned/{}.npz", [0, 1, 2, 4, 8], 1,
    label="$q={}$")
axs[0, 0].set_ylabel("Learned Features", fontsize=12)

_plot_ablations(
    axs[1], "summary/embedding/{}.npz", [4, 8, 16, 32, 64], 32,
    label="$r={}$")
axs[1, 0].set_ylabel("Embedding", fontsize=12)

_plot_ablations(
    axs[2], "summary/interference/{}.npz", [1, 2, 4, 8, 16], 2,
    label="$s={}$")
axs[2, 0].set_ylabel("Interference Types", fontsize=12)

_plot_ablations(
    axs[3], "summary/weight/{}.npz", [0.1, 0.2, 0.5, 1.0, 2.0], 0.5,
    label="$\\beta={}$")
axs[3, 0].set_ylabel("Interference Weight", fontsize=12)

for ax in axs[:, -1]:
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
axs[0, 0].set_title("No Interference")
axs[0, 1].set_title("2-way Interference")
axs[0, 2].set_title("3-way Interference")
axs[0, 3].set_title("4-way Interference")

fig.tight_layout()
fig.savefig("figures/hyperparameters.pdf")
