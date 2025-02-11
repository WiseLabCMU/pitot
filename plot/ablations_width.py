"""Margin width / tightness ablations."""

import numpy as np
from matplotlib import pyplot as plt

import plot


def _plot_ablations(
    fmt: str, experiments: dict[str, str], linestyle=['-', '--', ':', '-.']
):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    data = {x: dict(np.load(fmt.format(x))) for x in experiments}

    for (x, v), ls in zip(experiments.items(), linestyle):
        y = data[x]["mf_width"][4].T
        y2 = (
            data[x]["if2_width"][4].T + data[x]["if3_width"][4].T
            + data[x]["if4_width"][4].T) / 3
        plot.plot_errorbar(axs[0], np.arange(10), y, label=v, linestyle=ls)
        plot.plot_errorbar(axs[1], np.arange(10), y2, label=v, linestyle=ls)

    axs[-1].legend()
    axs[0].set_title("Without Interference")
    axs[1].set_title("With Interference")
    axs[0].set_ylabel("Bound Tightness")
    for ax in axs:
        ax.grid()
        ax.set_xlabel("Miscoverage Rate $\\varepsilon$")
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_xticklabels([0.1, 0.08, 0.06, 0.04, 0.02])
    return fig, axs


fig, axs = _plot_ablations(
    "summary/conformal/{}.npz", {
        "optimal": "Pitot", "naive": "Naive CQR",
        "nonquantile": "Non-quantile"})
fig.tight_layout()
fig.savefig("figures/ablation_cqr.pdf")


fig, axs = _plot_ablations(
    "summary/{}.npz", {
        "conformal/optimal": "Pitot",
        "baseline/monolith": "Neural Network",
        "baseline/attention": "Attention",
        "baseline/factorization": "Matrix Factorization"
    })
fig.tight_layout()
fig.savefig("figures/baseline_width.pdf")
