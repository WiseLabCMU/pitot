"""Plot full margin ablations."""

import numpy as np
from matplotlib import pyplot as plt

import plot


def _plot_ablations(
    ax, fmt: str, experiments: dict[str, str],
    linestyle=['-', '--', ':', '-.'], idx: int = 0, mode='mf'
):
    data = {x: dict(np.load(fmt.format(x))) for x in experiments}

    for (x, v), ls in zip(experiments.items(), linestyle):
        if mode == 'mf':
            y = data[x]["mf_width"][idx].T
        else:
            y = (
                data[x]["if2_width"][idx].T + data[x]["if3_width"][idx].T
                + data[x]["if4_width"][idx].T) / 3
        plot.plot_errorbar(ax, np.arange(10), y, label=v, linestyle=ls)


for mode in ['mf', 'if']:
    fig, axs = plt.subplots(3, 3, figsize=(7, 7))
    for i, ax in enumerate(axs.reshape(-1)):
        _plot_ablations(
            ax, "summary/{}.npz", {
                "conformal/optimal": "Pitot",
                "baseline/monolith": "Neural Network",
                "baseline/attention": "Attention",
                "baseline/factorization": "Matrix Factorization"
            }, idx=i, mode=mode)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_xticklabels([0.1, 0.08, 0.06, 0.04, 0.02])
        ax.set_title("{}% Train Split".format(10 * (i + 1)))
        ax.grid(visible=True)
    fig.tight_layout()
    axs[-1, 0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.15), ncols=4)
    fig.savefig(
        'figures/baseline_width_{}.pdf'.format(mode), bbox_inches='tight')
