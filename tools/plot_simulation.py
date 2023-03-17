"""Plot simulation results."""

import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick


_desc = "Plot simulation results."


def _parse(p):
    p.add_argument("-o", "--out", help="Output directory.", default="figures")
    p.add_argument(
        "-p", "--path",
        help="Results directory.", default="results-simulation")
    return p


def _plot_result(ax, path):

    def _load(method):
        base_dir = os.path.join(path, method)
        splits = os.listdir(base_dir)
        splits = sorted([float(x.split(".")[0]) for x in splits])

        mean = []
        stderr = []
        for s in splits:
            fname = str(int(s)) + ".npz"
            oracle = np.load(os.path.join(path, "oracle", fname))
            npz = np.load(os.path.join(base_dir, fname))
            rel = (np.mean(npz["latency"] / oracle["latency"], axis=1))
            mean.append(np.mean(rel))
            stderr.append(np.sqrt(np.var(rel, ddof=1) / (rel.shape[0] - 1)))

        return np.array(splits), np.array(mean), np.array(stderr)

    methods = {
        "pitot": "Pitot",
        "mlp": "Single Network",
        "linear": "Linear Factorization",
    }
    res = {m: _load(m) for m in methods}

    styles = ['.-', '.:', '.--', '.-.']
    for (k, (splits, mean, stderr)), fmt in zip(res.items(), styles):
        ax.errorbar(
            splits, mean, yerr=stderr * 2,
            capsize=2, fmt=fmt, label=methods[k])


def _main(args):

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    for ax, margin in zip(axs, ["p1", "p5"]):
        _plot_result(ax, os.path.join(args.path, margin))
        ax.axhline(1.0, linestyle='--', label='Oracle', color='black')
        ax.grid()
        ax.set_xlabel("Number of Tasks")
    axs[0].set_ylabel("Relative Latency")
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axs[1].legend()

    axs[0].set_title("99% Certainty")
    axs[1].set_title("95% Certainty")

    for ax in axs:
        ax.set_ylim(0.9, 2.75)

    for tick in axs[1].yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(args.out, "simulation.png"), dpi=400)
