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


def _main(args):

    def _load(method):
        base_dir = os.path.join(args.path, method)
        splits = os.listdir(base_dir)
        splits = sorted([float(x.split(".")[0]) for x in splits])

        mean = []
        stderr = []
        for s in splits:
            fname = str(int(s)) + ".npz"
            oracle = np.load(os.path.join(args.path, "oracle", fname))
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

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    styles = ['.-', '.:', '.--', '.-.']
    for (k, (splits, mean, stderr)), fmt in zip(res.items(), styles):
        ax.errorbar(
            splits, mean, yerr=stderr * 2,
            capsize=2, fmt=fmt, label=methods[k])
    ax.axhline(1.0, linestyle='--', label='Oracle', color='black')
    ax.grid()
    ax.set_xlabel("Number of Tasks")
    ax.set_ylabel("Relative Latency")
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout(pad=0.5)

    fig.savefig(os.path.join(args.out, "simulation.png"), dpi=400)
