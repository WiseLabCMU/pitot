"""Plot for embedded-only error."""

import json
import os
import numpy as np
from jax import numpy as jnp
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from prediction import Dataset, Objective


_desc = "Plot error for only embedded devices."


def _parse(p):
    p.add_argument("-p", "--path", default="results", help="Path to results.")
    p.add_argument(
        "-d", "--data", default="data/data.npz", help="Dataset directory.")
    return p


def _load_split(ds, method):

    with open(method + ".json") as f:
        config = json.load(f)

    _obj = config["config"]["objectives"][0]
    objective = Objective.from_config(ds, _obj)
    result = np.load(method + ".npz")

    def index(idx):
        return idx[ds.x[idx][:, 0] == 47]

    means = []
    for res, split in zip(result["C_hat"], result["mf_test"]):
        _filtered = index(split)
        means.append(objective.mape(jnp.array(res), jnp.array(_filtered)))
    return np.mean(means), np.sqrt(np.var(means, ddof=1))


def _load(ds, method):

    splits = sorted([
        float(x.replace(".npz", "")) for x in os.listdir(method)
        if x.endswith(".npz")])
    mean, stderr = list(zip(*[
        _load_split(ds, os.path.join(method, str(s))) for s in tqdm(splits)]))
    return np.array(mean).tolist(), np.array(stderr).tolist()


def _main(args):

    methods = {
        "embedding/128": "Pitot",
        "baseline/mlp": "Single Network",
        "linear/128": "Linear Factorization",
        "baseline/device_mlp": "Per-Device",
    }

    ds = Dataset.from_npz(args.data)
    data = {
        method: _load(ds, os.path.join(args.path, method))
        for method in methods
    }
    splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    for (k, (mean, stderr)), fmt in zip(data.items(), ['.-', '.:', '.--']):
        ax.errorbar(
            splits, mean, yerr=stderr, label=methods[k], capsize=3, fmt=fmt)
    ax.grid()
    ax.legend()
    ax.set_xlabel("Training Split")
    ax.set_ylabel("Mean Absolute Error (Embedded)")
    ax.set_ylim(0, 1.2)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout(pad=0.5)
    fig.savefig("figures/comparisons_d.png", dpi=400)
