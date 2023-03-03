"""Plot interference histograms."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from libsilverline import ArgumentParser
from forecast import Dataset


def plot_interference(
        baseline="data.npz", data="summary.csv", out="",
        mode="hist", left=-25.0, right=150.0):
    """Plot interference histograms by runtime.

    Parameters
    ----------
    baseline : str
        Path to non-interference baseline dataset.
    data : str
        Path to interference dataset (CSV). The file extension `.csv` is
        appended if not present.
    out : str
        Where to save plot. If empty, uses the same base path as data.
    mode : str
        Plot type; hist (standard) or trace (for debugging sessions mostly)
    left : float
        Left xlim.
    right : float
        Right xlim.
    """
    if not data.endswith(".csv"):
        data = data + ".csv"

    ds = Dataset(baseline)
    df = pd.read_csv(data)
    df["percent"] = 100 * (np.exp(df["diff"]) - 1)
    print("Total:", len(df))
    print("Unique pairs:", len(df["module"].unique()))

    fig, axs = plt.subplots(5, 4, figsize=(16, 16))

    for rt, ax in zip(ds.runtimes, axs.reshape(-1)):
        y = np.array(df[df['runtime'] == rt]['percent'])
        ax.set_title(rt)

        if mode == "hist":
            ax.hist(y, bins=np.linspace(left, right, 50), density=True)
            ax.axvline(0, color='black', linestyle='--')
            ax.set_yticks([])

        else:
            ax.plot(y)

    fig.tight_layout()

    if out == "":
        out = data.replace(".csv", ".png")
    fig.savefig(out)


def _parse():
    p = ArgumentParser()
    p.add_to_parser(
        "interference", plot_interference, "interference",
        aliases={"baseline": ["-b"], "data": ["-d"], "out": ["-o"]})
    return p


def _main(args):
    plot_interference(**args["interference"])


if __name__ == '__main__':
    _main(_parse().parse_args())
