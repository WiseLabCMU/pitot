"""Draw figures."""

import json
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os


_desc = "Draw ablation figures used in the paper."


def _parse(p):
    p.add_argument(
        "-f", "--figures", nargs='+', default=["matrix", "interference"],
        help="Figures to draw.")
    p.add_argument("-o", "--out", help="Output directory.", default="figures")
    p.add_argument(
        "-p", "--path", help="Results directory.", default="results")
    p.add_argument("-i", "--dpi", help="Out image DPI.", default=400, type=int)
    return p


def _load(p):
    with open(p) as f:
        return json.load(f)


def _plot(
    ax, pattern: str, values: list, labels: list = None, relative=None,
    format: list[str] = ['.-', '.:', '.--', '.-.'], key="mf", legend=True,
    baseline=True
) -> None:
    labels = values if labels is None else labels
    data = {v: _load(pattern.format(v)) for v in values}

    if relative is not None:
        if relative == "baseline":
            norm = np.array(data[values[0]]["baseline_mean"])
        else:
            norm = np.array(data[relative][key]["mean"])
    else:
        norm = 1.0

    if baseline:
        ax.errorbar(
            np.array(data[values[0]]["splits"]),
            np.array(data[values[0]][key]["baseline_mean"]) / norm,
            yerr=np.array(data[values[0]][key]["baseline_std"]) / norm,
            label="Baseline", capsize=2, fmt=format[-1])

    for v, label, fmt in zip(values, labels, format):
        ax.errorbar(
            np.array(data[v]["splits"]),
            np.array(data[v][key]["mean"]) / norm,
            yerr=np.array(data[v][key]["std"]) / norm * 2 / 5, label=label,
            capsize=2, fmt=fmt)

    if legend:
        ax.legend()
    ax.grid()


class Figures:
    """Paper figures."""

    @staticmethod
    def matrix(args):
        """Ablation plots."""
        fig, axs = plt.subplots(1, 2, figsize=(6, 4))
        _plot(
            axs[0], args.path + "/embedding/{}/summary.json",
            [32, 64, 128, 256, 512],
            labels=['$r=32$', '$r=64$', '$r=128$', '$r=256$', '$r=256$'],
            relative=512,
            format=['.-', '.:', '.--', '.-.', 'x-'])
        _plot(
            axs[1], args.path + "/features/{}/summary.json",
            [0, 2, 4, 8, 16], relative=4,
            labels=['$q=0$', '$q=2$', '$q=4$', '$q=8$', '$q=16$'],
            format=['.-', '.:', 'x-', '.--', '.-.'])

        axs[0].set_ylabel("Relative Error")
        axs[0].set_title("Embedding Dimension")
        axs[1].set_title("Learned Features")

        for ax in axs:
            ax.set_xlabel("Training Split")
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        fig.tight_layout()
        return {"ablations_mf": fig}

    @staticmethod
    def interference(args):
        """Ablation plots for interference types."""
        fig, axs = plt.subplots(1, 2, figsize=(6, 4))

        _plot(
            axs[0], args.path + "/interference/{}/summary.json",
            [1, 2, 3, 4], labels=['$s=1$', '$s=2$', '$s=3$', '$s=4$'],
            relative=2, format=['.-', 'x:', '.--', '.-.'])
        _plot(
            axs[1], args.path + "/interference/{}/summary.json",
            [1, 2, 3, 4], labels=['$s=1$', '$s=2$', '$s=3$', '$s=4$'],
            relative=2, format=['.-', 'x:', '.--', '.-.'], key='if')

        axs[0].set_ylabel("Relative Error")

        axs[0].set_title("Non-interference Error")
        axs[1].set_title("Interference Error")
        fig.tight_layout()
        return {"ablations_if": fig}


def _main(args):
    for name in args.figures:
        figs = getattr(Figures, name)(args)
        for k, v in figs.items():
            out = os.path.join(args.out, k + ".png")
            print("Created:", out)
            v.savefig(out, dpi=args.dpi)
