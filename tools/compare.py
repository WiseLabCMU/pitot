"""Draw figures."""

import json
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os


_desc = "Draw comparison figures used in the paper."


def _parse(p):
    p.add_argument(
        "-f", "--figures", nargs='+',
        default=["comparison", "ablations_mf", "ablations_if", "interference"],
        help="Figures to draw.")
    p.add_argument("-o", "--out", help="Output directory.", default="figures")
    p.add_argument(
        "-p", "--path", help="Results directory.", default="results")
    p.add_argument("-i", "--dpi", help="Out image DPI.", default=400, type=int)
    return p


def _load(p):
    with open(p) as f:
        return json.load(f)


def _sharey(axs):
    for ax in axs[1:]:
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)


def _fmt_axes(fig, axs):
    for ax in axs:
        ax.set_xlabel("Training Split")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()


def _plot(
    ax, values: list, labels: list = None, relative=None, pattern="results/{}",
    format: list[str] = ['.-', '.:', '.--', '.-.'], key="mf", legend=True,
    baseline=None
) -> None:
    labels = values if labels is None else labels
    data = {
        v: _load(os.path.join(pattern.format(v), "summary.json"))
        for v in values}

    if relative is not None:
        if relative == "baseline":
            norm = np.array(data[values[0]]["baseline_mean"])
        else:
            norm = np.array(data[relative][key]["mean"])
    else:
        norm = 1.0

    for v, label, fmt in zip(values, labels, format):
        ax.errorbar(
            np.array(data[v]["splits"]),
            np.array(data[v][key]["mean"]) / norm,
            yerr=np.array(data[v][key]["std"]) / norm * 2 / 5, label=label,
            capsize=2, fmt=fmt)

    if baseline is not None:
        ax.errorbar(
            np.array(data[values[-1]]["splits"]),
            np.array(data[values[-1]][key]["baseline_mean"]) / norm,
            yerr=np.array(data[values[-1]][key]["baseline_std"]) / norm,
            label=baseline, capsize=2, fmt=format[-1])

    if legend:
        ax.legend()
    ax.grid()


class Figures:
    """Paper figures."""

    @staticmethod
    def comparison(args):
        """Baseline comparisons."""
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        _plot(
            axs[0], ["embedding/128", "linear/128", "baseline/mlp"],
            labels=[
                "Pitot", "Linear Matrix Factorization",
                "Single Neural Network"])
        _plot(
            axs[1], ["embedding/128", "paragon/512", "baseline/device_mlp"],
            labels=["Pitot", "Paragon", "Per-Device MLP"],
            baseline="Rank 1 Baseline")
        _plot(
            axs[2], [
                "embedding/128", "baseline/module_only",
                "baseline/platform_only", "linear/128"],
            labels=[
                "All Features (Pitot)", "Module Features Only",
                "Platform Features Only", "No Features (Linear)"],
            relative="embedding/128")

        axs[0].set_ylabel("Mean Absolute Error")
        axs[1].set_ylabel("Mean Absolute Error")
        axs[2].set_ylabel("Relative Error")

        _fmt_axes(fig, axs)
        return {"comparisons": fig}

    @staticmethod
    def ablations_mf(args):
        """Ablation plots."""
        fig, axs = plt.subplots(1, 2, figsize=(6, 4))
        _plot(
            axs[0], [32, 64, 128, 256, 512],
            pattern=args.path + "/embedding/{}",
            labels=['$r=32$', '$r=64$', '$r=128$', '$r=256$', '$r=512$'],
            relative=512,
            format=['.-', '.:', '.--', '.-.', 'x-'])
        _plot(
            axs[1], [0, 2, 4, 8, 16],
            pattern=args.path + "/features/{}",
            labels=['$q=0$', '$q=2$', '$q=4$', '$q=8$', '$q=16$'],
            relative=4,
            format=['.-', '.:', 'x-', '.--', '.-.'])

        axs[0].set_ylabel("Relative Error")
        axs[0].set_title("Embedding Dimension")
        axs[1].set_title("Learned Features")

        _sharey(axs)
        _fmt_axes(fig, axs)
        for ax in axs:
            ax.set_ylim(0.95, 1.5)
        return {"ablations_mf": fig}

    @staticmethod
    def ablations_if(args):
        """Ablation plots."""
        fig, axs = plt.subplots(1, 2, figsize=(6, 4))
        _plot(
            axs[0], [1, 2, 3, 4],
            pattern=args.path + "/interference/{}",
            labels=['$s=1$', '$s=2$', '$s=3$', '$s=4$'],
            relative=2, format=['.-', 'x:', '.--', '.-.'], legend=False)
        _plot(
            axs[1], [1, 2, 3, 4],
            pattern=args.path + "/interference/{}",
            labels=['$s=1$', '$s=2$', '$s=3$', '$s=4$'],
            relative=2, format=['.-', 'x:', '.--', '.-.'], key='if')

        axs[0].set_ylabel("Relative Error")
        axs[0].set_title("Non-interference Error")
        axs[1].set_title("Interference Error")

        _sharey(axs)
        _fmt_axes(fig, axs)
        for ax in axs:
            ax.set_ylim(0.85, 1.16)
        return {"ablations_if": fig}

    @staticmethod
    def interference(args):
        """Interference evaluation (2-way)."""
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        methods = [
            "interference/2", "interference/discard", "interference/ignore"]
        methods3 = [
            "interference3/2", "interference3/discard", "interference3/ignore",
            "interference3/no-smt"]
        names = ["Interference-Aware", "Discard", "Ignore"]
        names3 = ["Interference-Aware", "Discard", "Ignore", "SMT Disabled"]
        _plot(axs[0], methods, labels=names, legend=False)
        _plot(axs[1], methods, labels=names, key="if", legend=False)
        _plot(axs[2], methods3, labels=names3, key="if")
        _sharey(axs)

        axs[0].set_ylabel("Percent Error")

        axs[0].set_title("Non-interference Error")
        axs[1].set_title("Interference Error")
        axs[2].set_title("Interference Error (3-way)")

        _sharey(axs)
        _fmt_axes(fig, axs)
        for ax in axs:
            ax.set_ylim(0, 0.35)
        return {"interference": fig}


def _main(args):
    for name in args.figures:
        figs = getattr(Figures, name)(args)
        for k, v in figs.items():
            out = os.path.join(args.out, k + ".png")
            print("Created:", out)
            v.savefig(out, dpi=args.dpi)
