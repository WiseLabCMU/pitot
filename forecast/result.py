"""Result plotting."""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Line2D


class Result:
    """Set of replicates with a single experiment type."""

    def __init__(self, path, dataset, name="Matrix Factorization"):
        self.path = path
        self.dataset = dataset
        self.name = name

        self.summary = dict(np.load(path.replace(".npz", "_summary.npz")))

    def plot_training(self, ax, keys=None, names=None):
        """Plot train/val/test curves.

        Each outer replicate is plotted as a separate curve, with inner
        k-fold replicates being averaged.
        """
        data = np.load(self.path)

        if keys is None:
            keys = ["train_loss", "mf_val_loss", "mf_test_loss"]
            names = ["train", "val", "test"]
        colors = ['C{}'.format(i) for i in range(len(keys))]

        for key, color in zip(keys, colors):
            ax.plot(np.mean(data[key], axis=-1).T, color=color)

        ax.legend([Line2D([0], [0], color=c, lw=2) for c in colors], names)

    def compare_plot(self, ax, full=False, bins=50):
        """Compare and plot as histogram.

        If `full`, the errors in test sets across all replicates are
        aggregated, and plotted as a histogram.
        Otherwise, the mean absolute error for each replicate is plotted.
        """
        if full:
            x1 = self.summary["baseline_full"].reshape(-1)
            x2 = self.summary["error_full"].reshape(-1)
        else:
            x1 = self.summary["baseline"]
            x2 = self.summary["error"]

        left = np.minimum(np.min(x1), np.min(x2))
        right = np.maximum(np.max(x1), np.max(x2))
        bins = np.linspace(left, right, bins)

        ax.hist(x1, bins=bins, label='Baseline', alpha=0.5, density=True)
        ax.hist(x2, bins=bins, label=self.name, alpha=0.5, density=True)

        ax.axvline(np.mean(x1), linestyle='--', color='C0')
        ax.axvline(np.mean(x2), linestyle='--', color='C1')


class Method:
    """Ablation experiments for a set of hyperparameters."""

    def __init__(self, path, dataset, name="Matrix Factorization"):
        _str = [
            p for p in os.listdir(path)
            if p.endswith(".npz") and "summary" not in p]
        _float = [float(sp.rstrip('.npz')) for sp in _str]
        _str, _float = zip(*sorted(zip(_str, _float)))

        self.results = [
            Result(os.path.join(path, sp), dataset, name) for sp in _str]
        self.splits = np.array(_float)

    @staticmethod
    def _boxplot(ax, x, data, color, **kwargs):
        boxplot = ax.boxplot(
            data.T, patch_artist=True, positions=x, **kwargs)
        for median in boxplot['medians']:
            median.set_color(color)
            median.set_linewidth(2)
        for box in boxplot['boxes']:
            box.set_facecolor('white')

    @staticmethod
    def _errorbar(ax, x, data, stddev=2, **kwargs):
        y = np.mean(data, axis=1)
        yerr = np.sqrt(np.var(data, axis=1)) * stddev
        ax.errorbar(x, y, yerr=yerr, **kwargs)

    def compare(
            self, ax, color='C0', boxplot=True, key="error"):
        """Add boxplots for mean absolute error on replicates to axes."""
        data = np.array([res.summary[key] for res in self.results])
        if boxplot:
            self._boxplot(ax, self.splits, data, color, widths=0.05)
        else:
            self._errorbar(
                ax, self.splits, data, color=color, capsize=5,
                fmt='o-', stddev=2)

        ax.set_xticks(self.splits)
        ax.set_xlim(0, 1)
        ax.set_xticklabels(["{}%".format(int(p * 100)) for p in self.splits])
        ax.set_xlabel("Train Split")
        ax.set_ylabel("Mean Absolute Error")

    def histogram(self, axs=None, **kwargs):
        """Plot histograms for all experiments."""
        if axs is None:
            fig, axs = plt.subplots(3, 3, figsize=(16, 12))

        for ax, res, split in zip(axs.reshape(-1), self.results, self.splits):
            res.compare_plot(ax, **kwargs)
            psplit = int(split * 100)
            ax.set_title("train={}% / test={}%".format(psplit, 100 - psplit))

        axs[0, 0].legend(loc='upper left')

    def training(self, axs=None, **kwargs):
        """Plot training curves for all experiments."""
        if axs is None:
            fig, axs = plt.subplots(3, 3, figsize=(16, 12))

        for ax, res, split in zip(axs.reshape(-1), self.results, self.splits):
            res.plot_training(ax, **kwargs)
            psplit = int(split * 100)
            ax.set_title("train={}% / test={}%".format(psplit, 100 - psplit))

    def bounds(self, ax, percentiles=[95, 90, 80, 50]):
        """Plot percentile absolute error bounds."""
        colors = ['C{}'.format(i) for i in range(len(percentiles))]

        per_base = np.array([
            np.percentile(res.summary["baseline_full"], percentiles)
            for res in self.results])
        per = np.array([np.percentile(
            res.summary["error_full"], percentiles) for res in self.results])
        sizes = [res.summary["error_full"].shape[1] for res in self.results]

        for x, xb, c, p in zip(per.T, per_base.T, colors, percentiles):
            ax.plot(x, marker='.', label="{}%".format(p), color=c)
            ax.plot(xb, marker='.', linestyle='dashed', color=c)

        ax.set_xticks(np.arange(len(self.splits)))
        ax.set_xticklabels([
            "{}%\nn={}".format(int(sp * 100), int(s))
            for sp, s in zip(self.splits, sizes)])
        ax.set_xlabel("Train Split")
        ax.set_ylabel("Absolute Error Bound")
        ax.legend(loc='upper right')
