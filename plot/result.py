"""Result plotting."""

import os

from jax import vmap
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Line2D


class Result:
    """Set of replicates with a single experiment type."""

    def __init__(self, path, dataset, name="Matrix Factorization"):
        self.path = path
        self.dataset = dataset
        self.name = name

        data = np.load(self.path)
        self.baseline = np.array(vmap(self.dataset.error)(
            data["baseline"], indices=data["test_split"]))
        self.error = np.array(vmap(self.dataset.error)(
            np.mean(data["predictions"], axis=1),
            indices=data["test_split"]))

        self.error_full = np.array(
            np.mean(data["predictions"], axis=1) - self.dataset.matrix)
        self.baseline_full = np.array(data["baseline"] - self.dataset.matrix)

    def plot_training(self, ax):
        """Plot train/val/test curves."""
        data = np.load(self.path)

        keys = ['train_loss', 'val_loss', 'test_loss']
        names = ['train', 'val', 'test']
        colors = ['C0', 'C1', 'C2']

        for key, color in zip(keys, colors):
            ax.plot(np.mean(data[key], axis=-1).T, color=color)

        ax.legend([Line2D([0], [0], color=c, lw=2) for c in colors], names)

    def compare_plot(self, ax, full=False, bins=50):
        """Compare and plot as histogram."""
        if full:
            x1 = self.baseline_full.reshape(-1)
            x2 = self.error_full.reshape(-1)
        else:
            x1 = self.baseline
            x2 = self.error

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
        _str = [p for p in os.listdir(path) if p.endswith(".npz")]
        _float = [float(sp.rstrip('.npz')) for sp in _str]
        _str, _float = zip(*sorted(zip(_str, _float)))

        self.results = [
            Result(os.path.join(path, sp), dataset, name) for sp in _str]
        self.splits = np.array(_float)

    @staticmethod
    def _boxplot(ax, x, data, color, **kwargs):
        boxplot = ax.boxplot(data.T, patch_artist=True, positions=x, **kwargs)
        for median in boxplot['medians']:
            median.set_color(color)
            median.set_linewidth(2)
        for box in boxplot['boxes']:
            box.set_facecolor('white')

    @staticmethod
    def _errorbar(ax, x, data, stddev=2, **kwargs):
        y = np.mean(data, axis=1)
        yerr = np.sqrt(np.var(data, axis=1) / (data.shape[1] - 1)) * stddev
        ax.errorbar(x, y, yerr=yerr, **kwargs)

    def compare(self, ax, color='C0', baseline=False, boxplot=True):
        """Add boxplots to axes."""
        data = np.array([
            (res.baseline if baseline else res.error)
            for res in self.results])
        pos = np.arange(len(self.splits))
        if boxplot:
            self._boxplot(ax, pos, data, color, widths=0.4)
        else:
            self._errorbar(
                ax, pos, data, color=color, capsize=5, fmt='o-', stddev=2)

        ax.set_xticks(pos)
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
