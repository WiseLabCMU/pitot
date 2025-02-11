"""Figure elements."""

import matplotlib.ticker as mtick
import numpy as np
from jaxtyping import Num


def plot_errorbar(
    ax, x: Num[np.ndarray, "N"], y: Num[np.ndarray, "N samples"], label: str,
    stderr: bool = True, percent: bool = True, **kwargs
) -> None:
    """Draw errorbars."""
    mean = np.nanmean(y, axis=1)
    std = np.nanstd(y, axis=1, ddof=1) / np.sqrt(y.shape[1] if stderr else 1.0)

    if percent:
        mean = mean * 100
        std = std * 100
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, pos: str(int(x)) + '%'))

    ax.errorbar(
        x, mean, yerr=2 * std, label=label,
        capsize=4, marker='.', **kwargs)


def format_xsplits(ax) -> None:
    """Format axes with data split x-axis."""
    ax.set_xticks([1, 3, 5, 7])
    ax.set_xticklabels(["20%", "40%", "60%", "80%"])
    ax.grid(visible=True)
