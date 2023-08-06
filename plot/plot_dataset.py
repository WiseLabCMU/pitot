"""Poster-style data plot."""

import numpy as np
from matplotlib import pyplot as plt
from prediction import Matrix, Index


_desc = "Generate large matrix and side information plot."


def _parse(p):
    p.add_argument("-p", "--path", default="data/data.npz", help="Data file.")
    p.add_argument(
        "-o", "--out", default="data/dataset.pdf", help="Output file.")


def _main(args):

    fig = plt.figure(constrained_layout=True)
    fig, axs = plt.subplots(
        2, 2, figsize=(40, 60),
        gridspec_kw={"height_ratios": [4, 3], "width_ratios": [1, 4.5]})

    matrix = Matrix.from_npz(args.path, rows="platform", cols="module")
    platforms = Matrix.from_npz(
        args.path, data="platform_data", rows="platform",
        cols="platform_features")
    _modules = Matrix.from_npz(
        args.path, data="module_data", rows="module", cols="module_features")
    _opcodes_hex = Index(_modules.cols.key, display=[
        "x{:02X}".format(x) for x in _modules.cols.key])
    modules = Matrix(_modules.data, _modules.rows, _opcodes_hex)

    with np.errstate(divide='ignore'):
        matrix = matrix @ np.log
        modules = modules @ np.log

    matrix.plot(axs[0, 1], ylabel=False, xlabel=False, aspect='auto')
    modules.plot(
        axs[1, 1], ylabel=True, xlabel=True, transpose=True, aspect='auto')
    platforms.plot(
        axs[0, 0], ylabel=True, xlabel=True, transpose=False, aspect='auto')

    axs[1, 0].axis('off')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    fig.tight_layout(h_pad=-8, w_pad=1)

    fig.savefig(args.out)
