"""Platform information aggregation."""

import json
import numpy as np
from matplotlib import pyplot as plt

from dataset import Index, Matrix


_desc = "Generate platform features."


def _int(x):
    if x == 0:
        return np.nan
    else:
        return x


def _logsize(x):
    """Log base 2 of size."""
    if x == 0:
        return np.nan
    else:
        return np.log2(int(x))


def _loghz(x):
    if x == 0:
        return np.nan
    else:
        return np.log(int(x) / 10**6)


def _parse(p):
    p.add_argument(
        "-p", "--path", help="Path to runtimes .json file.",
        default="runtimes.json")
    p.add_argument(
        "-m", "--matrix", help="Main matrix file.", default="matrix.npz")
    p.add_argument("-o", "--out", help="Output file.", default="platforms")
    p.add_argument(
        "-d", "--plot", help="Draw plot.", action='store_true', default=False)
    return p


def _main(args):
    with open(args.path) as f:
        src = json.load(f)

    architectures = np.array(sorted(np.unique(
        [v['platform']['cpu']['cpu'] for _, v in src.items()])))

    def _one_hot_uarch(x):
        return (architectures == x).astype(np.float32)

    def _transform(m):
        return [
            *_one_hot_uarch(m['cpu']['cpu']),     # Architecture
            _loghz(m['cpu']['cpufreq']) - 7,      # Log CPU Frequency / 1GHz
            _logsize(m['mem']['l1d_size']) - 16,  # Log L1D
            _logsize(m['mem']['l1i_size']) - 16,  # Log L1I
            _logsize(m['mem']['l2_size']) - 21,   # Log L2
            _logsize(m['mem']['l2_line']) - 10,   # Log L2 line size
            _int(m['mem']['l2_assoc']) - 7,       # L2 associativity
            _logsize(m['mem']['l3_size']) - 21,   # L3 size in MiB
        ]

    devices = {v['name']: _transform(v['platform']) for v in src.values()}
    platforms = np.load(args.matrix)["platform"]
    runtimes = np.array(sorted(list(set(
        [p.split(".")[0] for p in platforms]))))

    def _one_hot_runtime(x):
        return (runtimes == x).astype(np.float32)

    res = []
    for row in platforms:
        runtime, device = row.split(".")
        res.append(devices[device] + list(_one_hot_runtime(runtime)))
    res = np.array(res)

    labels = ["uarch=" + x for x in architectures] + [
        "cpufreq", "l1d_size", "l1i_size", "l2_size",
        "l2_line", "l2_assoc", "l3_size"
    ] + ["runtime=" + x for x in runtimes]

    np.savez(args.out + ".npz", data=res, feature=labels, platform=platforms)

    if args.plot:
        fig, ax = plt.subplots(figsize=(40, 40))
        Matrix(data=res, rows=Index(platforms), cols=Index(labels)).plot(ax)
        fig.savefig(args.out + ".png", bbox_inches="tight", pad_inches=0.2)
