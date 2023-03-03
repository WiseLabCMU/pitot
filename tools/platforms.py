"""Platform information aggregation."""

import json
import numpy as np
from matplotlib import pyplot as plt

from prediction import Index, Matrix


_desc = "Generate platform features."


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


def _int(x, offset=0):
    return [1 if x != 0 else -1, 0 if x == 0 else x - offset]


def _logsize(x, offset=0, factor=1.0):
    return [1 if x != 0 else -1, 0 if x == 0 else np.log2(x) - offset]


def _loghz(x):
    return np.log(int(x) / 10**6)


class OneHot:
    """One Hot (-1, 1) encoding."""

    def __init__(self, values, name="var"):
        self.values = np.array(sorted(np.unique(values)))
        self.display = ["{}={}".format(name, x) for x in self.values]

    def encode(self, x):
        """Generate encoding."""
        return (self.values == x).astype(np.float32) * 2 - 1


def _main(args):
    with open(args.path) as f:
        src = json.load(f)

    uarch = OneHot(
        [v['platform']['cpu']['cpu'] for _, v in src.items()], name="uarch")
    l2_assoc = OneHot(
        [v['platform']['mem']['l2_assoc'] for _, v in src.items()],
        name="l2_assoc")

    def _transform(m):
        return [
            *uarch.encode(m['cpu']['cpu']),              # Architecture
            _loghz(m['cpu']['cpufreq']) - 7,             # Log CPUFreq
            *_logsize(m['mem']['l1d_size'], offset=16),  # Log L1D
            *_logsize(m['mem']['l1i_size'], offset=16),  # Log L1I
            *_logsize(m['mem']['l2_size'], offset=21),   # Log L2
            *_logsize(m['mem']['l2_line'], offset=10),   # Log L2 line size
            *l2_assoc.encode(m['mem']['l2_assoc']),      # L2 associativity
            *_logsize(m['mem']['l3_size'], offset=21)  # Log L3 size
        ]

    devices = {v['name']: _transform(v['platform']) for v in src.values()}
    platforms = np.load(args.matrix)["platform"]
    runtimes = OneHot([p.split(".")[0] for p in platforms], name="rt")

    res = []
    for row in platforms:
        runtime, device = row.split(".")
        res.append(devices[device] + list(runtimes.encode(runtime)))
    res = np.array(res)

    labels = [
        *uarch.display, "cpufreq",
        "ind:l1d_size", "l1d_size", "ind:l1i_size", "l1i_size",
        "ind:l2_size", "l2_size", "ind:l2_line", "l2_line", *l2_assoc.display,
        "ind:l3_size", "l3_size",
        *runtimes.display
    ]

    np.savez(args.out + ".npz", data=res, feature=labels, platform=platforms)

    if args.plot:
        fig, ax = plt.subplots(figsize=(40, 40))
        Matrix(data=res, rows=Index(platforms), cols=Index(labels)).plot(ax)
        fig.savefig(args.out + ".png", bbox_inches="tight", pad_inches=0.2)
