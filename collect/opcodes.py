"""Aggregate benchmark module opcode counts."""

import os
import numpy as np
import json
from matplotlib import pyplot as plt

from beartype.typing import Callable, Optional, Any

from ._common import workload_name


def apply_recursive(
    path: str, func: Callable[[str], Optional[Any]],
    exclude: set[str] = {"runtimes.json"}
) -> list:
    """Apply function recursively in file system."""
    res = []
    for p in os.listdir(path):
        pp = os.path.join(path, p)
        if os.path.isdir(pp):
            res += apply_recursive(pp, func, exclude=exclude)
        else:
            if p not in exclude:
                d = func(pp)
                if d is not None:
                    res.append(d)
    return res


def _load(path):
    with open(path) as f:
        data = json.load(f)
    t = np.array(data["opcodes"], dtype=np.uint32)
    return t, workload_name(data["module"])


def _parse(p):
    p.add_argument(
        "-s", "--src", help="Path to dataset.", default="data-raw/opcodes")
    p.add_argument(
        "-o", "--out", help="Output file.", default="data/_opcodes.npz")
    p.add_argument(
        "-p", "--plot", help="Draw plot.", action='store_true', default=False)
    return p


def _main(args):
    opcodes, names = list(zip(*apply_recursive(args.src, _load)))
    names = np.array(names)
    order = np.argsort(names)
    opcodes = np.log(np.array(opcodes) + 1).astype(np.float32)
    nonconstant = np.any(opcodes != opcodes[0], axis=0)

    labels = ["x{:02X}".format(x) for x in np.where(nonconstant)[0]]
    data = opcodes[:, nonconstant]

    np.savez(args.out, data=data[order], names=names[order], labels=labels)

    if args.plot:
        fig, ax = plt.subplots(figsize=(40, 40))
        ax.imshow(data)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        fig.savefig(
            args.out.replace(".npz", ".png"),
            bbox_inches="tight", pad_inches=0.2)
