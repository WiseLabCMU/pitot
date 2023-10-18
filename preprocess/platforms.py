"""Aggregate platform data."""

import json
import numpy as np
from matplotlib import pyplot as plt

from ._common import platform_name


class LogSize:
    """Log size encoding, with an indicator for presence."""

    def __init__(self, name: str, offset: int = 0):
        self.offset = offset
        self.display = [name, "{}:ind".format(name)]

    def encode(self, x) -> list[float]:
        """Generate encoding."""
        return [
            0. if x == 0 else np.log2(x) - self.offset, 1. if x != 0 else -1.]


class LogHz:
    """Log frequency (in MHz)."""

    def __init__(self, name: str, offset: float = 0.0):
        self.display = [name]
        self.offset = offset

    def encode(self, x) -> list[float]:
        """Generate encoding."""
        return [np.log(int(x) / 10**6) - self.offset]


class OneHot:
    """One Hot (-1, 1) encoding."""

    def __init__(self, values, name="var"):
        self.values = np.array(sorted(np.unique(values)))
        self.display = ["{}={}".format(name, x) for x in self.values]

    def encode(self, x) -> list[float]:
        """Generate encoding."""
        return list((self.values == x).astype(np.float32) * 2 - 1)


DEFAULT_RUNTIMES = [
    "iwasm-a", "wasmer-a-ll", "wasmer-a-cl", "wasmtime-a",
    "iwasm-i", "wasm3-i", "wasmedge-i",
    "wasmtime-j", "wasmer-j-cl", "wasmer-j-sp"]


def _parse(p):
    p.add_argument(
        "-s", "--src", nargs='+', help="Path to `manifest.json` files.")
    p.add_argument(
        "-r", "--runtimes", default=DEFAULT_RUNTIMES, nargs='+',
        help="WebAssembly runtimes to include.")
    p.add_argument(
        "-o", "--out", help="Path to output file.",
        default="data/_platforms.npz")
    p.add_argument(
        "-p", "--plot", help="Draw plot.", action='store_true', default=False)
    return p


def _main(args):
    # Platform data
    rtdata = {}
    for cfg in args.src:
        with open(cfg) as f:
            rtdata.update(json.load(f))

    fields = {
        ("cpu", "cpu"): OneHot(
            [v['platform']['cpu']['cpu'] for v in rtdata.values()],
            name="uarch"),
        ("cpu", "cpufreq"): LogHz("cpufreq", offset=7.0),
        ("mem", "l1d_size"): LogSize("l1d_size", offset=16),
        ("mem", "l1i_size"): LogSize("l1i_size", offset=16),
        ("mem", "l2_size"): LogSize("l2_size", offset=21),
        ("mem", "l2_line"): LogSize("l2_line", offset=10),
        ("mem", "l2_assoc"): OneHot(
            [v['platform']['mem']['l2_assoc'] for v in rtdata.values()],
            name="l2_assoc"),
        ("mem", "l3_size"): LogSize("l3_size", offset=21)
    }
    runtimes = OneHot(args.runtimes, name="runtime")

    devices = sorted([
        (d['name'], sum([
            v.encode(d["platform"][k[0]][k[1]]) for k, v in fields.items()
        ], [])) for d in rtdata.values()], key=lambda x: x[0])

    data = [v + runtimes.encode(r) for r in args.runtimes for _, v in devices]
    data = np.array(data).astype(np.float32)
    names = [platform_name(v, r) for r in args.runtimes for v, _ in devices]
    labels = sum([v.display for v in fields.values()], []) + runtimes.display

    np.savez(args.out, data=data, names=names, labels=labels)

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
