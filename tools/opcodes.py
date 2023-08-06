"""Process opcode data."""

import numpy as np
import json
from matplotlib import pyplot as plt

from prediction import apply_recursive, Index, Matrix


_desc = "Aggregate benchmark module opcode counts."
_indices = {}


def _load(path):
    with open(path) as f:
        data = json.load(f)
    module = data["module"]["file"].replace("wasm/", "").replace(".wasm", "")

    argv = tuple(data["module"].get("args", {}).get("argv", []))
    if argv:
        if module not in _indices:
            _indices[module] = {}
        if argv not in _indices[module]:
            _indices[module][argv] = len(_indices[module])
        module += "_{}".format(_indices[module][argv])

    return np.array(data["opcodes"], dtype=np.uint32), module


def _parse(p):
    p.add_argument(
        "-p", "--path", help="Path to dataset.", default="data/opcodes")
    p.add_argument(
        "-m", "--matrix", help="Main matrix file.", default="matrix.npz")
    p.add_argument(
        "-o", "--out", help="Output (base) path.", default="opcodes")
    p.add_argument(
        "-d", "--plot", help="Draw plot.", action='store_true', default=False)
    return p


def _main(args):
    data, files = list(zip(*apply_recursive(args.path, _load)))

    target_modules = set(np.load(args.matrix)["module"])
    mask = np.array([x in target_modules for x in files])
    data = np.array(data)[mask]
    files = np.array(files)[mask]
    nonzero = np.where(np.sum(data, axis=0) > 10)[0]

    opcodes = Index(
        nonzero, display=["{:02x}".format(int(i)) for i in nonzero])
    mat = Matrix(
        data=data[:, nonzero], rows=Index(files), cols=opcodes
    )[np.argsort(files)]

    mat.save(args.out + ".npz", rows="module", cols="opcode")

    if args.plot:
        fig, ax = plt.subplots(figsize=(40, 40))
        (mat @ (lambda x: np.log(x + 1))).plot(ax, xlabel=True, ylabel=True)
        fig.savefig(args.out + ".png", bbox_inches="tight", pad_inches=0.2)
