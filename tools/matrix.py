"""Process matrix dataset."""

import os
import numpy as np
import json
from matplotlib import pyplot as plt
from functools import partial

from prediction import apply_recursive, Index, Matrix


_desc = "Create execution time matrix from data traces."


def _load(path, runtimes: dict = {}):
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Invalid JSON: {}".format(path))
        return None

    t = np.array(data["utime"]) + np.array(data["stime"])
    t = t[t > 0]

    # Remove containing folder (wasm/, aot/, ...) and extension (.wasm, .aot)
    benchmark_ext = data["module"]["file"].split(os.path.sep)[1:]
    module = os.path.splitext(os.path.join(*benchmark_ext))[0]

    if len(t) < 2:
        return None
    else:
        return {
            "device": runtimes.get(data["module"]["parent"])["name"],
            "module": module,
            "runtime": data["module"]["name"].split('.')[-1],
            "mean": np.mean(t[1:]),
        }


def _parse(p):
    p.add_argument(
        "-p", "--path", nargs='+', default=["data/matrix"],
        help="Path to dataset.")
    p.add_argument(
        "-o", "--out", help="Output (base) path.", default="matrix")
    p.add_argument(
        "-d", "--plot", help="Draw plot.", action='store_true', default=False)
    p.add_argument(
        "-f", "--filter", action='store_true', default=False,
        help="Filter invalid devices")
    p.add_argument(
        "--exclude", nargs='+', help="Excluded devices.",
        default=[
            "iwasm-aot.hc-21", "iwasm-aot.hc-25", "iwasm-aot.hc-27",
            "wasmer-singlepass.hc-15", "wasmer-singlepass.hc-19"
        ])
    return p


def _main(args):

    sl_runtimes = {}
    for p in args.path:
        with open(os.path.join(p, "runtimes.json")) as f:
            sl_runtimes.update(json.load(f))

    traces = []
    for p in args.path:
        traces += apply_recursive(
            p, partial(_load, runtimes=sl_runtimes),
            exclude={"runtimes.json", "README.md"})

    devices = Index.from_objects(traces, "device")
    runtimes = Index.from_objects(traces, "runtime")
    modules = Index.from_objects(traces, "module")

    data = np.zeros(
        (len(runtimes), len(devices), len(modules)), dtype=np.float32)
    for d in traces:
        i = runtimes[d["runtime"]]
        j = devices[d["device"]]
        k = modules[d["module"]]
        data[i, j, k] = d["mean"]

    matrix = Matrix(
        data=data.reshape(-1, data.shape[-1]),
        rows=runtimes @ devices, cols=modules)

    if args.filter:
        filter = (np.sum(matrix.data > 0, axis=1) > 0)
        for k in args.exclude:
            filter[matrix.rows[k]] = False

        matrix = matrix[filter, :]

    matrix.save(args.out + ".npz", rows="platform", cols="module")

    if args.plot:
        fig, ax = plt.subplots(1, 1, figsize=(40, 40))
        with np.errstate(divide='ignore'):
            (matrix @ np.log).plot(ax, xlabel=True, ylabel=True)
        fig.savefig(args.out + ".png", bbox_inches='tight', pad_inches=0.2)
