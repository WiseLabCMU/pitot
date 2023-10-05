"""Process matrix dataset."""

import os
import numpy as np
import json
from matplotlib import pyplot as plt
from functools import partial

from prediction import apply_recursive, Index, Matrix


_desc = "Create execution time matrix from data traces."
_indices = {}


def _load(path, runtimes: dict = {}):
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Invalid JSON: {}".format(path))
        return None

    # t=0 is used as the error condition.
    t = np.array(data["wall"])
    t = t[t > 0]

    # Remove containing folder (wasm/, aot/, ...) and extension (.wasm, .aot)
    benchmark_ext = data["module"]["file"].split(os.path.sep)[1:]
    module = os.path.splitext(os.path.join(*benchmark_ext))[0]

    # Add global index for args
    argv = tuple(data["module"].get("args", {}).get("argv", []))
    if argv:
        if module not in _indices:
            _indices[module] = {}
        if argv not in _indices[module]:
            _indices[module][argv] = len(_indices[module])
        module += "_{}".format(_indices[module][argv])

    runtime = data["module"]["name"].split('.')[-1]
    if len(t) == 0:
        return None
    else:
        return {
            "device": runtimes.get(data["module"]["parent"])["name"],
            "module": module, "runtime": runtime, "mean": np.mean(t),
        }


def _parse(p):
    p.add_argument(
        "-p", "--path", nargs='+', default=["data/matrix"],
        help="Paths to dataset.")
    p.add_argument(
        "-o", "--out", help="Output (base) path.", default="matrix")
    p.add_argument(
        "-d", "--plot", help="Draw plot.", action='store_true', default=False)
    p.add_argument(
        "-f", "--filter", action='store_true', default=False,
        help="Filter invalid devices")
    p.add_argument(
        "--exclude", nargs='+', help="Excluded devices.",
        default=["iwasm-a.hc-21", "iwasm-a.hc-25", "iwasm-a.hc-27"])
    p.add_argument(
        "-c", "--mincount", default=1, type=int,
        help="Minimum number of entries threshold in each row/column.")
    return p


def _main(args):

    traces = []
    for p in args.path:
        if os.path.isdir(p):
            with open(os.path.join(p, "runtimes.json")) as f:
                rt_manifest = json.load(f)
            traces += apply_recursive(
                p, partial(_load, runtimes=rt_manifest),
                exclude={"runtimes.json", "README.md"})
        else:
            with open(p) as f:
                data = json.load(f)
            for _, entry in data.items():
                for k, v in entry["data"].items():
                    traces.append({
                        "device": entry["device"],
                        "runtime": entry["runtime"],
                        "module": k,
                        "mean": v
                    })

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
        filter = (np.sum(matrix.data > 0, axis=1) >= args.mincount)
        filter2 = (np.sum(matrix.data > 0, axis=0) >= args.mincount)

        for k in args.exclude:
            try:
                filter[matrix.rows[k]] = False
            except KeyError:
                pass

        matrix = matrix[filter, :][:, filter2]

    print("Created dataset: {}.npz @ {} (n={})".format(
        args.out, matrix.data.shape, np.sum(matrix.data > 0)))
    matrix.save(args.out + ".npz", rows="platform", cols="module")

    if args.plot:
        fig, ax = plt.subplots(1, 1, figsize=(40, 40))
        with np.errstate(divide='ignore'):
            (matrix @ np.log).plot(ax, xlabel=True, ylabel=True)
        fig.savefig(args.out + ".png", bbox_inches='tight', pad_inches=0.2)
