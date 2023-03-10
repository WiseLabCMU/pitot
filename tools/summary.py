"""Summarize experiments."""

import os
import json
import jax
import numpy as np
from tqdm import tqdm

from prediction import Objective, Dataset


_desc = "Summarize experiment results."


def _parse(p):
    p.add_argument("-p", "--path", default="results", help="Input directory.")
    return p


def _load(path):
    with open(path.replace(".npz", ".json")) as f:
        config = json.load(f)

    _obj = config["config"]["objectives"]
    dataset = Dataset.from_npz(config["dataset"], log=_obj[0]["log"])
    objectives = [Objective.from_config(dataset, c) for c in _obj]
    result = np.load(path)

    summary = {}
    for obj, cfg in zip(objectives, _obj):
        y_pred = result[cfg["save"] if cfg["save"] else "C_hat"]
        indices = result[cfg["name"] + "_test"]
        error = jax.vmap(obj.perror)(y_pred, indices).reshape(-1)
        mape = jax.vmap(obj.mape)(y_pred, indices)
        summary[cfg["name"]] = {
            "mean": np.mean(mape),
            "std": np.sqrt(np.var(mape)),
            "std_all": np.sqrt(np.var(error)),
            "replicates": mape.tolist()
        }
        if "C_bar" in result:
            baseline = jax.vmap(obj.mape)(result["C_bar"], indices)
            summary[cfg["name"]].update({
                "baseline": baseline.tolist(),
                "baseline_mean": np.mean(baseline),
                "baseline_std": np.sqrt(np.var(baseline))
            })
    return summary


def _recurse(path):
    contents = os.listdir(path)
    if any(c.endswith(".npz") for c in contents):
        splits = sorted([
            float(c.replace(".npz", ""))
            for c in contents if c.endswith('.npz')])

        results = [
            _load(os.path.join(path, "{}.npz".format(x)))
            for x in tqdm(splits, desc=path)]
        results = {
            obj: {
                k: np.stack([x[obj][k] for x in results]).tolist()
                for k in subdict
            } for obj, subdict in results[0].items()
        }
        results["splits"] = splits

        with open(os.path.join(path, "summary.json"), 'w') as f:
            json.dump(results, f, indent=4)
    else:
        for c in contents:
            _recurse(os.path.join(path, c))


def _main(args):
    _recurse(args.path)
