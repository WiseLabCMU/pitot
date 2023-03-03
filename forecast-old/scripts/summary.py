"""Create results summary."""

import os
from libsilverline import ArgumentParser
from functools import partial
from tqdm import tqdm

import numpy as np
from jax import vmap


from forecast.dataset import Dataset


def _vresults(ds, data, indices, **kwargs):
    return (
        np.array(vmap(partial(ds.error, **kwargs))(data, indices=indices)),
        np.array(vmap(
            partial(ds.perror, full=True, **kwargs))(data, indices=indices)))


def _summary(ds, path):
    data = np.load(path)

    baseline, baseline_full = _vresults(ds, data["C_bar"], data["mf_test"])
    error, error_full = _vresults(
        ds, np.mean(data["C_hat"], axis=1), data["mf_test"])
    interference, interference_full = _vresults(
        ds, np.mean(data["C_ijk_hat"], axis=1), data["if_test"], mode="if")

    np.savez(path.replace(".npz", "_summary.npz"), **{
        "baseline": baseline, "baseline_full": baseline_full,
        "error": error, "error_full": error_full,
        "interference": interference, "interference_full": interference_full
    })


def summarize(
        data="data.npz", if_data="if.npz", key="mean", offset=1000. * 1000.,
        path=["results"], expand=0):
    """Create summary statistics for results.

    Parameters
    ----------
    data : str
        Non-interference dataset.
    if_data : str
        Interference dataset.
    key : str
        Dataset key.
    offset : float
        Execution time offset multiplier.
    path : str
        Method results directories.
    expand : int
        Search directories for this many levels as well.
    """
    ds = Dataset(data=data, if_data=if_data, key=key, offset=offset)

    for p in path:
        if expand > 0:
            dirs = [
                os.path.join(p, x) for x in os.listdir(p)
                if os.path.isdir(os.path.join(p, x))]
            summarize(
                data=data, if_data=if_data, key=key, offset=offset, path=dirs,
                expand=expand - 1)
        else:
            print(p)
            results = [
                x for x in os.listdir(p)
                if (x.endswith(".npz") and not x.endswith("_summary.npz"))]
            for r in tqdm(results):
                _summary(ds, os.path.join(p, r))


def _parse():
    p = ArgumentParser()
    p.add_to_parser("summarize", summarize, "summarize", exclude=["path"])
    p.add_argument("--path", nargs='+', default=["results"])
    return p


def _main(args):
    summarize(path=args["path"], **args["summarize"])
