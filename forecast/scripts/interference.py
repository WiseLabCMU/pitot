"""Get interference summary CSV from multiple experiments."""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from libsilverline import ArgumentParser
from forecast import Dataset, Session


def _load_df(path, ds, key="cpu_time"):
    session = Session(path)

    def _baseline(row):
        return float(ds.matrix[
            ds.modules_dict[row["file"]],
            ds.runtimes_dict[row["runtime"]]])

    def _mean(row):
        return np.mean(
            session
            .get(module_id=row["module_id"])
            .arrays(keys=[key])[key][2:])

    def _interferer(row):
        modules = row["module"].split('.')
        modules.remove(row["file"])
        return ".".join(modules)

    df = session.manifest
    tqdm.pandas(desc=path)
    df["mean"] = df.progress_apply(_mean, axis=1)

    df["interferer"] = df.apply(_interferer, axis=1)
    df["baseline"] = df.apply(_baseline, axis=1)
    df["diff"] = np.log(df["mean"]) - df["baseline"]
    df["source"] = path
    return df


def summarize(
        data="data", baseline="data.npz", out="", key="cpu_time"):
    """Create interference summary table.

    Parameters
    ----------
    data : str
        Data directory; each folder inside is a Session.
    baseline : str
        Baseline non-interference results.
    out : str
        Path to save to. If empty, uses the same base path as 'data'.
    key : str
        Key in dataset to look at.
    """
    ds = Dataset(baseline)
    paths = [os.path.join(data, p) for p in os.listdir(data)]
    dfs = [_load_df(p, ds, key=key) for p in paths if os.path.isdir(p)]

    if out == "":
        out = data
    res = pd.concat(dfs)
    res.to_csv(out + ".csv", index=False)

    file_idx = res.apply(lambda row: ds.modules_dict[row["file"]], axis=1)
    if_idx = res.apply(lambda row: ds.modules_dict[row["interferer"]], axis=1)
    rt_idx = res.apply(lambda row: ds.runtimes_dict[row["runtime"]], axis=1)
    np.savez(
        out + ".npz",
        module=np.array(file_idx).astype(np.int16),
        interferer=np.array(if_idx).astype(np.int16),
        runtime=np.array(rt_idx).astype(np.int8),
        mean=np.array(res["mean"]),
        baseline=np.array(res["baseline"]))


def _parse():
    p = ArgumentParser()
    p.add_to_parser(
        "summarize", summarize, "summarize",
        aliases={"baseline": ["-b"], "data": ["-d"], "out": ["-o"]})
    return p


def _main(args):
    summarize(**args["summarize"])
