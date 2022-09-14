"""Get interference summary CSV from multiple experiments."""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from libsilverline import ArgumentParser
from dataset import Dataset, Session


def _load_df(path, ds, key="cpu_time"):
    session = Session(path)

    def _baseline(row):
        return ds.matrix[
            ds.modules_dict[row["file"]],
            ds.runtimes_dict[row["runtime"]]]

    def _mean(row):
        return np.log(np.mean(
            session
            .get(module_id=row["module_id"])
            .arrays(keys=[key])[key]))

    df = session.manifest
    tqdm.pandas(desc=path)
    df["mean"] = df.progress_apply(_mean, axis=1)
    df["baseline"] = df.apply(_baseline, axis=1)
    df["diff"] = df["mean"] - df["baseline"]
    df["source"] = path
    return df


def summarize(
        data="data", baseline="data.npz", out="summary.csv", key="cpu_time"):
    """Create interference summary table.

    Parameters
    ----------
    data : str
        Data directory; each folder inside is a Session.
    baseline : str
        Baseline non-interference results.
    out : str
        Path to save to.
    key : str
        Key in dataset to look at.
    """
    ds = Dataset(baseline)
    paths = [os.path.join(data, p) for p in os.listdir(data)]
    dfs = [_load_df(p, ds, key=key) for p in paths if os.path.isdir(p)]
    pd.concat(dfs).to_csv(out, index=False)


def _parse():
    p = ArgumentParser()
    p.add_to_parser(
        "summarize", summarize, "summarize",
        aliases={"baseline": ["-b"], "data": ["-d"], "out": ["-o"]})
    return p


def _main(args):
    summarize(**args["summarize"])


if __name__ == '__main__':
    _main(_parse().parse_args())
