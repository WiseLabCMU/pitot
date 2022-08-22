"""Get interference summary CSV from multiple experiments."""

import argparse
import pandas as pd
import numpy as np
import os
from dataset import Dataset, Session
from tqdm import tqdm


def _load_df(path, ds):
    session = Session(path)

    def _baseline(row):
        return ds.matrix[
            ds.modules_dict[row["file"]],
            ds.runtimes_dict[row["runtime"]]]

    def _mean(row):
        return np.log(np.mean(
            session
            .get(module_id=row["module_id"])
            .arrays(keys=["cpu_time"])["cpu_time"]))

    df = session.manifest
    df["mean"] = df.apply(_mean, axis=1)
    df["baseline"] = df.apply(_baseline, axis=1)
    df["diff"] = df["mean"] - df["baseline"]
    df["source"] = path
    return df


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("path", default="interference")
    p.add_argument("--dataset", default="data.npz")
    p.add_argument("--out", default="summary.csv")
    args = p.parse_args()

    ds = Dataset(args.dataset)
    paths = [os.path.join(args.path, d) for d in os.listdir(args.path)]
    paths = [p for p in paths if os.path.isdir(p)]
    dfs = [_load_df(path, ds) for path in tqdm(paths)]
    pd.concat(dfs).to_csv(args.out, index=False)
