"""Prediction experiments."""

import os
import numpy as np
from tqdm import tqdm
from functools import partial
from collections import namedtuple
from argparse import ArgumentParser
import optax

from dataset import Dataset
from prediction import CrossValidationTrainer
from prediction.models import *


Method = namedtuple(
    "Method", ["constr", "kwargs", "optimizer", "epochs", "epoch_size"])

COMMON = {"epochs": 250, "epoch_size": 100, "optimizer": optax.adam(0.001)}
KWARGS_COMMON = {"layers": [64, 32], "dim": 4, "scale": 0.01}
SPARSITY = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _models(name, dataset):
    return {
        "embedding": Method(
            constr=embedding,
            kwargs={
                "runtime_data": dataset.runtime_data,
                "module_data": dataset.module_data,
                **KWARGS_COMMON},
            **COMMON),
        "runtime_only": Method(
            constr=embedding,
            kwargs={"runtime_data": dataset.runtime_data, **KWARGS_COMMON},
            **COMMON),
        "module_only": Method(
            constr=embedding,
            kwargs={"module_data": dataset.module_data, **KWARGS_COMMON},
            **COMMON),
        "linear4": Method(
            constr=linear,
            kwargs={"dim": 4, "scale": 0.01},
            **COMMON),
        "linear8": Method(
            constr=linear,
            kwargs={"dim": 8, "scale": 0.01},
            **COMMON),
        "linear16": Method(
            constr=linear,
            kwargs={"dim": 16, "scale": 0.01},
            **COMMON),
        "linear32": Method(
            constr=linear,
            kwargs={"dim": 32, "scale": 0.01},
            **COMMON),
        "linear64": Method(
            constr=linear,
            kwargs={"dim": 64, "scale": 0.01},
            **COMMON),
        "simple_mlp": Method(
            constr=MLPOnly,
            kwargs={
                "runtime_data": dataset.runtime_data,
                "module_data": ds.module_data},
            **COMMON),
    }[name]


def _experiment(name, method, p, replicates=100, baseline=True, dataset=None):
    trainer = CrossValidationTrainer(
        ds, partial(method.constr, shape=ds.shape, **method.kwargs),
        optimizer=method.optimizer, replicates=replicates, k=25,
        epochs=method.epochs, epoch_size=method.epoch_size, batch=64)

    pbar = partial(tqdm, desc="{} : {}".format(name, p))
    results = trainer.train_replicates(p=p, tqdm=pbar, do_baseline=baseline)

    model_dir = os.path.join("results", name)
    os.makedirs(model_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(model_dir, "{}.npz".format(p)), **results)


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("methods", nargs="+", default=[])
    p.add_argument("--replicates", "-n", type=int, default=100)
    p.add_argument("--dataset", "-d", default="data_cfs.npz")
    p.add_argument("--no-baseline", dest="baseline", action='store_false')
    p.set_defaults(baseline=True)
    args = p.parse_args()

    ds = Dataset(args.dataset, offset=1000. * 1000.)
    for name in args.methods:
        model_dir = os.path.join("results", name)
        for s in SPARSITY:
            _experiment(
                name, _models(name, ds), s, replicates=args.replicates,
                baseline=args.baseline, dataset=ds)
