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


ds = Dataset("data.npz", offset=1000. * 1000.)
Method = namedtuple(
    "Method", ["constr", "kwargs", "optimizer", "epochs", "epoch_size"])

COMMON = {"epochs": 100, "epoch_size": 100, "optimizer": optax.adam(0.001)}
KWARGS_COMMON = {"layers": [64, 32], "dim": 4, "scale": 0.01}

MODELS = {
    "embedding": Method(
        constr=embedding,
        kwargs={
            "runtime_data": ds.runtime_data, "module_data": ds.module_data,
            **KWARGS_COMMON},
        **COMMON),
    "runtime_only": Method(
        constr=embedding,
        kwargs={"runtime_data": ds.runtime_data, **KWARGS_COMMON},
        **COMMON),
    "module_only": Method(
        constr=embedding,
        kwargs={"module_data": ds.module_data, **KWARGS_COMMON},
        **COMMON),
    "linear": Method(
        constr=linear,
        kwargs={"dim": 32, "scale": 0.01},
        **COMMON),
    "simple_mlp": Method(
        constr=MLPOnly,
        kwargs={
            "runtime_data": ds.runtime_data, "module_data": ds.module_data},
        **COMMON),
}

SPARSITY = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _experiment(name, method, p, replicates=100):
    trainer = CrossValidationTrainer(
        ds, partial(method.constr, shape=ds.shape, **method.kwargs),
        optimizer=method.optimizer,
        epochs=method.epochs, epoch_size=method.epoch_size, batch=64)

    pbar = partial(tqdm, desc="{} : {}".format(name, p))
    results = trainer.train_replicates(
        replicates=replicates, p=p, k=25, tqdm=pbar)

    model_dir = os.path.join("results", name)
    os.makedirs(model_dir, exist_ok=True)
    np.savez_compressed(os.path.join(model_dir, "{}.npz".format(p)), **results)


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("methods", nargs="+", default=[])
    p.add_argument("--replicates", type=int, default=100)
    args = p.parse_args()

    for name in args.methods:
        model_dir = os.path.join("results", name)
        for s in SPARSITY:
            _experiment(name, MODELS[name], s, replicates=args.replicates)
