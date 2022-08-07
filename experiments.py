"""Prediction experiments."""

import os
import numpy as np
from tqdm import tqdm
from functools import partial
from collections import namedtuple
import optax

from dataset import Dataset
from prediction import CrossValidationTrainer
from prediction.models import *


ds = Dataset("data.npz", offset=1000. * 1000.)
Method = namedtuple(
    "Method", ["constr", "kwargs", "optimizer", "epochs", "epoch_size"])

MODELS = {
    "embedding": Method(
        constr=embedding,
        kwargs={
            "runtime_data": ds.runtime_data, "module_data": ds.module_data,
            "layers": [64, 32], "dim": 4, "scale": 0.1},
        optimizer=optax.adam(0.001),
        epochs=100,
        epoch_size=100),
    "linear": Method(
        constr=linear,
        kwargs={"dim": 32, "scale": 0.01},
        optimizer=optax.adam(0.001),
        epochs=100,
        epoch_size=100),
}

SPARSITY = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def _experiment(name, method, p):
    trainer = CrossValidationTrainer(
        ds, partial(method.constr, shape=ds.shape, **method.kwargs),
        optimizer=method.optimizer,
        epochs=method.epochs, epoch_size=method.epoch_size, batch=64)

    pbar = partial(tqdm, desc="{} : {}".format(name, p))
    results = trainer.train_replicates(replicates=100, p=p, k=25, tqdm=pbar)

    model_dir = os.path.join("results", name)
    os.makedirs(model_dir, exist_ok=True)
    np.savez_compressed(os.path.join(model_dir, "{}.npz".format(p)), **results)


if __name__ == "__main__":

    for name, model in MODELS.items():
        model_dir = os.path.join("results", name)
        for s in SPARSITY:
            _experiment(name, model, s)
