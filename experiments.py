"""Prediction experiments."""

import os
import numpy as np
from tqdm import tqdm
from jax import random, vmap
from functools import partial

from dataset import Dataset
from prediction import CrossValidationTrainer
from prediction.models import *


ds = Dataset("data.npz", offset=1000. * 1000.)


MODELS = {
    "linear": (MFLinear, {"dim": 32, "scale": 0.01}),
    "logsumexp": (MFLogSumExp, {"dim": 32, "scale": 0.01}),
    "embedding": (MFEmbedding, {
        "runtime_data": ds.runtime_data, "module_data": ds.module_data,
        "layers": [64, 32], "dim": 4, "scale": 0.1})
}

SPARSITY = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def _experiment(model, constr, kwargs, p):
    trainer = CrossValidationTrainer(
        ds, partial(constr, name=model, samples=ds.shape, **kwargs),
        optimizer=constr.optimizer,
        epochs=constr.epochs, epoch_size=constr.epoch_size, batch=64)

    results = trainer.train_replicates(
        random.PRNGKey(42), replicates=1000, p=p, k=25,
        tqdm=partial(tqdm, desc="{} : {}".format(model, p)))
    np.savez_compressed(os.path.join(model_dir, "{}.npz".format(p)), **results)


if __name__ == "__main__":

    key = random.PRNGKey(42)

    for model, (constr, kwargs) in MODELS.items():
        model_dir = os.path.join("results", model)
        os.makedirs(model_dir, exist_ok=True)

        for s in SPARSITY:
            _experiment(model, constr, kwargs, s)
