"""Run networking simulations."""

import json
import os

import numpy as np
from jax import random

from prediction import Objective
from simulation import SimulatedOrchestrator, NetworkSimulation


_desc = "Run networking simulations."


def _parse(p):
    p.add_argument(
        "-p", "--path", default="results",
        help="Path to training checkpoints.")
    p.add_argument("-o", "--out", default="results-simulation")
    p.add_argument(
        "-d", "--dataset", default="data/data.npz", help="Path to dataset.")
    p.add_argument("-k", "--key", default=42, type=int, help="Random key.")
    p.add_argument(
        "-j", "--jobs", nargs='+', type=int,
        default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        help="Number of jobs; each will receive the same random key.")
    p.add_argument(
        "-n", "--replicates", type=int, default=10,
        help="Number of replicates to run.")
    return p


def _evaluate(key, args, sim, num_jobs, orchestrators):
    """Perform main evaluation."""
    keys = random.split(key, args.replicates)

    jobs = [sim.create_jobs(k, jobs=num_jobs) for k in keys]
    for name, orch in orchestrators.items():
        base_dir = os.path.join(args.out, name)
        os.makedirs(base_dir, exist_ok=True)
        latency, utilization = orch.evaluate(jobs)
        np.savez(
            os.path.join(base_dir, str(num_jobs) + ".npz"),
            latency=latency, utilization=utilization)


def _main(args):

    key = random.PRNGKey(args.key)

    k1, k2 = random.split(key, 2)
    sim = NetworkSimulation(key=k1, dataset=args.dataset, job_scale=0.02)

    with open(os.path.join(args.path, "embedding/128/0.1.json")) as f:
        cfg = json.load(f)
        mf_obj = Objective.from_config(
            sim.dataset, cfg["config"]["objectives"][0])

    def _create(predictor=None, **kwargs):
        if predictor is not None:
            predictor = os.path.join(args.path, predictor, "0.1.npz")
        return SimulatedOrchestrator(
            mf_obj, sim, predictor=predictor, **kwargs)

    orchestrators = {
        "oracle": _create(predictor=None),
        "pitot": _create(predictor="embedding/128"),
        "linear": _create(predictor="linear/128"),
        "mlp": _create(predictor="baseline/mlp"),
        "baseline": _create(predictor="embedding/128", key="C_bar")
    }
    for j in args.jobs:
        _evaluate(k2, args, sim, j, orchestrators)
