"""Run networking simulations."""

import json
import os
import time

import numpy as np

from prediction import Objective
from simulation import NetworkSimulation, SimulatedOrchestrator
from simulation.algorithms import (
    ilp, ilp2, greedy, mask_direct_path, mask_same_cluster)


_desc = "Run networking simulations."


def _parse(p):

    g = p.add_argument_group("Predictor")
    g.add_argument(
        "-p", "--path", default="results", help="Path to model checkpoints.")
    g.add_argument(
        "-d", "--dataset", default="data/data.npz", help="Path to dataset.")
    g.add_argument(
        "--percentile", type=int, default=5, help="Calibration percentile.")

    g = p.add_argument_group("Evaluation")
    g.add_argument(
        "-r", "--replicates", type=int, default=100,
        help="Number of replicates to run.")
    g.add_argument("-k", "--key", default=42, type=int, help="Random key.")
    g.add_argument("-o", "--out", default="results-simulation")
    g.add_argument(
        "-j", "--jobs", type=int, default=100,
        help="Number of jobs; each will receive the same random key.")
    g.add_argument(
        "-n", "--nodes", type=int, default=200, help="Number of nodes.")
    g.add_argument(
        "-s", "--scale", type=float, default=0.02, help="Job scale (problem "
        "difficulty / system load); relative to `nodes/jobs`.")
    g.add_argument(
        "-t", "--limit", type=float, default=30.0,
        help="Time limit for solving (terminates once exceeded).")

    return p


def _evaluate(args, sim, name, orchestrator):
    """Perform main evaluation."""
    base_dir = os.path.join(args.out, name)
    os.makedirs(base_dir, exist_ok=True)

    jobs = [
        sim.jobspec.simulate(np.random.default_rng(i), n=args.jobs)
        for i in range(args.replicates)]
    latency, util, runtime = orchestrator.evaluate(jobs, limit=args.limit)

    res = orchestrator.summarize_results(latency, util, jobs)
    res["t"] = runtime

    out = os.path.join(base_dir, "j={},n={},s={},p={}.json".format(
        args.jobs, args.nodes, args.scale, args.percentile))
    with open(out, 'w') as f:
        print("mean/stderr", np.mean(res["mean"]), np.sqrt(np.var(res["mean"]) / args.replicates) * 2)
        print("l2/l1", np.mean(res["layer2"]), np.mean(res["layer1"]))
        json.dump(res, f, indent=4)


def _main(args):

    _start = time.time()
    scale = args.scale * args.nodes / args.jobs
    sim = NetworkSimulation(
        random=args.key, dataset=args.dataset, job_scale=scale,
        n=args.nodes, n0=int(args.nodes / 20), n1=int(args.nodes / 4))

    with open(os.path.join(args.path, "embedding/128/0.1.json")) as f:
        cfg = json.load(f)
        mf_obj = Objective.from_config(
            sim.dataset, cfg["config"]["objectives"][0])

    def _create(predictor=None, **kwargs):
        if predictor is not None:
            predictor = os.path.join(args.path, predictor, "0.1.npz")
        return SimulatedOrchestrator(
            sim, objective=mf_obj, predictor=predictor, p=args.percentile,
            algorithm=greedy, proposal=None, **kwargs)

    orchestrators = {
        "oracle": _create(predictor=None),
        "pitot": _create(predictor="embedding/128"),
        "paragon": _create(predictor="paragon/128", log_predictor=False),
        "mlp": _create(predictor="baseline/mlp"),
        "baseline": _create(predictor="embedding/128", key="C_bar")
    }
    print("Created simulation: {:.3f}s".format(time.time() - _start))

    _start = time.time()
    for name, orchestrator in orchestrators.items():
        _evaluate(args, sim, name, orchestrator)
    print("Finished experiments: {:.3f}s".format(time.time() - _start))
