"""Orchestration algorithms."""

import numpy as np
import cvxpy as cp

from beartype.typing import Optional, Callable
from jaxtyping import Float32, Int16, UInt8, Bool
from numpy import ndarray as Array

from .jobs import Jobs


#: Valid assignment mask
CandidateMask = Optional[Bool[Array, "Nj Np"]]


#: Orchestration algorithm type that maps a distributed system and job set to
#: platform assignments.
#:
#: Parameters
#: ----------
#: latency: Latency of job i running on platform j
#: utilization: Utilization of job i running on platform j
#: capacity: Capacity of platform j
#: mask: Mask for which assignments should be considered; can be ignored.
#:
#: Returns
#: -------
#: Platform assignments for each module (max: 32767 jobs) along with
#: remaining capacity. If a job is infeasible (not enough capacity), its
#: platform index is -1.
#:
OrchestrationAlgorithm = Callable[
    [
        Float32[Array, "Nj Np"],
        Float32[Array, "Nj Np"],
        Float32[Array, "Np"],
        CandidateMask
    ],
    Optional[Int16[Array, "Nj"]]]


#: Algorithm for limiting possible orchestration candidates.
#:
#: Parameters
#: ----------
#: jobs: input jobs.
#: tid: tower that each platform is connected to (0 if cloud).
#: layer: the tower layer (0=cloud, 1=tower, 2=edge).
#:
#: Returns
#: -------
#: Mask for each (job, platform) assignment that is True for potential
#: deployment candidates.
#:
CandidateAlgorithm = Callable[
    [Jobs, Int16[Array, "Np"], UInt8[Array, "Np"]], CandidateMask]


def unconstrained(
    latency: Float32[Array, "Nj Np"], util: Float32[Array, "Nj Np"],
    capacity: Float32[Array, "Np"], mask: CandidateMask = None
) -> Optional[Int16[Array, "Nj"]]:
    """Unconstrained orchestration which ignores resource constraints.

    Used as a general baseline for what a reasonable latency is.
    """
    return np.argmin(latency, axis=1)


def greedy(
    latency: Float32[Array, "Nj Np"], util: Float32[Array, "Nj Np"],
    capacity: Float32[Array, "Np"], mask: CandidateMask = None
) -> Optional[Int16[Array, "Nj"]]:
    """Greedy orchestration algortihm with a least utilization heuristic."""
    heuristic = np.exp(np.nanmean(np.log(util), axis=1))

    assn = np.ones(latency.shape[0], dtype=np.int16) * -1
    for i in np.argsort(heuristic):
        job_latency: Float32[Array, "Np"] = latency[i]
        job_latency[capacity < util[i]] = np.inf

        choice = np.argmin(job_latency)
        if job_latency[choice] != np.inf:
            assn[i] = choice
            capacity[choice] -= util[i][choice]

    return assn


def ilp(
    latency: Float32[Array, "Nj Np"], util: Float32[Array, "Nj Np"],
    capacity: Float32[Array, "Np"], mask: CandidateMask = None
) -> Optional[Int16[Array, "Nj"]]:
    """Solution using integer linear programming.

    The objective is weighted by the (inverse) unconstrained latency, i.e.
    bin-packing overhead relative to the unconstrained maximum. Masks are
    applied by adding an additional constraint.
    """
    alpha = latency / np.min(latency, axis=1).reshape(-1, 1)
    assn = cp.Variable(latency.shape, integer=True)
    objective = cp.Minimize(cp.sum(cp.multiply(alpha, assn)))
    constraints = [
        assn >= 0, assn <= 1,
        cp.sum(assn, axis=1) == 1,
        cp.sum(cp.multiply(assn, util), axis=0) <= capacity
    ]
    if mask is not None:
        mask[np.isinf(latency)] = False
        constraints.append(assn[~mask] == 0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CBC, maximumSeconds=5, numberThreads=64)

    if assn.value is None:
        return None
    else:
        return np.argmax(assn.value, axis=1).astype(np.int16)


def ilp2(
    latency: Float32[Array, "Nj Np"], util: Float32[Array, "Nj Np"],
    capacity: Float32[Array, "Np"], mask: CandidateMask = None
) -> Optional[Int16[Array, "Nj"]]:
    """Solution using integer linear programming.

    The objective is weighted by the (inverse) unconstrained latency, i.e.
    bin-packing overhead relative to the unconstrained maximum. The mask is
    applied by removing those variables.
    """
    Nj, Np = latency.shape
    if mask is None:
        mask = np.ones_like(latency, dtype=bool)
    mask[np.isinf(latency)] = False

    alpha = (latency / np.min(latency, axis=1).reshape(-1, 1))[mask]
    assn = cp.Variable(np.sum(mask), integer=True)

    # Column, row index mask of each potential placement
    colmask = np.tile(np.arange(Np).reshape(1, -1), (Nj, 1))[mask]
    colidx = np.eye(Np)[np.tile(np.arange(Np).reshape(1, -1), (Nj, 1))[mask]].T
    rowidx = np.eye(Nj)[np.tile(np.arange(Nj).reshape(-1, 1), (1, Np))[mask]].T

    objective = cp.Minimize(cp.sum(cp.multiply(alpha, assn)))
    constraints = [
        assn >= 0, assn <= 1,
        # Each job is assigned to exactly one platform
        cp.matmul(rowidx, assn) == 1,
        # Each platform doesn't exceed its capacity
        cp.matmul(colidx, cp.multiply(util[mask], assn)) <= capacity
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CBC, maximumSeconds=5, numberThreads=64)
    if assn.value is None:
        return None
    else:
        assn_unravel = np.zeros_like(latency)
        assn_unravel[mask] = assn.value
        return np.argmax(assn_unravel, axis=1).astype(np.int16)


def mask_none(
    jobs: Jobs, tid: Int16[Array, "Np"], layer: UInt8[Array, "Np"]
) -> CandidateMask:
    """Don't apply any masking."""
    return None


def mask_direct_path(
    jobs: Jobs, tid: Int16[Array, "Np"], layer: UInt8[Array, "Np"]
) -> CandidateMask:
    """Create candidate mask for the "on direct path" heuristic.

    Candidates must be on the direct path in the network between the source
    and destination node, or in the cloud.
    """
    j = np.arange(layer.shape[0])
    is_src = (jobs.src.reshape(-1, 1) == j.reshape(1, -1))
    is_dst = (jobs.dst.reshape(-1, 1) == j.reshape(1, -1))
    is_srctwr = (
        (tid.reshape(1, -1) == tid[jobs.src].reshape(-1, 1))
        & (layer == 1).reshape(1, -1))
    is_dsttwr = (
        (tid.reshape(1, -1) == tid[jobs.dst].reshape(-1, 1))
        & (layer == 1).reshape(1, -1))
    is_cloud = (tid == 0).reshape(1, -1)
    return is_src | is_dst | is_srctwr | is_dsttwr | is_cloud


def mask_same_cluster(
    jobs: Jobs, tid: Int16[Array, "Np"], layer: UInt8[Array, "Np"]
) -> CandidateMask:
    """Create candidate mask for the "in same tower or direct path" heuristic.

    Candidates can be:
    - On the direct path in the network between source and destination node
    - Have the same tower as the source or destination node
    - In the cloud
    """
    j = np.arange(layer.shape[0])
    is_src = (jobs.src.reshape(-1, 1) == j.reshape(1, -1))
    is_dst = (jobs.dst.reshape(-1, 1) == j.reshape(1, -1))
    is_srctwr = (tid.reshape(1, -1) == tid[jobs.src].reshape(-1, 1))
    is_dsttwr = (tid.reshape(1, -1) == tid[jobs.dst].reshape(-1, 1))
    is_cloud = (tid == 0).reshape(1, -1)
    return is_src | is_dst | is_srctwr | is_dsttwr | is_cloud
