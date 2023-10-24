"""Interference matrix interpretation."""

import numpy as np
from matplotlib import pyplot as plt
from jax import numpy as jnp
import jax

from matplotlib.ticker import PercentFormatter


embeddings = np.load("summary/_embeddings.npz")
data = np.load("data/data.npz")


def _norm(x1, x2):
    return jnp.linalg.norm(jnp.matmul(x1.T, x2), ord=2)


def _get_interference(i):
    mask = z['i_platform'] == i
    diff = np.log(z['t'][mask]) - np.log(baseline[i, z['i_workload'][mask]])
    return np.exp(np.nanmean(np.maximum(0.0, diff)))


if_embedding = jax.vmap(jax.vmap(_norm))(embeddings['Vs'], embeddings['Vg'])
z = np.load("data/if2.npz")
mat = np.load("data/data.npz")
baseline = np.zeros((len(mat["n_platform"]), len(mat["n_workload"])))
baseline[mat["i_platform"], mat["i_workload"]] = mat["t"]
if_actual = np.array(
    [_get_interference(i) for i in range(if_embedding.shape[1])])

architectures = {
    'AMD x86': {'hc-10', 'hc-13', 'hc-14', 'hc-16'},
    'Intel x86': {
        'hc-11', 'hc-12', 'hc-15', 'hc-17', 'hc-18',
        'hc-19', 'hc-31', 'hc-33', 'hc-34', 'hc-35'},
    'ARM A-class': {
        'hc-20', 'hc-21', 'hc-23', 'hc-24', 'hc-25', 'hc-26', 'hc-27',
        'hc-28', 'hc-28', 'hc-29', 'hc-43'},
    'RISC-V': {'hc-42'},
    'ARM M-class': {'hc-80'}
}

architecture_index = np.ones(len(data["n_platform"])) * -1
for i, (k, v) in enumerate(architectures.items()):
    members = np.array([
        i for i, val in enumerate(data["n_platform"])
        if val.split(':')[1] in v])
    architecture_index[members] = i

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
markers = ['.', 'x', '+', '1', '2']
for (i, k), m in zip(enumerate(architectures), markers):
    mask = (architecture_index == i)
    ax.scatter(
        np.mean(if_embedding, axis=0)[mask],
        100 * (if_actual[mask] - 1),
        c='C{}'.format(i), label=k, marker=m)
    ax.set_xscale('log')
    ax.set_yscale('log')

ax.grid(visible=True)
ax.legend(loc='lower right')
ax.set_ylabel("Measured Mean Interference Slowdown")
ax.set_xlabel("Learned $\mathbb{E}[||\mathbf{F}_j||_2]$")
ax.yaxis.set_major_formatter(PercentFormatter())
fig.tight_layout()
fig.savefig("figures/interference_norm.pdf")
