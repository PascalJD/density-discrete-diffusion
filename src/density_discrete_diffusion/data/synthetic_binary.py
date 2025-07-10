from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

__all__ = ["SyntheticBinary"]


@dataclass
class SyntheticBinary:
    """Synthetic dataset on {-1, 1}^d with an exact user-defined PMF.

    Parameters
    ----------
    dim : int
        Dimension d (number of binary coordinates).
    pmf : jax.numpy.ndarray
        Flat vector of length 2**d that sums to 1.  Entries must be ordered.
    num_samples : int
        Number of i.i.d. samples to materialize in memory.
    seed : int, optional
        RNG seed for reproducibility (default 0).
    """
    name: str
    dim: int
    pmf: jnp.ndarray  # (2**dim,)
    num_samples: int
    batch_size: int
    seed: int = 0

    # filled in __post_init__
    samples:     jnp.ndarray = field(init=False)  # (N, d)
    joint:       jnp.ndarray = field(init=False)  # (2,)*d
    marginals:   List[jnp.ndarray] = field(init=False)  # list of length d
    _rng: jax.Array = field(init=False)

    def __post_init__(self):
        self.pmf = jnp.asarray(self.pmf, dtype=jnp.float32)
        assert self.pmf.shape == (2 ** self.dim,), (
            f"pmf must have length 2**dim = {2**self.dim}"
        )
        assert jnp.isclose(self.pmf.sum(), 1.0), "PMF must sum to 1."

        # Cache the full joint table of shape (2,)*d
        self.joint = self.pmf.reshape([2] * self.dim)
        self.marginals = [
            self.joint.sum(axis=tuple(i for i in range(self.dim) if i != k))
            for k in range(self.dim)
        ]

        # Materialize num_samples draws
        lookup = jnp.asarray(list(product([1, -1], repeat=self.dim)),
                             dtype=jnp.float32)  # (2**d, d)
        key = jax.random.key(self.seed)
        idx = jax.random.choice(key, lookup.shape[0],
                                shape=(self.num_samples,), p=self.pmf)
        self.samples = lookup[idx]

        self._rng = jax.random.PRNGKey(self.seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int) -> jnp.ndarray:
        return self.samples[i]
    
    def __iter__(self):
        return self

    def __next__(self):
        self._rng, subkey = jax.random.split(self._rng)
        return self.sample_batch(subkey, self.batch_size)

    def sample_batch(self, key, batch_size: int) -> jnp.ndarray:
        """Return a batch of size batch_size (sampling with replacement)."""
        idx = jax.random.randint(key, (batch_size,), 0, self.num_samples)
        return self.samples[idx]

    def joint_pmf(self) -> jnp.ndarray:
        """Alias returning the cached (2,)*d tensor."""
        return self.joint

    def marginal(self, k: int) -> jnp.ndarray:
        """Return P(X_k) for 0 ≤ k < d."""
        return self.marginals[k]

    def wandb_figures(
        self,
        samples: jnp.ndarray | None = None,  # (N,d)  final samples
        traj: jnp.ndarray | None = None  # (T,N,d) entire path
    ) -> dict[str, matplotlib.figure.Figure]:
        """Return a dict {name: mpl Figure} for logging/visualisation."""
        figs = {}

        figs["data_marginals"] = _plot_marginals(self)
        jfig = _plot_joint_3d(self)
        if jfig:
            figs["data_joint"] = jfig

        if samples is not None:
            fig, ax = plt.subplots()
            ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=.7)
            xs = [-1, 1]
            ax.set_xticks(xs); ax.set_yticks(xs)
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
            ax.set_aspect("equal")
            ax.set_title("Model samples (t = T)")
            figs["model_samples"] = fig
            if self.dim == 2:
                figs["model_joint"] = _plot_joint_3d_empirical(samples)

        if traj is not None:
            timesteps = jnp.arange(traj.shape[0])  # (T,)
            for dim in range(traj.shape[2]):  # loop over d
                fig_d, ax_d = plt.subplots()
                for k in range(traj.shape[1]):  # loop over particles
                    ax_d.plot(timesteps, traj[:, k, dim], lw=.5, alpha=.5)
                ax_d.set_xlabel("timestep")
                ax_d.set_ylabel(f"x[{dim}]")
                ax_d.set_title(f"generative trajectories; dim {dim}")
                figs[f"traj_dim{dim}"] = fig_d

        return figs

def _plot_marginals(ds, k=4):
    k = min(k, ds.dim)
    fig, ax = plt.subplots()
    bar_w = 0.8 / k
    xs = jnp.arange(2)  # {-1, +1}
    for d in range(k):
        ax.bar(xs - 0.4 + d * bar_w,
               ds.marginal(d), bar_w, label=f"P(X{d+1})")
    ax.set_xticks(xs, ["+1", "-1"])
    ax.set_ylabel("Probability")
    ax.set_title(f"First {k} marginals")
    ax.legend()
    return fig

def _plot_joint_3d(ds):
    if ds.dim != 2:
        return None
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    joint = ds.joint_pmf()

    xpos, ypos = jnp.meshgrid(jnp.array([0, 1]), jnp.array([0, 1]), indexing='ij')
    xpos, ypos = xpos.flatten(), ypos.flatten()
    zpos = jnp.zeros_like(xpos)
    dx = dy = 0.5 * jnp.ones_like(zpos)
    dz = joint.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
    ax.set_xticks([0.25, 1.25], ["+1", "-1"])
    ax.set_yticks([0.25, 1.25], ["+1", "-1"])
    ax.set_zlim(0, dz.max() * 1.2)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Probability")
    ax.set_title("Joint PMF")
    return fig

def _plot_joint_3d_empirical(samples: jnp.ndarray):
    """
    Build a 3-D bar plot of the empirical joint PMF of the generated samples.
    """
    signs = jnp.where(samples > 0, 1, -1)

    idx0 = (signs[:, 0] == -1).astype(int)      
    idx1 = (signs[:, 1] == -1).astype(int)       
    flat  = idx0 * 2 + idx1                       

    counts = jnp.bincount(flat, length=4)
    pmf_emp = (counts / counts.sum()).reshape(2, 2)

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    xpos, ypos = jnp.meshgrid(jnp.array([0, 1]), jnp.array([0, 1]),
                              indexing='ij')
    xpos, ypos = xpos.flatten(), ypos.flatten()
    zpos = jnp.zeros_like(xpos)
    dx = dy = 0.5 * jnp.ones_like(zpos)
    dz = pmf_emp.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
    ax.set_xticks([0.25, 1.25], ["+1", "-1"])
    ax.set_yticks([0.25, 1.25], ["+1", "-1"])
    ax.set_zlim(0, dz.max() * 1.2)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Probability")
    ax.set_title("Model joint PMF")
    return fig