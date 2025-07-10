from __future__ import annotations
import matplotlib.pyplot as plt
import jax.numpy as jnp
from mpl_toolkits.mplot3d import Axes3D 

__all__ = ["plot_marginals", "plot_joint_3d"]


def plot_marginals(dataset, k: int = 2, ax=None, title=None):
    k = min(k, dataset.dim)
    if ax is None:
        _, ax = plt.subplots()
    bar_w = 0.8 / k
    xs = jnp.arange(2)  # {-1, +1}
    for d in range(k):
        ax.bar(xs - 0.4 + d * bar_w,
               dataset.marginal(d), bar_w, label=f"P(X{d+1})")
    ax.set_xticks(xs, ["+1", "-1"])
    ax.set_ylabel("Probability")
    ax.set_title(title or f"First {k} marginals")
    ax.legend()


def plot_joint_3d(dataset, ax=None, title="Joint PMF on {-1,1}^2"):
    """3-D bar plot of full probability table."""
    joint = dataset.joint_pmf()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Coordinates: (x1,x2) ∈ {+1,-1}
    xpos, ypos = jnp.meshgrid(jnp.array([0, 1]), jnp.array([0, 1]), indexing='ij')
    xpos = xpos.flatten()
    ypos = ypos.flatten()
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
    ax.set_title(title)