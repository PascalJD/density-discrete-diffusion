# src/density_discrete_diffusion/sde/poisson_sde.py
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .base import BaseSDE
from .vp import VP


class PoissonDiffusion(BaseSDE):
    """Add marked Poisson jumps to a continuous-time SDE.
    """
    def __init__(
        self,
        base_sde: BaseSDE,
        jump_rate: float,
        mark_sampler,  # callable
        mark_effect,  # callable (x, z) -> x + gamma(x,z)
        name: str | None = None,
    ):
        self._base = base_sde
        self._lambda = float(jump_rate)
        self._sample_z = mark_sampler
        self._apply_gamma = mark_effect
        self.name = name or f"{base_sde.name}-poisson"

    def drift(self, x: Array, t: Array) -> Array:
        return self._base.drift(x, t)

    def diffusion(self, t: Array) -> Array:
        return self._base.diffusion(t)  

    def prior_sample(self, rng: PRNGKeyArray, shape) -> Array:
        return self._base.prior_sample(rng, shape)

    def sample_kernel(
        self, rng: PRNGKeyArray, x0: Array, t: Array
    ) -> Array:  # returns x_t sample
        """
        Stick-then-diffuse mixture
        """
        pass
