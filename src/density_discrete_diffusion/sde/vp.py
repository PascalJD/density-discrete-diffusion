# src/density_discrete_diffusion/sde/vp.py
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .base import BaseSDE


class VP(BaseSDE):

    def __init__(
        self, 
        name: str = "vp",
        T: float = 10.0,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        scale: float = 1.0
    ):
        # Assumed to be noising p_data -> p_prior
        self.name = name
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.scale = scale

    def _lerp(self, a: float | Array, b: float | Array, w: Array) -> Array:
        return (1.0 - w) * a + w * b
    
    def beta(self, t: Array) -> Array:
        s = jnp.asarray(t) / self.T
        return self._lerp(self.beta_min, self.beta_max, s)
    
    def drift_coefficient(self, t: Array) -> Array:
        return -0.5 * self.beta(t)
    
    def drift(self, x: Array, t: Array) -> Array:
        return self.drift_coefficient(t)[..., None] * x
    
    def diffusion(self, t: Array) -> Array:
        return self.scale * jnp.sqrt(self.beta(t))
    
    def prior_sample(self, rng: PRNGKeyArray, shape) -> Array:
        return jax.random.normal(rng, shape)
    
    def sample_kernel(self, rng: PRNGKeyArray, x0: Array, t: Array) -> tuple[Array, Array]:
        t = jnp.asarray(t)
        beta_t = self.beta(t)
        beta_0 = self.beta_min
        int_drift = -0.25 * (beta_t + beta_0) * t

        mean_coeff = jnp.exp(int_drift)
        var = (1.0 - jnp.exp(2.0 * int_drift)) * (self.scale ** 2)
        std = jnp.sqrt(jnp.maximum(var, 1e-12))  

        eps = jax.random.normal(rng, shape=x0.shape)
        return mean_coeff[..., None] * x0 + std[..., None] * eps