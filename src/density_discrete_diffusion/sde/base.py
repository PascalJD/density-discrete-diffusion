# src/density_discrete_diffusion/sde/base.py
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

class BaseSDE(ABC):
    """Abstract SDE interfaces."""

    @abstractmethod
    def drift(self, x: Array, t: float) -> Array: ...

    @abstractmethod
    def diffusion(self, t: float) -> float | Array: ...

    def sde_step(self, rng: PRNGKeyArray, x: Array, t: float, dt: float):
        """Euler-Maruyama forward step."""
        noise = jax.random.normal(rng, shape=x.shape)
        return x + self.drift(x, t) * dt + jnp.sqrt(dt) * self.diffusion(t) * noise
    
    def pf_ode_step(self, rng: PRNGKeyArray, x: Array, t: float, dt: float):
        raise NotImplementedError

    @abstractmethod
    def prior_sample(self, rng: PRNGKeyArray, shape) -> Array: ...

    @abstractmethod
    def sample_kernel(self, x0: Array, t: float) -> tuple[Array, Array]: