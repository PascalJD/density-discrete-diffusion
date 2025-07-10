import jax
import jax.numpy as jnp

from density_discrete_diffusion.sde.base import BaseSDE


class KarrasSDE(BaseSDE):
    def __init__(self, T: float = 20.0):
        self.T = T

    def drift(self, x, t):
        return 0.0 * x   

    def diffusion(self, t):
        return jnp.sqrt(2.0 * t)

    def marginal_params(self, x0, t):
        return x0, t

    def prior_sample(self, rng, shape):
        return jax.random.normal(rng, shape) * self.T