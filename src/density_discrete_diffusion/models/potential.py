from __future__ import annotations
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from .phi_net import PhiMLP
from .psi_net import PsiMLP


class Potential(nn.Module):

    T: float = 20.0
    tau_max: float = 1.0
    tau_min: float = 0.05
    eps: float = 1e-5
    hidden_dims_phi: Sequence[int] = (128, 128)
    num_time_freqs: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, t, *, return_diag=False):
        t = jnp.asarray(t)
        s = t / self.T 

        # Schedule & relaxed sign
        tau = self.tau_max + (self.tau_min - self.tau_max) * s 
        z = jnp.tanh(x / tau[..., None])
        
        # Energy E_θ and rho
        z_detached = jax.lax.stop_gradient(jnp.sign(x))
        quad = jnp.sum((x - z_detached) ** 2, axis=-1)                             
        quad_term = quad / (2.0 * (self.T - t + self.eps))  
        psi_term = PsiMLP()(z)             
        log_rho = -(quad_term + psi_term)

        # Phi 
        phi_net = PhiMLP(hidden_dims=self.hidden_dims_phi,
                         num_time_freqs=self.num_time_freqs)
        phi = phi_net(x, s)

        # Potential
        V = s * log_rho + s * (1.0 - s) * phi

        if return_diag:
            return V, {
                "tau": tau,
                # "quad_term": quad_term,
                "psi_term": psi_term,
                "log_rho": log_rho,
            }
        return jnp.squeeze(V)