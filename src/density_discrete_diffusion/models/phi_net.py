from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import flax.linen as nn


class TimeEncoder(nn.Module):
    """Learnable Fourier feature time embedding."""
    num_freqs: int = 8  # produces 2 * num_freqs features

    def setup(self):
        # Phase offsets are learnable; frequencies are fixed (log‑spaced).
        self.phase = self.param("phase",
                                nn.initializers.zeros, (1, self.num_freqs))
        # 0.1 … 100 covers several decades and works well in practice.
        self.coeff = jnp.logspace(jnp.log10(0.1), jnp.log10(100.0),
                                  num=self.num_freqs)[None]

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        angles = t * self.coeff + self.phase 
        sin = jnp.sin(angles)
        cos = jnp.cos(angles)
        return jnp.concatenate([sin, cos], axis=-1)


class PhiMLP(nn.Module):
    """
    time-conditioned phi MLP producing a scalar per sample.
    """
    hidden_dims: Sequence[int] = (128, 128)
    num_time_freqs: int = 8

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,  # (B, d)
        t: jnp.ndarray | float  # (B,) or scalar
    ) -> jnp.ndarray:  # (B,)

        # Time embedding
        time_emb = TimeEncoder(num_freqs=self.num_time_freqs)(t)  # (B, 2K) inline is fine (see setup vs compact)

        if x.ndim == 2 and time_emb.shape[0] == 1 and x.shape[0] > 1:
            time_emb = jnp.broadcast_to(time_emb, (x.shape[0], time_emb.shape[1]))
        
        # broadcast x if it is un‑batched but time_emb has a batch dim
        if x.ndim == 1 and time_emb.ndim == 2:
            x = x[None, :]  

        # Concatenate x and time features
        h = jnp.concatenate([x, time_emb], axis=-1)

        # MLP → scalar
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.gelu(h)
        h = nn.Dense(1)(h)  # (B, 1)
        return h.squeeze(-1)  # (B,)