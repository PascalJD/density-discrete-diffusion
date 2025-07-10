from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import flax.linen as nn



class PsiMLP(nn.Module):
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:  # z : (B, d)
        h = z
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.gelu(h)
        h = nn.Dense(1)(h)  # (B, 1)
        return h.squeeze(-1)  # (B,)