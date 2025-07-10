from __future__ import annotations
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from density_discrete_diffusion.sde.base import BaseSDE


class FPLossOut(NamedTuple):
    residual: jnp.ndarray  # (B,)
    loss: jnp.ndarray  # ()


def laplacian(V_scalar_fn):
    def grad_V(x, t):
        return jax.grad(lambda y: V_scalar_fn(y, t))(x)  # (d,)

    def lap_fn(x, t):
        d = x.shape[-1]
        eye = jnp.eye(d, dtype=x.dtype)

        # hvp with each basis vector
        def hvp(v):
            _, hv = jax.jvp(lambda y: grad_V(y, t), (x,), (v,))
            return hv  # (d,)

        diag_elems = jax.vmap(lambda v: jnp.dot(hvp(v), v))(eye)  # (d,)
        return jnp.sum(diag_elems)  # scalar

    return jax.vmap(lap_fn)


def divergence(vec_fn: Callable, x):
    def single(xi):
        return jnp.trace(jax.jacfwd(vec_fn)(xi))  # scalar
    return jax.vmap(single)(x) if x.ndim > 1 else single(x)[None]


def fp_residual_loss(
    params,
    model_apply: Callable, 
    sde: BaseSDE,
    x: jnp.ndarray, 
    t: jnp.ndarray, 
) -> FPLossOut:
    """
    Computes the log-FP residual and its MSE loss, using the noising drift f.
    """

    V_fn = lambda x_, t_: model_apply(params, x_, t_)  # scalar

    gradV_fn = jax.grad(V_fn, argnums=0)                  
    dVdt_fn = jax.grad(V_fn, argnums=1)
    lap_fn = laplacian(V_fn)  

    def residual_single(xi, ti):
        gradV = gradV_fn(xi, ti)  # (d,)
        dVdt = dVdt_fn(xi, ti)  # scalar
        f_val = sde.drift(xi, sde.T - ti)  # (d,)
        sigma = sde.diffusion(sde.T - ti)  # scalar
        div_f  = divergence(lambda y: sde.drift(y, sde.T - ti), xi)[0]
        lapV = lap_fn(xi[None, ...], ti[None, ...])[0]  # scalar
        gradV_sq = jnp.dot(gradV, gradV)
        return (
            dVdt 
            - div_f 
            - jnp.dot(gradV, f_val) 
            + 0.5 * sigma**2 * (lapV + gradV_sq)
        )

    residual = jax.vmap(residual_single)(x, t)  # (B,)
    return FPLossOut(residual=residual, loss=jnp.mean(residual**2))