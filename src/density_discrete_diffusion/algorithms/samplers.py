from __future__ import annotations
import jax
import jax.numpy as jnp
from density_discrete_diffusion.sde.base import BaseSDE


def euler_maruyama(
    sde: BaseSDE,
    x_0,
    schedule,
    rng,
    *,
    score_fn=None, 
    generative: bool = True,
    pf_ode: bool = False,
):
    x_0 = x_0.astype(jnp.float32)

    def step(x, args):
        t_next, t_curr, key = args
        dt = t_next - t_curr  # Positive for generative traversal, negative otherwise
        sigma  = sde.diffusion(sde.T - t_curr)  # Noising is time-reversed, sorry
        f = sde.drift(x, sde.T - t_curr)

        if generative and score_fn is not None:
            if pf_ode:
                mu = -f + 0.5 * (sigma ** 2) * score_fn(x, t_curr)
            else:
                mu = -f + (sigma ** 2) * score_fn(x, t_curr)
        x_next = x + mu * dt 
        if not pf_ode:
            noise = jax.random.normal(key, x.shape)
            x_next = x_next + sigma * jnp.sqrt(jnp.abs(dt)) * noise
        return x_next, x_next

    keys = jax.random.split(rng, schedule.shape[0] - 1)
    _, traj = jax.lax.scan(step, x_0, (schedule[1:], schedule[:-1], keys))
    traj = jnp.vstack((x_0[None, ...], traj))
    return traj  # Shape (len(schedule), *x_0.shape)