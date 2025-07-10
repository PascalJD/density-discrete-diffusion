from __future__ import annotations
from functools import partial
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm
import logging, os, functools, pathlib, time
import orbax.checkpoint as ocp
import wandb

from density_discrete_diffusion.sde.base import BaseSDE
from density_discrete_diffusion.algorithms.samplers import euler_maruyama
from .losses import fp_residual_loss

log = logging.getLogger(__name__)
logging.getLogger("orbax.checkpoint").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


class TrainState(NamedTuple):
    params: any
    opt_state: optax.OptState
    rng: jax.random.KeyArray


class Trainer:
    def __init__(
        self,
        model_apply: Callable,
        init_params,
        sde: BaseSDE,
        *,
        cfg_ckpt: dict | None = None,
        cfg_wandb: dict | None = None,
        lr: float = 3e-4,
        batch_size: int = 512,
        training_steps: int = 50000,
        sampling_steps: int = 50,
        seed: int = 0,
        log_every: int = 1000,
        save_every: int = 1000,
        visualize_every: int = 1000,
    ):
        self.apply = model_apply
        self.sde   = sde
        self.opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-4))
        self.state = TrainState(
            params=init_params,
            opt_state=self.opt.init(init_params),
            rng=jax.random.PRNGKey(seed),
        )
        self.cfg_ckpt = cfg_ckpt
        self.cfg_wandb = cfg_wandb
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.sampling_steps = sampling_steps
        
        self.log_every = log_every
        self.save_every = save_every
        self.visualize_every = visualize_every

        # Checkpointing
        ckpt_dir = pathlib.Path(self.cfg_ckpt.get("dir",  "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

        self.ckpt_mgr = ocp.CheckpointManager(
            ckpt_dir,
            self.ckptr,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=self.cfg_ckpt.get("max_to_keep", 4),
                create=True,
                best_fn=None,
            ),
        )
        
        # Wandb
        self.use_wandb = bool(self.cfg_wandb)
        if self.use_wandb and not wandb.run:
            wandb.init(**self.cfg_wandb, config={"lr": lr})

        # JIT
        self._train_step = jax.jit(partial(self._train_step_impl, sde=self.sde))

    def fit(self, dataset):
        self._demo_iter = dataset
        self.dataset   = dataset
        pbar = tqdm(range(self.training_steps))
        for step in pbar:
            x_T = next(dataset)
            self.state, metrics = self._train_step(self.state, x_T)

            if step % self.log_every == 0:
                msg = f"step={step:>7d}  loss={metrics['loss']:.3e}  " \
                      f"|residual|={metrics['residual_rms']:.3e}"
                log.info(msg)
                pbar.set_description(msg)

                if self.use_wandb:
                    wandb.log(metrics, step=step)

            if step % self.save_every == 0 and step:
                self.ckpt_mgr.save(
                    step, 
                    self.state, 
                    force=self.cfg_ckpt.get("async_save", True)
                )

            if self.use_wandb and step % self.visualize_every == 0:
                self._log_figures(step)
    
    def _train_step_impl(self, state: TrainState, x_T, *, sde):
        rng, key_t, key_eps = jax.random.split(state.rng, 3)

        t = jax.random.uniform(key_t, (self.batch_size,), minval=0.0, maxval=sde.T)

        # Sample X_t given x_T
        loc, std = sde.marginal_params(x_T, self.sde.T - t)  
        eps = jax.random.normal(key_eps, shape=loc.shape)
        x_t = loc + std[..., None] * eps 

        # Loss and its gradient
        def loss_fn(params):
            out = fp_residual_loss(params, self.apply, sde, x_t, t)
            return out.loss, out
        (loss_val, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # SGD step
        updates, new_opt_state = self.opt.update(grads, state.opt_state, params=state.params)
        new_params = optax.apply_updates(state.params, updates)

        # New state
        new_state = TrainState(params=new_params, opt_state=new_opt_state, rng=rng)

        # Sanity metrics 
        g_norm = optax.global_norm(grads)
        updates, new_opt_state = self.opt.update(grads, state.opt_state, params=state.params)
        u_norm = optax.global_norm(updates)
        p_norm = optax.global_norm(state.params)

        metrics = {
            "loss": loss_val,
            "residual_rms": jnp.sqrt(jnp.mean(out.residual**2)),
            "grad_norm": g_norm,
            "update_norm": u_norm,
            "param_norm": p_norm,
        }
        return new_state, metrics
    
    def _log_figures(self, step):
        rng, key = jax.random.split(self.state.rng)
        x_T = next(self._demo_iter)  # (N,d)
        schedule = jnp.linspace(0., self.sde.T, self.sampling_steps)
        x_0 = self.sde.prior_sample(key, x_T.shape)

        rng, key = jax.random.split(rng)
        params_frozen = jax.tree.map(jax.lax.stop_gradient, self.state.params)
        score = lambda x, t: jax.grad(
                lambda y: self.apply(params_frozen, y, t).sum())(x)
        traj = euler_maruyama(self.sde, x_0, schedule, key,
                              score_fn=score, generative=True, pf_ode=False)
        samples = traj[-1]

        imgs = {}
        if hasattr(self, "dataset") and hasattr(self.dataset, "wandb_figures"):
            for name, fig in self.dataset.wandb_figures(
                    samples=samples, traj=traj).items():
                imgs[name] = wandb.Image(fig)
        wandb.log(imgs, step=step)
        plt.close("all")