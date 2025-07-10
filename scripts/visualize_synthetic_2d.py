from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb

from density_discrete_diffusion.data.synthetic_binary import SyntheticBinary
from density_discrete_diffusion.sde.karras import KarrasSDE
from density_discrete_diffusion.utils.plotting import plot_marginals, plot_joint_3d
from density_discrete_diffusion.algorithms.samplers import euler_maruyama


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    ds_cfg = cfg.data
    sde_cfg = cfg.sde

    if ds_cfg.dim != 2:
        raise ValueError(
            f"This script expects a 2-D dataset, got dim={ds_cfg.dim} "
            f"(check config: {OmegaConf.to_yaml(ds_cfg)})"
        )

    dataset = SyntheticBinary(
        dim=ds_cfg.dim,
        pmf=jnp.asarray(ds_cfg.pmf, dtype=jnp.float32),
        num_samples=ds_cfg.num_samples,
        seed=ds_cfg.seed,
    )

    run = wandb.init(
        project="ddc_discrete",
        name=f"{ds_cfg.name}-viz",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Set up the SDE and integration schedule
    sde = instantiate(cfg.sde)
    schedule = jnp.linspace(0, sde.T, num=sde_cfg.steps)

    key = jax.random.PRNGKey(ds_cfg.seed)
    key, keygen = jax.random.split(key)
    samples = dataset.sample_batch(key, batch_size=100)

    keys = jax.random.split(keygen, samples.shape[0])

    em_jit = jax.jit(euler_maruyama, static_argnums=(0,4))

    trajectories = jax.vmap(em_jit, in_axes=(None, 0, None, 0, None))(sde, samples, schedule, keys, False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for trajectory in trajectories:
        ax.plot(schedule, trajectory[:, 0], alpha=0.7)

    ax.set_title("Trajectories of the first dimension over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("X_0")
    ax.grid(True)

    wandb.log({"trajectories_time": wandb.Image(fig)})
    plt.close(fig)

    # Marginals 
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_marginals(dataset, ax=ax, title="Marginals of X₁ and X₂")
    wandb.log({"marginals": wandb.Image(fig)})
    plt.close(fig)

    # 3d plot
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    plot_joint_3d(dataset, ax=ax, title="Joint PMF on ±1²")
    wandb.log({"joint_pmf": wandb.Image(fig)})
    plt.close(fig)

    run.finish()

if __name__ == "__main__":
    main()