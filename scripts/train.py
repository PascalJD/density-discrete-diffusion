#!/usr/bin/env python
from __future__ import annotations
import hydra, omegaconf
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: omegaconf.DictConfig):
    import jax, jax.numpy as jnp

    sde = instantiate(cfg.sde)
    model = instantiate(cfg.model, T=cfg.sde.T)
    params = model.init(
        jax.random.PRNGKey(0), jnp.zeros((1,2)), jnp.zeros((1,1))
    )
    dataset = instantiate(cfg.data, batch_size=cfg.trainer.batch_size)
    trainer = instantiate(cfg.trainer,
                          model_apply=model.apply,
                          init_params=params,
                          sde=sde,
                          _convert_="all", 
                          )

    trainer.fit(
        dataset=dataset
    )

if __name__ == "__main__":
    main()