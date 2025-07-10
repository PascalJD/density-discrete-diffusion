import jax.numpy as jnp
from jax import random

from density_discrete_diffusion.data.synthetic_binary import SyntheticBinary


def _check_dataset(ds: SyntheticBinary):
    # 1. shapes
    assert ds.samples.shape == (ds.num_samples, ds.dim)
    assert ds.joint.shape == (2,) * ds.dim
    assert len(ds.marginals) == ds.dim

    # 2. support is {-1, +1}
    assert jnp.all(jnp.isin(ds.samples, jnp.array([-1, 1])))

    # 3. probabilities sum to 1
    assert jnp.isclose(ds.pmf.sum(), 1.0)
    assert jnp.isclose(ds.joint.sum(), 1.0)

    # 4. Marginals are consistent with joint
    for k in range(ds.dim):
        expected = ds.joint.sum(tuple(i for i in range(ds.dim) if i != k))
        assert jnp.allclose(expected, ds.marginal(k))

    # 5. sample_batch returns correct shape and values
    key = random.PRNGKey(0)
    batch = ds.sample_batch(key, 16)
    assert batch.shape == (16, ds.dim)
    assert jnp.all(jnp.isin(batch, jnp.array([-1, 1])))


def test_1d():
    # (+1,)  (-1,)
    pmf = jnp.array([0.3, 0.7])  
    ds = SyntheticBinary(dim=1, pmf=pmf, num_samples=512, seed=0)
    _check_dataset(ds)


def test_2d():
    # (+1,+1) (+1,-1) (-1,+1) (-1,-1)
    pmf = jnp.array([0.4, 0.1, 0.1, 0.4])
    ds = SyntheticBinary(dim=2, pmf=pmf, num_samples=1024, seed=1)
    _check_dataset(ds)


def test_5d_uniform():
    # Uniform 5d
    pmf = jnp.full(32, 1 / 32)  
    ds = SyntheticBinary(dim=5, pmf=pmf, num_samples=2048, seed=2)
    _check_dataset(ds)