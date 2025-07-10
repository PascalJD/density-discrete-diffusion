import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

def check_finite(tensor, name):
    # Works in both jit and eager.  Raises FloatingPointError as soon as a NaN/Inf
    return _check_finite(tensor, f"{name} contains NaN/Inf")

def _nan_like(y):
    """Return an array/tensor of NaNs with the same shape & dtype as y."""
    return jnp.full_like(y, jnp.nan)

def _check_finite(x, msg: str = "NaN/Inf detected"):
    """
    Raise FloatingPointError (via JAX_DEBUG_NANS) the instant x contains
    non-finite values.  Works inside and outside jit; zero overhead in XLA.
    """
    # tree of bools indicating which leaves are finite
    is_finite_tree = tree_map(lambda a: jnp.all(jnp.isfinite(a)), x)
    is_finite_all  = jnp.all(jnp.array(jax.tree_util.tree_leaves(is_finite_tree)))

    def _fail(y):
        # runs only if non-finite detected
        jax.debug.print("{}.", msg)
        return tree_map(_nan_like, y) # same shapes/dtypes
    def _ok(y):
        return y

    return jax.lax.cond(is_finite_all, _ok, _fail, x)