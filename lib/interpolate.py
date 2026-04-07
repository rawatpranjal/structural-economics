"""JAX-compatible interpolation utilities."""

import jax.numpy as jnp
from jax import Array


def linear_interp(x_grid: Array, y_values: Array, x_new: Array | float) -> Array:
    """Linear interpolation on a 1D grid.

    Args:
        x_grid: Sorted grid of x values, shape (n,).
        y_values: Corresponding y values, shape (n,).
        x_new: Points at which to interpolate. Scalar or array.

    Returns:
        Interpolated values at x_new.
    """
    return jnp.interp(x_new, x_grid, y_values)
