"""JAX-compatible interpolation utilities."""

import jax.numpy as jnp
import numpy as np
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


def bilinear_interp(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    x_new: float,
    y_new: float,
) -> float:
    """Bilinear interpolation on a 2D regular grid.

    Args:
        x_grid: Sorted x grid, shape (nx,).
        y_grid: Sorted y grid, shape (ny,).
        values: Function values on the grid, shape (nx, ny).
        x_new: x-coordinate to interpolate at.
        y_new: y-coordinate to interpolate at.

    Returns:
        Interpolated value.
    """
    x_new = np.clip(x_new, x_grid[0], x_grid[-1])
    y_new = np.clip(y_new, y_grid[0], y_grid[-1])

    ix = np.searchsorted(x_grid, x_new) - 1
    iy = np.searchsorted(y_grid, y_new) - 1
    ix = np.clip(ix, 0, len(x_grid) - 2)
    iy = np.clip(iy, 0, len(y_grid) - 2)

    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]

    tx = (x_new - x0) / (x1 - x0) if x1 != x0 else 0.0
    ty = (y_new - y0) / (y1 - y0) if y1 != y0 else 0.0

    v00 = values[ix, iy]
    v10 = values[ix + 1, iy]
    v01 = values[ix, iy + 1]
    v11 = values[ix + 1, iy + 1]

    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + \
           (1 - tx) * ty * v01 + tx * ty * v11
