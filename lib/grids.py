"""Grid construction utilities for state and control spaces."""

import jax.numpy as jnp
from jax import Array


def uniform_grid(x_min: float, x_max: float, n: int) -> Array:
    """Create a uniformly spaced grid."""
    return jnp.linspace(x_min, x_max, n)


def exponential_grid(x_min: float, x_max: float, n: int, density: float = 3.0) -> Array:
    """Create a grid with more points concentrated near x_min.

    Args:
        x_min: Lower bound.
        x_max: Upper bound.
        n: Number of grid points.
        density: Controls concentration near x_min. Higher = more points near x_min.
            1.0 gives a uniform grid.
    """
    raw = jnp.linspace(0, 1, n)
    curved = raw ** density
    return x_min + (x_max - x_min) * curved


def chebyshev_nodes(x_min: float, x_max: float, n: int) -> Array:
    """Create Chebyshev nodes on [x_min, x_max]."""
    k = jnp.arange(1, n + 1)
    nodes = jnp.cos((2 * k - 1) / (2 * n) * jnp.pi)
    return 0.5 * (x_max + x_min) + 0.5 * (x_max - x_min) * nodes
