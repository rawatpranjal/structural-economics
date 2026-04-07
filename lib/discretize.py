"""Discretization methods for continuous stochastic processes."""

import jax.numpy as jnp
from jax import Array
from scipy.stats import norm


def rouwenhorst(n: int, mu: float, sigma: float, rho: float) -> tuple[Array, Array, Array]:
    """Rouwenhorst method for discretizing an AR(1) process.

    Discretizes: z' = mu + rho * (z - mu) + sigma * epsilon, epsilon ~ N(0,1)

    Args:
        n: Number of grid points.
        mu: Unconditional mean.
        sigma: Unconditional standard deviation of the innovation.
        rho: Persistence parameter.

    Returns:
        grid: (n, 1) array of grid points.
        trans: (n, n) transition matrix.
        dist: (n, 1) ergodic distribution.
    """
    width = jnp.sqrt((n - 1) * sigma**2 / (1 - rho**2))
    grid = jnp.linspace(mu - width, mu + width, n).reshape(n, 1)

    p0 = (1 + rho) / 2
    trans = jnp.array([[p0, 1 - p0], [1 - p0, p0]])

    if n > 2:
        for _ in range(n - 2):
            size = trans.shape[0]
            zeros_col = jnp.zeros((size, 1))
            zeros_row = jnp.zeros((1, size + 1))
            trans = (
                p0 * jnp.block([[trans, zeros_col], [zeros_row]])
                + (1 - p0) * jnp.block([[zeros_col, trans], [zeros_row]])
                + (1 - p0) * jnp.block([[zeros_row], [trans, zeros_col]])
                + p0 * jnp.block([[zeros_row], [zeros_col, trans]])
            )
        trans = trans / trans.sum(axis=1, keepdims=True)

    dist = jnp.ones((1, n)) / n
    for i in range(1, 101):
        dist = dist @ jnp.linalg.matrix_power(trans, i)
    dist = dist / dist.sum()

    return grid, trans, dist.T


def tauchen(rho: float, sigma: float, n: int, m: float = 3.0) -> tuple[Array, Array]:
    """Tauchen method for discretizing an AR(1) process.

    Discretizes: z' = rho * z + sigma * epsilon, epsilon ~ N(0,1)

    Args:
        rho: Persistence parameter.
        sigma: Standard deviation of innovation.
        n: Number of grid points.
        m: Width of grid in standard deviations.

    Returns:
        grid: (n,) array of grid points.
        trans: (n, n) transition matrix.
    """
    sigma_z = sigma / jnp.sqrt(1 - rho**2)
    grid = jnp.linspace(-m * sigma_z, m * sigma_z, n)
    step = grid[1] - grid[0]

    trans_np = jnp.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j == 0:
                trans_np = trans_np.at[i, j].set(
                    norm.cdf((grid[j] - rho * grid[i] + step / 2) / sigma)
                )
            elif j == n - 1:
                trans_np = trans_np.at[i, j].set(
                    1 - norm.cdf((grid[j] - rho * grid[i] - step / 2) / sigma)
                )
            else:
                trans_np = trans_np.at[i, j].set(
                    norm.cdf((grid[j] - rho * grid[i] + step / 2) / sigma)
                    - norm.cdf((grid[j] - rho * grid[i] - step / 2) / sigma)
                )

    return grid, trans_np


def discrete_normal(n: int, mu: float, sigma: float, width: float = 3.0) -> tuple[float, Array, Array]:
    """Equally spaced approximation to a normal distribution.

    Args:
        n: Number of points.
        mu: Mean.
        sigma: Standard deviation.
        width: Multiple of sigma for grid width.

    Returns:
        error: Approximation error in standard deviation.
        grid: (n, 1) array of grid points.
        probs: (n, 1) array of probabilities.
    """
    grid = jnp.linspace(mu - width * sigma, mu + width * sigma, n).reshape(n, 1)

    if n == 2:
        probs = 0.5 * jnp.ones((n, 1))
    else:
        probs = jnp.zeros((n, 1))
        probs = probs.at[0, 0].set(
            norm.cdf(float(grid[0] + 0.5 * (grid[1] - grid[0])), mu, sigma)
        )
        for i in range(1, n - 1):
            probs = probs.at[i, 0].set(
                norm.cdf(float(grid[i] + 0.5 * (grid[i + 1] - grid[i])), mu, sigma)
                - norm.cdf(float(grid[i] - 0.5 * (grid[i] - grid[i - 1])), mu, sigma)
            )
        probs = probs.at[n - 1, 0].set(1 - float(probs[:n - 1, 0].sum()))

    ex = float(grid.T @ probs)
    sdx = float(jnp.sqrt((grid.T**2) @ probs - ex**2))
    error = sdx - sigma

    return error, grid, probs
