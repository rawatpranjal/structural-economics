"""Value function iteration solver."""

from typing import Callable

import jax.numpy as jnp
from jax import Array


def solve_vfi(
    bellman: Callable[[Array], tuple[Array, Array]],
    v_init: Array,
    tol: float = 1e-6,
    max_iter: int = 1000,
    verbose: bool = True,
) -> tuple[Array, Array, dict]:
    """Generic value function iteration.

    Args:
        bellman: Function that takes V and returns (V_new, policy).
        v_init: Initial guess for value function.
        tol: Convergence tolerance (sup norm).
        max_iter: Maximum iterations.
        verbose: Print convergence info.

    Returns:
        v_star: Converged value function.
        policy: Optimal policy at convergence.
        info: Dict with 'iterations', 'converged', 'error'.
    """
    v = v_init.copy()

    for iteration in range(1, max_iter + 1):
        v_new, policy = bellman(v)
        error = float(jnp.max(jnp.abs(v_new - v)))

        if verbose and iteration % 50 == 0:
            print(f"  VFI iteration {iteration:4d}, error = {error:.2e}")

        v = v_new

        if error < tol:
            if verbose:
                print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            return v, policy, {"iterations": iteration, "converged": True, "error": error}

    print(f"  VFI did NOT converge after {max_iter} iterations (error = {error:.2e})")
    return v, policy, {"iterations": max_iter, "converged": False, "error": error}
