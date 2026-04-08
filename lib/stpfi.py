"""Simultaneous Transition and Policy Function Iteration (STPFI) solver.

Implements the algorithm from Cao, Luo, and Nie (2023), "Global DSGE Models",
Review of Economic Dynamics 51, 199-225.

The STPFI algorithm solves for policy and transition functions simultaneously
by iterating on the system of first-order conditions, market clearing, and
consistency equations at each collocation point.
"""

from typing import Callable

import numpy as np


def solve_stpfi(
    step: Callable[[dict, dict], tuple[dict, dict, float]],
    policy_init: dict[str, np.ndarray],
    trans_init: dict[str, np.ndarray],
    tol: float = 1e-6,
    max_iter: int = 500,
    dampen: float = 1.0,
    verbose: bool = True,
) -> tuple[dict, dict, dict]:
    """Simultaneous Transition and Policy Function Iteration.

    At each iteration, solves the full system of equilibrium equations
    (Euler equations, market clearing, consistency equations) at every
    collocation point, updating policy and transition functions jointly.

    Args:
        step: Function (policy, trans) -> (policy_new, trans_new, max_change).
            Solves the equation system at all collocation points for one
            iteration and returns updated policy/transition function arrays
            along with the maximum absolute residual of the equation system.
        policy_init: Initial guess for policy functions as {name: array}.
        trans_init: Initial guess for transition functions as {name: array}.
        tol: Convergence tolerance on policy function sup-norm change.
        max_iter: Maximum iterations.
        dampen: Dampening parameter in (0, 1]. 1.0 = full update,
            <1.0 = weighted average with previous iteration.
        verbose: Print convergence info.

    Returns:
        policy: Converged policy functions.
        trans: Converged transition functions.
        info: Dict with 'iterations', 'converged', 'error', 'residual'.
    """
    policy = {k: v.copy() for k, v in policy_init.items()}
    trans = {k: v.copy() for k, v in trans_init.items()}

    for iteration in range(1, max_iter + 1):
        policy_new, trans_new, max_residual = step(policy, trans)

        # Compute sup-norm change across all policy functions
        max_change = 0.0
        for k in policy:
            diff = np.max(np.abs(policy_new[k] - policy[k]))
            max_change = max(max_change, diff)

        # Dampened update
        if dampen < 1.0:
            for k in policy:
                policy_new[k] = dampen * policy_new[k] + (1.0 - dampen) * policy[k]
            for k in trans:
                trans_new[k] = dampen * trans_new[k] + (1.0 - dampen) * trans[k]

        policy = policy_new
        trans = trans_new

        if verbose and iteration % 10 == 0:
            print(f"  STPFI iteration {iteration:4d}, "
                  f"change = {max_change:.2e}, residual = {max_residual:.2e}")

        if max_change < tol:
            if verbose:
                print(f"  STPFI converged in {iteration} iterations "
                      f"(change = {max_change:.2e}, residual = {max_residual:.2e})")
            return policy, trans, {
                "iterations": iteration,
                "converged": True,
                "error": max_change,
                "residual": max_residual,
            }

    if verbose:
        print(f"  STPFI did NOT converge after {max_iter} iterations "
              f"(change = {max_change:.2e}, residual = {max_residual:.2e})")
    return policy, trans, {
        "iterations": max_iter,
        "converged": False,
        "error": max_change,
        "residual": max_residual,
    }
