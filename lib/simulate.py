"""Markov chain simulation and ergodic distribution utilities."""

import numpy as np


def simulate_markov(
    trans_matrix: np.ndarray,
    n_periods: int,
    initial_state: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a discrete Markov chain.

    Args:
        trans_matrix: Transition matrix, shape (n_states, n_states).
            Row i gives probabilities of transitioning from state i.
        n_periods: Number of periods to simulate.
        initial_state: Index of the starting state.
        seed: Random seed for reproducibility.

    Returns:
        Array of state indices, shape (n_periods,).
    """
    rng = np.random.default_rng(seed)
    n_states = trans_matrix.shape[0]
    cum_trans = np.cumsum(trans_matrix, axis=1)

    states = np.zeros(n_periods, dtype=int)
    states[0] = initial_state

    draws = rng.uniform(size=n_periods - 1)
    for t in range(n_periods - 1):
        states[t + 1] = np.searchsorted(cum_trans[states[t]], draws[t])
        states[t + 1] = min(states[t + 1], n_states - 1)

    return states


def compute_ergodic_histogram(
    values: np.ndarray,
    bins: int = 50,
    burn: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram of ergodic distribution from simulated values.

    Args:
        values: Simulated time series, shape (n_periods,).
        bins: Number of histogram bins.
        burn: Number of initial periods to discard.

    Returns:
        bin_centers: Center of each bin, shape (bins,).
        densities: Normalized density for each bin, shape (bins,).
    """
    data = values[burn:]
    counts, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts
