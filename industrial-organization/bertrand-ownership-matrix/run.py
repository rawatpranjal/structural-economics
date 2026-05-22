#!/usr/bin/env python3
"""Multi-Product Bertrand-Nash Pricing and the Ownership Matrix.

Demonstrates how the ownership matrix Omega encodes a multi-product firm's
pricing FOC as a linear system. Covers:
  1. Pre-merger Bertrand-Nash equilibrium via fixed-point iteration.
  2. Marginal-cost recovery by inverting the FOC at observed prices.
  3. Post-merger counterfactual prices under the updated Omega.

Reference: Berry, Levinsohn & Pakes (1995), Nevo (2000), Werden & Froeb (1994).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail

# =============================================================================
# CONSTANTS
# =============================================================================

ALPHA: float = 1.5
X: np.ndarray = np.array([2.0, 1.8, 1.5, 2.2, 1.6])   # product utility intercepts
TRUE_MC: np.ndarray = np.array([1.0, 0.9, 0.8, 1.2, 0.85])

# Pre-merger: Firm A=0, B=1, C=2, D=3; Firm D owns products 3 and 4.
OWNERS_PRE: list[int] = [0, 1, 2, 3, 3]
# Post-merger: Firm A and B merge into AB=0; labels for C and D unchanged.
OWNERS_POST: list[int] = [0, 0, 2, 3, 3]


# =============================================================================
# DEMAND
# =============================================================================

def shares_logit(p: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """Logit shares: s_j = exp(x_j - alpha p_j) / (1 + sum_k exp(x_k - alpha p_k))."""
    v = np.exp(x - alpha * p)
    return v / (1.0 + np.sum(v))


def jacobian_logit(p: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """Demand Jacobian Delta of shape (J, J): Delta[k, j] = d s_k / d p_j.

    Diagonal:  Delta[j, j] = -alpha s_j (1 - s_j)
    Off-diag:  Delta[k, j] =  alpha s_k s_j  for k != j
    """
    s = shares_logit(p, x, alpha)
    Delta = alpha * np.outer(s, s)     # off-diagonal terms: alpha s_k s_j
    np.fill_diagonal(Delta, -alpha * s * (1.0 - s))
    return Delta


# =============================================================================
# OWNERSHIP
# =============================================================================

def ownership_matrix(owners: list[int]) -> np.ndarray:
    """Omega[j, k] = 1 if owners[j] == owners[k], else 0."""
    o = np.asarray(owners)
    return (o[:, None] == o[None, :]).astype(float)


# =============================================================================
# EQUILIBRIUM SOLVER
# =============================================================================

def solve_bertrand_nash(
    c: np.ndarray,
    x: np.ndarray,
    alpha: float,
    owners: list[int],
    p_init: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
    record: bool = False,
) -> tuple[np.ndarray, int, list[float]]:
    """Fixed-point iteration for Bertrand-Nash prices.

    Iterates p <- c - inv(Omega * Delta(p).T) @ s(p) with 0.5-damping.
    Returns (p_star, n_iter, history) where history holds step norms.
    """
    Omega = ownership_matrix(owners)
    p = p_init.copy() if p_init is not None else c + 0.5
    history: list[float] = []

    for i in range(max_iter):
        s = shares_logit(p, x, alpha)
        Delta = jacobian_logit(p, x, alpha)
        M = Omega * Delta.T                         # Hadamard product, (J, J)
        p_raw = c - np.linalg.solve(M, s)           # solve M @ (p - c) = -s
        p_new = 0.5 * p_raw + 0.5 * p              # damped step for stability
        step = float(np.linalg.norm(p_new - p))
        if record:
            history.append(step)
        p = p_new
        if step < tol:
            break

    return p, i + 1, history


# =============================================================================
# COST RECOVERY
# =============================================================================

def recover_marginal_costs(
    p_obs: np.ndarray,
    s_obs: np.ndarray,
    Delta_obs: np.ndarray,
    owners: list[int],
) -> np.ndarray:
    """Recover c from observed (p, s, Delta).

    FOC: s + (Omega * Delta.T) @ (p - c) = 0
    Rearranges to: c = p + inv(Omega * Delta.T) @ s
    """
    Omega = ownership_matrix(owners)
    M = Omega * Delta_obs.T
    return p_obs + np.linalg.solve(M, s_obs)


# =============================================================================
# FIGURES
# =============================================================================

def plot_prices_pre_post(
    prices_pre: np.ndarray,
    prices_post: np.ndarray,
    true_mc: np.ndarray,
    path: str,
) -> None:
    """Bar chart: pre-merger price, post-merger price, and marginal cost per product."""
    J = len(prices_pre)
    x = np.arange(J)
    w = 0.26

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, prices_pre, w, label="Pre-merger price", color="#4878CF", edgecolor="white")
    ax.bar(x,     prices_post, w, label="Post-merger price", color="#D65F5F", edgecolor="white")
    ax.bar(x + w, true_mc,    w, label="Marginal cost",    color="#6ACC65", edgecolor="white")

    # Annotate percent price changes for affected products.
    for j in range(J):
        pct = (prices_post[j] - prices_pre[j]) / prices_pre[j] * 100
        color = "#D65F5F" if pct > 0.01 else "#444444"
        ax.text(x[j], prices_post[j] + 0.01, f"+{pct:.1f}%",
                ha="center", va="bottom", fontsize=8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Product {j}" for j in range(J)], fontsize=9)
    ax.set_ylabel("Price")
    ax.set_title("Bertrand-Nash Prices Before and After Merger\n"
                 "A+B merge (products 0 and 1); D owns products 3 and 4 in both scenarios")
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, path, dpi=150)


def plot_ownership_heatmaps(
    Omega_pre: np.ndarray,
    Omega_post: np.ndarray,
    path: str,
) -> None:
    """Side-by-side binary heatmaps of the pre- and post-merger ownership matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    titles = ["Pre-merger Omega", "Post-merger Omega"]
    matrices = [Omega_pre, Omega_post]

    for ax, mat, title in zip(axes, matrices, titles):
        J = mat.shape[0]
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for i in range(J):
            for j in range(J):
                ax.text(j, i, str(int(mat[i, j])),
                        ha="center", va="center", fontsize=12,
                        color="white" if mat[i, j] else "#555555")
        ax.set_xticks(range(J))
        ax.set_yticks(range(J))
        ax.set_xticklabels([f"P{j}" for j in range(J)], fontsize=9)
        ax.set_yticklabels([f"P{j}" for j in range(J)], fontsize=9)
        ax.set_xlabel("Product j")
        ax.set_ylabel("Product k")
        ax.set_title(title)

    fig.suptitle("Ownership Matrix: Omega[j,k] = 1 when j and k share an owner", y=1.02)
    fig.tight_layout()
    save_figure(fig, path, dpi=150)


def plot_convergence(
    history_pre: list[float],
    history_post: list[float],
    path: str,
) -> None:
    """Semilog-y plot of fixed-point step norms for pre- and post-merger solves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, len(history_pre) + 1), history_pre,
                "o-", color="#4878CF", markersize=4, label="Pre-merger")
    ax.semilogy(range(1, len(history_post) + 1), history_post,
                "s-", color="#D65F5F", markersize=4, label="Post-merger")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step norm  ||p^{t+1} - p^t||")
    ax.set_title("Fixed-Point Convergence of the Bertrand-Nash Solver")
    ax.legend(frameon=False)
    save_figure(fig, path, dpi=150)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    setup_style()

    # ------------------------------------------------------------------
    # Pre-merger equilibrium
    # ------------------------------------------------------------------
    omega_pre = ownership_matrix(OWNERS_PRE)
    p_pre, n_pre, hist_pre = solve_bertrand_nash(
        TRUE_MC, X, ALPHA, OWNERS_PRE, record=True
    )
    s_pre = shares_logit(p_pre, X, ALPHA)

    # Cost recovery sanity check: invert FOC at the computed equilibrium.
    Delta_pre = jacobian_logit(p_pre, X, ALPHA)
    c_recovered = recover_marginal_costs(p_pre, s_pre, Delta_pre, OWNERS_PRE)
    assert np.allclose(c_recovered, TRUE_MC, atol=1e-8), (
        f"Cost recovery failed: max error {np.max(np.abs(c_recovered - TRUE_MC)):.2e}"
    )

    # ------------------------------------------------------------------
    # Post-merger equilibrium
    # ------------------------------------------------------------------
    omega_post = ownership_matrix(OWNERS_POST)
    p_post, n_post, hist_post = solve_bertrand_nash(
        TRUE_MC, X, ALPHA, OWNERS_POST, p_init=p_pre, record=True
    )
    s_post = shares_logit(p_post, X, ALPHA)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_prices_pre_post(p_pre, p_post, TRUE_MC, "figures/prices-pre-post.png")
    plot_ownership_heatmaps(omega_pre, omega_post, "figures/ownership-heatmaps.png")
    plot_convergence(hist_pre, hist_post, "figures/convergence.png")
    save_thumbnail("figures/prices-pre-post.png", "figures/thumb.png")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    markups_pre = p_pre - TRUE_MC
    markups_post = p_post - TRUE_MC
    pct_change = (p_post - p_pre) / p_pre * 100

    print("=" * 64)
    print("BERTRAND-NASH PRICING: PRE vs POST-MERGER")
    print("=" * 64)
    print(f"{'Product':<10} {'Pre price':>10} {'Post price':>11} {'Change %':>10} "
          f"{'Pre share':>10} {'Post share':>11} {'Pre markup':>11} {'Post markup':>12}")
    for j in range(len(TRUE_MC)):
        print(f"  P{j:<7}  {p_pre[j]:>10.4f} {p_post[j]:>11.4f} {pct_change[j]:>9.2f}%"
              f" {s_pre[j]:>10.4f} {s_post[j]:>11.4f}"
              f" {markups_pre[j]:>11.4f} {markups_post[j]:>12.4f}")

    print()
    print(f"Fixed-point iterations: pre-merger = {n_pre}, post-merger = {n_post}")
    print(f"Cost recovery exact match (atol=1e-8): "
          f"max error = {np.max(np.abs(c_recovered - TRUE_MC)):.2e}")
    print()
    print("Products 0 and 1 belong to merged firm AB. Their prices rise the most.")
    print("Products 3 and 4 (firm D, multi-product in both scenarios) also rise")
    print("because the merger relaxes competitive pressure from firm B on product 1,")
    print("which had cross-price effects on all rivals.")


if __name__ == "__main__":
    main()
