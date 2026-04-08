#!/usr/bin/env python3
"""GARP Testing via Warshall's Algorithm.

Tests the Generalized Axiom of Revealed Preference (GARP) using Warshall's
algorithm to compute the transitive closure of the direct revealed preference
relation. Includes Bronars (1987) power analysis and Afriat (1967) efficiency
index computation.

Reference: Varian (1982), Bronars (1987), Afriat (1967).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


# =============================================================================
# Core algorithms
# =============================================================================

def direct_revealed_preference(prices, quantities):
    """Build the direct revealed preference matrix R.

    R[i, j] = 1 iff bundle i is directly revealed preferred to bundle j,
    i.e., p_i . x_i >= p_i . x_j  (bundle j was affordable when i was chosen).

    Parameters
    ----------
    prices : ndarray, shape (T, K)
        Price vectors for T observations and K goods.
    quantities : ndarray, shape (T, K)
        Quantity vectors for T observations and K goods.

    Returns
    -------
    R : ndarray, shape (T, T)
        Direct revealed preference matrix (binary).
    expenditures : ndarray, shape (T,)
        Own expenditures p_i . x_i.
    """
    T = prices.shape[0]
    # expenditures[i] = p_i . x_i
    expenditures = np.sum(prices * quantities, axis=1)
    # cost_matrix[i, j] = p_i . x_j  (cost of bundle j at prices i)
    cost_matrix = prices @ quantities.T
    # R[i, j] = 1 iff p_i . x_i >= p_i . x_j
    R = (expenditures[:, None] >= cost_matrix).astype(int)
    return R, expenditures


def warshall_transitive_closure(R):
    """Compute transitive closure R* using Warshall's algorithm in O(T^3).

    R*[i, j] = 1 iff there is a chain i R k1 R k2 ... R j.

    Parameters
    ----------
    R : ndarray, shape (T, T)
        Direct relation matrix (binary).

    Returns
    -------
    R_star : ndarray, shape (T, T)
        Transitive closure (binary).
    steps : list of ndarray
        Intermediate matrices after each pivot (for visualization).
    """
    T = R.shape[0]
    R_star = R.copy()
    steps = [R.copy()]

    for k in range(T):
        for i in range(T):
            if R_star[i, k]:
                for j in range(T):
                    if R_star[k, j]:
                        R_star[i, j] = 1
        steps.append(R_star.copy())

    return R_star, steps


def check_garp(R_star, prices, quantities):
    """Check GARP violations.

    GARP is violated if i R* j (i is revealed preferred to j through a chain)
    AND p_j . x_j > p_j . x_i (j could have afforded i's bundle strictly cheaper,
    meaning j is strictly directly revealed preferred to i).

    Returns
    -------
    violations : list of (i, j) pairs where GARP is violated.
    """
    T = prices.shape[0]
    expenditures = np.sum(prices * quantities, axis=1)
    cost_matrix = prices @ quantities.T
    violations = []

    for i in range(T):
        for j in range(T):
            if i != j and R_star[i, j]:
                # Check if j is strictly directly revealed preferred to i
                # i.e., p_j . x_j > p_j . x_i  (bundle i was strictly cheaper at j's prices)
                if expenditures[j] > cost_matrix[j, i]:
                    violations.append((i, j))

    return violations


def find_violation_cycle(R_star, R, violations):
    """Find a cycle path for the first violation for visualization.

    Given violation (i, j): i R* j and j strictly RP i.
    We reconstruct the chain i -> ... -> j using BFS on R.
    """
    if not violations:
        return []
    i, j = violations[0]
    T = R.shape[0]

    # BFS from i to j through R
    visited = {i}
    queue = [(i, [i])]
    while queue:
        current, path = queue.pop(0)
        if current == j:
            return path + [i]  # close the cycle back to i (since j RP i)
        for nxt in range(T):
            if nxt not in visited and R[current, nxt] and current != nxt:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))

    # Fallback: direct
    return [i, j, i]


# =============================================================================
# Data generation
# =============================================================================

def generate_cobb_douglas_data(T, K, rng, alpha=None):
    """Generate consumption data from a Cobb-Douglas utility maximizer.

    A Cobb-Douglas consumer with utility u(x) = prod(x_k^alpha_k) always
    satisfies GARP, because the data is rationalizable.

    Parameters
    ----------
    T : int
        Number of observations.
    K : int
        Number of goods.
    rng : np.random.Generator
        Random number generator.
    alpha : ndarray, optional
        Budget share parameters (must sum to 1).

    Returns
    -------
    prices : ndarray, shape (T, K)
    quantities : ndarray, shape (T, K)
    """
    if alpha is None:
        alpha = np.ones(K) / K  # equal shares

    # Random prices and incomes
    prices = rng.uniform(0.5, 3.0, size=(T, K))
    incomes = rng.uniform(5.0, 15.0, size=T)

    # Cobb-Douglas demand: x_k = alpha_k * m / p_k
    quantities = alpha[None, :] * incomes[:, None] / prices

    return prices, quantities


def generate_random_data(T, K, rng):
    """Generate random (non-rational) consumption data.

    Random bundles on the budget hyperplane — no optimization.
    """
    prices = rng.uniform(0.5, 3.0, size=(T, K))
    incomes = rng.uniform(5.0, 15.0, size=T)

    # Random budget shares (on the simplex)
    shares = rng.dirichlet(np.ones(K), size=T)
    quantities = shares * incomes[:, None] / prices

    return prices, quantities


# =============================================================================
# Bronars power test
# =============================================================================

def bronars_power(prices, quantities_rational, n_sims, rng):
    """Compute Bronars (1987) power: fraction of random datasets that violate GARP.

    Uses the same prices as the rational data but replaces quantities with
    random Becker-type alternatives (uniform on the budget set).

    Parameters
    ----------
    prices : ndarray, shape (T, K)
    quantities_rational : ndarray, shape (T, K)
        The rational data (used only for budget computation).
    n_sims : int
        Number of random simulations.
    rng : np.random.Generator

    Returns
    -------
    power : float
        Fraction of random datasets that violate GARP.
    """
    T, K = prices.shape
    expenditures = np.sum(prices * quantities_rational, axis=1)
    n_violations = 0

    for _ in range(n_sims):
        # Random bundles on the budget set
        shares = rng.dirichlet(np.ones(K), size=T)
        q_random = shares * expenditures[:, None] / prices
        R, _ = direct_revealed_preference(prices, q_random)
        R_star, _ = warshall_transitive_closure(R)
        violations = check_garp(R_star, prices, q_random)
        if violations:
            n_violations += 1

    return n_violations / n_sims


def bronars_power_by_T(K, T_values, n_sims, rng):
    """Compute Bronars power as a function of T (number of observations)."""
    powers = []
    for T in T_values:
        p, q = generate_cobb_douglas_data(T, K, rng)
        pw = bronars_power(p, q, n_sims, rng)
        powers.append(pw)
        print(f"  Bronars power for T={T:3d}: {pw:.3f}")
    return np.array(powers)


# =============================================================================
# Afriat efficiency index
# =============================================================================

def afriat_efficiency_index(prices, quantities, n_grid=200):
    """Compute the Afriat (1967) efficiency index via binary search.

    The efficiency index e* is the largest e in [0, 1] such that the
    e-adjusted data satisfies GARP. The adjustment deflates expenditures:
    e * p_i . x_i >= p_i . x_j implies i R_e j.

    Parameters
    ----------
    prices : ndarray, shape (T, K)
    quantities : ndarray, shape (T, K)
    n_grid : int
        Grid points for binary search.

    Returns
    -------
    e_star : float
        The critical efficiency level.
    """
    T = prices.shape[0]
    expenditures = np.sum(prices * quantities, axis=1)
    cost_matrix = prices @ quantities.T

    def garp_at_e(e):
        # Adjusted direct RP: e * p_i . x_i >= p_i . x_j
        R_e = (e * expenditures[:, None] >= cost_matrix).astype(int)
        R_star, _ = warshall_transitive_closure(R_e)
        # Check strict violation: i R*_e j and p_j . x_j > e * p_j . x_i
        # Simplified: use standard GARP check on the adjusted relation
        for i in range(T):
            for j in range(T):
                if i != j and R_star[i, j]:
                    if e * expenditures[j] > cost_matrix[j, i]:
                        return False
        return True

    # Binary search for e*
    lo, hi = 0.0, 1.0
    for _ in range(30):  # ~30 iterations gives precision ~1e-9
        mid = (lo + hi) / 2
        if garp_at_e(mid):
            lo = mid
        else:
            hi = mid

    return lo


def afriat_distribution(K, T, n_datasets, rng):
    """Compute the distribution of Afriat efficiency indices for random data."""
    indices = []
    for s in range(n_datasets):
        p, q = generate_random_data(T, K, rng)
        e = afriat_efficiency_index(p, q)
        indices.append(e)
        if (s + 1) % 20 == 0:
            print(f"  Afriat index: {s+1}/{n_datasets} datasets computed")
    return np.array(indices)


# =============================================================================
# Main
# =============================================================================

def main():
    rng = np.random.default_rng(42)

    # Parameters
    K = 3       # number of goods
    T = 15      # number of observations

    # =========================================================================
    # Generate data
    # =========================================================================
    print("Generating Cobb-Douglas (rational) data...")
    alpha = np.array([0.4, 0.35, 0.25])
    prices_cd, quantities_cd = generate_cobb_douglas_data(T, K, rng, alpha=alpha)

    print("Generating random (irrational) data...")
    prices_rand, quantities_rand = generate_random_data(T, K, rng)

    # =========================================================================
    # GARP test on Cobb-Douglas data
    # =========================================================================
    print("\nTesting GARP on Cobb-Douglas data...")
    R_cd, exp_cd = direct_revealed_preference(prices_cd, quantities_cd)
    R_star_cd, steps_cd = warshall_transitive_closure(R_cd)
    violations_cd = check_garp(R_star_cd, prices_cd, quantities_cd)
    print(f"  Direct RP edges: {R_cd.sum() - T}")
    print(f"  Transitive closure edges: {R_star_cd.sum() - T}")
    print(f"  GARP violations: {len(violations_cd)}")

    # =========================================================================
    # GARP test on random data
    # =========================================================================
    print("\nTesting GARP on random data...")
    R_rand, exp_rand = direct_revealed_preference(prices_rand, quantities_rand)
    R_star_rand, steps_rand = warshall_transitive_closure(R_rand)
    violations_rand = check_garp(R_star_rand, prices_rand, quantities_rand)
    print(f"  Direct RP edges: {R_rand.sum() - T}")
    print(f"  Transitive closure edges: {R_star_rand.sum() - T}")
    print(f"  GARP violations: {len(violations_rand)}")

    # =========================================================================
    # Bronars power as function of T
    # =========================================================================
    print("\nComputing Bronars power as function of T...")
    T_values = np.array([5, 8, 10, 15, 20, 25, 30, 40, 50])
    n_sims_power = 200
    powers = bronars_power_by_T(K, T_values, n_sims_power, rng)

    # =========================================================================
    # Afriat efficiency index distribution
    # =========================================================================
    print("\nComputing Afriat efficiency index distribution...")
    n_afriat = 100
    afriat_indices = afriat_distribution(K, T, n_afriat, rng)
    print(f"  Mean Afriat index: {afriat_indices.mean():.4f}")
    print(f"  Std Afriat index:  {afriat_indices.std():.4f}")
    print(f"  Fraction with e*=1 (consistent): {(afriat_indices > 0.999).mean():.3f}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "GARP Testing via Warshall's Algorithm",
        "Testing the Generalized Axiom of Revealed Preference using transitive closure.",
    )

    report.add_overview(
        "The Generalized Axiom of Revealed Preference (GARP) is the fundamental testable "
        "implication of utility maximization. Given a dataset of prices and chosen bundles, "
        "GARP asks: could this data have been generated by a consumer maximizing a well-behaved "
        "utility function subject to a budget constraint?\n\n"
        "Testing GARP requires computing the *transitive closure* of the direct revealed "
        "preference relation. Warshall's algorithm does this in $O(T^3)$ time, making GARP "
        "testing practical even for large datasets. We also implement the Bronars (1987) power "
        "test and the Afriat (1967) efficiency index."
    )

    report.add_equations(
        r"""
**Direct Revealed Preference:** Bundle $x_i$ is *directly revealed preferred* to $x_j$ if

$$p_i \cdot x_i \ge p_i \cdot x_j$$

i.e., bundle $j$ was affordable when $i$ was chosen.

**Transitive Closure (Warshall):** $i \; R^* \; j$ iff there exists a chain $i \; R \; k_1 \; R \; k_2 \; R \cdots R \; j$.

$$R^{(k)}[i,j] = R^{(k-1)}[i,j] \;\lor\; \bigl(R^{(k-1)}[i,k] \;\land\; R^{(k-1)}[k,j]\bigr)$$

**GARP Violation:** The data violates GARP if there exist $i, j$ such that $i \; R^* \; j$ (revealed preferred through a chain) and $p_j \cdot x_j > p_j \cdot x_i$ (bundle $i$ was strictly cheaper at $j$'s prices).

**Afriat Efficiency Index:** The largest $e \in [0,1]$ such that the $e$-adjusted data satisfies GARP, where the adjusted relation uses $e \cdot p_i \cdot x_i \ge p_i \cdot x_j$.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $K$ | {K} | Number of goods |\n"
        f"| $T$ | {T} | Number of observations |\n"
        f"| $\\alpha$ | ({alpha[0]}, {alpha[1]}, {alpha[2]}) | Cobb-Douglas budget shares |\n"
        f"| Bronars sims | {n_sims_power} | Random datasets per T value |\n"
        f"| Afriat datasets | {n_afriat} | Random datasets for efficiency index |"
    )

    report.add_solution_method(
        "**Warshall's Algorithm:** Starting from the direct revealed preference matrix $R$, "
        "we iterate over each observation $k$ as a potential intermediate node. For each pair "
        "$(i, j)$, if $i$ can reach $k$ and $k$ can reach $j$, then $i$ can reach $j$. After "
        "$T$ pivots, the matrix $R^*$ encodes all transitive preference chains.\n\n"
        "**Complexity:** $O(T^3)$ for the transitive closure, $O(T^2 K)$ for building the "
        "direct preference matrix.\n\n"
        f"**Results:** Cobb-Douglas data: {len(violations_cd)} GARP violations (as expected). "
        f"Random data: {len(violations_rand)} GARP violations detected."
    )

    # --- Figure 1: Warshall algorithm visualization ---
    # Show the direct R and transitive closure R* side by side, plus a couple of intermediate steps
    n_show = min(10, T)  # show a subset for readability
    fig1, axes1 = plt.subplots(1, 4, figsize=(18, 4.5))

    step_indices = [0, T // 3, 2 * T // 3, T]
    step_labels = ["Direct $R$ (step 0)", f"After pivot $k={T//3}$",
                   f"After pivot $k={2*T//3}$", f"Transitive $R^*$ (step {T})"]

    # Use random data steps for more interesting visualization
    display_steps = steps_rand
    for ax, si, label in zip(axes1, step_indices, step_labels):
        mat = display_steps[si][:n_show, :n_show]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Observation $j$")
        ax.set_ylabel("Observation $i$")
        ax.set_xticks(range(n_show))
        ax.set_yticks(range(n_show))
        ax.set_xticklabels(range(n_show), fontsize=7)
        ax.set_yticklabels(range(n_show), fontsize=7)
        # Annotate cells
        for ii in range(n_show):
            for jj in range(n_show):
                ax.text(jj, ii, int(mat[ii, jj]), ha="center", va="center",
                        fontsize=6, color="white" if mat[ii, jj] > 0.5 else "black")
    fig1.suptitle("Warshall's Algorithm: Step-by-Step Transitive Closure", fontsize=13, y=1.02)
    fig1.tight_layout()
    report.add_figure("figures/warshall-steps.png",
                      "Warshall algorithm: progressive construction of the transitive closure from direct preferences",
                      fig1,
                      description="Each panel shows the preference matrix after using a new observation as a pivot node. "
                      "Watch the matrix fill in as indirect preferences are discovered: if $i$ prefers $k$ and $k$ prefers $j$, "
                      "then $i$ indirectly prefers $j$. The final matrix encodes all transitive preference chains needed for GARP testing.")

    # --- Figure 2: GARP violation detection ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: Cobb-Douglas (no violations)
    mat_cd = R_star_cd[:n_show, :n_show].copy().astype(float)
    im_cd = ax2a.imshow(mat_cd, cmap="Greens", vmin=0, vmax=1, aspect="equal")
    ax2a.set_title("Cobb-Douglas (GARP Satisfied)", fontsize=11)
    ax2a.set_xlabel("Observation $j$")
    ax2a.set_ylabel("Observation $i$")
    ax2a.set_xticks(range(n_show))
    ax2a.set_yticks(range(n_show))
    ax2a.set_xticklabels(range(n_show), fontsize=7)
    ax2a.set_yticklabels(range(n_show), fontsize=7)

    # Right: Random data (with violations highlighted)
    mat_rand = R_star_rand[:n_show, :n_show].copy().astype(float)
    # Create a custom colormap: 0=white, 1=blue, 2=red (violation)
    for vi, vj in violations_rand:
        if vi < n_show and vj < n_show:
            mat_rand[vi, vj] = 2.0

    cmap_viol = mcolors.ListedColormap(["white", "#4a90d9", "#e74c3c"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm_viol = mcolors.BoundaryNorm(bounds, cmap_viol.N)
    im_rand = ax2b.imshow(mat_rand, cmap=cmap_viol, norm=norm_viol, aspect="equal")
    ax2b.set_title(f"Random Data ({len(violations_rand)} GARP Violations)", fontsize=11)
    ax2b.set_xlabel("Observation $j$")
    ax2b.set_ylabel("Observation $i$")
    ax2b.set_xticks(range(n_show))
    ax2b.set_yticks(range(n_show))
    ax2b.set_xticklabels(range(n_show), fontsize=7)
    ax2b.set_yticklabels(range(n_show), fontsize=7)

    # Add cycle annotation if violations exist
    if violations_rand:
        cycle = find_violation_cycle(R_star_rand, R_rand, violations_rand)
        cycle_str = " -> ".join([str(c) for c in cycle])
        ax2b.set_xlabel(f"Observation $j$\nViolation cycle: {cycle_str}", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="white", edgecolor="black", label="No relation"),
                       Patch(facecolor="#4a90d9", label="$R^*$ (revealed preferred)"),
                       Patch(facecolor="#e74c3c", label="GARP violation")]
    ax2b.legend(handles=legend_elements, loc="upper left", fontsize=8,
                bbox_to_anchor=(0.0, -0.18), ncol=3)

    fig2.tight_layout()
    report.add_figure("figures/garp-violations.png",
                      "GARP violation detection: Cobb-Douglas data (left, no violations) vs random data (right, violations in red)",
                      fig2,
                      description="The clean green matrix for Cobb-Douglas data confirms that utility-maximizing behavior never produces GARP violations -- "
                      "a direct consequence of Afriat's theorem. Red cells in the random data matrix pinpoint the exact observation pairs where the consumer's choices "
                      "are mutually contradictory, making rationalization by any utility function impossible.")

    # --- Figure 3: Bronars power as function of T ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(T_values, powers, "b-o", linewidth=2, markersize=6)
    ax3.set_xlabel("Number of observations $T$")
    ax3.set_ylabel("Bronars power (fraction violating GARP)")
    ax3.set_title("Bronars (1987) Power of GARP Test")
    ax3.set_ylim(-0.05, 1.05)
    ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax3.fill_between(T_values, powers, alpha=0.15, color="blue")
    report.add_figure("figures/bronars-power.png",
                      "Bronars power: fraction of random datasets violating GARP, increasing with T",
                      fig3,
                      description="Bronars power answers the question: if the consumer were choosing randomly instead of optimizing, how often would we catch them? "
                      "The rapid rise to near 100% means that GARP has strong discriminating power -- passing the test with many observations is strong evidence of purposeful behavior.")

    # --- Figure 4: Afriat efficiency index distribution ---
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.hist(afriat_indices, bins=25, color="#4a90d9", edgecolor="white", alpha=0.85, density=True)
    ax4.axvline(x=afriat_indices.mean(), color="red", linestyle="--", linewidth=2,
                label=f"Mean = {afriat_indices.mean():.3f}")
    ax4.axvline(x=1.0, color="green", linestyle="-", linewidth=2, alpha=0.7,
                label="Perfect rationality ($e^*=1$)")
    ax4.set_xlabel("Afriat Efficiency Index $e^*$")
    ax4.set_ylabel("Density")
    ax4.set_title(f"Distribution of Afriat Efficiency Index (T={T}, K={K})")
    ax4.legend()
    ax4.set_xlim(0.4, 1.05)
    report.add_figure("figures/afriat-distribution.png",
                      "Afriat efficiency index distribution for random data: distance from rationalizability",
                      fig4,
                      description="The Afriat efficiency index $e^*$ measures how close data is to being rationalizable: $e^*=1$ means fully rational, "
                      "and lower values indicate greater departure from utility maximization. Even random data often achieves $e^*$ close to 1, "
                      "highlighting that small datasets may lack the power to distinguish optimization from noise.")

    # --- Table: Direct preference R and transitive closure R* ---
    n_tab = min(10, T)
    rows = []
    for i in range(n_tab):
        r_row = "".join([str(R_rand[i, j]) for j in range(n_tab)])
        rs_row = "".join([str(R_star_rand[i, j]) for j in range(n_tab)])
        rows.append({"Obs": i, "R (direct)": r_row, "R* (transitive)": rs_row,
                      "New edges": R_star_rand[i, :n_tab].sum() - R_rand[i, :n_tab].sum()})

    df = pd.DataFrame(rows)
    report.add_table("tables/preference-matrices.csv",
                     "Direct Revealed Preference R and Transitive Closure R* (random data, first 10 obs)",
                     df,
                     description="The 'New edges' column shows how many indirect preferences Warshall's algorithm discovers beyond the direct relation. "
                     "Observations with many new edges are central nodes in the preference graph -- they participate in long transitive chains.")

    report.add_takeaway(
        "GARP is the testable implication of utility maximization. Warshall's algorithm "
        "efficiently computes the transitive closure, making GARP testing practical even "
        "for large datasets. The Afriat efficiency index measures 'how close' data is to "
        "being rationalizable.\n\n"
        "**Key insights:**\n"
        "- Cobb-Douglas data always satisfies GARP — this is a direct consequence of Afriat's "
        "theorem: any dataset rationalizable by a locally non-satiated utility function must "
        "satisfy GARP.\n"
        "- Bronars power increases with the number of observations $T$. With more data points, "
        "random behavior is increasingly likely to produce preference cycles, giving GARP "
        "more opportunities to reject.\n"
        "- The Afriat efficiency index provides a continuous measure of 'near-rationality'. "
        "Even random data often has $e^*$ close to 1, reflecting the difficulty of detecting "
        "irrationality from small samples.\n"
        "- Warshall's $O(T^3)$ complexity is far more efficient than checking all possible "
        "chains explicitly, making it the standard algorithm for empirical revealed preference "
        "analysis."
    )

    report.add_references([
        "Varian, H. (1982). The nonparametric approach to demand analysis. *Econometrica*, 50(4), 945-973.",
        "Bronars, S. (1987). The power of nonparametric tests of preference maximization. *Econometrica*, 55(3), 693-698.",
        "Afriat, S. (1967). The construction of utility functions from expenditure data. *International Economic Review*, 8(1), 67-77.",
        "Warshall, S. (1962). A theorem on Boolean matrices. *Journal of the ACM*, 9(1), 11-12.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
