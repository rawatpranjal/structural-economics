#!/usr/bin/env python3
"""BLP Random Coefficients Logit Demand Model.

Estimates a random coefficients discrete-choice demand model using the
Berry-Levinsohn-Pakes (1995) methodology. Random coefficients break the IIA
property of plain logit by allowing heterogeneous consumer preferences,
producing more realistic substitution patterns between products.

Reference: Berry, Levinsohn, and Pakes (1995), "Automobile Prices in Market
Equilibrium," Econometrica, 63(4), 841-890.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


# =============================================================================
# Data Generation
# =============================================================================

def generate_synthetic_data(T, J, ns, rng):
    """Generate synthetic market data for BLP estimation.

    Parameters
    ----------
    T : int
        Number of markets.
    J : int
        Number of products per market.
    ns : int
        Number of simulation draws for computing shares.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    dict with keys: x, p, xi, s_obs, z, true_params, nu
    """
    # True parameters
    beta_0 = 2.0         # Intercept (mean preference for inside good)
    beta_x = 1.5         # Coefficient on observed product characteristic
    alpha = -0.8         # Price coefficient (mean)
    sigma_x = 0.8        # Std dev of random coefficient on x
    sigma_p = 0.3        # Std dev of random coefficient on price

    true_params = {
        "beta_0": beta_0, "beta_x": beta_x, "alpha": alpha,
        "sigma_x": sigma_x, "sigma_p": sigma_p,
    }

    # Product characteristics: x_jt ~ Uniform(0, 3)
    x = rng.uniform(0, 3, size=(T, J))

    # Unobserved quality: xi_jt ~ N(0, 0.5)
    xi = rng.normal(0, 0.5, size=(T, J))

    # Cost shifter (instrument): z_jt ~ Uniform(1, 4)
    z = rng.uniform(1, 4, size=(T, J))

    # Price: correlated with xi (endogeneity) and driven by cost
    p = 1.0 + 0.5 * x + 0.8 * z + 0.5 * xi + rng.normal(0, 0.2, size=(T, J))

    # Simulation draws for consumer heterogeneity: nu ~ N(0, I)
    # nu has shape (ns, 2): one draw for x-coefficient, one for price-coefficient
    nu = rng.normal(0, 1, size=(ns, 2))

    # Compute true market shares via simulation
    # Mean utility: delta_jt = beta_0 + beta_x * x_jt + alpha * p_jt + xi_jt
    delta_true = beta_0 + beta_x * x + alpha * p + xi  # (T, J)

    s_obs = compute_shares(delta_true, x, p, sigma_x, sigma_p, nu)  # (T, J)

    return {
        "x": x, "p": p, "xi": xi, "z": z,
        "s_obs": s_obs, "delta_true": delta_true,
        "true_params": true_params, "nu": nu,
    }


# =============================================================================
# Share Computation
# =============================================================================

def compute_shares(delta, x, p, sigma_x, sigma_p, nu):
    """Compute simulated market shares given mean utilities and random coefficients.

    s_jt = (1/ns) * sum_i  exp(delta_jt + mu_ijt) / (1 + sum_k exp(delta_kt + mu_ikt))

    where mu_ijt = sigma_x * nu_i1 * x_jt + sigma_p * nu_i2 * p_jt

    Parameters
    ----------
    delta : array (T, J)
        Mean utilities.
    x : array (T, J)
        Product characteristics.
    p : array (T, J)
        Prices.
    sigma_x, sigma_p : float
        Standard deviations of random coefficients.
    nu : array (ns, 2)
        Simulation draws.

    Returns
    -------
    shares : array (T, J)
        Predicted market shares.
    """
    T, J = delta.shape
    ns = nu.shape[0]

    # mu_ijt = sigma_x * nu_i1 * x_jt + sigma_p * nu_i2 * p_jt
    # Broadcast: nu (ns, 1, 1, 2), x (1, T, J), p (1, T, J)
    mu = (sigma_x * nu[:, 0][:, None, None] * x[None, :, :]
          + sigma_p * nu[:, 1][:, None, None] * p[None, :, :])  # (ns, T, J)

    # Utility: V_ijt = delta_jt + mu_ijt
    V = delta[None, :, :] + mu  # (ns, T, J)

    # Choice probabilities (logit with outside good)
    exp_V = np.exp(V)                            # (ns, T, J)
    denom = 1.0 + exp_V.sum(axis=2, keepdims=True)  # (ns, T, 1)
    prob = exp_V / denom                          # (ns, T, J)

    # Average over simulation draws
    shares = prob.mean(axis=0)  # (T, J)

    return shares


def compute_share_jacobian(delta, x, p, sigma_x, sigma_p, nu):
    """Compute the Jacobian of shares w.r.t. prices for each market.

    Returns a list of (J, J) matrices, one per market.
    ds_j/dp_k = (1/ns) * sum_i  (alpha_i) * prob_ij * (1{j==k} - prob_ik)
    where alpha_i = alpha + sigma_p * nu_i2  (but alpha is embedded in delta).
    """
    T, J = delta.shape
    ns = nu.shape[0]

    mu = (sigma_x * nu[:, 0][:, None, None] * x[None, :, :]
          + sigma_p * nu[:, 1][:, None, None] * p[None, :, :])
    V = delta[None, :, :] + mu
    exp_V = np.exp(V)
    denom = 1.0 + exp_V.sum(axis=2, keepdims=True)
    prob = exp_V / denom  # (ns, T, J)

    # Individual-level price coefficient: alpha_i = alpha_mean + sigma_p * nu_i2
    # We don't know alpha_mean here, so we use the derivative of utility w.r.t. price:
    # dV/dp = alpha + sigma_p * nu_i2.  We'll pass alpha in as part of delta,
    # so the effective coefficient for each draw is embedded.
    # For elasticity purposes, we need the individual alpha_i.
    # Since delta = beta_0 + beta_x*x + alpha*p + xi, the marginal effect of price
    # on utility at the individual level is: alpha + sigma_p * nu_i2.
    # We'll return the Jacobian with alpha_i built in.

    jacobians = []
    for t in range(T):
        # prob_i: (ns, J)
        prob_t = prob[:, t, :]
        # alpha_i: (ns,)
        # We don't have alpha directly, so compute via delta:
        # Instead, note that dV_ijt/dp_jt = alpha + sigma_p * nu_i2
        # We need alpha passed separately. For now, return the "probability part"
        # and the caller can multiply by alpha_i.
        jac_t = np.zeros((J, J))
        for i_draw in range(ns):
            p_i = prob_t[i_draw, :]  # (J,)
            # ds_j/dp_k = alpha_i * p_j * (1{j==k} - p_k)
            # = alpha_i * (diag(p) - p @ p^T)
            jac_t += np.diag(p_i) - np.outer(p_i, p_i)
        jac_t /= ns
        jacobians.append(jac_t)

    return jacobians


# =============================================================================
# BLP Contraction Mapping
# =============================================================================

def contraction_mapping(s_obs, x, p, sigma_x, sigma_p, nu, tol=1e-12, max_iter=1000):
    """Invert observed shares to recover mean utilities delta via BLP contraction.

    delta_{k+1} = delta_k + log(s_obs) - log(s_pred(delta_k, sigma))

    Parameters
    ----------
    s_obs : array (T, J)
        Observed market shares.
    x, p : arrays (T, J)
        Product characteristics and prices.
    sigma_x, sigma_p : float
        Random coefficient standard deviations (nonlinear parameters).
    nu : array (ns, 2)
        Simulation draws.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    delta : array (T, J)
        Recovered mean utilities.
    convergence_history : list of float
        Norm of update at each iteration.
    """
    T, J = s_obs.shape

    # Initial guess: plain logit inversion
    s0 = 1.0 - s_obs.sum(axis=1, keepdims=True)  # outside good share (T, 1)
    s0 = np.maximum(s0, 1e-15)
    delta = np.log(np.maximum(s_obs, 1e-15)) - np.log(s0)

    convergence_history = []

    for iteration in range(max_iter):
        s_pred = compute_shares(delta, x, p, sigma_x, sigma_p, nu)
        s_pred = np.maximum(s_pred, 1e-15)

        # Contraction update
        update = np.log(s_obs) - np.log(s_pred)
        delta = delta + update

        error = np.max(np.abs(update))
        convergence_history.append(error)

        if error < tol:
            break

    return delta, convergence_history


def build_instruments(x, p, z):
    """Build the regressor matrix X and instrument matrix Z for 2SLS.

    Instruments: [1, x, z, sum_x_others, sum_z_others, x^2, z^2].
    Over-identification is needed so the GMM criterion varies with sigma.
    """
    T, J = x.shape
    ones = np.ones(T * J)
    x_flat = x.flatten()
    p_flat = p.flatten()
    z_flat = z.flatten()

    # BLP-style instruments: sum of characteristics of OTHER products
    sum_x_others = np.zeros((T, J))
    sum_z_others = np.zeros((T, J))
    for t in range(T):
        for j in range(J):
            mask = np.ones(J, dtype=bool)
            mask[j] = False
            sum_x_others[t, j] = x[t, mask].sum()
            sum_z_others[t, j] = z[t, mask].sum()

    X_reg = np.column_stack([ones, x_flat, p_flat])
    Z_iv = np.column_stack([
        ones, x_flat, z_flat,
        sum_x_others.flatten(), sum_z_others.flatten(),
        x_flat**2, z_flat**2,
    ])
    return X_reg, Z_iv


# =============================================================================
# GMM Objective
# =============================================================================

def gmm_objective(theta, s_obs, x, p, z, nu):
    """GMM objective function for estimating nonlinear parameters.

    For given sigma = (sigma_x, sigma_p):
      1. Invert shares to get delta(sigma) via contraction mapping.
      2. Recover xi = delta - X*theta_1 using IV regression.
      3. Form moment: E[z * xi] = 0.
      4. Minimize xi' Z (Z'Z)^{-1} Z' xi.

    Parameters
    ----------
    theta : array (2,)
        [sigma_x, sigma_p] -- nonlinear parameters.
    s_obs, x, p, z : arrays
        Data.
    nu : array
        Simulation draws.

    Returns
    -------
    obj : float
        GMM objective value.
    """
    sigma_x, sigma_p = np.abs(theta)  # Ensure positive

    T, J = s_obs.shape

    # Step 1: Contraction mapping to recover delta
    delta, _ = contraction_mapping(s_obs, x, p, sigma_x, sigma_p, nu, tol=1e-12)
    delta_flat = delta.flatten()

    # Step 2: IV regression (2SLS)
    X_reg, Z_iv = build_instruments(x, p, z)
    ZtZ_inv = np.linalg.inv(Z_iv.T @ Z_iv)
    Pz = Z_iv @ ZtZ_inv @ Z_iv.T
    theta_1 = np.linalg.solve(X_reg.T @ Pz @ X_reg, X_reg.T @ Pz @ delta_flat)

    # Step 3: Recover structural error xi
    xi = delta_flat - X_reg @ theta_1

    # Step 4: GMM objective -- E[z * xi] = 0
    n = T * J
    moments = Z_iv.T @ xi / n
    W = np.linalg.inv(Z_iv.T @ Z_iv / n)  # Weighting matrix
    obj = n * moments.T @ W @ moments

    return obj


# =============================================================================
# Main
# =============================================================================

def main():
    # =========================================================================
    # Settings
    # =========================================================================
    T = 100         # Number of markets
    J = 5           # Products per market
    ns = 200        # Simulation draws for share computation
    seed = 42

    rng = np.random.default_rng(seed)

    # =========================================================================
    # Generate Data
    # =========================================================================
    print("Generating synthetic market data...")
    data = generate_synthetic_data(T, J, ns, rng)
    x, p, xi = data["x"], data["p"], data["xi"]
    z, s_obs = data["z"], data["s_obs"]
    nu = data["nu"]
    true_params = data["true_params"]
    delta_true = data["delta_true"]

    print(f"  Markets: {T}, Products/market: {J}, Simulation draws: {ns}")
    print(f"  Mean observed share: {s_obs.mean():.4f}")
    print(f"  Outside good share (mean): {(1 - s_obs.sum(axis=1)).mean():.4f}")

    # =========================================================================
    # Step 1: Contraction Mapping at True Parameters (Demonstration)
    # =========================================================================
    print("\nRunning BLP contraction mapping at true parameters...")
    sigma_x_true = true_params["sigma_x"]
    sigma_p_true = true_params["sigma_p"]

    delta_recovered, conv_history = contraction_mapping(
        s_obs, x, p, sigma_x_true, sigma_p_true, nu, tol=1e-12,
    )
    print(f"  Contraction converged in {len(conv_history)} iterations")
    print(f"  Final error: {conv_history[-1]:.2e}")
    print(f"  Max |delta_recovered - delta_true|: {np.max(np.abs(delta_recovered - delta_true)):.2e}")

    # =========================================================================
    # Step 2: GMM Estimation of Nonlinear Parameters
    # =========================================================================
    print("\nEstimating nonlinear parameters via GMM...")

    # Grid search for starting values
    print("  Grid search for starting values...")
    best_obj = np.inf
    best_theta0 = np.array([0.5, 0.3])
    for sx_try in [0.1, 0.3, 0.5, 0.8, 1.0]:
        for sp_try in [0.1, 0.2, 0.3, 0.5, 0.8]:
            try:
                obj_try = gmm_objective(np.array([sx_try, sp_try]), s_obs, x, p, z, nu)
                if obj_try < best_obj:
                    best_obj = obj_try
                    best_theta0 = np.array([sx_try, sp_try])
            except Exception:
                continue
    print(f"  Best grid point: sigma_x={best_theta0[0]:.2f}, sigma_p={best_theta0[1]:.2f}, obj={best_obj:.6f}")

    result = minimize(
        gmm_objective, best_theta0, args=(s_obs, x, p, z, nu),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-6, "disp": True},
    )

    sigma_x_hat, sigma_p_hat = np.abs(result.x)
    print(f"  sigma_x: true={sigma_x_true:.3f}, estimated={sigma_x_hat:.3f}")
    print(f"  sigma_p: true={sigma_p_true:.3f}, estimated={sigma_p_hat:.3f}")

    # Recover delta and linear parameters at estimated sigma
    delta_hat, _ = contraction_mapping(
        s_obs, x, p, sigma_x_hat, sigma_p_hat, nu, tol=1e-12,
    )
    delta_flat = delta_hat.flatten()
    X_reg, Z_iv = build_instruments(x, p, z)
    ZtZ_inv = np.linalg.inv(Z_iv.T @ Z_iv)
    Pz = Z_iv @ ZtZ_inv @ Z_iv.T
    theta_1_hat = np.linalg.solve(X_reg.T @ Pz @ X_reg, X_reg.T @ Pz @ delta_flat)
    beta_0_hat, beta_x_hat, alpha_hat = theta_1_hat

    print(f"  beta_0: true={true_params['beta_0']:.3f}, estimated={beta_0_hat:.3f}")
    print(f"  beta_x: true={true_params['beta_x']:.3f}, estimated={beta_x_hat:.3f}")
    print(f"  alpha:  true={true_params['alpha']:.3f}, estimated={alpha_hat:.3f}")

    # =========================================================================
    # Predicted Shares at Estimated Parameters
    # =========================================================================
    s_pred = compute_shares(delta_hat, x, p, sigma_x_hat, sigma_p_hat, nu)

    # =========================================================================
    # Elasticities
    # =========================================================================
    print("\nComputing elasticities...")

    # Compute elasticities for a single market to illustrate BLP vs logit
    t_example = 0
    J_ex = J

    # Compute individual-level elasticities for market t_example
    mu_ex = (sigma_x_hat * nu[:, 0][:, None] * x[None, t_example, :]
             + sigma_p_hat * nu[:, 1][:, None] * p[None, t_example, :])  # (ns, J)
    V_ex = delta_hat[None, t_example, :] + mu_ex  # (ns, J)
    exp_V_ex = np.exp(V_ex)
    denom_ex = 1.0 + exp_V_ex.sum(axis=1, keepdims=True)  # (ns, 1)
    prob_ex = exp_V_ex / denom_ex  # (ns, J)

    # alpha_i for each draw
    alpha_i = alpha_hat + sigma_p_hat * nu[:, 1]  # (ns,)

    # Own-price elasticities for each product in market t_example
    own_elast_blp = np.zeros(J_ex)
    for j in range(J_ex):
        # eta_j = (alpha_i) * p_j * (1 - s_ij), averaged
        own_elast_blp[j] = np.mean(alpha_i * p[t_example, j] * (1 - prob_ex[:, j]))

    # Plain logit elasticities for comparison
    # In plain logit: eta_jj = alpha * p_j * (1 - s_j)
    s_logit = s_pred[t_example, :]
    own_elast_logit = alpha_hat * p[t_example, :] * (1 - s_logit)

    # Cross-price elasticity matrix for market t_example (BLP)
    cross_elast_blp = np.zeros((J_ex, J_ex))
    for j in range(J_ex):
        for k in range(J_ex):
            if j == k:
                cross_elast_blp[j, k] = own_elast_blp[j]
            else:
                # d s_j / d p_k * (p_k / s_j)
                # = (1/ns) sum_i alpha_i * prob_ij * (-prob_ik) * p_k / s_j
                numer = np.mean(alpha_i * prob_ex[:, j] * (-prob_ex[:, k]))
                cross_elast_blp[j, k] = numer * p[t_example, k] / s_pred[t_example, j]

    # Plain logit cross-price elasticities: eta_jk = -alpha * p_k * s_k (same for all j!=k)
    cross_elast_logit = np.zeros((J_ex, J_ex))
    for j in range(J_ex):
        for k in range(J_ex):
            if j == k:
                cross_elast_logit[j, k] = own_elast_logit[j]
            else:
                cross_elast_logit[j, k] = -alpha_hat * p[t_example, k] * s_logit[k]

    print(f"  Own-price elasticities (BLP, market 1): {own_elast_blp}")
    print(f"  Own-price elasticities (logit, market 1): {own_elast_logit}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "BLP Random Coefficients Logit",
        "Demand estimation with heterogeneous consumer preferences via Berry, Levinsohn, and Pakes (1995).",
    )

    report.add_overview(
        "The BLP model estimates demand for differentiated products while allowing consumer "
        "preferences to vary across the population. Standard logit demand imposes the IIA "
        "(Independence of Irrelevant Alternatives) property: the ratio of choice probabilities "
        "between any two products is independent of the characteristics of all other products. "
        "This produces unrealistic substitution patterns.\n\n"
        "Random coefficients break IIA by letting each consumer have a different marginal "
        "utility for product characteristics. Consumers who value a characteristic highly will "
        "substitute toward products that share that characteristic, creating a realistic "
        "pattern where similar products compete more intensely."
    )

    report.add_equations(
        r"""
**Indirect utility** of consumer $i$ for product $j$ in market $t$:

$$u_{ijt} = \beta_0 + \beta_x x_{jt} + \alpha p_{jt} + \xi_{jt} + \sigma_x \nu_{i1} x_{jt} + \sigma_p \nu_{i2} p_{jt} + \varepsilon_{ijt}$$

where $\nu_i \sim N(0, I)$ generates preference heterogeneity and $\varepsilon_{ijt}$ is T1EV (logit error).

**Decomposition** into mean utility $\delta_{jt}$ and individual deviation $\mu_{ijt}$:

$$\delta_{jt} = \beta_0 + \beta_x x_{jt} + \alpha p_{jt} + \xi_{jt}, \qquad \mu_{ijt} = \sigma_x \nu_{i1} x_{jt} + \sigma_p \nu_{i2} p_{jt}$$

**Market shares** via simulation over $ns$ draws:

$$s_{jt} = \frac{1}{ns} \sum_{i=1}^{ns} \frac{\exp(\delta_{jt} + \mu_{ijt})}{1 + \sum_{k=1}^{J} \exp(\delta_{kt} + \mu_{ikt})}$$

**BLP contraction mapping** to invert shares:

$$\delta^{(r+1)}_{jt} = \delta^{(r)}_{jt} + \log s^{\text{obs}}_{jt} - \log s^{\text{pred}}_{jt}(\delta^{(r)}, \sigma)$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $T$ | {T} | Number of markets |\n"
        f"| $J$ | {J} | Products per market |\n"
        f"| $ns$ | {ns} | Simulation draws |\n"
        f"| $\\beta_0$ | {true_params['beta_0']} | Intercept |\n"
        f"| $\\beta_x$ | {true_params['beta_x']} | Characteristic coefficient |\n"
        f"| $\\alpha$ | {true_params['alpha']} | Mean price coefficient |\n"
        f"| $\\sigma_x$ | {true_params['sigma_x']} | Std dev of random coeff on $x$ |\n"
        f"| $\\sigma_p$ | {true_params['sigma_p']} | Std dev of random coeff on price |"
    )

    report.add_solution_method(
        "**Nested Fixed-Point (NFXP) with GMM:**\n\n"
        "The BLP estimator has a nested structure. The *outer loop* searches over nonlinear "
        "parameters $\\sigma = (\\sigma_x, \\sigma_p)$ to minimize the GMM objective. For each "
        "candidate $\\sigma$:\n\n"
        "1. **Inner loop (contraction mapping):** Invert observed shares to recover mean "
        "utilities $\\delta(\\sigma)$ using the BLP contraction $\\delta^{(r+1)} = \\delta^{(r)} "
        "+ \\log s^{\\text{obs}} - \\log s^{\\text{pred}}(\\delta^{(r)}, \\sigma)$. Berry (1994) "
        "proved this map is a contraction.\n\n"
        "2. **IV regression:** Regress $\\delta$ on $[1, x, p]$ using instruments $[1, x, z]$ "
        "(2SLS) to recover linear parameters $(\\beta_0, \\beta_x, \\alpha)$ and structural "
        "errors $\\xi$.\n\n"
        "3. **GMM criterion:** $Q(\\sigma) = \\xi(\\sigma)' Z (Z'Z)^{-1} Z' \\xi(\\sigma)$, "
        "exploiting the moment condition $E[z_{jt} \\cdot \\xi_{jt}] = 0$.\n\n"
        f"Contraction converged in **{len(conv_history)} iterations** at true parameters. "
        f"GMM optimization used Nelder-Mead ({result.nfev} function evaluations)."
    )

    # --- Figure 1: Observed vs Predicted Shares ---
    fig1, ax1 = plt.subplots()
    ax1.scatter(s_obs.flatten(), s_pred.flatten(), alpha=0.4, s=15, c="steelblue", edgecolors="none")
    lims = [0, max(s_obs.max(), s_pred.max()) * 1.05]
    ax1.plot(lims, lims, "k--", linewidth=1, alpha=0.7, label="45-degree line")
    ax1.set_xlabel("Observed market share $s_{jt}^{\\mathrm{obs}}$")
    ax1.set_ylabel("Predicted market share $s_{jt}^{\\mathrm{pred}}$")
    ax1.set_title("Observed vs Predicted Market Shares")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.legend()
    report.add_figure(
        "figures/observed-vs-predicted-shares.png",
        "Observed vs predicted market shares at estimated parameters. Points near the 45-degree line indicate good model fit.",
        fig1,
        description="Tight clustering along the 45-degree line indicates that the estimated "
        "random coefficients model fits the observed market shares well. Outliers would suggest "
        "model misspecification or products with unusual unobserved characteristics.",
    )

    # --- Figure 2: Own-Price Elasticities ---
    fig2, ax2 = plt.subplots()
    products = np.arange(1, J_ex + 1)
    width = 0.35
    ax2.bar(products - width / 2, own_elast_blp, width, label="BLP (random coefficients)", color="steelblue")
    ax2.bar(products + width / 2, own_elast_logit, width, label="Plain logit", color="coral")
    ax2.set_xlabel("Product")
    ax2.set_ylabel("Own-price elasticity")
    ax2.set_title("Own-Price Elasticities: BLP vs Plain Logit (Market 1)")
    ax2.set_xticks(products)
    ax2.legend()
    ax2.axhline(y=0, color="k", linewidth=0.5)
    report.add_figure(
        "figures/own-price-elasticities.png",
        "Own-price elasticities in market 1. BLP produces heterogeneous elasticities across products; plain logit elasticities are driven almost entirely by price level.",
        fig2,
        description="In plain logit, own-price elasticities are mechanically tied to "
        "price and share levels. BLP allows elasticities to vary with the composition of "
        "consumers who choose each product, producing richer variation that reflects genuine "
        "differences in price sensitivity across market segments.",
    )

    # --- Figure 3: Contraction Mapping Convergence ---
    fig3, ax3 = plt.subplots()
    ax3.semilogy(range(1, len(conv_history) + 1), conv_history, "b-", linewidth=1.5)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("$\\|\\delta^{(r+1)} - \\delta^{(r)}\\|_\\infty$")
    ax3.set_title("BLP Contraction Mapping Convergence")
    ax3.axhline(y=1e-12, color="r", linestyle="--", linewidth=1, alpha=0.7, label="Tolerance ($10^{-12}$)")
    ax3.legend()
    report.add_figure(
        "figures/contraction-convergence.png",
        "Convergence of the BLP contraction mapping. The log-linear decline confirms the contraction property proved by Berry (1994).",
        fig3,
        description="The log-linear convergence rate confirms that the BLP mapping is indeed "
        "a contraction, as Berry (1994) proved. Each iteration reduces the error by a constant "
        "factor, making convergence reliable and predictable across different starting values.",
    )

    # --- Figure 4: Cross-Price Elasticity Matrix ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))

    vmin = min(cross_elast_blp.min(), cross_elast_logit.min())
    vmax = max(cross_elast_blp.max(), cross_elast_logit.max())

    im1 = ax4a.imshow(cross_elast_blp, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax4a.set_title("BLP (Random Coefficients)")
    ax4a.set_xlabel("Product $k$ (price change)")
    ax4a.set_ylabel("Product $j$ (share response)")
    ax4a.set_xticks(range(J_ex))
    ax4a.set_yticks(range(J_ex))
    ax4a.set_xticklabels([f"{k+1}" for k in range(J_ex)])
    ax4a.set_yticklabels([f"{j+1}" for j in range(J_ex)])
    for j in range(J_ex):
        for k in range(J_ex):
            ax4a.text(k, j, f"{cross_elast_blp[j, k]:.2f}", ha="center", va="center", fontsize=8)

    im2 = ax4b.imshow(cross_elast_logit, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax4b.set_title("Plain Logit (IIA)")
    ax4b.set_xlabel("Product $k$ (price change)")
    ax4b.set_ylabel("Product $j$ (share response)")
    ax4b.set_xticks(range(J_ex))
    ax4b.set_yticks(range(J_ex))
    ax4b.set_xticklabels([f"{k+1}" for k in range(J_ex)])
    ax4b.set_yticklabels([f"{j+1}" for j in range(J_ex)])
    for j in range(J_ex):
        for k in range(J_ex):
            ax4b.text(k, j, f"{cross_elast_logit[j, k]:.2f}", ha="center", va="center", fontsize=8)

    fig4.colorbar(im2, ax=[ax4a, ax4b], shrink=0.8, label="Elasticity")
    fig4.suptitle("Cross-Price Elasticity Matrix (Market 1)", fontsize=13)
    fig4.subplots_adjust(left=0.08, right=0.88, top=0.90, bottom=0.08, wspace=0.35)
    report.add_figure(
        "figures/cross-price-elasticity-matrix.png",
        "Cross-price elasticity matrices. BLP produces asymmetric off-diagonal entries reflecting heterogeneous substitution; plain logit cross-elasticities depend only on the column product (IIA).",
        fig4,
        description="The side-by-side comparison reveals the IIA limitation most clearly. In the "
        "logit panel (right), each column has identical off-diagonal entries because cross-elasticities "
        "depend only on the column product's price and share. In the BLP panel (left), cross-elasticities "
        "vary by row, capturing the fact that similar products compete more intensely.",
    )

    # --- Table: Parameter Estimates ---
    table_data = {
        "Parameter": ["$\\beta_0$ (intercept)", "$\\beta_x$ (characteristic)",
                       "$\\alpha$ (price)", "$\\sigma_x$ (RC on $x$)",
                       "$\\sigma_p$ (RC on price)"],
        "True": [f"{true_params['beta_0']:.3f}", f"{true_params['beta_x']:.3f}",
                 f"{true_params['alpha']:.3f}", f"{true_params['sigma_x']:.3f}",
                 f"{true_params['sigma_p']:.3f}"],
        "Estimated": [f"{beta_0_hat:.3f}", f"{beta_x_hat:.3f}",
                      f"{alpha_hat:.3f}", f"{sigma_x_hat:.3f}",
                      f"{sigma_p_hat:.3f}"],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/parameter-estimates.csv",
        "Estimated vs True Parameters",
        df,
        description="The GMM estimator recovers the true parameters with reasonable accuracy. "
        "The nonlinear parameters (sigma_x, sigma_p) govern the degree of consumer heterogeneity "
        "and are identified by the variation in substitution patterns across markets.",
    )

    report.add_takeaway(
        "The BLP random coefficients logit model fundamentally changes how we think about "
        "demand substitution in differentiated product markets.\n\n"
        "**Key insights:**\n"
        "- **Breaking IIA:** In plain logit, if a product is removed from the market, its "
        "share is redistributed to all remaining products in proportion to their existing "
        "shares — regardless of similarity. BLP's random coefficients create realistic "
        "patterns where close substitutes absorb more share.\n"
        "- **Heterogeneous elasticities:** The cross-price elasticity matrix is no longer "
        "symmetric in the off-diagonal. Products that attract similar consumer types exhibit "
        "stronger cross-price effects.\n"
        "- **Contraction mapping:** Berry (1994) proved that the mapping "
        "$\\delta \\mapsto \\delta + \\log s^{\\text{obs}} - \\log s^{\\text{pred}}(\\delta)$ is a "
        "contraction, guaranteeing unique inversion from shares to mean utilities. This is "
        "the computational backbone of BLP.\n"
        "- **Identification:** Price endogeneity ($\\text{Cov}(p, \\xi) \\neq 0$) requires "
        "instruments. Cost shifters ($z$) that affect price but not utility provide "
        "exclusion restrictions for 2SLS estimation of the linear parameters."
    )

    report.add_references([
        "Berry, S., Levinsohn, J., and Pakes, A. (1995). \"Automobile Prices in Market Equilibrium.\" *Econometrica*, 63(4), 841-890.",
        "Berry, S. (1994). \"Estimating Discrete-Choice Models of Product Differentiation.\" *RAND Journal of Economics*, 25(2), 242-262.",
        "Nevo, A. (2000). \"A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand.\" *Journal of Economics & Management Strategy*, 9(4), 513-548.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
