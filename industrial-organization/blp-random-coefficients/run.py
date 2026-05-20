#!/usr/bin/env python3
"""BLP random-coefficients logit demand.

Generates synthetic differentiated-products markets. A BLP contraction and
IV/GMM estimate taste heterogeneity, then compare elasticities with plain logit.

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


def market_elasticity_matrix(delta_t, x_t, p_t, sigma_x, sigma_p, alpha, nu):
    """Elasticity matrix for one market under random-coefficients logit."""
    mu = (
        sigma_x * nu[:, 0][:, None] * x_t[None, :]
        + sigma_p * nu[:, 1][:, None] * p_t[None, :]
    )
    V = delta_t[None, :] + mu
    exp_V = np.exp(V)
    prob = exp_V / (1.0 + exp_V.sum(axis=1, keepdims=True))
    shares = prob.mean(axis=0)
    alpha_i = alpha + sigma_p * nu[:, 1]

    J = len(p_t)
    elasticity = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                derivative = np.mean(alpha_i * prob[:, j] * (1.0 - prob[:, j]))
            else:
                derivative = np.mean(-alpha_i * prob[:, j] * prob[:, k])
            elasticity[j, k] = derivative * p_t[k] / shares[j]
    return elasticity


def logit_elasticity_matrix(alpha, prices, shares):
    """Elasticity matrix for the plain-logit benchmark."""
    J = len(prices)
    elasticity = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                elasticity[j, k] = alpha * prices[j] * (1.0 - shares[j])
            else:
                elasticity[j, k] = -alpha * prices[k] * shares[k]
    return elasticity


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
    grid_evals = 0
    for sx_try in [0.1, 0.3, 0.5, 0.8, 1.0]:
        for sp_try in [0.1, 0.2, 0.3, 0.5, 0.8]:
            try:
                obj_try = gmm_objective(np.array([sx_try, sp_try]), s_obs, x, p, z, nu)
                grid_evals += 1
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

    # Compute elasticities for a single market to illustrate BLP vs logit.
    t_example = 0
    J_ex = J

    true_elast_blp = market_elasticity_matrix(
        delta_true[t_example, :],
        x[t_example, :],
        p[t_example, :],
        sigma_x_true,
        sigma_p_true,
        true_params["alpha"],
        nu,
    )
    cross_elast_blp = market_elasticity_matrix(
        delta_hat[t_example, :],
        x[t_example, :],
        p[t_example, :],
        sigma_x_hat,
        sigma_p_hat,
        alpha_hat,
        nu,
    )
    cross_elast_logit = logit_elasticity_matrix(alpha_hat, p[t_example, :], s_pred[t_example, :])

    own_elast_true = np.diag(true_elast_blp)
    own_elast_blp = np.diag(cross_elast_blp)
    own_elast_logit = np.diag(cross_elast_logit)
    max_delta_error = np.max(np.abs(delta_recovered - delta_true))
    max_own_elast_error = np.max(np.abs(own_elast_blp - own_elast_true))

    print(f"  Own-price elasticities (BLP, market 1): {own_elast_blp}")
    print(f"  Own-price elasticities (true model, market 1): {own_elast_true}")
    print(f"  Own-price elasticities (logit, market 1): {own_elast_logit}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Differentiated-Products Demand with BLP",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "An antitrust analyst needs to know where demand goes after one product "
        "raises price. Market shares alone do not answer that question.\n\n"
        "The object is differentiated-products demand with heterogeneous consumers. "
        "Products are closer substitutes when they attract similar buyers.\n\n"
        "The main difficulty is that market shares are aggregate outcomes. The analyst "
        "does not observe each consumer's taste draw, but the substitution calculation "
        "depends on the distribution of those tastes.\n\n"
        "The computation estimates those tastes from market shares. A BLP contraction "
        "recovers mean utility for each trial dispersion, and IV/GMM chooses "
        "dispersion."
    )

    report.add_equations(
        r"""
Consumer $i$ in market $t$ chooses among $J$ inside goods and an outside good.

Think of $x_{jt}$ as a product attribute such as quality, style, or size.

The indirect utility from inside product $j$ is:

$$u_{ijt} = \beta_0 + \beta_x x_{jt} + \alpha p_{jt} + \xi_{jt} + \sigma_x \nu_{i1} x_{jt} + \sigma_p \nu_{i2} p_{jt} + \varepsilon_{ijt}$$

Here $x_{jt}$ is an observed product characteristic, $p_{jt}$ is price,
$\xi_{jt}$ is unobserved quality, $\nu_i \sim N(0,I)$, and
$\varepsilon_{ijt}$ is Type-I extreme value.

The outside good has utility
normalized to zero.

Mean utility and individual taste enter separately:

$$\delta_{jt} = \beta_0 + \beta_x x_{jt} + \alpha p_{jt} + \xi_{jt}, \qquad \mu_{ijt} = \sigma_x \nu_{i1} x_{jt} + \sigma_p \nu_{i2} p_{jt}$$

For a candidate $\sigma=(\sigma_x,\sigma_p)$, simulated market shares are:

$$s_{jt} = \frac{1}{ns} \sum_{i=1}^{ns} \frac{\exp(\delta_{jt} + \mu_{ijt})}{1 + \sum_{k=1}^{J} \exp(\delta_{kt} + \mu_{ikt})}$$

The BLP contraction finds the mean utilities that make predicted shares equal
observed shares:

$$\delta^{(r+1)}_{jt} = \delta^{(r)}_{jt} + \log s^{\text{obs}}_{jt} - \log s^{\text{pred}}_{jt}(\delta^{(r)}, \sigma)$$

Given $\delta(\sigma)$, the linear demand equation is:

$$\delta_{jt} = X_{jt}\theta_1 + \xi_{jt}, \qquad X_{jt}=(1,x_{jt},p_{jt})$$

Here $\theta_1 = (\beta_0, \beta_x, \alpha)$ collects the linear demand coefficients.

The identifying moments are $E[Z_{jt}\xi_{jt}]=0$. The instruments include
a cost shifter and sums of rival characteristics, so price can be endogenous
through $\mathrm{Cov}(p_{jt},\xi_{jt}) \ne 0$.
"""
    )

    report.add_model_setup(
        "The example has 100 independent markets with five products per market. "
        "Each product has an observed characteristic, an unobserved quality draw, "
        "a cost shifter, and a price. Price loads on both cost and unobserved quality, "
        "so the IV step has an actual endogeneity problem to solve.\n\n"
        f"| Object | Value | Role |\n"
        f"|-----------|-------|-------------|\n"
        f"| $T$ | {T} | Markets |\n"
        f"| $J$ | {J} | Products per market |\n"
        f"| $ns$ | {ns} | Simulation draws used for shares |\n"
        f"| $\\beta_0$ | {true_params['beta_0']} | Mean inside-good utility |\n"
        f"| $\\beta_x$ | {true_params['beta_x']} | Mean taste for $x$ |\n"
        f"| $\\alpha$ | {true_params['alpha']} | Mean price coefficient |\n"
        f"| $\\sigma_x$ | {true_params['sigma_x']} | Dispersion in taste for $x$ |\n"
        f"| $\\sigma_p$ | {true_params['sigma_p']} | Dispersion in price sensitivity |"
    )

    report.add_solution_method(
        "The estimator is a nested fixed point with GMM. The outer search chooses the "
        "taste-dispersion parameters $\sigma=(\sigma_x,\sigma_p)$. For each trial "
        "$\sigma$, the inner contraction finds the mean utilities $\delta(\sigma)$ "
        "that reproduce the observed shares.\n\n"
        "It helps to separate two jobs. The contraction is an inversion: it finds the "
        "product-level mean utilities that rationalize the observed shares for the "
        "current taste distribution. The IV/GMM step is identification: it asks whether "
        "the implied unobserved quality is orthogonal to cost and rival-characteristic "
        "instruments. The elasticity matrix is computed only after both jobs are done.\n\n"
        "```text\n"
        "Inputs: observed shares s_obs, characteristics x, prices p, instruments Z, draws nu\n"
        "Choose trial nonlinear parameters sigma = (sigma_x, sigma_p)\n"
        "Initialize delta with the simple-logit inversion log(s_jt) - log(s_0t)\n"
        "Repeat until the share residual is small:\n"
        "    predict shares s_pred(delta, sigma) by averaging over taste draws nu\n"
        "    update delta <- delta + log(s_obs) - log(s_pred)\n"
        "Run 2SLS of delta(sigma) on (1, x, p) using Z\n"
        "Compute xi(sigma) and Q(sigma) = n g(sigma)' W g(sigma), where g = Z' xi / n\n"
        "Search over sigma and keep the minimizer\n"
        "Output: sigma_hat, theta_1_hat, xi_hat, elasticities\n"
        "```\n\n"
        "The contraction is the share inversion. It asks what common product utility "
        "must be present for the model to match observed shares after averaging over "
        "consumer heterogeneity.\n\n"
        "The GMM step then checks whether the recovered "
        "unobserved qualities are orthogonal to excluded cost and rival-characteristic "
        "instruments.\n\n"
        f"At the true nonlinear parameters, the contraction converged in "
        f"**{len(conv_history)} iterations** with max "
        f"$|\\delta^{{\\mathrm{{recovered}}}}-\\delta^{{\\mathrm{{true}}}}|="
        f"{max_delta_error:.2e}$.\n\n"
        f"The GMM search first ran a coarse starting grid, where the grid "
        f"evaluated the objective {grid_evals} times. The Nelder-Mead refinement "
        f"from the best grid point then evaluated the objective {result.nfev} "
        f"more times. The convergence diagnostics in the Results table record "
        f"these counts so they can be checked against a fresh run."
    )

    report.add_results(
        "The estimated model matches the simulated market shares closely. The "
        "elasticity comparison is the harder check. It asks whether estimated "
        "heterogeneity changes substitution in the right direction."
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
        "Observed and predicted market shares at estimated parameters.",
        fig1,
        description="The share fit lies on the 45-degree line. The contraction makes "
        "predicted shares match observed shares at the chosen dispersion. This plot "
        "checks the inversion, not the substitution pattern.",
    )

    # --- Figure 2: Own-Price Elasticities ---
    fig2, ax2 = plt.subplots()
    products = np.arange(1, J_ex + 1)
    width = 0.25
    ax2.bar(products - width, own_elast_true, width, label="True model", color="darkgreen")
    ax2.bar(products, own_elast_blp, width, label="Estimated BLP", color="steelblue")
    ax2.bar(products + width, own_elast_logit, width, label="Plain logit", color="coral")
    ax2.set_xlabel("Product")
    ax2.set_ylabel("Own-price elasticity")
    ax2.set_title("Own-Price Elasticities in Market 1")
    ax2.set_xticks(products)
    ax2.legend()
    ax2.axhline(y=0, color="k", linewidth=0.5)
    report.add_figure(
        "figures/own-price-elasticities.png",
        "Own-price elasticities in market 1 under the true model, estimated BLP model, and plain logit benchmark.",
        fig2,
        description="The true-model bars are available because the data are simulated. "
        "Estimated BLP follows the product-level pattern. "
        f"The largest own-elasticity error is {max_own_elast_error:.3f} in this market. "
        "Plain logit has no consumer-specific price coefficient.",
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
        "Convergence of the BLP contraction mapping.",
        fig3,
        description="The inner fixed point is stable. The update norm falls steadily "
        "on the log scale, so the inversion can sit inside GMM.",
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
        "Cross-price elasticity matrices for estimated BLP and plain logit in market 1.",
        fig4,
        description="The cross-elasticity matrix is the main economic object. In plain "
        "logit, each column has identical off-diagonal entries. A price increase sends "
        "proportional demand to each rival. In BLP, off-diagonal entries vary because "
        "products attract different buyers.",
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
        description="The parameter table checks the simulation truth. The nonlinear "
        "dispersion estimates are less exact than the linear coefficients. They are "
        "also what break IIA.",
    )

    # --- Table: Convergence Diagnostics ---
    diag_df = pd.DataFrame({
        "Diagnostic": [
            "contraction_iters",
            "max_delta_error",
            "grid_evals",
            "gmm_nfev",
            "max_own_elast_error",
        ],
        "Value": [
            f"{len(conv_history)}",
            f"{max_delta_error:.2e}",
            f"{grid_evals}",
            f"{result.nfev}",
            f"{max_own_elast_error:.3f}",
        ],
    })
    report.add_table(
        "tables/convergence-diagnostics.csv",
        "Convergence and Search Diagnostics",
        diag_df,
        description="These are the runtime counts and accuracy checks the prose "
        "refers to: contraction iterations at the true parameters, the max "
        "recovered-versus-true mean-utility error, starting-grid objective "
        "evaluations, Nelder-Mead objective evaluations, and the largest "
        "own-elasticity error in market 1. Committing them lets a re-run "
        "verify the numbers in the text against an on-disk artifact.",
    )

    report.add_takeaway(
        "BLP changes the estimated substitution object. The contraction lets each "
        "candidate $\\sigma$ fit observed shares. IV/GMM chooses heterogeneity using "
        "moments for recovered unobserved quality. With heterogeneity, substitution "
        "no longer has to follow existing shares."
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
