#!/usr/bin/env python3
"""GP regression with squared-exponential and Matern-5/2 kernels on a 1-D test function.

The target is f(x) = x sin(x) on [0, 10]. We observe 15 noisy training points,
fit a GP with each kernel by maximising the log marginal likelihood over the
hyperparameters, and compare posterior means, credible bands, and marginal-
likelihood curves. A prior-vs-posterior panel shows how conditioning on data
collapses the prior uncertainty.

References:
- Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian Processes for Machine Learning, MIT Press.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning, Springer, S6.4.
- Kennedy, M. C. and O'Hagan, A. (2001). Bayesian calibration of computer models. JRSS B 63(3), 425-464.
- Snoek, J., Larochelle, H., and Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NIPS 25.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


# =============================================================================
# Constants
# =============================================================================
SEED = 0
N_TRAIN = 15
NOISE_OBS = 0.5
X_LO, X_HI = 0.0, 10.0
N_TEST = 200
JITTER = 1e-8


# =============================================================================
# True function
# =============================================================================
def f_true(x: np.ndarray) -> np.ndarray:
    """Target: f(x) = x sin(x)."""
    x = np.asarray(x, dtype=float)
    return x * np.sin(x)


# =============================================================================
# Kernels
# =============================================================================
def rbf_kernel(x1: np.ndarray, x2: np.ndarray, ell: float, sigma_f: float) -> np.ndarray:
    """Squared-exponential (RBF) kernel.

    k(x, x') = sigma_f^2 exp(-||x - x'||^2 / (2 ell^2))
    """
    x1 = np.atleast_1d(x1).astype(float)
    x2 = np.atleast_1d(x2).astype(float)
    d2 = (x1[:, None] - x2[None, :]) ** 2
    return (sigma_f ** 2) * np.exp(-0.5 * d2 / (ell ** 2))


def matern52_kernel(x1: np.ndarray, x2: np.ndarray, ell: float, sigma_f: float) -> np.ndarray:
    """Matern-5/2 kernel.

    k(x, x') = sigma_f^2 (1 + sqrt(5) d/ell + 5 d^2 / (3 ell^2)) exp(-sqrt(5) d/ell)
    where d = ||x - x'||.
    """
    x1 = np.atleast_1d(x1).astype(float)
    x2 = np.atleast_1d(x2).astype(float)
    d = np.abs(x1[:, None] - x2[None, :])
    r = np.sqrt(5.0) * d / ell
    return (sigma_f ** 2) * (1.0 + r + r ** 2 / 3.0) * np.exp(-r)


KERNELS = {
    "RBF": rbf_kernel,
    "Matern-5/2": matern52_kernel,
}


# =============================================================================
# GP core: Cholesky-based posterior and marginal likelihood
# =============================================================================
class GaussianProcess:
    """GP regressor with a pluggable kernel and zero prior mean.

    Hyperparameters are tuned externally via :meth:`optimize_hyperparams`.
    Internal computations use Cholesky factorisation throughout; no explicit
    matrix inversion is performed.
    """

    def __init__(
        self,
        kernel_fn,
        ell: float = 1.0,
        sigma_f: float = 1.0,
        sigma_n: float = NOISE_OBS,
    ) -> None:
        self.kernel_fn = kernel_fn
        self.ell = ell
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        # Set after calling fit()
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self._cho = None
        self._alpha: np.ndarray | None = None

    def _build_K(self, X: np.ndarray) -> np.ndarray:
        K = self.kernel_fn(X, X, self.ell, self.sigma_f)
        K += (self.sigma_n ** 2 + JITTER) * np.eye(len(X))
        return K

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """Condition the GP on training data (X, y)."""
        self.X_train = np.atleast_1d(X).astype(float)
        self.y_train = np.atleast_1d(y).astype(float)
        K = self._build_K(self.X_train)
        self._cho = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._cho, self.y_train)
        return self

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and standard deviation at test points X_star."""
        X_star = np.atleast_1d(X_star).astype(float)
        K_s = self.kernel_fn(self.X_train, X_star, self.ell, self.sigma_f)  # (n, n*)
        mu = K_s.T @ self._alpha
        v = cho_solve(self._cho, K_s)  # (n, n*)
        K_ss_diag = self.sigma_f ** 2 + JITTER  # k(x*, x*) for RBF/Matern with equal args
        var = K_ss_diag - np.sum(K_s * v, axis=0)
        var = np.clip(var, 0.0, None)
        return mu, np.sqrt(var)

    def log_marginal_likelihood(self) -> float:
        """Log marginal likelihood log p(y | X, theta) via the Cholesky factor.

        log p = -0.5 y^T (K + sigma_n^2 I)^{-1} y
                - 0.5 log det(K + sigma_n^2 I)
                - (n/2) log(2 pi)
        """
        L = self._cho[0]  # lower-triangular Cholesky factor
        n = len(self.y_train)
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
        lml = (
            -0.5 * float(self.y_train @ self._alpha)
            - 0.5 * log_det_K
            - 0.5 * n * np.log(2.0 * np.pi)
        )
        return lml

    def sample_prior(self, X_star: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw prior samples f ~ GP(0, k) at X_star.

        Returns array of shape (n_samples, len(X_star)).
        """
        X_star = np.atleast_1d(X_star).astype(float)
        K = self.kernel_fn(X_star, X_star, self.ell, self.sigma_f)
        K += JITTER * np.eye(len(X_star))
        L = np.linalg.cholesky(K)
        z = rng.standard_normal((len(X_star), n_samples))
        return (L @ z).T  # (n_samples, n_test)

    def sample_posterior(self, X_star: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw posterior samples at X_star.

        Returns array of shape (n_samples, len(X_star)).
        """
        X_star = np.atleast_1d(X_star).astype(float)
        mu, _ = self.predict(X_star)
        K_s = self.kernel_fn(self.X_train, X_star, self.ell, self.sigma_f)
        v = cho_solve(self._cho, K_s)
        K_ss = self.kernel_fn(X_star, X_star, self.ell, self.sigma_f)
        K_ss += JITTER * np.eye(len(X_star))
        cov_post = K_ss - K_s.T @ v
        cov_post = 0.5 * (cov_post + cov_post.T)  # enforce symmetry
        eigvals = np.linalg.eigvalsh(cov_post)
        if eigvals.min() < -1e-6:
            cov_post += (-eigvals.min() + JITTER) * np.eye(len(X_star))
        L = np.linalg.cholesky(cov_post + JITTER * np.eye(len(X_star)))
        z = rng.standard_normal((len(X_star), n_samples))
        return mu[None, :] + (L @ z).T  # (n_samples, n_test)


# =============================================================================
# Hyperparameter optimisation via log marginal likelihood
# =============================================================================
def optimize_hyperparams(
    kernel_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sigma_n: float = NOISE_OBS,
    n_restarts: int = 8,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Maximise log marginal likelihood over (ell, sigma_f) with sigma_n fixed.

    Uses L-BFGS-B on log-parameters to enforce positivity.  Runs n_restarts
    random starts to avoid poor local optima.

    Returns (ell_opt, sigma_f_opt, lml_opt).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    def neg_lml(log_params: np.ndarray) -> float:
        ell = float(np.exp(log_params[0]))
        sf = float(np.exp(log_params[1]))
        try:
            gp = GaussianProcess(kernel_fn, ell=ell, sigma_f=sf, sigma_n=sigma_n)
            gp.fit(X_train, y_train)
            return -gp.log_marginal_likelihood()
        except np.linalg.LinAlgError:
            return 1e10

    best_lml = -np.inf
    best_ell, best_sf = 1.0, 1.0

    # Bounds: log(ell) in [log(0.05), log(20)], log(sigma_f) in [log(0.01), log(20)]
    bounds = [(np.log(0.05), np.log(20.0)), (np.log(0.01), np.log(20.0))]

    # Fixed starting points plus random restarts
    x0_list: list[np.ndarray] = [np.array([np.log(1.0), np.log(1.0)])]
    for _ in range(n_restarts - 1):
        x0_list.append(rng.uniform([b[0] for b in bounds], [b[1] for b in bounds]))

    for x0 in x0_list:
        res = minimize(neg_lml, x0=x0, method="L-BFGS-B", bounds=bounds)
        if res.success or res.fun < -best_lml * (-1.0):
            lml_val = -res.fun
            if lml_val > best_lml:
                best_lml = lml_val
                best_ell = float(np.exp(res.x[0]))
                best_sf = float(np.exp(res.x[1]))

    return best_ell, best_sf, best_lml


# =============================================================================
# Marginal-likelihood curve along ell axis
# =============================================================================
def lml_vs_ell(
    kernel_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sigma_f: float,
    sigma_n: float,
    ell_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate log marginal likelihood at each ell in ell_grid, sigma_f fixed."""
    lmls = np.empty(len(ell_grid))
    for i, ell in enumerate(ell_grid):
        try:
            gp = GaussianProcess(kernel_fn, ell=float(ell), sigma_f=sigma_f, sigma_n=sigma_n)
            gp.fit(X_train, y_train)
            lmls[i] = gp.log_marginal_likelihood()
        except np.linalg.LinAlgError:
            lmls[i] = np.nan
    return lmls


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    t0 = time.perf_counter()

    rng = np.random.default_rng(SEED)

    # -------------------------------------------------------------------------
    # Training data: random draw from [0, 10] + Gaussian noise
    # -------------------------------------------------------------------------
    X_train = np.sort(rng.uniform(X_LO, X_HI, N_TRAIN))
    y_train = f_true(X_train) + rng.normal(0.0, NOISE_OBS, N_TRAIN)

    # Test grid
    X_test = np.linspace(X_LO, X_HI, N_TEST)
    f_test = f_true(X_test)

    # -------------------------------------------------------------------------
    # Fit GP for each kernel
    # -------------------------------------------------------------------------
    results: dict[str, dict] = {}

    for kernel_name, kernel_fn in KERNELS.items():
        ell_opt, sf_opt, lml_opt = optimize_hyperparams(
            kernel_fn, X_train, y_train, sigma_n=NOISE_OBS, n_restarts=10, rng=rng
        )
        gp = GaussianProcess(kernel_fn, ell=ell_opt, sigma_f=sf_opt, sigma_n=NOISE_OBS)
        gp.fit(X_train, y_train)
        mu_test, sd_test = gp.predict(X_test)
        rmse = float(np.sqrt(np.mean((mu_test - f_test) ** 2)))
        results[kernel_name] = {
            "gp": gp,
            "ell": ell_opt,
            "sigma_f": sf_opt,
            "lml": lml_opt,
            "mu": mu_test,
            "sd": sd_test,
            "rmse": rmse,
        }

    # -------------------------------------------------------------------------
    # Marginal-likelihood curves vs ell (sigma_f fixed at its optimum)
    # -------------------------------------------------------------------------
    ell_grid = np.logspace(np.log10(0.05), np.log10(20.0), 120)
    lml_curves: dict[str, np.ndarray] = {}
    for kernel_name, kernel_fn in KERNELS.items():
        sf_opt = results[kernel_name]["sigma_f"]
        lml_curves[kernel_name] = lml_vs_ell(
            kernel_fn, X_train, y_train, sf_opt, NOISE_OBS, ell_grid
        )

    # -------------------------------------------------------------------------
    # Prior and posterior samples from the RBF GP
    # -------------------------------------------------------------------------
    gp_rbf = results["RBF"]["gp"]
    n_samples = 3
    prior_samples = gp_rbf.sample_prior(X_test, n_samples, rng)
    post_samples = gp_rbf.sample_posterior(X_test, n_samples, rng)

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    setup_style()

    # Figure 1: posterior fit - two panels (RBF and Matern)
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, kernel_name in zip(axes1, ["RBF", "Matern-5/2"]):
        r = results[kernel_name]
        mu, sd = r["mu"], r["sd"]
        ax.plot(X_test, f_test, color="tab:gray", linestyle="--", linewidth=1.5, label="True f(x)")
        ax.plot(X_test, mu, color="tab:blue", linewidth=2.0, label="Posterior mean")
        ax.fill_between(
            X_test,
            mu - 1.96 * sd,
            mu + 1.96 * sd,
            color="tab:blue",
            alpha=0.18,
            label="95% credible band",
        )
        ax.scatter(X_train, y_train, color="tab:orange", s=40, zorder=5, label="Training obs.")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ell_str = f"{r['ell']:.2f}"
        sf_str = f"{r['sigma_f']:.2f}"
        ax.set_title(f"{kernel_name} kernel  (ell={ell_str}, sigma_f={sf_str})")
        ax.legend(fontsize=9, loc="upper left")
    fig1.suptitle("GP posterior fit: RBF vs Matern-5/2 on f(x) = x sin(x)", y=1.01)
    fig1.tight_layout()
    save_figure(fig1, "figures/posterior-fit.png", dpi=150)

    # Figure 2: log marginal likelihood vs ell (log x-axis)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = {"RBF": "tab:blue", "Matern-5/2": "tab:orange"}
    for kernel_name, lmls in lml_curves.items():
        ax2.plot(ell_grid, lmls, color=colors[kernel_name], linewidth=2.0, label=kernel_name)
        ell_opt = results[kernel_name]["ell"]
        idx_opt = int(np.nanargmax(lmls))
        ax2.axvline(ell_opt, color=colors[kernel_name], linestyle=":", linewidth=1.2,
                    alpha=0.8, label=f"{kernel_name} opt ell={ell_opt:.2f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Length scale ell (log scale)")
    ax2.set_ylabel("Log marginal likelihood")
    ax2.set_title("Log marginal likelihood vs length scale (sigma_f fixed at optimum)")
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    save_figure(fig2, "figures/marginal-likelihood-curve.png", dpi=150)

    # Figure 3: 2x2 grid -- prior samples, posterior samples, posterior mean band, blank
    sample_colors = ["tab:purple", "tab:green", "tab:red"]
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10), sharey=False)

    ax_prior = axes3[0, 0]
    ax_post = axes3[0, 1]
    ax_band = axes3[1, 0]
    ax_blank = axes3[1, 1]

    # Panel 1 (top-left): prior samples
    for i in range(n_samples):
        ax_prior.plot(X_test, prior_samples[i], color=sample_colors[i], linewidth=1.5,
                      alpha=0.85, label=f"Sample {i + 1}")
    ax_prior.set_xlabel("x")
    ax_prior.set_ylabel("f(x)")
    ax_prior.set_title("Prior samples (RBF GP)")
    ax_prior.legend(fontsize=9)

    # Panel 2 (top-right): posterior samples
    for i in range(n_samples):
        ax_post.plot(X_test, post_samples[i], color=sample_colors[i], linewidth=1.5,
                     alpha=0.85, label=f"Sample {i + 1}")
    ax_post.scatter(X_train, y_train, color="tab:orange", s=40, zorder=5, label="Training obs.")
    ax_post.plot(X_test, f_test, color="tab:gray", linestyle="--", linewidth=1.2, label="True f(x)")
    ax_post.set_xlabel("x")
    ax_post.set_title("Posterior samples (RBF GP)")
    ax_post.legend(fontsize=9)

    # Panel 3 (bottom-left): posterior mean + 2sd band (RBF)
    r_rbf = results["RBF"]
    mu_rbf, sd_rbf = r_rbf["mu"], r_rbf["sd"]
    ax_band.plot(X_test, f_test, color="tab:gray", linestyle="--", linewidth=1.5, label="True f(x)")
    ax_band.plot(X_test, mu_rbf, color="tab:blue", linewidth=2.0, label="Posterior mean")
    ax_band.fill_between(
        X_test,
        mu_rbf - 2.0 * sd_rbf,
        mu_rbf + 2.0 * sd_rbf,
        color="tab:blue",
        alpha=0.18,
        label="Mean +/- 2sd",
    )
    ax_band.scatter(X_train, y_train, color="tab:orange", s=40, zorder=5, label="Training obs.")
    ax_band.set_xlabel("x")
    ax_band.set_ylabel("f(x)")
    ax_band.set_title("Posterior mean and uncertainty band (RBF GP)")
    ax_band.legend(fontsize=9)

    # Panel 4 (bottom-right): blank
    ax_blank.set_visible(False)

    fig3.suptitle("Prior vs posterior: how conditioning collapses GP uncertainty", y=1.01)
    fig3.tight_layout()
    save_figure(fig3, "figures/prior-vs-posterior-samples.png", dpi=150)

    # Thumbnail from figure 1
    save_thumbnail("figures/posterior-fit.png", "figures/thumb.png")

    # -------------------------------------------------------------------------
    # Stdout summary table
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    header = (
        f"{'Kernel':<14}  {'ell':>8}  {'sigma_f':>9}  {'sigma_n':>8}  "
        f"{'Log ML':>10}  {'RMSE':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for kernel_name in ["RBF", "Matern-5/2"]:
        r = results[kernel_name]
        print(
            f"{kernel_name:<14}  {r['ell']:>8.4f}  {r['sigma_f']:>9.4f}  "
            f"{NOISE_OBS:>8.4f}  {r['lml']:>10.3f}  {r['rmse']:>8.5f}"
        )
    print(sep)
    print(f"Elapsed: {elapsed:.2f}s")
    print("Figures: figures/posterior-fit.png, figures/marginal-likelihood-curve.png,")
    print("         figures/prior-vs-posterior-samples.png, figures/thumb.png")


if __name__ == "__main__":
    main()
