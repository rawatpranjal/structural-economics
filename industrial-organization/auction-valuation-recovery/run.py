#!/usr/bin/env python3
"""Recover private values from first-price auction bids.

This tutorial simulates a symmetric IPV first-price auction, hides the values,
and applies the GPV bid inversion to recover pseudo-values from observed bids.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import beta, gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def equilibrium_bid_grid(
    value_grid: np.ndarray,
    n_bidders: int,
    dist: beta,
) -> np.ndarray:
    """Compute the symmetric risk-neutral first-price bid schedule."""
    cdf_power = dist.cdf(value_grid) ** (n_bidders - 1)
    integral = cumulative_trapezoid(cdf_power, value_grid, initial=0.0)
    bid_grid = value_grid - np.divide(
        integral,
        cdf_power,
        out=np.zeros_like(value_grid),
        where=cdf_power > 1e-12,
    )
    bid_grid[0] = 0.0
    return np.maximum.accumulate(bid_grid)


def simulate_auctions(
    n_auctions: int,
    n_bidders: int,
    seed: int,
) -> pd.DataFrame:
    """Draw values and equilibrium bids for repeated IPV first-price auctions."""
    rng = np.random.default_rng(seed)
    dist = beta(a=2.0, b=5.0)
    value_grid = np.linspace(1e-5, 0.99999, 5000)
    bid_grid = equilibrium_bid_grid(value_grid, n_bidders, dist)

    values = rng.beta(2.0, 5.0, size=(n_auctions, n_bidders))
    bids = np.interp(values, value_grid, bid_grid)

    return pd.DataFrame({
        "auction": np.repeat(np.arange(n_auctions), n_bidders),
        "bidder": np.tile(np.arange(n_bidders), n_auctions),
        "value": values.ravel(),
        "bid": bids.ravel(),
    })


def empirical_cdf_at_sample(x: np.ndarray) -> np.ndarray:
    """Evaluate the empirical CDF at each observed sample point."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks / len(x)


def recover_pseudo_values(
    bids: np.ndarray,
    n_bidders: int,
    trim_quantile: float,
) -> pd.DataFrame:
    """Apply the GPV-style inversion to observed first-price bids."""
    bid_cdf = empirical_cdf_at_sample(bids)
    kde = gaussian_kde(bids)
    bid_density = kde(bids)
    lower, upper = np.quantile(bids, [trim_quantile, 1.0 - trim_quantile])
    keep = (bids >= lower) & (bids <= upper) & (bid_density > 1e-8)
    pseudo_value = bids + bid_cdf / ((n_bidders - 1) * bid_density)

    return pd.DataFrame({
        "bid": bids,
        "bid_cdf": bid_cdf,
        "bid_density": bid_density,
        "pseudo_value": pseudo_value,
        "kept": keep,
    })


def ecdf_grid(sample: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate an empirical CDF on a fixed grid."""
    sample_sorted = np.sort(sample)
    return np.searchsorted(sample_sorted, grid, side="right") / len(sample_sorted)


def summarize_distribution(
    true_values: np.ndarray,
    recovered_values: np.ndarray,
) -> pd.DataFrame:
    """Compare distribution moments and quantiles."""
    rows = []
    for label, data in [
        ("True values", true_values),
        ("Recovered pseudo-values", recovered_values),
    ]:
        rows.append({
            "Series": label,
            "Mean": np.mean(data),
            "Std. dev.": np.std(data, ddof=1),
            "P10": np.quantile(data, 0.10),
            "Median": np.quantile(data, 0.50),
            "P90": np.quantile(data, 0.90),
        })
    return pd.DataFrame(rows).round(3)


def binned_errors(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Average recovery error by bid decile."""
    out = df.copy()
    out["Bid decile"] = pd.qcut(out["bid"], n_bins, labels=[f"D{i}" for i in range(1, n_bins + 1)])
    return (
        out.groupby("Bid decile", observed=True)
        .agg(
            bid_midpoint=("bid", "mean"),
            mean_error=("error", "mean"),
            mae=("abs_error", "mean"),
            count=("error", "size"),
        )
        .reset_index()
    )


def main() -> None:
    n_auctions = 3_000
    n_bidders = 4
    trim_quantile = 0.05
    seed = 20260508

    setup_style()
    auctions = simulate_auctions(n_auctions, n_bidders, seed)
    recovered = recover_pseudo_values(
        auctions["bid"].to_numpy(),
        n_bidders=n_bidders,
        trim_quantile=trim_quantile,
    )
    data = pd.concat([auctions, recovered.drop(columns=["bid"])], axis=1)
    data = data[data["kept"]].copy()
    data["error"] = data["pseudo_value"] - data["value"]
    data["abs_error"] = np.abs(data["error"])

    true_trimmed_values = data["value"].to_numpy()
    recovered_values = data["pseudo_value"].to_numpy()
    distribution_table = summarize_distribution(true_trimmed_values, recovered_values)
    diagnostics = pd.DataFrame([{
        "Auctions": n_auctions,
        "Bidders": n_bidders,
        "Observed bids": len(auctions),
        "Kept bids": len(data),
        "Trimmed share": round(1.0 - len(data) / len(auctions), 3),
        "RMSE": round(float(np.sqrt(np.mean(data["error"] ** 2))), 3),
        "MAE": round(float(data["abs_error"].mean()), 3),
        "Correlation": round(float(np.corrcoef(data["value"], data["pseudo_value"])[0, 1]), 3),
    }])
    error_bins = binned_errors(data)

    print("Auction valuation recovery tutorial")
    print(f"  Auctions: {n_auctions}")
    print(f"  Bidders per auction: {n_bidders}")
    print(f"  Kept bids after trimming: {len(data)} / {len(auctions)}")
    print(f"  RMSE: {diagnostics['RMSE'].iloc[0]:.3f}")
    print(f"  Correlation: {diagnostics['Correlation'].iloc[0]:.3f}")

    report = ModelReport(
        "Recovering Auction Values from First-Price Bids",
        "Use the first-price auction equilibrium condition to turn observed bids into pseudo-values.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "What do bids reveal about values? In a first-price auction the answer is not "
        "the bid itself. Bidders shade below value because paying more reduces surplus "
        "when they win.\n\n"
        "The object is the latent distribution of private values. The data contain all "
        "bids and the number of bidders, but the econometrician does not observe values.\n\n"
        "The tutorial simulates a known-truth auction, discards the values during "
        "estimation, estimates the bid distribution, and inverts the first-order "
        "condition into pseudo-values. The known truth is used only to check recovery."
    )

    report.add_equations(r"""
There are $n$ symmetric risk-neutral bidders in each first-price sealed-bid
auction. Bidder $i$ observes a private value $v_i$ drawn independently from a
common distribution $F_v$ and submits bid $b_i$. The highest bidder wins and
pays its own bid.

Let $s(v)$ be a strictly increasing symmetric equilibrium bid rule. A bidder
with value $v$ who bids as if its type were $x$ earns

$$
\pi(v,x) = (v-s(x))F_v(x)^{n-1}.
$$

The first-order condition at $x=v$ is

$$
s'(v)F_v(v) + (n-1)(s(v)-v)f_v(v) = 0.
$$

This condition implies the equilibrium bid schedule

$$
s(v)=v-\frac{\int_0^v F_v(t)^{n-1}dt}{F_v(v)^{n-1}}.
$$

For estimation, write the same condition in terms of the observed bid
distribution. If $G$ and $g$ are the CDF and density of bids, then monotonicity
gives $G(b)=F_v(v)$ and $g(b)=f_v(v)/s'(v)$. Solving for value gives the
GPV-style pseudo-value inversion:

$$
\hat v_i = b_i + \frac{\hat G(b_i)}{(n-1)\hat g(b_i)}.
$$

The density estimate is least stable near the bid support boundaries, so the
exercise trims low and high bid quantiles before evaluating recovery.
""")

    report.add_model_setup(
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        f"| Auctions | {n_auctions:,} | Independent repetitions |\n"
        f"| Bidders per auction | {n_bidders} | Fixed competition level in the inversion |\n"
        "| Value distribution | Beta(2,5) on [0,1] | Known truth for simulation only |\n"
        "| Auction format | First-price sealed bid | Winner pays own bid |\n"
        "| Reserve price | None | Lower support starts at zero |\n"
        "| Observed by estimator | Bids and bidder count | Values are hidden during recovery |\n"
        f"| Boundary trim | {trim_quantile:.0%} in each tail | Avoids unstable density estimates |"
    )

    report.add_solution_method(
        "The computation has two parts. First it creates auction data from the symmetric "
        "equilibrium. Then it acts like an econometrician who only sees bids.\n\n"
        "```text\n"
        "Algorithm: GPV-style value recovery\n"
        "Input: all bids b_i from auctions with n bidders\n"
        "Output: pseudo-values vhat_i and an estimated value distribution\n\n"
        "1. Simulate private values from a known distribution F_v.\n"
        "2. Compute the monotone first-price equilibrium bid rule s(v).\n"
        "3. Store the bids b_i = s(v_i), then hide the values from the estimator.\n"
        "4. Estimate the bid CDF Ghat(b_i) with empirical ranks.\n"
        "5. Estimate the bid density ghat(b_i) with a kernel density estimate.\n"
        "6. Drop the lowest and highest bid tails.\n"
        "7. Recover pseudo-values vhat_i = b_i + Ghat(b_i) / [(n-1) ghat(b_i)].\n"
        "8. Compare the pseudo-value distribution with the hidden true values.\n"
        "```\n\n"
        "The key economic input is monotonicity. It lets the rank of a bid stand in for "
        "the rank of a value."
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(
        auctions["value"],
        bins=45,
        density=True,
        alpha=0.42,
        color="#4C78A8",
        label="Latent values",
    )
    ax1.hist(
        auctions["bid"],
        bins=45,
        density=True,
        alpha=0.42,
        color="#F58518",
        label="Observed bids",
    )
    ax1.set_xlabel("Dollars")
    ax1.set_ylabel("Density")
    ax1.set_title("Observed Bids Are Shaded Below Latent Values")
    ax1.legend()
    report.add_results(
        "The bid distribution is shifted left of the value distribution. That gap is "
        "bid shading. A bid is therefore not a direct measure of willingness to pay."
    )
    report.add_figure(
        "figures/bids-versus-values.png",
        "Observed bid density compared with the latent value density",
        fig1,
    )

    grid = np.linspace(
        min(true_trimmed_values.min(), recovered_values.min()),
        max(true_trimmed_values.max(), recovered_values.max()),
        250,
    )
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(grid, ecdf_grid(true_trimmed_values, grid), color="#4C78A8", label="True values")
    ax2.plot(
        grid,
        ecdf_grid(recovered_values, grid),
        color="#E45756",
        linestyle="--",
        label="Recovered pseudo-values",
    )
    ax2.set_xlabel("Value")
    ax2.set_ylabel("CDF")
    ax2.set_title("Recovered Values Track the True Value Distribution")
    ax2.legend()
    report.add_results(
        "After the inversion, the recovered pseudo-value CDF is close to the true value "
        "CDF over the trimmed support. The match is not exact because the bid density "
        "is estimated nonparametrically."
    )
    report.add_figure(
        "figures/recovered-cdf.png",
        "Recovered pseudo-value CDF compared with true values",
        fig2,
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.axhline(0.0, color="black", linewidth=1.1)
    ax3.plot(
        error_bins["bid_midpoint"],
        error_bins["mean_error"],
        marker="o",
        color="#E45756",
        label="Mean error",
    )
    ax3.plot(
        error_bins["bid_midpoint"],
        error_bins["mae"],
        marker="s",
        color="#54A24B",
        label="Mean absolute error",
    )
    ax3.set_xlabel("Bid decile midpoint")
    ax3.set_ylabel("Recovery error")
    ax3.set_title("Recovery Error by Bid Region")
    ax3.legend()
    report.add_results(
        "Errors are smallest in the middle of the bid support and larger near the "
        "remaining boundaries. That pattern is why empirical GPV applications usually "
        "pay close attention to trimming, bandwidths, and boundary correction."
    )
    report.add_figure(
        "figures/recovery-error-by-bid.png",
        "Pseudo-value recovery error by bid decile",
        fig3,
    )

    report.add_table(
        "tables/value-recovery-summary.csv",
        "True and Recovered Value Distributions",
        distribution_table,
    )
    report.add_table(
        "tables/recovery-diagnostics.csv",
        "Recovery Diagnostics",
        diagnostics,
    )

    report.add_takeaway(
        "Observed first-price bids mix values with strategic shading. Under symmetric "
        "IPV assumptions and monotone bidding, the equilibrium first-order condition "
        "turns the bid CDF and density into pseudo-values.\n\n"
        "The exercise also shows the cost of the method. Recovery depends on a density "
        "estimate, so the edges of the bid support are fragile. Trimming is not cosmetic; "
        "it is part of making the structural inversion usable."
    )

    report.add_references([
        "[Perrigne, I. and Vuong, Q. (2019). Econometrics of Auctions and Nonlinear Pricing. *Annual Review of Economics*, 11, 27-54.](https://doi.org/10.1146/annurev-economics-080218-025702)",
        "[Guerre, E., Perrigne, I., and Vuong, Q. (2000). Optimal Nonparametric Estimation of First-Price Auctions. *Econometrica*, 68(3), 525-574.](https://doi.org/10.1111/1468-0262.00123)",
        "[Gentry, M., Komarova, T., and Schiraldi, P. (2023). Preferences and Performance in Simultaneous First-Price Auctions: A Structural Analysis. *Review of Economic Studies*, 90(2), 852-878.](https://doi.org/10.1093/restud/rdac030)",
        "[Hickman, B. R., Hubbard, T. P., and Saglam, Y. (2012). Structural Econometric Methods in Auctions: A Guide to the Literature. *Journal of Econometric Methods*, 1(1), 67-106.](https://doi.org/10.1515/2156-6674.1019)",
        "[Krishna, V. (2009). *Auction Theory*, 2nd ed. Academic Press.](https://shop.elsevier.com/books/auction-theory/krishna/978-0-12-374507-1)",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
