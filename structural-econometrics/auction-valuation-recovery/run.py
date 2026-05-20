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
from lib.plotting import save_figure, save_thumbnail, setup_style


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
    all_bids = simulate_auctions(n_auctions, n_bidders, seed)
    recovered = recover_pseudo_values(
        all_bids["bid"].to_numpy(),
        n_bidders=n_bidders,
        trim_quantile=trim_quantile,
    )
    data = pd.concat([all_bids, recovered.drop(columns=["bid"])], axis=1)
    data = data[data["kept"]].copy()
    data["error"] = data["pseudo_value"] - data["value"]
    data["abs_error"] = np.abs(data["error"])

    true_trimmed_values = data["value"].to_numpy()
    recovered_values = data["pseudo_value"].to_numpy()
    distribution_table = summarize_distribution(true_trimmed_values, recovered_values)
    diagnostics = pd.DataFrame([{
        "Auctions": n_auctions,
        "Bidders": n_bidders,
        "Observed bids": len(all_bids),
        "Kept bids": len(data),
        "Trimmed share": round(1.0 - len(data) / len(all_bids), 3),
        "RMSE": round(float(np.sqrt(np.mean(data["error"] ** 2))), 3),
        "MAE": round(float(data["abs_error"].mean()), 3),
        "Correlation": round(float(np.corrcoef(data["value"], data["pseudo_value"])[0, 1]), 3),
    }])
    error_bins = binned_errors(data)

    print("Auction valuation recovery tutorial")
    print(f"  Auctions: {n_auctions}")
    print(f"  Bidders per auction: {n_bidders}")
    print(f"  Kept bids after trimming: {len(data)} / {len(all_bids)}")
    print(f"  RMSE: {diagnostics['RMSE'].iloc[0]:.3f}")
    print(f"  Correlation: {diagnostics['Correlation'].iloc[0]:.3f}")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(
        all_bids["value"],
        bins=45,
        density=True,
        alpha=0.42,
        color="#4C78A8",
        label="Latent values",
    )
    ax1.hist(
        all_bids["bid"],
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
    save_figure(fig1, "figures/bids-versus-values.png", dpi=150)

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
    save_figure(fig2, "figures/recovered-cdf.png", dpi=150)

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
    save_figure(fig3, "figures/recovery-error-by-bid.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    distribution_table.to_csv("tables/value-recovery-summary.csv", index=False)
    diagnostics.to_csv("tables/recovery-diagnostics.csv", index=False)

    save_thumbnail("figures/bids-versus-values.png", "figures/thumb.png")
    print(f"Done: 3 figures, 2 tables")


if __name__ == "__main__":
    main()
