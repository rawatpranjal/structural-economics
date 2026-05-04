#!/usr/bin/env python3
"""AR(1) forecasting for Treasury yields."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def load_ten_year() -> pd.DataFrame:
    """Load the ten-year Treasury yield from the static Treasury CMT dataset."""
    path = Path(__file__).resolve().parents[1] / "_data" / "daily-treasury-rates.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "10 Yr"]].rename(columns={"10 Yr": "Yield"})


def fit_ar1(series: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Estimate y[t+1] = alpha + rho y[t] + error."""
    y_next = series[1:]
    y_lag = series[:-1]
    design = np.column_stack([np.ones_like(y_lag), y_lag])
    alpha, rho = np.linalg.lstsq(design, y_next, rcond=None)[0]
    fitted = design @ np.array([alpha, rho])
    return float(alpha), float(rho), fitted


def forecast_test(df: pd.DataFrame, train_share: float = 0.7) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit on an initial sample and evaluate one-step forecasts out of sample."""
    series = df["Yield"].to_numpy()
    train_n = int(train_share * len(series))
    alpha, rho, fitted_train = fit_ar1(series[:train_n])

    rows = []
    for t in range(train_n - 1, len(series) - 1):
        current = series[t]
        actual = series[t + 1]
        ar1_forecast = alpha + rho * current
        rw_forecast = current
        rows.append(
            {
                "Date": df.loc[t + 1, "Date"],
                "Actual": actual,
                "AR(1)": ar1_forecast,
                "No-change": rw_forecast,
                "AR(1) error": actual - ar1_forecast,
                "No-change error": actual - rw_forecast,
            }
        )
    test = pd.DataFrame(rows)
    params = {
        "alpha": alpha,
        "rho": rho,
        "train_n": train_n,
        "fitted_train_rmse": float(np.sqrt(np.mean((series[1:train_n] - fitted_train) ** 2))),
    }
    return params, test


def accuracy_table(test: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    """Return compact forecast accuracy diagnostics."""
    rows = []
    for model, error_col in [("AR(1)", "AR(1) error"), ("No-change", "No-change error")]:
        errors = test[error_col].to_numpy()
        rows.append(
            {
                "Model": model,
                "RMSE (bp)": f"{100.0 * np.sqrt(np.mean(errors**2)):.2f}",
                "MAE (bp)": f"{100.0 * np.mean(np.abs(errors)):.2f}",
                "Mean error (bp)": f"{100.0 * np.mean(errors):.2f}",
                "Test obs.": len(test),
            }
        )
    rows.append(
        {
            "Model": "Estimated rho",
            "RMSE (bp)": f"{params['rho']:.3f}",
            "MAE (bp)": "",
            "Mean error (bp)": "",
            "Test obs.": int(params["train_n"]),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    setup_style()
    df = load_ten_year()
    params, test = forecast_test(df)
    accuracy = accuracy_table(test, params)

    print("AR(1) rate forecasting")
    print(accuracy.to_string(index=False))

    report = ModelReport(
        "AR(1) Forecasting for Treasury Yields",
        "A persistent-yield benchmark compared with a no-change forecast.",
    )

    report.add_overview(
        "Interest rates are often highly persistent: today's level contains substantial "
        "information about tomorrow's level. An AR(1) model puts that persistence in one "
        "coefficient, which controls how much the current yield carries into the next "
        "one-step prediction.\n\n"
        "Because rates move slowly at daily horizons, the no-change forecast is the natural "
        "benchmark. A fitted autoregression has to improve on the forecast that simply sets "
        "tomorrow's yield equal to today's yield. The data are a static 1990 Treasury CMT "
        "snapshot, so the exercise is a compact benchmark rather than a full term-structure "
        "forecasting model."
    )

    report.add_equations(
        r"""
The AR(1) model is

$$
y_{t+1} = \alpha + \rho y_t + \epsilon_{t+1}.
$$

The one-step-ahead forecast is

$$
\widehat{y}_{t+1|t} = \widehat{\alpha} + \widehat{\rho} y_t.
$$

When $|\rho| < 1$, the stationary mean is

$$
\mu = \frac{\alpha}{1-\rho}.
$$
"""
    )

    report.add_model_setup(
        f"| Object | Value |\n"
        f"|--------|-------|\n"
        f"| Series | 10-year Treasury yield |\n"
        f"| Data | Static 1990 Treasury CMT snapshot |\n"
        f"| Training share | 70% |\n"
        f"| Estimated $\\rho$ | {params['rho']:.3f} |\n"
        f"| Benchmark | No-change forecast $y_{{t+1}} = y_t$ |"
    )

    report.add_solution_method(
        "AR(1) coefficients are estimated by least squares on the first 70% of the sample. "
        "Rolling one-step forecasts are evaluated on the remaining dates. The comparison "
        "forecast simply sets tomorrow's yield equal to today's yield."
    )

    fig1, ax1 = plt.subplots(figsize=(7.4, 5.2))
    ax1.plot(df["Date"], df["Yield"], label="Observed", color="tab:blue")
    ax1.plot(test["Date"], test["AR(1)"], label="AR(1) forecast", color="tab:orange")
    ax1.plot(test["Date"], test["No-change"], label="No-change forecast", color="tab:green", alpha=0.8)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("10-year yield (%)")
    ax1.set_title("One-Step Forecasts for the Ten-Year Yield")
    ax1.legend()
    report.add_figure(
        "figures/ar1-forecast.png",
        "Observed ten-year yield and one-step forecasts",
        fig1,
        description=(
            "With highly persistent rates, AR(1) and no-change forecasts can be close. That "
            "is an empirical feature: persistence makes simple benchmarks hard to beat at "
            "short horizons."
        ),
    )

    fig2, ax2 = plt.subplots(figsize=(7.4, 4.8))
    ax2.plot(test["Date"], 100.0 * test["AR(1) error"], label="AR(1)")
    ax2.plot(test["Date"], 100.0 * test["No-change error"], label="No-change", alpha=0.8)
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Forecast error (bp)")
    ax2.set_title("Forecast Errors")
    ax2.legend()
    report.add_figure(
        "figures/forecast-errors.png",
        "One-step forecast errors",
        fig2,
        description=(
            "Forecast errors are the object to compare, not visual fit alone. A model with "
            "more structure should earn its place by improving out-of-sample errors."
        ),
    )

    report.add_table(
        "tables/forecast-accuracy.csv",
        "Forecast accuracy",
        accuracy,
    )

    report.add_takeaway(
        "AR(1) models are useful first forecasting tools because they force the persistence "
        "question into one parameter. For interest rates, a no-change forecast is often a "
        "tough baseline, so a more elaborate model should be judged against it."
    )
    report.add_references(
        [
            "[QuantEcon. AR(1) Processes.](https://intro.quantecon.org/ar1_processes.html)",
        ]
    )
    report.write()


if __name__ == "__main__":
    main()
