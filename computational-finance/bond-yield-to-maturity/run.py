#!/usr/bin/env python3
"""Bond prices and yield-to-maturity calculations.

This tutorial converts the old bond-yield notebook into one reproducible script.
It treats YTM as the internal rate of return that prices promised cash flows,
then shows how price, coupon, maturity, and face value move the solution.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


@dataclass(frozen=True)
class CashFlowInstrument:
    """A named fixed-income instrument represented by dated cash flows."""

    name: str
    price: float
    times: np.ndarray
    cashflows: np.ndarray
    note: str


def present_value(cashflows: np.ndarray, times: np.ndarray, yield_rate: float) -> float:
    """Discount arbitrary cash flows at an annual effective yield."""
    return float(np.sum(cashflows / (1.0 + yield_rate) ** times))


def solve_ytm(cashflows: np.ndarray, times: np.ndarray, price: float) -> float:
    """Find the annual effective yield that prices a cash-flow stream."""

    def gap(yield_rate: float) -> float:
        return present_value(cashflows, times, yield_rate) - price

    return float(brentq(gap, -0.95, 2.0))


def coupon_cashflows(face: float, coupon_rate: float, maturity: int) -> tuple[np.ndarray, np.ndarray]:
    """Return annual coupon-bond payment times and cash flows."""
    times = np.arange(1, maturity + 1, dtype=float)
    cashflows = np.full(maturity, coupon_rate * face, dtype=float)
    cashflows[-1] += face
    return times, cashflows


def fixed_payment_cashflows(payment: float, maturity: int) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed annual loan-payment cash flows."""
    times = np.arange(1, maturity + 1, dtype=float)
    cashflows = np.full(maturity, payment, dtype=float)
    return times, cashflows


def instrument_examples() -> list[CashFlowInstrument]:
    """Create the short examples from the source notebook in a common format."""
    coupon_times, coupon_flows = coupon_cashflows(face=100.0, coupon_rate=0.10, maturity=10)
    mortgage_times, mortgage_flows = fixed_payment_cashflows(payment=9439.29, maturity=20)
    arbitrary_times = np.arange(1, 8, dtype=float)

    return [
        CashFlowInstrument(
            "Simple loan",
            1000.0,
            np.array([6.0]),
            np.array([1100.0]),
            "One final principal-plus-interest payment.",
        ),
        CashFlowInstrument(
            "Discount bond",
            95.0,
            np.array([5.0]),
            np.array([100.0]),
            "No coupon; all payoff comes at maturity.",
        ),
        CashFlowInstrument(
            "Perpetuity",
            99.0,
            np.array([1.0]),
            np.array([5.0]),
            "Closed-form yield is coupon divided by price.",
        ),
        CashFlowInstrument(
            "Fixed-payment loan",
            100000.0,
            mortgage_times,
            mortgage_flows,
            "Equal annual payments amortize the loan.",
        ),
        CashFlowInstrument(
            "Coupon bond",
            95.0,
            coupon_times,
            coupon_flows,
            "Coupon stream plus final face value.",
        ),
        CashFlowInstrument(
            "Arbitrary cash flow",
            486.84,
            arbitrary_times,
            np.full(7, 100.0),
            "YTM is still a root of the present-value equation.",
        ),
    ]


def ytm_summary(instruments: list[CashFlowInstrument]) -> pd.DataFrame:
    """Return a compact table of YTM examples."""
    rows = []
    for item in instruments:
        if item.name == "Perpetuity":
            ytm = item.cashflows[0] / item.price
            pv_check = item.price
        else:
            ytm = solve_ytm(item.cashflows, item.times, item.price)
            pv_check = present_value(item.cashflows, item.times, ytm)
        rows.append(
            {
                "Instrument": item.name,
                "Price": f"{item.price:.2f}",
                "YTM": f"{100.0 * ytm:.2f}%",
                "PV at YTM": f"{pv_check:.2f}",
                "Interpretation": item.note,
            }
        )
    return pd.DataFrame(rows)


def coupon_bond_price(face: float, coupon_rate: float, maturity: int, yield_rate: np.ndarray) -> np.ndarray:
    """Price an annual coupon bond over a vector of yields."""
    times, cashflows = coupon_cashflows(face, coupon_rate, maturity)
    return np.sum(cashflows[None, :] / (1.0 + yield_rate[:, None]) ** times[None, :], axis=1)


def main() -> None:
    setup_style()
    instruments = instrument_examples()
    summary = ytm_summary(instruments)

    print("Bond yield-to-maturity examples")
    print(summary[["Instrument", "Price", "YTM"]].to_string(index=False))

    report = ModelReport(
        "Bond Prices and Yield to Maturity",
        "Pricing promised cash flows and solving for the yield that rationalizes price.",
    )

    report.add_overview(
        "Yield to maturity is the discount rate that makes the present value of a bond's "
        "promised payments equal to its current price. That makes it a useful summary rate, "
        "but it is not a guaranteed realized return unless the promised payments arrive and "
        "the holding-period assumptions are satisfied.\n\n"
        "The source notebook listed several debt instruments separately. This version puts "
        "them in one cash-flow representation so the same present-value logic handles simple "
        "loans, discount bonds, fixed-payment loans, coupon bonds, and arbitrary cash flows."
    )

    report.add_equations(
        r"""
For promised cash flows $C_t$ paid at dates $t = 1,\ldots,T$, price is

$$
P = \sum_{t=1}^{T} \frac{C_t}{(1+y)^t}.
$$

The yield to maturity is the value of $y$ that solves this equation for an
observed price $P$. A perpetuity has the closed-form price

$$
P = \frac{C}{y},
\qquad
y = \frac{C}{P}.
$$

For a coupon bond with face value $F$, coupon rate $c$, and maturity $T$,

$$
P = \sum_{t=1}^{T} \frac{cF}{(1+y)^t} + \frac{F}{(1+y)^T}.
$$
"""
    )

    report.add_model_setup(
        "| Object | Value |\n"
        "|--------|-------|\n"
        "| Coupon-bond face value | 100 |\n"
        "| Baseline coupon rate | 6% |\n"
        "| Baseline maturity | 10 years |\n"
        "| Yield grid | 0.5% to 14% |\n"
        "| Root finder | Brent method on the present-value gap |"
    )

    report.add_solution_method(
        "All finite instruments are converted into payment times and cash-flow amounts. "
        "For a candidate yield, the script discounts each cash flow and compares the sum "
        "with the observed price. The YTM is found with a scalar root finder. The plots "
        "then hold the cash-flow schedule fixed while varying price, coupon, or yield."
    )

    yields = np.linspace(0.005, 0.14, 180)
    fig1, ax1 = plt.subplots(figsize=(7.4, 5.2))
    for coupon_rate in [0.02, 0.06, 0.10]:
        prices = coupon_bond_price(100.0, coupon_rate, 10, yields)
        ax1.plot(100.0 * yields, prices, label=f"{100.0 * coupon_rate:.0f}% coupon")
    ax1.axhline(100.0, color="black", linestyle="--", linewidth=1.0, label="Par")
    ax1.set_xlabel("Yield to maturity (%)")
    ax1.set_ylabel("Bond price")
    ax1.set_title("Higher Yields Lower Fixed-Cash-Flow Prices")
    ax1.legend()
    report.add_figure(
        "figures/price-yield-curve.png",
        "Coupon-bond price as yield changes",
        fig1,
        description=(
            "The inverse price-yield relationship is mechanical: a higher discount rate lowers "
            "the present value of the same promised cash flows. Premium and discount status "
            "depend on how the coupon compares with the market yield."
        ),
    )

    prices = np.linspace(80.0, 120.0, 160)
    times, flows = coupon_cashflows(face=100.0, coupon_rate=0.06, maturity=10)
    implied_yields = np.array([solve_ytm(flows, times, price) for price in prices])
    fig2, ax2 = plt.subplots(figsize=(7.4, 5.0))
    ax2.plot(prices, 100.0 * implied_yields, color="tab:blue")
    ax2.axvline(100.0, color="black", linestyle="--", linewidth=1.0)
    ax2.axhline(6.0, color="black", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("Observed price")
    ax2.set_ylabel("Implied YTM (%)")
    ax2.set_title("Observed Price Pins Down the Implied Yield")
    report.add_figure(
        "figures/implied-yield-by-price.png",
        "Yield to maturity implied by price",
        fig2,
        description=(
            "For the same 6% coupon bond, a price below par implies a YTM above the coupon "
            "rate, while a price above par implies a YTM below the coupon rate."
        ),
    )

    report.add_table(
        "tables/instrument-summary.csv",
        "Yield-to-maturity examples",
        summary,
        description=(
            "The same present-value equation handles the source notebook's different debt "
            "instruments once each one is written as cash flows."
        ),
    )

    report.add_takeaway(
        "YTM is best read as an implied discount rate for promised cash flows. It is useful "
        "because it compresses a price and cash-flow schedule into one number, but that "
        "compression hides reinvestment, default, call, tax, and holding-period issues."
    )
    report.add_references(
        [
            "[OpenStax. 10.2 Bond Valuation.](https://openstax.org/books/principles-finance/pages/10-2-bond-valuation)",
            "[CFA Institute. Fixed-Income Bond Valuation: Prices and Yields.](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/fixed-income-bond-valuation-prices-and-yields)",
        ]
    )
    report.write()


if __name__ == "__main__":
    main()
