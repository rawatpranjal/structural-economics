#!/usr/bin/env python3
"""Bond prices and yield-to-maturity calculations.

Fixed-income instruments are represented as dated cash-flow claims. The
calculation solves for the discount rate that makes promised payments match
market price.
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
    pattern: str
    note: str
    perpetuity: bool = False


def present_value(cashflows: np.ndarray, times: np.ndarray, yield_rate: float) -> float:
    """Discount arbitrary cash flows at an annual effective yield."""
    return float(np.sum(cashflows / (1.0 + yield_rate) ** times))


def solve_ytm(cashflows: np.ndarray, times: np.ndarray, price: float) -> float:
    """Find the annual effective yield that prices a cash-flow stream."""

    def gap(yield_rate: float) -> float:
        return present_value(cashflows, times, yield_rate) - price

    return float(brentq(gap, -0.95, 2.0))


def instrument_value(item: CashFlowInstrument, yield_rate: float) -> float:
    """Evaluate the price of an instrument at a candidate annual yield."""
    if item.perpetuity:
        return float(item.cashflows[0] / yield_rate)
    return present_value(item.cashflows, item.times, yield_rate)


def instrument_ytm(item: CashFlowInstrument) -> float:
    """Return YTM using closed form where it is simple."""
    if item.perpetuity:
        return float(item.cashflows[0] / item.price)
    if len(item.cashflows) == 1:
        return float((item.cashflows[0] / item.price) ** (1.0 / item.times[0]) - 1.0)
    return solve_ytm(item.cashflows, item.times, item.price)


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
    """Create short debt-instrument examples in a common cash-flow format."""
    coupon_times, coupon_flows = coupon_cashflows(face=100.0, coupon_rate=0.06, maturity=10)
    mortgage_times, mortgage_flows = fixed_payment_cashflows(payment=9439.29, maturity=20)
    arbitrary_times = np.arange(1, 8, dtype=float)

    return [
        CashFlowInstrument(
            "Simple loan",
            1000.0,
            np.array([6.0]),
            np.array([1100.0]),
            "1100 paid in year 6",
            "One terminal principal-plus-interest payment.",
        ),
        CashFlowInstrument(
            "Discount bond",
            95.0,
            np.array([5.0]),
            np.array([100.0]),
            "100 paid in year 5",
            "No coupon; all payoff comes at maturity.",
        ),
        CashFlowInstrument(
            "Perpetuity",
            99.0,
            np.array([1.0]),
            np.array([5.0]),
            "5 paid every year forever",
            "Closed-form yield is coupon divided by price.",
            perpetuity=True,
        ),
        CashFlowInstrument(
            "Fixed-payment loan",
            100000.0,
            mortgage_times,
            mortgage_flows,
            "9439.29 paid for 20 years",
            "Equal annual payments amortize the loan.",
        ),
        CashFlowInstrument(
            "Coupon bond",
            95.0,
            coupon_times,
            coupon_flows,
            "6 per year plus 100 in year 10",
            "Coupon stream plus final face value.",
        ),
        CashFlowInstrument(
            "Arbitrary cash flow",
            486.84,
            arbitrary_times,
            np.full(7, 100.0),
            "100 paid for 7 years",
            "YTM is still a root of the present-value equation.",
        ),
    ]


def ytm_summary(instruments: list[CashFlowInstrument]) -> pd.DataFrame:
    """Return a compact table of YTM examples."""
    rows = []
    for item in instruments:
        ytm = instrument_ytm(item)
        residual = instrument_value(item, ytm) - item.price
        rows.append(
            {
                "Instrument": item.name,
                "Payment pattern": item.pattern,
                "Price": f"{item.price:.2f}",
                "YTM": f"{100.0 * ytm:.2f}%",
                "PV residual": f"{residual:.2e}",
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
        "Promised fixed-income cash flows, present values, and implied discount rates.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A fixed-income security is first a claim on promised dollars at dated horizons. "
        "Its price asks how much those promises are worth today. The yield to maturity "
        "turns the same information around: given the price, it reports the single annual "
        "discount rate that makes the promised cash flows add back up to that price.\n\n"
        "That compression is useful, but it is also easy to overread. YTM is an internal "
        "rate for a stated cash-flow schedule, not a statement about realized holding-period "
        "returns, reinvestment rates, default, taxes, or calls. This tutorial keeps the "
        "cash flows deterministic so the economic object stays clean. The data-rich term-"
        "structure analogue is the [Treasury yield curve](../treasury-yield-curve/); "
        "predictability questions appear later in the [Fama-Bliss-style regression](../fama-bliss-forward-regression/)."
    )

    report.add_equations(
        r"""
Let a bond or loan promise payments $C_m>0$ at annual dates $\tau_m$, for
$m=1,\ldots,M$. With annual effective yield $y$, its present value is

$$
PV(y)=\sum_{m=1}^{M}\frac{C_m}{(1+y)^{\tau_m}},\qquad y>-1.
$$

The yield to maturity for observed price $P$ is the root $y^{*}$ of

$$
G(y)=PV(y)-P=0.
$$

For positive promised cash flows, $PV(y)$ is strictly decreasing in $y$:

$$
PV'(y)=-\sum_{m=1}^{M}\frac{\tau_m C_m}{(1+y)^{\tau_m+1}}<0.
$$

This monotonicity gives a unique YTM when the price lies in the attainable
range. Some special cases have closed forms. A single terminal payment
$C_T$ at date $T$ implies

$$
y=\left(\frac{C_T}{P}\right)^{1/T}-1.
$$

A perpetuity with annual payment $C$ has

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
        "| Object | Value | Role |\n"
        "|--------|-------|------|\n"
        "| Face value $F$ | 100 | Par payoff for the coupon-bond examples |\n"
        "| Baseline coupon rate $c$ | 6% | Annual coupon used in the price and YTM figures |\n"
        "| Baseline maturity $T$ | 10 years | Horizon for the coupon-bond figures |\n"
        "| Yield grid | 0.5% to 14% | Range used only for plotting exact present values |\n"
        "| YTM root bracket | -95% to 200% | Bracket for the annual effective yield |\n"
        "| Cash-flow examples | 6 instruments | Loans, discount bonds, perpetuities, coupons, and annuities |"
    )

    report.add_solution_method(
        "Once the dated cash flows are fixed, pricing is analytic. The only numerical "
        "step is the one-dimensional inversion from price to yield. The code evaluates "
        "the present-value gap and uses a bracketing root finder; the sign change preserves "
        "the economic monotonicity of price in yield.\n\n"
        "```text\n"
        "Algorithm: yield to maturity for a promised cash-flow claim\n"
        "Input: price P, payment dates tau_m, payments C_m, yield bracket [y_low, y_high]\n"
        "Output: implied annual yield y_star\n"
        "Define G(y) = sum_m C_m / (1 + y)^tau_m - P\n"
        "Check that G(y_low) and G(y_high) have opposite signs\n"
        "repeat:\n"
        "    choose a trial yield inside the bracket using Brent's step\n"
        "    evaluate the present-value gap G(y)\n"
        "    shrink the bracket while keeping the sign change\n"
        "until the gap and bracket width are numerically small\n"
        "return y_star\n"
        "```\n\n"
        "The plotted price-yield curves are not simulations. They are exact present values "
        "for a fixed coupon schedule evaluated over a grid of yields. For the YTM examples, "
        "the residual $PV(y^{*})-P$ is the numerical check."
    )

    yields = np.linspace(0.005, 0.14, 180)
    fig1, ax1 = plt.subplots(figsize=(7.4, 5.2))
    for coupon_rate in [0.02, 0.06, 0.10]:
        prices = coupon_bond_price(100.0, coupon_rate, 10, yields)
        ax1.plot(100.0 * yields, prices, label=f"{100.0 * coupon_rate:.0f}% coupon")
    ax1.axhline(100.0, color="black", linestyle="--", linewidth=1.0, label="Par")
    ax1.set_xlabel("Yield to maturity (%)")
    ax1.set_ylabel("Bond price")
    ax1.set_title("Price-Yield Schedule for 10-Year Coupon Bonds")
    ax1.legend()
    report.add_results(
        "Holding the promised payments fixed, a higher discount rate lowers the price. "
        "The par line separates premium from discount bonds. A 6% coupon bond sells "
        "at par when the market yield is 6%; if the required yield is higher, the same "
        "cash-flow claim must trade below par."
    )
    report.add_figure(
        "figures/price-yield-curve.png",
        "Price-yield schedule for 10-year coupon bonds",
        fig1,
        description=(
            "Coupon rate shifts the level of promised payments. The inverse price-yield "
            "relationship is common across the three schedules."
        ),
    )

    prices = np.linspace(80.0, 120.0, 160)
    times, flows = coupon_cashflows(face=100.0, coupon_rate=0.06, maturity=10)
    implied_yields = np.array([solve_ytm(flows, times, price) for price in prices])
    baseline_price = 95.0
    baseline_ytm = solve_ytm(flows, times, baseline_price)
    fig2, ax2 = plt.subplots(figsize=(7.4, 5.0))
    ax2.plot(prices, 100.0 * implied_yields, color="tab:blue")
    ax2.scatter([baseline_price], [100.0 * baseline_ytm], color="tab:red", zorder=3, label="Price 95")
    ax2.axvline(100.0, color="black", linestyle="--", linewidth=1.0)
    ax2.axhline(6.0, color="black", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("Observed price")
    ax2.set_ylabel("Implied YTM (%)")
    ax2.set_title("Implied Yield for a 6% Coupon Bond")
    ax2.legend()
    report.add_results(
        "The inversion works in the other direction: for the same 6% coupon schedule, "
        "lower observed prices imply higher yields. At price 95, the implied annual YTM "
        f"is **{100.0 * baseline_ytm:.2f}%**, above the coupon rate because the buyer earns "
        "both coupons and capital gain back to face value."
    )
    report.add_figure(
        "figures/implied-yield-by-price.png",
        "Yield to maturity implied by price for a 6% coupon bond",
        fig2,
        description=(
            "The vertical line is par and the horizontal line is the coupon rate. Their "
            "intersection is the par-bond case; away from par, YTM absorbs the capital "
            "gain or loss implied by the purchase price."
        ),
    )

    report.add_table(
        "tables/instrument-summary.csv",
        "YTM calculations for different promised cash-flow patterns",
        summary,
        description=(
            "The table applies the same price equation across debt instruments. Closed-form "
            "cases and root-solved cases are mixed deliberately; the common object is the "
            "present-value residual, which is essentially zero at the reported yield."
        ),
    )

    report.add_takeaway(
        "Yield to maturity is best read as an implied discount rate for a promised cash-flow "
        "schedule. It puts loans, discount bonds, coupon bonds, and annuities into a common "
        "present-value language. The cost is compression: one yield hides the timing of "
        "payments and says little by itself about realized returns when reinvestment, "
        "default, calls, taxes, or interim sale prices matter."
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
