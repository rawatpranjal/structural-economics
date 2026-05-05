# Bond Prices and Yield to Maturity

> Promised fixed-income cash flows, present values, and implied discount rates.

## Overview

A fixed-income security is first a claim on promised dollars at dated horizons. Its price asks how much those promises are worth today. The yield to maturity turns the same information around: given the price, it reports the single annual discount rate that makes the promised cash flows add back up to that price.

That compression is useful, but it is also easy to overread. YTM is an internal rate for a stated cash-flow schedule, not a statement about realized holding-period returns, reinvestment rates, default, taxes, or calls. This tutorial keeps the cash flows deterministic so the economic object is clean. The data-rich term-structure analogue is the [Treasury yield curve](../treasury-yield-curve/); predictability questions appear later in the [Fama-Bliss-style regression](../fama-bliss-forward-regression/).

## Equations

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

## Model Setup

| Object | Value | Role |
|--------|-------|------|
| Face value $F$ | 100 | Par payoff for the coupon-bond examples |
| Baseline coupon rate $c$ | 6% | Annual coupon used in the price and YTM figures |
| Baseline maturity $T$ | 10 years | Horizon for the coupon-bond figures |
| Yield grid | 0.5% to 14% | Range used only for plotting exact present values |
| YTM root bracket | -95% to 200% | Bracket for the annual effective yield |
| Cash-flow examples | 6 instruments | Loans, discount bonds, perpetuities, coupons, and annuities |

## Solution Method

Once the dated cash flows are fixed, pricing is analytic. The only numerical step is the one-dimensional inversion from price to yield. The code evaluates the present-value gap and uses a bracketing root finder; the sign change matters because it preserves the economic monotonicity of price in yield.

```text
Algorithm: yield to maturity for a promised cash-flow claim
Input: price P, payment dates tau_m, payments C_m, yield bracket [y_low, y_high]
Output: implied annual yield y_star
Define G(y) = sum_m C_m / (1 + y)^tau_m - P
Check that G(y_low) and G(y_high) have opposite signs
repeat:
    choose a trial yield inside the bracket using Brent's step
    evaluate the present-value gap G(y)
    shrink the bracket while keeping the sign change
until the gap and bracket width are numerically small
return y_star
```

The plotted price-yield curves are not simulations. They are exact present values for a fixed coupon schedule evaluated over a grid of yields. For the YTM examples, the residual $PV(y^{*})-P$ is the relevant numerical check.

## Results

Holding the promised payments fixed, a higher discount rate lowers the price. The par line helps separate premium from discount bonds. A 6% coupon bond sells at par when the market yield is 6%; if the required yield is higher, the same cash-flow claim must trade below par.

Coupon rate shifts the level of promised payments, but the inverse price-yield relationship is common across the three schedules.

<img src="figures/price-yield-curve.png" alt="Price-yield schedule for 10-year coupon bonds" width="80%">

The inversion works in the other direction: for the same 6% coupon schedule, lower observed prices imply higher yields. At price 95, the implied annual YTM is **6.70%**, above the coupon rate because the buyer earns both coupons and capital gain back to face value.

The vertical line is par and the horizontal line is the coupon rate. Their intersection is the par-bond case; away from par, YTM absorbs the capital gain or loss implied by the purchase price.

<img src="figures/implied-yield-by-price.png" alt="Yield to maturity implied by price for a 6% coupon bond" width="80%">

The table applies the same price equation across debt instruments. Closed-form cases and root-solved cases are mixed deliberately: the common object is the present-value residual, which is essentially zero at the reported yield.

**YTM calculations for different promised cash-flow patterns**

| Instrument          | Payment pattern                |     Price | YTM    |   PV residual | Interpretation                                     |
|:--------------------|:-------------------------------|----------:|:-------|--------------:|:---------------------------------------------------|
| Simple loan         | 1100 paid in year 6            |   1000    | 1.60%  |     -1.14e-13 | One terminal principal-plus-interest payment.      |
| Discount bond       | 100 paid in year 5             |     95    | 1.03%  |      0        | No coupon; all payoff comes at maturity.           |
| Perpetuity          | 5 paid every year forever      |     99    | 5.05%  |      0        | Closed-form yield is coupon divided by price.      |
| Fixed-payment loan  | 9439.29 paid for 20 years      | 100000    | 7.00%  |     -9.88e-09 | Equal annual payments amortize the loan.           |
| Coupon bond         | 6 per year plus 100 in year 10 |     95    | 6.70%  |     -1.78e-10 | Coupon stream plus final face value.               |
| Arbitrary cash flow | 100 paid for 7 years           |    486.84 | 10.00% |      2.34e-10 | YTM is still a root of the present-value equation. |

## Takeaway

Yield to maturity is best read as an implied discount rate for a promised cash-flow schedule. It is useful because it puts loans, discount bonds, coupon bonds, and annuities into a common present-value language. The cost is compression: one yield hides the timing of payments and says little by itself about realized returns when reinvestment, default, calls, taxes, or interim sale prices matter.

## References

- [OpenStax. 10.2 Bond Valuation.](https://openstax.org/books/principles-finance/pages/10-2-bond-valuation)
- [CFA Institute. Fixed-Income Bond Valuation: Prices and Yields.](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/fixed-income-bond-valuation-prices-and-yields)
