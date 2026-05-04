# Huggett Equilibrium and the Risk-Free Rate

> A continuous-time incomplete-markets economy where idiosyncratic income risk and borrowing limits determine the equilibrium bond return.

## Overview

Huggett (1993) asks how a risk-free asset is priced when households cannot perfectly insure idiosyncratic income risk. Each household can save or borrow in a single bond, but debt is limited by $a \geq \underline a$. Since the bond is in zero net supply, the interest rate has to make the cross-sectional demand for assets add up to zero.

The economic force is precautionary saving. Low-income spells make the borrowing limit valuable, so households try to carry buffer wealth into bad states. In equilibrium that desired asset demand is offset by a lower risk-free rate. The tutorial uses the continuous-time HJB/KFE representation from Achdou et al. (2022) because it makes the two equilibrium objects explicit: the household drift in asset space and the stationary density that drift induces.

## Equations

There are two income states, $i \in \{L,H\}$, with income $z_i$ and Poisson
switching intensity $\lambda_i$ into the other state $j$. Assets are denoted by
$a$, the bond return by $r$, and the stationary density by $g_i(a)$.

Household assets evolve according to
$$\dot a = s_i(a) = z_i + r a - c_i(a), \qquad a \geq \underline a.$$

The HJB equation is
$$\rho V_i(a) =
\max_{c > 0}
\Bigl[
\frac{c^{1-\sigma}}{1-\sigma}
+ V_i'(a)\,[z_i + r a - c]
+ \lambda_i [V_j(a)-V_i(a)]
\Bigr].$$

The consumption rule comes from the first-order condition
$$c_i(a)=\left[V_i'(a)\right]^{-1/\sigma},$$
so the asset drift is $s_i(a)=z_i+ra-c_i(a)$. At the borrowing limit the state
constraint requires $s_i(\underline a)\geq 0$.

The stationary distribution solves the KFE
$$0 =
-\frac{\partial}{\partial a}\left[s_i(a)g_i(a)\right]
-\lambda_i g_i(a)+\lambda_j g_j(a),$$
with total mass normalized to one. The zero-net-supply bond market clears when
$$S(r)=\int_{\underline a}^{\bar a} a\,[g_L(a)+g_H(a)]\,da=0.$$

## Model Setup

The calibration is deliberately small: two income states, symmetric switching, and a one-dimensional asset grid. This keeps the focus on how incomplete insurance pins down the risk-free rate.

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\rho$   | 0.05 | Rate of time preference |
| $\sigma$ | 2.0 | Relative risk aversion |
| $z$       | [0.1, 0.2] | Income states |
| $\lambda$ | [1.2, 1.2] | Income-state switching rates |
| $\underline{a}$ | -0.15 | Borrowing constraint |
| $\bar{a}$       | 5.0 | Upper bound on assets |
| Asset grid points | 500 | Uniform spacing |

## Solution Method

For a candidate interest rate, the HJB is solved by an implicit finite-difference
step. The only delicate choice is the derivative of $V_i(a)$. If assets drift to
the right, the update uses the forward derivative; if assets drift to the left,
it uses the backward derivative. This upwind choice is what keeps the borrowing
constraint from being crossed numerically.

The implicit update can be written as
$$\left[(\Delta^{-1}+\rho)I-A^n\right]V^{n+1}
=u(c^n)+\Delta^{-1}V^n,$$
where $A^n$ is the Markov generator over asset and income states implied by the
current drift. After the HJB step converges, the same generator is used in the
KFE, $A^\top g=0$, with $\int g\,da=1$.

```text
Inputs: candidate r, asset grid a, income states z, switching intensities lambda
Output: policies c_i(a), s_i(a), density g_i(a), aggregate assets S(r)

1. Guess V_i^0(a).
2. Until the HJB sup norm is small:
   a. Compute forward and backward derivatives of V_i^n(a).
   b. Convert each derivative into candidate consumption using u'(c)=V_i'(a).
   c. Select the derivative whose implied asset drift points into the grid.
   d. Build the generator A^n for asset drift plus income switching.
   e. Solve the sparse implicit system for V_i^{n+1}(a).
3. Solve A(r)' g = 0 and normalize the density to integrate to one.
4. Compute S(r)=integral a[g_L(a)+g_H(a)] da.
5. Use bisection on r until S(r) is close to zero.
```

There is no closed-form equilibrium to overlay as ground truth in this
truncated-grid Huggett problem. The relevant numerical checks are therefore the
HJB fixed-point tolerance and the zero-net-supply bond-market residual.

The final HJB step converged in **8 iterations** (sup-norm change 1.56e-07). Bisection gives **$r^{*}=0.03192$**, with a market-clearing residual of **2.86e-06** in aggregate assets.

## Results

The value functions put the income state and the borrowing limit in one picture. High income raises continuation value at every asset level. Near $\underline a$, both curves become steep because one more unit of wealth relaxes the state constraint and buys insurance against staying in the low-income state.

<img src="figures/value-function.png" alt="Value functions by asset holdings and income state at the equilibrium interest rate" width="80%">

The savings policies show the direction of movement in asset space. Low-income households run assets down unless they are already wealthy; high-income households rebuild buffer wealth. The zero crossings are local asset targets for a fixed income state, but income switching keeps households moving across the two drift fields.

<img src="figures/savings-policy.png" alt="Asset drift policies by income state at the equilibrium interest rate" width="80%">

The KFE turns those drift policies into a cross-sectional distribution. Density is largest near the borrowing limit because low-income households drift toward it and cannot continue borrowing once they arrive. The right tail is thin but nonzero, coming from households that have spent enough time in the high-income state to accumulate assets.

<img src="figures/wealth-distribution.png" alt="Stationary asset densities by income state" width="80%">

The market-clearing plot is the equilibrium argument. Higher $r$ makes saving more attractive, so aggregate asset demand rises. Because the bond is in zero net supply, equilibrium is the point where the curve crosses $S(r)=0$. That crossing is below $\rho$: the interest rate has to fall enough to offset precautionary asset demand.

<img src="figures/bond-market.png" alt="Aggregate asset demand as a function of the interest rate" width="80%">

The table summarizes the equilibrium, not a separate calibration target. Mean assets are numerically zero because the interest rate has been chosen to clear the zero-net-supply bond market. The mass near $\underline a$ is the visible trace of incomplete insurance in this two-state economy.

**Equilibrium Values**

| Variable                       |    Value |
|:-------------------------------|---------:|
| Equilibrium interest rate r*   | 0.03192  |
| Mean wealth E[a]               | 0        |
| Mean income E[z]               | 0.15     |
| Mean consumption E[c]          | 0.15     |
| Mass near borrowing limit      | 0.1447   |
| Prob(z = z_low)                | 0.5      |
| Prob(z = z_high)               | 0.5      |
| Bond-market residual abs(E[a]) | 2.86e-06 |
| HJB iterations                 | 8        |
| HJB final sup-norm change      | 1.56e-07 |

## Takeaway

The main lesson is the Huggett pricing mechanism. With incomplete insurance, households want to self-insure by holding the risk-free bond; with zero net supply, the equilibrium return must fall until aggregate desired assets are zero. The HJB/KFE machinery is useful because it keeps the individual saving drift and the induced wealth distribution in the same equilibrium calculation. This is the continuous-time counterpart to the discrete-time incomplete-markets logic in the [Aiyagari saving tutorial](../../dynamic-programming/aiyagari/), but here the asset in fixed supply is a bond rather than aggregate capital.

## References

- Huggett, M. (1993). "The risk-free rate in heterogeneous-agent incomplete-insurance economies." *Journal of Economic Dynamics and Control* 17(5-6), 953-969.
- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89(1), 45-86.
- Moll, B. "Lecture notes on continuous-time heterogeneous-agent models." https://benjaminmoll.com/lectures/
