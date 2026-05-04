# McCall Job Search and the Reservation Wage

> Sequential wage-offer search, option value, and threshold acceptance.

## Overview

The McCall model is a small partial-equilibrium model of unemployment with a sharp economic object: the value of waiting for a better wage offer. An unemployed worker observes one offer at a time. Accepting locks in that wage forever; rejecting pays the unemployment benefit $b$ today and preserves the right to draw again tomorrow.

The optimal policy is a reservation rule. The worker accepts offers at or above a cutoff $w^{*}$ and rejects offers below it. The cutoff is not a taste parameter; it is an endogenous price of search, pinned down by the wage-offer distribution, the benefit level, and the discount factor. This is the worker-side building block behind richer search models such as [search and matching unemployment](../diamond-mortensen-pissarides/). It also echoes the logic in [income-risk saving](../consumption-savings/): a current choice is valuable because it changes exposure to future states.

## Equations

Let $W$ denote a wage offer drawn from distribution $F$, and let $w$ be the
current realization. The worker discounts next period by $\beta \in (0,1)$.
Accepting $w$ gives the permanent value

$$A(w)=\frac{w}{1-\beta}.$$

Rejecting gives the common continuation value

$$C=b+\beta \mathbb{E}_{F}[V(W')],$$

so the Bellman equation is

$$V(w)=\max\bigl[ \frac{w}{1-\beta},\; b+\beta \mathbb{E}_{F}[V(W')] \bigr].$$

The reservation wage is the offer that makes the worker indifferent:

$$\frac{w^{*}}{1-\beta}=C.$$

Using this indifference condition inside the Bellman equation gives the scalar
fixed point

$$w^{*}=(1-\beta)b+\beta \mathbb{E}_{F}[\max\{W',w^{*}\}].$$

This last equation is useful for interpretation. A higher $b$ raises the value
of rejection directly. A higher $\beta$ raises the option value of future draws.
A thicker right tail also raises $w^{*}$ because rejecting a mediocre offer buys
exposure to rare high wages.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| Discount factor $\beta$ | 0.95 | Weight on future search opportunities |
| Flow benefit $b$ | 1.0 | Payoff while unemployed |
| Wage law | $\log W \sim N(0.0,1.0^2)$ | Offer distribution |
| Mean offer $\mathbb{E}[W]$ | 1.6487 | Reference level, not an upper bound on $w^{*}$ |
| Main wage grid | 50 bins | Equiprobable bins represented by conditional means |
| Continuous benchmark | lognormal tail moments | Held-out reservation-wage check |
| VFI tolerance | 1e-08 | Sup-norm stopping rule |

## Solution Method

On a finite offer grid, the Bellman iteration is especially simple because the value of rejecting does not depend on the current offer. Each iteration computes one expected continuation value and then compares it with the lifetime value of accepting each grid wage.

```text
Algorithm: finite-grid McCall VFI
Input: wages w_i, probabilities p_i, beta, benefit b, tolerance epsilon
Output: value function V_i and reservation wage w*
Initialize V_i = w_i / (1 - beta)
repeat for n = 0, 1, 2, ...:
    C_n = b + beta * sum_i p_i V_i
    V_i_new = max{w_i / (1 - beta), C_n} for every wage i
    error = max_i |V_i_new - V_i|
    set V_i = V_i_new
until error < epsilon
set w* = (1 - beta) * (b + beta * sum_i p_i V_i)
```

The continuous benchmark uses the scalar reservation-wage equation rather than the value function. For a candidate cutoff $r$, compute $m(r)=\mathbb{E}[\max\{W,r\}]$ under the lognormal distribution and find the root of $r-(1-\beta)b-\beta m(r)=0$ by bracketing.

```text
Algorithm: continuous reservation-wage benchmark
Input: beta, b, lognormal parameters mu and sigma, tolerance epsilon
Output: continuous-distribution cutoff r
Define m(r) = E_F[max{W, r}] using lognormal tail moments
Find a bracket [low, high] with residual(low) < 0 < residual(high)
Solve residual(r) = r - (1 - beta)*b - beta*m(r) = 0
```

The finite-grid VFI converged in **178 iterations** with sup-norm error **9.84e-09**. The baseline cutoff is $w^{*}_{grid}=4.7054$, compared with the continuous lognormal benchmark $w^{*}_{cont}=4.7055$.

## Results

The first figure shows the reservation rule in value units. Accepting is linear in the current offer, while rejecting is flat because it depends only on the distribution of future offers. In the baseline calibration the finite grid accepts about **6.0%** of grid offers. The continuous benchmark accepts **6.1%** of offers.

<img src="figures/accept-vs-reject.png" alt="Accept and reject values with finite-grid and continuous reservation wages." width="80%">

Patience makes the worker more selective because the future draw arrives almost as valuable as the current payoff. The cutoff can rise above the mean offer in a right-skewed distribution; the mean is a reference point, not a bound on optimal selectivity.

<img src="figures/wstar-vs-beta.png" alt="Reservation wage by discount factor with continuous benchmark." width="80%">

Benefits move the outside option one for one only in the limiting case where search has no upside. With a non-degenerate wage offer distribution, a higher benefit also preserves the option to wait, so the cutoff remains above the 45-degree line over this range.

<img src="figures/wstar-vs-benefits.png" alt="Reservation wage by unemployment benefit with continuous benchmark." width="80%">

The parameter grid separates two margins. Increasing $b$ improves the payoff from unemployment today. Increasing $\beta$ raises the present value of future draws. Both margins reduce the probability that a fresh offer is accepted.

**Reservation wages and continuous-benchmark acceptance rates**

|   beta |   b |   w* grid |   w* cont. |   grid gap |   Accept % (cont.) |   VFI iter. |
|-------:|----:|----------:|-----------:|-----------:|-------------------:|------------:|
|   0.9  | 0.5 |    3.3118 |     3.3126 |    -0.0007 |               11.6 |          86 |
|   0.9  | 1   |    3.5654 |     3.5656 |    -0.0002 |               10.2 |          95 |
|   0.9  | 2   |    4.1194 |     4.1196 |    -0.0002 |                7.8 |         106 |
|   0.95 | 0.5 |    4.4718 |     4.4794 |    -0.0076 |                6.7 |         176 |
|   0.95 | 1   |    4.7054 |     4.7055 |    -0.0001 |                6.1 |         178 |
|   0.95 | 2   |    5.1727 |     5.1946 |    -0.0219 |                5   |         181 |
|   0.99 | 0.5 |    8.1646 |     8.1789 |    -0.0143 |                1.8 |         694 |
|   0.99 | 1   |    8.3324 |     8.3631 |    -0.0308 |                1.7 |         696 |
|   0.99 | 2   |    8.6679 |     8.7514 |    -0.0834 |                1.5 |         699 |

## Takeaway

The McCall model is useful because it turns unemployment duration into a reservation-price problem. The worker rejects low offers not because the current benefit is large by itself, but because rejecting preserves a valuable claim on future wage draws. In this calibration the right tail is important enough that the cutoff can sit above the mean offer, a point the finite-grid and continuous benchmarks make explicit. The same acceptance-threshold logic is the partial-equilibrium core of larger search models with matching frictions, endogenous vacancies, and equilibrium wage determination.

## References

- McCall, J.J. (1970). "Economics of Information and Job Search." *Quarterly Journal of Economics*, 84(1), 113-126.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 6.
- Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press.
- Pissarides, C.A. (2000). *Equilibrium Unemployment Theory*. MIT Press.
