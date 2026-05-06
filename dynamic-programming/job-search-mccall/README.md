# McCall Job Search and the Reservation Wage

> Sequential wage-offer search as an optimal-stopping problem with a scalar continuation value.

## Overview

An unemployed worker draws one wage offer per period from a known distribution. Accepting an offer locks in that wage forever; rejecting pays the unemployment benefit $b$ and rolls the dice again next period. The question is when to stop searching.

Two features make this the cleanest optimal-stopping problem in the catalog. First, the only state is the current offer $w$, and once $w$ is rejected it is forgotten, so the value of rejection does not depend on $w$ at all. The Bellman equation reduces to a comparison between a linear function of $w$ (acceptance) and a scalar (rejection), which forces the policy to be a cutoff $w^{\ast}$. Second, that cutoff is characterized by a one-dimensional fixed point that can be solved without iterating on the full value function, giving a closed enough benchmark to audit any discretization.

Economically, $w^{\ast}$ is the price of search: the wage at which the marginal continuation gain from waiting equals the foregone earnings from rejecting today. Patience, the benefit level, and the right tail of the offer distribution all push it up; impatience and bad benefits pull it down.

This is the worker-side primitive behind frictional unemployment. The same threshold logic, with vacancies and a matching function added, drives [Diamond-Mortensen-Pissarides search and matching](../diamond-mortensen-pissarides/). On the recursive-methods side, [cake eating](../cake-eating/) shares the scalar Bellman structure without choice under uncertainty, while [income risk and buffer-stock saving](../consumption-savings/) keeps the continuation expectation but adds a continuous endogenous state. The wage discretization here uses the same conditional-mean trick that [shock discretization](../shock-discretization/) applies to AR(1) processes.

## Equations

Let $W$ be a wage offer with distribution $F$, and let $w$ denote the current
realization. The worker discounts at $\beta\in(0,1)$. Accepting locks in a
permanent income stream worth

$$A(w)=\frac{w}{1-\beta}.$$

Rejecting yields the unemployment benefit $b$ today plus the continuation
value of being unemployed tomorrow. Because today's offer is discarded on
rejection, that continuation does not depend on $w$:

$$C=b+\beta\,\mathbb{E}_{F}[V(W')].$$

The Bellman equation is

$$V(w)=\max\bigl\{\,\frac{w}{1-\beta},\; C\,\bigr\}.$$

Since $A(w)$ is strictly increasing in $w$ and $C$ is constant, the optimal
policy is the threshold $w^{\ast}$ defined by indifference,
$A(w^{\ast})=C$, i.e.

$$\frac{w^{\ast}}{1-\beta}=b+\beta\,\mathbb{E}_{F}[V(W')].$$

Plugging $V(W')=\max\{W'/(1-\beta),\,C\}$ back in and using
$C=w^{\ast}/(1-\beta)$ gives a scalar fixed point in $w^{\ast}$ alone:

$$w^{\ast}=(1-\beta)\,b+\beta\,\mathbb{E}_{F}\!\left[\max\{W',\,w^{\ast}\}\right].$$

Three margins read off this equation directly. A higher $b$ raises the floor
on the right-hand side. A higher $\beta$ scales up the continuation term and
makes the worker more selective. And a thicker right tail of $F$ lifts
$\mathbb{E}_{F}[\max\{W',w^{\ast}\}]$ above $w^{\ast}$ even when most of the
mass sits below it, which is why the cutoff can settle far above the mean
offer in fat-tailed calibrations.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| Discount factor $\beta$ | 0.95 | Weight on the next draw |
| Flow benefit $b$ | 1.0 | Per-period payoff while unemployed |
| Wage law | $\log W\sim N(0.0,1.0^2)$ | Lognormal offer distribution |
| Median offer | 1.0000 | $e^{\mu}$ for the lognormal |
| Mean offer $\mathbb{E}[W]$ | 1.6487 | Reference level, not a bound on $w^{\ast}$ |
| Wage grid | 50 equiprobable bins | Each bin represented by its conditional mean |
| Continuous benchmark | exact lognormal moments | Ground-truth cutoff via scalar fixed point |
| VFI tolerance | 1e-08 | Sup-norm stopping rule |

## Solution Method

**Why VFI is essentially scalar here.** The Bellman operator $T$ acting on a candidate $V$ is

$$(TV)(w)=\max\bigl\{\,\frac{w}{1-\beta},\,b+\beta\,\mathbb{E}_{F}[V(W')]\,\bigr\}.$$

It is a $\beta$-contraction in the sup norm, so iterates converge to the unique fixed point. The novelty is that the continuation term is a single number $C=b+\beta\,\mathbb{E}_{F}[V]$, recomputed once per sweep. Each iteration is therefore one inner product and one elementwise max, no interpolation and no per-state expectation.

**Discretization.** The continuous lognormal is replaced by an $n_w=50$-bin discrete law with equal probabilities $1/n_w$ and support points equal to the conditional mean of each bin. This preserves $\mathbb{E}[W]$ exactly and keeps the tail moments the reservation wage actually depends on. Quantile midpoints would compress the right tail and pull $w^{\ast}$ downward.

```text
Algorithm  Finite-grid McCall VFI
Inputs   wages w_1,...,w_n; probabilities p_1,...,p_n;
           discount beta in (0,1); benefit b; tolerance epsilon
Outputs  value V_i and reservation wage w*

Initialise V_i <- w_i / (1 - beta)             # accept-everything guess
repeat n = 0, 1, 2, ...:
    C  <- b + beta * sum_i p_i V_i             # one expectation per sweep
    V_i_new <- max{ w_i / (1 - beta), C }      # elementwise threshold update
    err <- max_i | V_i_new - V_i |
    V_i <- V_i_new
stop when err < epsilon
w* <- (1 - beta) * (b + beta * sum_i p_i V_i)  # invert C = w* / (1 - beta)
```

**Continuous benchmark.** With the lognormal offer law the scalar fixed-point equation

$$r = (1-\beta)\,b+\beta\,m(r),\qquad m(r)=\mathbb{E}_{F}[\max\{W,r\}],$$

has a closed-form $m(r)$ in terms of the standard-normal CDF. Bracketing and Brent's method give $r$ to machine precision and provide ground truth against the grid solution.

```text
Algorithm  Continuous reservation-wage benchmark
Inputs   beta in (0,1); benefit b; lognormal parameters mu, sigma;
           tolerance epsilon
Output   reservation wage r

Define m(r) = r * F(r) + e^{mu + sigma^2/2} * (1 - Phi((log r - mu - sigma^2)/sigma))
Define residual(r) = r - (1 - beta)*b - beta * m(r)
Find a bracket [lo, hi] with residual(lo) < 0 < residual(hi)
Solve residual(r) = 0 by Brent's method to tolerance epsilon
```

At the baseline calibration the finite-grid VFI converges in **178 iterations** to sup-norm error **9.84e-09**, giving $w^{\ast}_{\text{grid}}=4.7054$. The continuous benchmark returns $w^{\ast}_{\text{cont}}=4.7055$, so the discretization error is **9.1e-05** in absolute terms. The two curves overlay almost everywhere in the comparative statics below; the table at the end shows where the gap is large enough to read off.

## Results

The reservation rule reads directly off the value functions. The rising line is the permanent-income value of accepting the current offer, $w/(1-\beta)$. The flat dashed line is the rejection value $C$, which by construction does not depend on $w$. They cross at the cutoff. The shaded region marks acceptable offers, and the two vertical lines show how close the 50-bin grid gets to the continuous-distribution benchmark. In this calibration the worker accepts about **6.0%** of grid offers and **6.1%** of continuous offers, so expected unemployment duration is roughly **16 periods** before an acceptable draw arrives.

<img src="figures/accept-vs-reject.png" alt="Accept and reject values with finite-grid and continuous reservation wages." width="80%">

Where the cutoff sits relative to the offer distribution makes the fat-tail intuition concrete. The mean and median of $W$ are below $w^{\ast}$, yet rejecting almost-mean offers is optimal because the right tail of $f(w)$ — visible above $w^{\ast}$ — is thick enough that one good draw eventually compensates for many rejected ones. Acceptance is a tail event by design.

<img src="figures/cutoff-on-density.png" alt="Lognormal offer density with the reservation wage and acceptance region." width="80%">

Patience compounds the option value of waiting. As $\beta\to 1$ the worker treats the next draw as nearly equivalent to today's, so the cutoff explodes upward and pushes deep into the right tail. The horizontal benefit line and the mean offer are reference points only: neither bounds $w^{\ast}$ from above when the right tail is thick enough. The grid solution tracks the closed-form benchmark everywhere on this range; the small gap at high $\beta$ is where discretization bites the most because the cutoff probes parts of the tail that the 50-bin grid only crudely resolves.

<img src="figures/wstar-vs-beta.png" alt="Reservation wage by discount factor with continuous benchmark." width="80%">

Generosity of benefits raises the cutoff, but never one for one. If search had no upside, $w^{\ast}$ would track the $45^{\circ}$ line: the worker would accept any offer above the outside option. With a non-degenerate offer distribution the slope is strictly less than one because each extra dollar of $b$ also raises the value of drawing again, partially offsetting the change in the floor. The vertical gap to the $45^{\circ}$ line is the option value of search at that benefit level.

<img src="figures/wstar-vs-benefits.png" alt="Reservation wage by unemployment benefit with continuous benchmark." width="80%">

Reading the table separates the two margins. Increasing $b$ shifts the floor on the rejection value upward; increasing $\beta$ scales up the option value of the next draw. Both push $w^{\ast}$ up, drag acceptance rates down, and lengthen expected unemployment duration $1/\Pr[W\geq w^{\ast}]$. The discretization error is modest at moderate $\beta$ and grows visibly at $\beta=0.99$, where the cutoff probes parts of the right tail that the 50-bin grid resolves poorly. Refining the grid, or using the scalar fixed-point solver directly, closes the gap at negligible cost.

**Reservation wages, acceptance rates, and expected unemployment duration**

|   beta |   b |   w* grid |   w* cont. |   grid gap |   Accept % (cont.) |   E[duration] |   VFI iter. |
|-------:|----:|----------:|-----------:|-----------:|-------------------:|--------------:|------------:|
|   0.9  | 0.5 |    3.3118 |     3.3126 |    -0.0007 |               11.6 |           8.7 |          86 |
|   0.9  | 1   |    3.5654 |     3.5656 |    -0.0002 |               10.2 |           9.8 |          95 |
|   0.9  | 2   |    4.1194 |     4.1196 |    -0.0002 |                7.8 |          12.8 |         106 |
|   0.95 | 0.5 |    4.4718 |     4.4794 |    -0.0076 |                6.7 |          15   |         176 |
|   0.95 | 1   |    4.7054 |     4.7055 |    -0.0001 |                6.1 |          16.5 |         178 |
|   0.95 | 2   |    5.1727 |     5.1946 |    -0.0219 |                5   |          20.1 |         181 |
|   0.99 | 0.5 |    8.1646 |     8.1789 |    -0.0143 |                1.8 |          56.2 |         694 |
|   0.99 | 1   |    8.3324 |     8.3631 |    -0.0308 |                1.7 |          59.4 |         696 |
|   0.99 | 2   |    8.6679 |     8.7514 |    -0.0834 |                1.5 |          66.5 |         699 |

## Takeaway

Sequential search collapses unemployment duration into a single endogenous price, the reservation wage. Two margins move it: a higher outside option $b$ and more patience $\beta$, both of which raise $w^{\ast}$ and lengthen expected duration. The cutoff sits well above the mean offer in this calibration because the lognormal right tail is thick enough that rejecting a near-mean draw buys real exposure to high wages. Two computational points generalize beyond this model. First, when the rejection value is offer-independent the Bellman operator is essentially scalar, so VFI here is a useful warm-up rather than a performance bottleneck. Second, the same threshold logic is the partial-equilibrium core of the [Diamond-Mortensen-Pissarides](../diamond-mortensen-pissarides/) matching framework, where free entry and a matching function pin down vacancies and wages on top of exactly this worker problem.

## References

- McCall, J.J. (1970). "Economics of Information and Job Search." *Quarterly Journal of Economics*, 84(1), 113-126.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 6.
- Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press.
- Pissarides, C.A. (2000). *Equilibrium Unemployment Theory*. MIT Press, 2nd edition.
