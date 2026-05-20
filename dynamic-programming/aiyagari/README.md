# Aiyagari Saving and Capital-Market Clearing

## Overview

In Aiyagari (1994), households face persistent idiosyncratic income risk and cannot borrow. They save in one risk-free asset to smooth consumption across income states.

The object is a stationary general equilibrium. Households choose an asset policy, firms demand capital, and the interest rate makes aggregate assets equal capital demand.

The computation nests two fixed points. Value function iteration gives household saving at a candidate rate. Bisection updates the rate until the capital market clears.

## Equations

**Household.** Let $a_t\in[\underline a,\bar a]$ be beginning-of-period assets and
$z_t$ idiosyncratic labor efficiency. With prices $(r,w)$ and a no-borrowing
constraint $\underline a=0$, the household chooses next-period assets
$a_{t+1}=a'$ and consumes

$$c_t = (1+r)a_t + w z_t - a_{t+1},\qquad c_t>0.$$

Preferences are time-separable CRRA,

$$U_0 = \mathbb{E}_0\sum_{t=0}^{\infty}\beta^t u(c_t),\qquad
u(c)=\frac{c^{1-\sigma}}{1-\sigma},$$

with $\beta\in(0,1)$ and $\sigma>0$.

Log productivity is a Gaussian AR(1),

$$\log z_{t+1} = \rho\log z_t + \varepsilon_{t+1},\qquad
\varepsilon_{t+1}\sim\mathcal{N}(0,\sigma_\varepsilon^2),$$

approximated by an $N$-state Rouwenhorst chain. The grid is $\{z_j\}$, and
$P_{jk}=\Pr(z_{t+1}=z_k\mid z_t=z_j)$. The chain matches the AR(1) variance
and persistence, and is normalized so $\mathbb{E}[z]=1$.

The Bellman equation is

$$
V(a,z_j) = \max_{a'\in[\underline a,(1+r)a+wz_j)}
[u((1+r)a+wz_j-a') + \beta\sum_k P_{jk} V(a',z_k)],
$$

The solution gives asset policy $g_a(a,z_j)$. It also gives consumption policy
$c^{\ast}(a,z_j) = (1+r)a + w z_j - g_a(a,z_j)$. The borrowing constraint binds
whenever $g_a(a,z_j)=\underline a$.

**Stationary cross-section.** Given $g_a$ and $P$, the long-run distribution
$\mu$ over $(a,z)$ satisfies the operator equation

$$
\mu(a',z_k) = \sum_j P_{jk}\sum_{i:\,g_a(a_i,z_j)=a'} \mu(a_i,z_j),
$$

Here $i$ indexes nodes of the asset grid $\{a_i\}$. Aggregate household assets are
$K^s(r) = \sum_{i,j} a_i\,\mu(a_i,z_j)$.

**Firm.** Cobb-Douglas technology $Y = K^{\alpha} L^{1-\alpha}$ with capital
share $\alpha$ and depreciation $\delta$ delivers competitive factor prices

$$r(K) = \alpha(\tfrac{K}{L})^{\alpha-1}-\delta,\qquad
w(K) = (1-\alpha)(\tfrac{K}{L})^{\alpha}.$$

With aggregate efficient labor normalized to $L=1$, capital demand at $r$ is

$$K^d(r) = (\tfrac{r+\delta}{\alpha})^{1/(\alpha-1)}.$$

**Stationary equilibrium.** A stationary equilibrium contains price $r^{\ast}$,
wage $w^{\ast}$, policy $g_a$, and distribution $\mu$. The household problem is
solved at $(r^{\ast},w^{\ast})$. The distribution is invariant under
$(g_a,P)$. The capital market clears, $K^s(r^{\ast}) = K^d(r^{\ast})$, with
household capital supply

$$K^s(r^{\ast}) = \sum_{i,j} a_i\,\mu(a_i,z_j).$$

Bisection stops once the relative gap falls below tolerance, so the run delivers
$K^s(r^{\ast}) \approx K^d(r^{\ast})$ rather than exact equality. The diagnostics
table reports both sides; the small residual between them is the tolerance gap,
not a model object.

A standard result is $r^{\ast}<1/\beta-1$. At or above that rate,
precautionary saving becomes unbounded. The economy has excess capital supply.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| Discount factor $\beta$ | 0.96 | Annual time preference |
| Impatience benchmark $1/\beta-1$ | 0.0417 | Complete-markets ceiling on $r^{\ast}$ |
| CRRA $\sigma$ | 2.0 | Curvature; controls precautionary motive |
| Capital share $\alpha$ | 0.36 | Cobb-Douglas exponent on $K$ |
| Depreciation $\delta$ | 0.08 | Pinning $K^d(r)$ |
| Income persistence $\rho$ | 0.9 | AR(1) coefficient on $\log z$ |
| Innovation s.d. $\sigma_\varepsilon$ | 0.2 | AR(1) shock scale |
| Income states $N$ | 7 | Rouwenhorst nodes for $\{z_j\}$ |
| Asset bracket | $[0,50]$ | $\underline a$ at no-borrowing limit |
| Asset grid (coarse) | 200 pts | Exponential, denser at $\underline a$ |
| Capital-market tolerance | 5e-04 | Stop when $\lvert K^s-K^d\rvert/K^d$ falls below |
| Bracket-width tolerance | 1e-06 | Backup stop on $r_H-r_L$ |
| VFI tolerance | 1e-07 | Sup-norm on $V$ |

## Solution Method

At a candidate $r$, firm first-order conditions imply $K^d(r)$ and $w(r)$. The household Bellman problem returns an asset policy. Forward iteration under that policy gives a stationary distribution. Aggregating assets gives household capital supply $K^s(r)$.

Bisection compares $K^s(r)$ with $K^d(r)$. If households save too much, the rate is too high. If they save too little, the rate is too low. The run stops after **12** bisection steps, with relative gap **4.94e-04**.

```text
Algorithm: stationary Aiyagari equilibrium
Inputs: primitives, asset grid, income chain, bracket [r_L, r_H]
Output: r*, w*, K*, household policy, stationary distribution

while not converged:
    r = (r_L + r_H) / 2
    compute K^d(r) and w(r) from firm first-order conditions
    solve the household Bellman problem at (r, w)
    iterate the joint distribution over (a, z) to stationarity
    compute K^s(r) from the stationary distribution
    if K^s(r) > K^d(r): set r_H = r
    else: set r_L = r
```

The final household VFI takes **188** iterations. The sup-norm residual is below 1e-07.

## Results

The firm schedule is analytic and slopes down. The household schedule solves the Bellman problem at each rate. The crossing is the stationary equilibrium. At the calibration $r^{\ast}=0.0260$, roughly 38% below the complete-markets benchmark.

<img src="figures/capital-market.png" alt="Capital demand and household supply schedules with the stationary equilibrium" width="80%">

Value functions rise with assets and income. Asset policies show stronger saving after good income states. Near zero assets, the borrowing limit creates a visible kink in the policy. Each income line eventually crosses below the 45-degree line. That crossing gives a buffer-stock target at the equilibrium rate.

<img src="figures/savings-policy.png" alt="Equilibrium value function and asset policy" width="80%">

The stationary distribution comes from the asset policy and income chain at $r^{\ast}=0.0260$. Mean wealth ($\mathbb{E}[a]=6.76$) exceeds median wealth ($\tilde a=4.47$). $2.5\%$ of households sit at the borrowing limit. The right tail comes from repeated high-income draws. With no ex-ante heterogeneity, the run produces Gini $G=0.526$.

<img src="figures/wealth-distribution.png" alt="Stationary wealth distribution and Lorenz curve at the equilibrium prices" width="80%">

The market-clearing gap is the bisection residual. It is numerical error, not a model object.

**Stationary equilibrium diagnostics**

| Variable                     |       Value |
|:-----------------------------|------------:|
| Interest rate $r^{\ast}$     |   0.025959  |
| Wage $w^{\ast}$              |   1.2734    |
| Aggregate capital $K^{\ast}$ |   6.7599    |
| Output $Y^{\ast}$            |   1.9897    |
| Capital-output ratio $K/Y$   |   3.3975    |
| Mean wealth $\mathbb{E}[a]$  |   6.7633    |
| Median wealth $\tilde a$     |   4.4728    |
| P90 wealth                   |  16.3145    |
| Gini                         |   0.5261    |
| Mass at constraint           |   0.0245    |
| Relative market-clearing gap |   0.0004939 |
| Bisection steps              |  12         |
| VFI iterations               | 188         |

## Takeaway

Precautionary saving turns a household policy into an aggregate capital supply curve. In this calibration, capital-market clearing gives $r^{\ast}=0.0260$, below $1/\beta-1=0.0417$. The lower rate is the price of incomplete insurance. The computation shows how VFI, a stationary distribution, and bisection close the model.

## References

- Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *Quarterly Journal of Economics*, 109(3), 659-684.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 18.
- **See also.** The continuous-time analogue with HJB and KFE solvers is in [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/), and the steady state here is reused as the household block of the HANK model in [`heterogeneous-agents/sequence-space-jacobian-hank/`](../../heterogeneous-agents/sequence-space-jacobian-hank/), where sequence-space Jacobians give the impulse responses to MIT shocks.
