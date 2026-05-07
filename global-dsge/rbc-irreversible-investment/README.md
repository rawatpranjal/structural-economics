# Capital Overhang from Irreversible Investment in RBC

> Installed capital is easy to add and hard to undo, so recessions can leave the economy stuck with too much capital.

## Overview

Think of a firm that installed machines before learning that productivity would be low for several years. It can stop adding machines, and the old capital can depreciate, but it cannot turn installed capital back into consumption goods. The RBC version of that situation is simple: the household chooses next period's capital after seeing productivity, while net investment cannot be negative.

That friction matters only in particular states. Near the deterministic steady state, replacement investment is positive and the constraint is slack. After a bad productivity draw with high installed capital, the standard model wants to run capital down faster than depreciation. The irreversible model cannot, so it carries a capital overhang until depreciation and future investment decisions bring the state back toward the usual region.

This tutorial sits between the local [linearized RBC](../../dsge/rbc/) shock-propagation example and the global [capital-tax RBC](../rbc-capital-tax/) tutorial. We solve the model globally because the object of interest is a state-space boundary, $I=0$, rather than a smooth response around the steady state. A local linear solution cannot show where that boundary binds or how much value is lost there.

## Equations

Let $K_t$ be beginning-of-period capital, $z_t$ productivity, $c_t$ consumption,
and $K_{t+1}$ next-period capital. Output is $Y_t=z_tK_t^\alpha$ and

$$\log z_{t+1}=\rho \log z_t+\varepsilon_{t+1},
\qquad \varepsilon_{t+1}\sim N(0,\sigma_\varepsilon^2).$$

The Bellman equation is

$$V(K,z)=\max_{K'\in \Gamma(K,z)}\Bigg[
\frac{\left[zK^\alpha+(1-\delta)K-K'\right]^{1-\sigma}}{1-\sigma}
+\beta \sum_{z'} P(z,z')V(K',z')\Bigg].$$

The standard RBC choice set is

$$\Gamma^{std}(K,z)=\{K'\geq 0:
zK^\alpha+(1-\delta)K-K'>0\}.$$

Irreversibility adds

$$I_t\equiv K_{t+1}-(1-\delta)K_t\geq 0,
\qquad
\Gamma^{irr}(K,z)=\{K'\geq (1-\delta)K:
zK^\alpha+(1-\delta)K-K'>0\}.$$

With multiplier $\lambda_t\geq 0$ on $K_{t+1}-(1-\delta)K_t\geq 0$,

$$u_c(c_t)-\lambda_t
=\beta E_t\left[
u_c(c_{t+1})\left(\alpha z_{t+1}K_{t+1}^{\alpha-1}+1-\delta\right)
-\lambda_{t+1}(1-\delta)
\right],$$

$$\lambda_t\geq 0,\qquad I_t\geq 0,\qquad \lambda_t I_t=0.$$

At the deterministic steady state the constraint is slack because
$I_{ss}=\delta K_{ss}>0$.
With the calibration below, $K_{ss}=37.989$, $Y_{ss}=3.704$, $C_{ss}=2.754$, and $I_{ss}=0.950$.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\beta$ | 0.99 | Discount factor |
| $\alpha$ | 0.36 | Capital share |
| $\sigma$ | 2.0 | CRRA coefficient |
| $\delta$ | 0.025 | Depreciation rate |
| $\rho$ | 0.9 | Persistence of log productivity |
| $\sigma_\varepsilon$ | 0.05 | Innovation std for log productivity |
| Capital grid | 72 points on [20.89, 60.78] | State grid and candidate $K'$ grid |
| TFP grid | 7 Tauchen states | Common shock grid for both models |
| Fine-grid check | 150 capital points | Used only to benchmark the low-productivity investment policy |
| Overhang experiment | $K_0=1.25K_{ss}$ plus a low-$z$ episode | A stress path, not a stationary moment |

## Solution Method

The computation uses global value function iteration. It solves a standard RBC model and an irreversible model on the same capital and productivity grids, then compares their policies at each state. For the irreversible model, the feasible set changes with current capital because the lower bound is $K'=(1-\delta)K$. The grid search adds that exact boundary choice even when it falls between grid nodes, since the economic kink is at zero net investment.

```text
Algorithm: global VFI with an irreversible-investment boundary
Input: grids K and Z, transition matrix P, primitives beta, alpha, sigma, delta
Output: value V(z,K), policies g_K(z,K), g_c(z,K), binding indicator b(z,K)
Precompute resources R(z,K)=z K^alpha+(1-delta)K and utility for all grid choices K'
repeat:
    for each productivity state z_i and capital state K_m:
        set A_std(K_m) to all feasible grid choices K'
        set A_irr(K_m) to choices in A_std with K' >= (1-delta)K_m
        add the exact boundary K'=(1-delta)K_m to A_irr when it is between grid nodes
        choose K' to maximize u(R(z_i,K_m)-K') + beta * sum_j P_ij V_n(z_j,K')
        set b(z_i,K_m)=1 if the boundary K'=(1-delta)K_m is chosen
    apply Howard improvement to the fixed policy
until the sup-norm Bellman update is below epsilon
Solve a finer-grid irreversible model and compare the low-z investment policy
Simulate the standard and irreversible policies on the same productivity paths
```

The irreversible model converged in **49** VFI iterations; the standard comparison converged in **49**. The fine-grid check changes the low-productivity investment policy by at most **0.2941** units of capital on the coarse grid.

## Results

The policy comparison shows where irreversibility binds. At low productivity and high capital, the standard model chooses negative investment and runs capital down. The irreversible policy flattens at $I=0$. The dotted fine-grid line in the investment panel is a local check that the coarse-grid kink is not a plotting artifact.

<img src="figures/policy-functions.png" alt="Investment and consumption policies for standard and irreversible RBC models" width="80%">

The binding set is away from the deterministic steady state. It sits in states with too much installed capital for the current productivity level. A stationary simulation can therefore spend little time constrained, while the constraint still shapes recession states.

<img src="figures/binding-region.png" alt="State-space region where nonnegative investment binds" width="80%">

The stress path starts with capital above steady state and then sends productivity to its lowest grid state. The standard model disinvests immediately. The irreversible model cannot do that, so investment sits at zero and capital remains high until depreciation works through the overhang. The gray band is the adverse productivity episode.

<img src="figures/overhang-experiment.png" alt="Stress-path comparison after a low-productivity episode" width="80%">

The value loss is concentrated near the same high-capital, low-productivity states where the boundary binds. Near the steady state the loss is small because normal replacement investment is positive. The friction mainly prices bad states reached with too much installed capital.

<img src="figures/value-difference.png" alt="Value-function difference between irreversible and standard RBC models" width="80%">

The stationary simulation starts at $K_{ss}$ and uses the same productivity draws for both policies. The binding frequency is modest because the economy does not often enter the high-capital, low-productivity region. That is a property of the calibration, not evidence that the constraint is irrelevant.

**Stationary Simulation Moments**

| Model        |   mean K |   std(Y) % |   std(C)/std(Y) |   mean I/Y | I=0 frequency   |
|:-------------|---------:|-----------:|----------------:|-----------:|:----------------|
| Irreversible |   39.499 |      17.11 |           0.729 |      0.251 | 0.42%           |
| Standard RBC |   39.499 |      17.11 |           0.729 |      0.251 | 0.00%           |

The numerical checks separate three objects: the global state-space binding set, the deliberately adverse overhang path, and the ordinary stationary simulation. Only the first two are designed to stress the kink.

**Solution and Boundary Checks**

| Check                              | Value   | Interpretation                                                   |
|:-----------------------------------|:--------|:-----------------------------------------------------------------|
| Irreversible VFI iterations        | 49      | Coarse-grid constrained solution                                 |
| Standard VFI iterations            | 49      | Unconstrained comparison on the same grid                        |
| Binding states on coarse grid      | 9.5%    | Fraction of (z,K) states where the policy chooses I=0            |
| Overhang episode binding frequency | 13.3%   | Share of periods with I=0 in the displayed recession experiment  |
| Stationary binding frequency       | 0.42%   | Share of simulated periods with I=0 from K_ss                    |
| Fine-grid low-z investment gap     | 0.2941  | Max absolute gap between 72- and 150-point irreversible policies |

The figures give the main economic message. The investment floor does not move the deterministic steady state, because replacement investment is strictly positive there. It changes the state-contingent policy. Once productivity is low enough relative to installed capital, the unconstrained Euler equation asks for disinvestment, but the feasible policy is pinned at $I=0$.

The comparison with the standard RBC model should therefore be read state by state. Around ordinary states, the two policies are close. In overhang states, the irreversible model has a kink, a binding multiplier, and a value loss. A global grid solution preserves those objects directly.

## Takeaway

Irreversibility is not a different steady-state theory of capital accumulation. It is a theory of bad states. When the economy has too much installed capital for current productivity, the standard RBC model adjusts by selling or scrapping capital immediately. The irreversible model adjusts only through depreciation and future low investment. For nearby DSGE applications, the practical lesson is that occasionally binding constraints matter because they create state-dependent kinks, not because they necessarily bind in the average period.

## References

- Abel, A. and Eberly, J. (1996). *Optimal Investment with Costly Reversibility*. Review of Economic Studies.
- Bertola, G. and Caballero, R. (1994). *Irreversibility and Aggregate Investment*. Review of Economic Studies.
- Cao, D., Luo, W., and Nie, G. (2023). *Global DSGE Models*. Review of Economic Dynamics.
- Khan, A. and Thomas, J. (2008). *Idiosyncratic Shocks and the Role of Nonconvexities*. Econometrica.
