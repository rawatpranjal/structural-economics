# Capital Tax Wedges in an RBC Model

> A revenue-neutral capital tax leaves aggregate resources unchanged but lowers the after-tax return that governs saving.

## Overview

Capital income taxation separates aggregate feasibility from private incentives. The government taxes capital income at rate $\tau_k$ and rebates the proceeds lump-sum. The representative economy still has the same resource constraint, but households save against an after-tax marginal product of capital. The distortion is entirely intertemporal: current consumption becomes cheaper relative to future consumption.

This tutorial is a tax-wedge companion to the global [RBC capital and labor](../../dynamic-programming/rbc/) example and the local [linearized RBC](../../dsge/rbc/) impulse-response example. The point is not shock propagation per se, but how a permanent wedge moves the exact deterministic steady state, the nonlinear capital policy, and simulated investment behavior.

## Equations

Let $K_t$ be aggregate capital at the start of period $t$, $z_t$ aggregate
TFP, $c_t$ consumption, and $K_{t+1}$ next-period capital. Preferences are

$$\mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t
\frac{c_t^{1-\sigma}}{1-\sigma}, \qquad \sigma>0,$$

with Cobb-Douglas output $Y_t=z_t K_t^\alpha$. Productivity follows

$$\log z_{t+1}=\rho \log z_t+\varepsilon_{t+1},
\qquad \varepsilon_{t+1}\sim N(0,\sigma_\varepsilon^2).$$

The government rebate means aggregate feasibility is the usual RBC resource
constraint,

$$c_t + K_{t+1} = z_t K_t^\alpha + (1-\delta)K_t.$$

The tax appears in the household Euler equation:

$$c_t^{-\sigma} =
\beta \mathbb{E}_t\left[
c_{t+1}^{-\sigma}
\left((1-\tau_k)\alpha z_{t+1}K_{t+1}^{\alpha-1}+1-\delta\right)
\right].$$

Thus the wedge changes the return to saving but not the goods available to the
economy in a given period.

At $z=1$, the exact deterministic steady state is

$$K_{ss}(\tau_k)=
\left(\frac{(1-\tau_k)\alpha}{1/\beta-1+\delta}\right)^{1/(1-\alpha)},$$

with $Y_{ss}=K_{ss}^{\alpha}$, $C_{ss}=Y_{ss}-\delta K_{ss}$, and
tax revenue $T_{ss}=\tau_k \alpha Y_{ss}$.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\beta$  | 0.99 | Discount factor |
| $\alpha$ | 0.36 | Capital share |
| $\sigma$ | 2.0 | CRRA coefficient |
| $\delta$ | 0.025 | Depreciation rate |
| $\rho$   | 0.95 | TFP persistence |
| $\sigma_\varepsilon$ | 0.01 | TFP innovation std |
| $\tau_k$ | [0.0, 0.1, 0.2, 0.3, 0.4] | Permanent tax rates compared |
| Capital grid | 40 points around each $K_{ss}(\tau_k)$ | State and $K'$ choice grid |
| TFP grid | 5 Tauchen states | Approximation to log productivity |
| Simulation periods | 5000 | Same shock seed for every tax regime, with 500 burn-in periods |

## Solution Method

The computation uses the resource-feasible Bellman problem to get a stable global policy on the $(z,K)$ grid, then refines consumption with the after-tax Euler equation. The first step is a good initializer because the rebate leaves the aggregate resource constraint unchanged. The second step introduces the capital-tax wedge.

```text
Algorithm: global policy iteration with a capital-tax wedge
Input: tax rate tau_k, grids K and Z, transition matrix P, primitives beta, alpha, sigma, delta
Output: value V(z,K), capital policy g_K(z,K), consumption policy g_c(z,K)
Compute the exact deterministic K_ss(tau_k) and build a capital grid around it
Discretize log productivity with Tauchen to obtain Z and P
Precompute feasible consumption c = z K^alpha + (1-delta)K - K' for every (z,K,K')
Initialize V_0(z,K)
repeat:
    for each state (z_i,K_m):
        choose K' on the grid to maximize u(c) + beta * sum_j P_ij V_n(z_j,K')
        record V_{n+1}, g_K, and g_c
    apply Howard improvement to the fixed policy
until the sup-norm value update is below epsilon
repeat Euler refinement:
    for each state (z_i,K_m):
        K_plus = g_K(z_i,K_m)
        M = sum_j P_ij g_c(z_j,K_plus)^(-sigma)
            * ((1-tau_k) alpha z_j K_plus^(alpha-1) + 1-delta)
        g_c_new(z_i,K_m) = (beta * M)^(-1/sigma)
        g_K_new(z_i,K_m) = z_i K_m^alpha + (1-delta)K_m - g_c_new(z_i,K_m)
until the consumption policy update is below epsilon
Simulate all tax regimes on the same productivity path
```

The exact deterministic steady state serves as a ground-truth benchmark for the long-run comparisons. The stochastic policy functions are numerical, and the table below separates the exact steady states from simulated means. Across the five tax regimes, VFI used at most **49** outer iterations and the Euler refinement used at most **223** iterations.

## Results

The exact steady-state formulas already show the size of the distortion. At $\tau_k=30\%$, deterministic capital is 42.7% below the no-tax value, output is 18.2% lower, and consumption is 9.7% lower. Consumption falls less because a lower capital stock also reduces replacement investment. The simulations use the same productivity sequence for every tax rate, so the level differences across paths are the tax wedge, not different shock histories.

The first comparison is analytical rather than simulated. Capital falls with $(1-\tau_k)^{1/(1-\alpha)}$, so the tax rate is amplified by the capital share. Output and consumption move less than capital, but the economy operates from a lower productive base.

<img src="figures/steady-state-tax.png" alt="Exact steady-state levels and losses by capital tax rate" width="80%">

At the median productivity state, the policy functions show the same wedge in decision-rule form. Higher taxes move the capital policy down and the consumption policy up: the household chooses less saving because tomorrow's marginal product is partly taxed away.

<img src="figures/policy-by-tax.png" alt="Capital and consumption policies at median TFP by capital tax rate" width="80%">

The simulated paths keep the productivity sequence fixed across regimes. The higher-tax economies therefore track the same booms and recessions from permanently lower capital and output levels.

<img src="figures/simulation-paths.png" alt="Simulated capital and output paths by capital tax rate" width="80%">

The distributional view is useful because the policy change is not only a new mean. Higher taxes shift the investment share and the capital-output ratio left across the stationary simulation, so the economy spends more time in states with a smaller productive base.

<img src="figures/investment-distributions.png" alt="Investment-rate and capital-output distributions by tax regime" width="80%">

The table keeps the closed-form steady-state benchmark separate from the simulated mean. The simulated mean capital is slightly above the deterministic value because productivity risk and the nonlinear policy shift the invariant distribution, but the ranking across tax regimes is unchanged.

**Exact Steady States and Simulated Moments by Tax Rate**

| Tax rate   |    K_ss |   Y_ss |   C_ss |   T_ss |   K_ss / K_ss(0) |   K loss % |   Mean K (sim) |   std(Y) % |
|:-----------|--------:|-------:|-------:|-------:|-----------------:|-----------:|---------------:|-----------:|
| 0%         | 37.9893 | 3.7041 | 2.7543 | 0      |            1     |        0   |        38.9739 |      6.664 |
| 10%        | 32.2229 | 3.4909 | 2.6853 | 0.1257 |            0.848 |       15.2 |        33.0755 |      6.714 |
| 20%        | 26.8064 | 3.2671 | 2.597  | 0.2352 |            0.706 |       29.4 |        27.5309 |      6.767 |
| 30%        | 21.7584 | 3.0307 | 2.4868 | 0.3273 |            0.573 |       42.7 |        22.3595 |      6.823 |
| 40%        | 17.101  | 2.779  | 2.3515 | 0.4002 |            0.45  |       55   |        17.5845 |      6.881 |

## Takeaway

The rebate closes the government budget, not the intertemporal wedge. Once the household prices saving with $(1-\tau_k)MPK$ rather than $MPK$, the economy carries less capital into every productivity state. The exact steady state is the cleanest way to see the long-run loss; the global policy functions show how the same force operates away from the steady state. The lesson for nearby DSGE applications is that fiscal wedges can be revenue-neutral in resources and still large in allocation.

## References

- Chamley, C. (1986). *Optimal Taxation of Capital Income in General Equilibrium*. Econometrica.
- Judd, K. (1985). *Redistributive Taxation in a Simple Perfect Foresight Model*. JPE.
- Cao, D., Luo, W., and Nie, G. (2023). *Global DSGE Models*. Review of Economic Dynamics.
