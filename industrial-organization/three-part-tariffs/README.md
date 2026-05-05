# Three-Part Tariffs and Forward-Looking Broadband Demand

> How data caps make monthly broadband demand a dynamic choice problem.

## Overview

Residential broadband contracts often combine three instruments: a fixed fee, an included data allowance, and a per-GB overage price. The allowance is not just a nonlinear price schedule. It creates a state variable inside the month. A GB used on day 3 lowers the remaining allowance on day 4, so the relevant marginal price includes the option value of keeping data for later.

The demand side here is small. A consumer solves a finite-horizon usage problem within the billing cycle, then heterogeneous types choose among a low-fee metered plan, a middle three-part plan, and an unlimited plan. The dynamic-choice logic is close to the continuation-value reasoning in [bus replacement](../dynamic-discrete-choice/), while the fixed-fee role connects to the two-part-tariff discussion in [vertical relationships](../vertical-relationships/).

## Equations

Let $t=1,\ldots,T$ index days in a billing cycle and let $C_{t-1}$ be cumulative
usage before day $t$. Under plan $k$, the consumer pays a fixed fee $F_k$, has
allowance $A_k$, and pays overage price $q_k$ per GB above the allowance.

Daily usage is $c_t \geq 0$. Type $h$ has gross daily utility

$$u(c_t;h) = h\log(1+c_t) - \frac{\psi}{2}c_t^2,$$

with $\psi>0$. Cumulative usage follows

$$C_t=C_{t-1}+c_t,\qquad C_0=0.$$

The incremental overage quantity created on day $t$ is

$$\Delta O_k(C_{t-1},c_t)
=\max\{0,C_{t-1}+c_t-A_k\}-\max\{0,C_{t-1}-A_k\}.$$

For a given plan and type, the within-cycle value function is

$$V_{k,t}(C_{t-1};h)=
\max_{c_t\in[0,\bar c]}
\left[
u(c_t;h)-q_k\Delta O_k(C_{t-1},c_t)
+V_{k,t+1}(C_t;h)
\right],$$

with terminal value $V_{k,T+1}(\cdot;h)=0$. The policy
$g_{k,t}(C_{t-1};h)$ gives daily usage.

Plan choice adds the fixed fee and the value of speed $B(s_k)$:

$$W_i(k)=V_{k,1}(0;h_i)+B(s_k)-F_k,\qquad
d_i=\arg\max_k W_i(k).$$

## Model Setup

The calibration uses a 30-day billing cycle, $\psi=0.34$, daily choices $c_t\in[0,6]$, and the speed shifter $B(s_k)=2.6\log(s_k)$. Consumer heterogeneity is a small discrete type distribution; the weights are used only for plan shares and average outcomes.

| Plan | Fixed fee | Allowance | Overage price | Speed |
|------|-----------|-----------|---------------|-------|
| Metered | 16 | 25 GB | 0.70 | 80 Mbps |
| Three-part | 46 | 85 GB | 1.60 | 200 Mbps |
| Unlimited | 52 | uncapped | 0.00 | 320 Mbps |

| Taste type $h_i$ | Weight |
|------------------|--------|
| 3.0 | 0.10 |
| 3.5 | 0.14 |
| 4.0 | 0.18 |
| 4.5 | 0.20 |
| 5.0 | 0.17 |
| 5.6 | 0.13 |
| 6.2 | 0.08 |

## Solution Method

For each type-plan pair, backward induction solves the finite-horizon usage problem on a grid for cumulative monthly usage. The fixed fee is excluded from the daily Bellman recursion because it is sunk after the plan is chosen; it enters only when comparing plans. The overage price enters inside the recursion because today's usage can move the consumer closer to the cap or past it.

```text
Algorithm: finite-horizon usage and plan choice
Input: plans (F_k, A_k, q_k, s_k), type distribution (h_i, omega_i), usage grid C
Output: daily policies g_{k,t}(C; h_i), chosen plans d_i, plan shares
for each type h_i and plan k:
    set V_{k,T+1}(C; h_i) = 0 for every cumulative-usage state C
    for t = T, T-1, ..., 1:
        for each state C on the cumulative-usage grid:
            for each feasible daily usage c in [0, c_bar]:
                C_next = C + c
                overage_increment = max(0, C_next - A_k) - max(0, C - A_k)
                payoff = u(c; h_i) - q_k * overage_increment + V_{k,t+1}(C_next; h_i)
            choose c that maximizes payoff and record g_{k,t}(C; h_i)
    compute W_i(k) = V_{k,1}(0; h_i) + B(s_k) - F_k
choose d_i = argmax_k W_i(k), then aggregate shares with weights omega_i
```

The focal policy uses a 0.5 GB grid. For the billing-cycle path, the same model is also solved on a 0.25 GB grid as a numerical benchmark.

## Results

The shadow price of the remaining allowance. Early in the month, a consumer near the cap cuts usage because each GB raises the chance of paying overage charges later. Near the end of the cycle, the same remaining allowance has less option value, so the policy relaxes.

<img src="figures/usage-policy.png" alt="Daily usage policy by day and remaining allowance" width="80%">

The simulated path stays close to the allowance without treating it as a hard constraint. The finer-grid benchmark matches total usage within **0.00 GB**, while the largest cumulative-path gap is **3.00 GB**. The dynamics come from the nonlinear contract, not from time-varying daily tastes.

<img src="figures/billing-cycle-usage.png" alt="Billing-cycle usage under the baseline grid and a finer-grid benchmark" width="80%">

Plan choice as a sorting problem. Low-usage types choose the low fixed fee, middle types value the allowance, and high-usage types pay for unlimited access. The circled points are the contracts selected by the discrete type distribution.

<img src="figures/plan-comparison.png" alt="Net consumer value by type and plan" width="80%">

**Plan-choice summary across consumer types**

| Plan       |   Share |   Average usage |   Average revenue |   Average consumer value |
|:-----------|--------:|----------------:|------------------:|-------------------------:|
| Metered    |    0.1  |          45     |                30 |                   52.384 |
| Three-part |    0.69 |          82.971 |                46 |                   99.233 |
| Unlimited  |    0.21 |         110.714 |                52 |                  163.701 |

## Takeaway

A three-part tariff changes demand before the cap is reached. The allowance has a shadow value because it can be spent later in the billing cycle, so a forward-looking consumer reacts to expected overage risk rather than only to the current marginal price. The contract menu sorts consumers by usage intensity: low types avoid the fixed fee, middle types buy the allowance, and high types choose unlimited access. The numerical benchmark check suggests that the focal path is not driven by the coarse grid: net consumer value differs from the finer-grid solution by **0.408**, about **0.4%** of the baseline value.

## References

- Nevo, A., Turner, J., and Williams, J. (2016). Usage-Based Pricing and Demand for Residential Broadband. *Econometrica*, 84(2), 411-443.
- Lecture 18 Slides 2023: Three-part tariffs and forward-looking broadband demand.
