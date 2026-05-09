# RBC Capital, Labor, and Business-Cycle Moments

## Overview

Aggregate productivity changes over time. A representative household owns capital, supplies labor, and chooses investment after observing productivity. The shock moves output directly and also changes work and saving.

The object is a stochastic RBC allocation. The state is capital and a two-state TFP process. The policies choose next-period capital and current labor.

The Bellman equation has no closed-form stochastic policy. We solve it on a global grid, simulate the economy, and compare simulated cycles with standard RBC moments.

## Equations

**Technology and resources.** Capital $k_t$, labor $l_t\in(0,1)$, and TFP $z_t$
produce output through Cobb-Douglas technology:

$$y_t = z_t\,k_t^{\alpha}\,l_t^{1-\alpha},\qquad \alpha\in(0,1),$$

The resource constraint is

$$c_t + k_{t+1} = z_t\,k_t^{\alpha}\,l_t^{1-\alpha} + (1-\delta)\,k_t,$$

with $c_t>0$ and $k_{t+1}\geq 0$. Investment is $i_t = k_{t+1} - (1-\delta)\,k_t$.

**Preferences.** Period utility uses log consumption and log leisure:

$$u(c,l)=\log c+\phi\log(1-l),\qquad \phi>0,$$

The household maximizes
$\mathbb{E}_0\sum_{t=0}^{\infty}\beta^t u(c_t,l_t)$.

**TFP process.** Productivity takes two values $z_t\in\{z_L,z_H\}=\{0.95,1.05\}$
with persistent symmetric transitions:

$$P_{ij}=\Pr(z_{t+1}=z_j\mid z_t=z_i),\qquad
P=\begin{pmatrix}0.95 & 0.05\\ 0.05 & 0.95\end{pmatrix}.$$

**Bellman equation.** Conditioning on the current state $(k,z_i)$, the household
solves:

$$V(k,z_i)=\max_{k',\,l\in(0,1)}[\log c+\phi\log(1-l)+\beta\sum_{j}P_{ij}\,V(k',z_j)],$$

subject to $c=z_i k^{\alpha} l^{1-\alpha}+(1-\delta)k-k'>0$. The policy
functions are $g_k(k,z)=k'$ and $g_l(k,z)=l$.

**Deterministic $z=1$ benchmark.** Setting $z\equiv 1$ in the stochastic
Bellman, the Euler condition for capital pins down the steady-state
capital-labor ratio,

$$\frac{k_{ss}}{l_{ss}}=(\frac{1/\beta-1+\delta}{\alpha})^{1/(\alpha-1)},$$

and the labor first-order condition pins down hours

$$l_{ss}=\frac{w_{ss}}{w_{ss}+\phi\,(c_{ss}/l_{ss})},\qquad
w_{ss}=(1-\alpha)(k_{ss}/l_{ss})^{\alpha}.$$

The stochastic policy fluctuates around this benchmark.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| $\beta$ | 0.99 | Discount factor (quarterly) |
| $\delta$ | 0.0233 | Depreciation rate |
| $\alpha$ | 0.3333 | Capital share in Cobb-Douglas |
| $\phi$ | 1.74 | Leisure weight in utility |
| $z\in\{z_L,z_H\}$ | $\{0.95,1.05\}$ | Two-state aggregate TFP |
| $P_{ii}$ | 0.95 | Probability of staying in the same TFP state |
| $k_{ss}$ | 10.4980 | Deterministic steady-state capital at $z=1$ |
| $l_{ss}$ | 0.3330 | Deterministic steady-state hours |
| $c_{ss}$ | 0.8073 | Deterministic steady-state consumption |
| $i_{ss}$ | 0.2446 | Deterministic steady-state investment |
| Capital grid | $[9.0,12.0]$, 50 pts | State and $k'$ choice grid |
| Labor grid | $[0.2,0.6]$, 50 pts | $l$ candidates |
| Fine benchmark | 200 capital, 100 labor pts | Audit only |
| Tolerance | 1e-05 | Sup-norm stopping rule for VFI |
| Simulation | 5000 periods after 500 burn-in | Stationary moments |

## Solution Method

**Bellman update.** The Bellman operator

$$(TV)(k,z_i)=\max_{(l,k')}[\log c(k,z_i,l,k')+\phi\log(1-l)+\beta\sum_{j}P_{ij}V(k',z_j)]$$

is a $\beta$-contraction. VFI applies it until the value function changes by less than the tolerance. For each state, the code evaluates every labor and next-capital pair. It masks negative consumption and takes a joint argmax. The selected indices define the two policy rules.

**Pseudocode.**

```text
Algorithm  Global VFI for the two-state RBC model
Inputs   capital grid K = {k_i}, labor grid L = {l_m}, TFP states {z_1,z_2},
           transition matrix P, primitives (beta, delta, alpha, phi),
           tolerance epsilon
Outputs  V(k_i, z_s), capital policy g_k(k_i, z_s), labor policy g_l(k_i, z_s)

Precompute  u_{i,s,m,j} <- log c + phi log(1 - l_m)
            with c = z_s k_i^alpha l_m^(1-alpha) + (1-delta) k_i - k_j,
            and u_{i,s,m,j} <- -infinity if c <= 0
Initialize  V_{i,s} <- (log c_guess + phi log(1 - l_guess)) / (1 - beta)
repeat n = 0, 1, 2, ...:
    EV_{j,s} <- sum_t P_{s,t} V_{j,t}                  # 1 mat-mat
    M_{i,s,m,j} <- u_{i,s,m,j} + beta * EV_{j,s}        # broadcast add
    (m*, j*)_{i,s} <- argmax over (m, j) of M_{i,s,m,j} # joint argmax
    V^new_{i,s}    <- max over (m, j) of M_{i,s,m,j}
    err            <- max_{i,s} | V^new_{i,s} - V_{i,s} |
    V              <- V^new
stop when err < epsilon
g_k(k_i, z_s) <- k_{j*_{i,s}};   g_l(k_i, z_s) <- l_{m*_{i,s}}
```

**Fine-grid audit.** The fine grid uses 200 capital nodes and 100 labor nodes on the same domain. It is an audit, not the policy used for simulation. The max relative value error is **2.1e-04**. The max capital-policy gap is **0.0461**. The max hours gap is **0.0150**.

The coarse VFI converged in **515 iterations** with sup-norm error **9.95e-06**. The fine-grid VFI converged in **525 iterations** with error **9.96e-06**.

## Results

The value function rises with capital. High TFP shifts the curve up because installed capital is more productive. The dotted fine-grid lines sit on the coarse-grid curves. The deterministic steady state sits between the two stochastic centers.

<img src="figures/value-function.png" alt="Value function by capital and TFP state" width="80%">

The capital policy stays near the 45-degree line, so capital moves slowly. High TFP raises next-period capital at each current capital level. Hours rise in high TFP states and fall slightly with capital. The fine grid shows the same policies with smoother steps.

<img src="figures/policy-functions.png" alt="Capital and labor policy functions" width="80%">

Output jumps when TFP changes. Consumption moves less because capital buffers resources. Investment absorbs most of the gap between output and consumption. Capital then adjusts slowly after each regime switch.

<img src="figures/simulation.png" alt="Simulated output, consumption, investment, and TFP" width="80%">

Consumption is smoother than output. Investment moves with output and is about four times as volatile. Hours are strongly procyclical. Capital is persistent because it accumulates past investment.

<img src="figures/comovements.png" alt="HP-filtered cyclical comovements" width="80%">

The table gives standard HP-filtered moments from the simulated economy. Investment is the most volatile flow. Consumption is smoother than output. Hours are strongly procyclical. Capital has high autocorrelation because it is a stock.

**Business-cycle moments, HP-filtered (lambda=1600), 5000-quarter simulation**

| Variable        |   Std Dev (%) |   Relative to Y |   Corr with Y |   Autocorr(1) |
|:----------------|--------------:|----------------:|--------------:|--------------:|
| Output (Y)      |          4.55 |            1    |          1    |          0.71 |
| Consumption (C) |          1.54 |            0.34 |          0.48 |          0.74 |
| Investment (I)  |         18.75 |            4.12 |          0.96 |          0.69 |
| Hours (L)       |          2.75 |            0.6  |          0.94 |          0.7  |
| Capital (K)     |          1.32 |            0.29 |          0.07 |          0.95 |

## Takeaway

The global-grid RBC model turns a two-state productivity shock into familiar business-cycle comovements. Investment is volatile, consumption is smooth, and hours are procyclical. Capital is the persistent state that carries shocks forward. The fine-grid audit shows the moments are not driven by coarse discretization.

## References

- Kydland, F. and Prescott, E. (1982). "Time to Build and Aggregate Fluctuations." *Econometrica*, 50(6), 1345-1370.
- King, R., Plosser, C., and Rebelo, S. (1988). "Production, Growth and Business Cycles: I. The Basic Neoclassical Model." *Journal of Monetary Economics*, 21(2-3), 195-232.
- Cooley, T. and Prescott, E. (1995). "Economic Growth and Business Cycles." In Cooley (ed.), *Frontiers of Business Cycle Research*, Princeton University Press.
- Hansen, G. (1985). "Indivisible Labor and the Business Cycle." *Journal of Monetary Economics*, 16(3), 309-327.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 12.
