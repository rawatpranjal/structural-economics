# RBC Capital, Labor, and Business-Cycle Moments

## Overview

Brock and Mirman (1972) introduced stochastic shocks into the neoclassical growth model but left labor supply fixed and made no contact with data. Kydland and Prescott (1982) asked whether a quantitative version of that model, with endogenous labor and a time-to-build investment lag, could replicate the second moments of the post-war U.S. business cycle.

The object is a *stationary RBC allocation*. The state is capital and a two-state TFP process. The policies choose next-period capital and current labor.

The Bellman equation has no closed-form stochastic policy. We solve it on a global grid, simulate the economy, and compare simulated cycles with standard RBC moments.

## Read before

- [Optimal growth model](../optimal-growth/README.md)
- [Consumption-savings under income risk](../consumption-savings/README.md)
- [Shock discretization with Rouwenhorst](../shock-discretization/README.md)

## Equations

Capital $`k_t`$, labor $`l_t\in(0,1)`$, and TFP $`z_t`$ produce output through Cobb-Douglas technology:

```math
y_t = z_t k_t^{\alpha} l_t^{1-\alpha}, \qquad \alpha\in(0,1).
```

The resource constraint is

```math
c_t + k_{t+1} = z_t k_t^{\alpha} l_t^{1-\alpha} + (1-\delta) k_t,
```

with $`c_t>0`$ and $`k_{t+1}\geq 0`$. Investment is $`i_t = k_{t+1} - (1-\delta) k_t`$.

Period utility uses log consumption and log leisure:

```math
u(c,l)=\log c+\phi\log(1-l), \qquad \phi>0.
```

The household maximizes

```math
\mathbb{E}_0\sum_{t=0}^{\infty}\beta^t u(c_t,l_t).
```

Productivity takes two values $`z_t\in\lbrace z_L,z_H\rbrace=\lbrace0.95,1.05\rbrace`$ with persistent symmetric transitions. Let $`P_{ij}=\Pr(z_{t+1}=z_j\mid z_t=z_i)`$:

```math
P=\begin{pmatrix}0.95 & 0.05\\ 0.05 & 0.95\end{pmatrix}.
```

Conditioning on state $`(k,z_i)`$, the *Bellman equation* is

```math
V(k,z_i)=\max_{k', l\in(0,1)}\left[\log c+\phi\log(1-l)+\beta\sum_{j}P_{ij} V(k',z_j)\right],
```

subject to $`c=z_i k^{\alpha} l^{1-\alpha}+(1-\delta)k-k'>0`$. The policy functions are $`g_k(k,z)=k'`$ and $`g_l(k,z)=l`$.

Setting $`z\equiv 1`$ in the Bellman, the Euler condition for capital pins down the steady-state capital-labor ratio:

```math
\frac{k_{ss}}{l_{ss}}=\left(\frac{1/\beta-1+\delta}{\alpha}\right)^{1/(\alpha-1)}.
```

The labor first-order condition then pins down hours:

```math
l_{ss}=\frac{w_{ss}}{w_{ss}+\phi(c_{ss}/l_{ss})}, \qquad
w_{ss}=(1-\alpha)(k_{ss}/l_{ss})^{\alpha}.
```

The stochastic policy fluctuates around this benchmark.

## Worked Numerical Example

The model has two interlocking first-order conditions at the deterministic steady state ($`z = 1`$, all expectations realized). Solving them in sequence pins down every object the VFI converges to.

The Euler equation for capital, evaluated at a constant interior path ($`c_{t+1} = c_t`$), requires the net return on capital to equal the inverse of the discount factor:

```math
\alpha (k/l)^{\alpha - 1} = \frac{1}{\beta} - 1 + \delta
= \frac{1}{0.99} - 1 + 0.0233 = 0.03340.
```

Rearranging isolates the capital-labor ratio. With $`\alpha = 1/3`$ the exponent on the left is $`-2/3`$, so inverting gives an exponent of $`3/2`$:

```math
(k/l)^{2/3} = \frac{\alpha}{0.03340} = \frac{1/3}{0.03340} = 9.982.
```

```math
k/l = 9.982^{3/2} = 31.54.
```

The wage follows from the labor marginal product at this ratio:

```math
w = (1-\alpha)(k/l)^{\alpha} = \frac{2}{3} \times 31.54^{1/3}
= \frac{2}{3} \times 3.160 = 2.107.
```

The labor first-order condition ties hours to consumption through the leisure weight $`\phi`$:

```math
\phi \frac{c}{1-l} = w \quad \Longrightarrow \quad
c = \frac{w(1-l)}{\phi} = \frac{2.107(1-l)}{1.74}.
```

The resource constraint at steady state gives a second expression for $`c`$ in terms of $`l`$. Output per worker is $`(k/l)^\alpha = 3.160`$, capital per worker is $`31.54`$, so investment per worker is $`\delta \times 31.54 = 0.7349`$, and consumption per worker is $`(3.160 - 0.7349) \times l = 2.425 l`$. Setting the two expressions equal:

```math
2.425 l = \frac{2.107(1-l)}{1.74} = 1.211(1-l).
```

```math
l(2.425 + 1.211) = 1.211, \qquad l = \frac{1.211}{3.636} = 0.333.
```

Capital and consumption recover immediately:

```math
k = 31.54 \times 0.333 = 10.50, \qquad c = 2.425 \times 0.333 = 0.808.
```

```math
\boxed{k_{ss} \approx 10.50, \quad l_{ss} \approx 0.333, \quad c_{ss} \approx 0.808.}
```

These three numbers are the fixed point that VFI converges to. The *Bellman operator* pins the capital-labor ratio through the Euler equation alone. Labor hours are determined only when the leisure weight $`\phi`$ enters through the labor-leisure tradeoff. Without endogenous labor there would be one equation in one unknown. Adding $`l`$ as a choice requires the second condition to close the system.

## Model Setup

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| Discount factor $`\beta`$ | 0.99 | Leisure weight $`\phi`$ | 1.74 |
| Depreciation $`\delta`$ | 0.0233 | TFP states $`\lbrace z_L, z_H\rbrace`$ | $`\lbrace0.95,1.05\rbrace`$ |
| Capital share $`\alpha`$ | 0.3333 | Persistence $`P_{ii}`$ | 0.95 |
| Steady-state capital $`k_{ss}`$ | 10.4980 | Steady-state hours $`l_{ss}`$ | 0.3330 |
| Capital grid $`[9.0,12.0]`$, 50 pts | state + choice | Labor grid $`[0.2,0.6]`$, 50 pts | $`l`$ candidates |
| Fine benchmark | 200 capital, 100 labor pts | VFI tolerance (sup-norm) | 1e-05 |
| Simulation | 5000 periods | Burn-in | 500 periods |

## Solution Method

*Value function iteration* applies the Bellman operator until the sup-norm change falls below tolerance. For each state the code evaluates every labor and next-capital pair, masks infeasible consumption, and takes a joint argmax. The selected indices define the two policy rules. A fine grid with 200 capital and 100 labor nodes serves as an audit benchmark.

```
          primitives, k grid, Markov chain (z, P)
                          |
                          v
          +---------- VFI loop ----------+
          |                              |
          |  V_k  -->  [ Bellman op ]  -->  V_{k+1}
          |                              |
          +------ err >= tol: repeat ----+
                          |
                      err < tol
                          v
                 V*(k,z), g_k(k,z), g_l(k,z)
```

```python
# VFI: precompute the full flow-utility tensor once, then iterate.
def solve_vfi(k_grid, l_grid, z_vals, P, beta, delta, alpha, phi, tol=1e-5):
    # flow_utility[i, s, m, j] = log c + phi*log(1-l_m)
    #   where c = z_s * k_i^alpha * l_m^(1-alpha) + (1-delta)*k_i - k_j
    flow_utility = precompute_flow(k_grid, l_grid, z_vals, delta, alpha, phi)

    V = initialize_guess(k_grid, z_vals, alpha, phi, beta)
    error_history = []

    while True:
        EV = V @ P.T                                   # expected continuation
        total = flow_utility + beta * EV.T[None, :, None, :]
        total_flat = total.reshape(n_k, n_z, n_l * n_k)
        V_new = np.max(total_flat, axis=2)             # Bellman operator
        err = np.max(np.abs(V_new - V))
        error_history.append(err)
        V = V_new
        if err < tol:
            break

    policy_flat = np.argmax(total_flat, axis=2)
    g_l = l_grid[policy_flat // n_k]
    g_k = k_grid[policy_flat % n_k]
    return V, g_k, g_l, error_history
```

## Results

The value function rises with capital. High TFP shifts the curve up because installed capital is more productive. The dotted fine-grid lines sit on the coarse-grid curves. The VFI sup-norm converges geometrically in log scale.

<img src="figures/value-function.png" alt="Value function by TFP state and VFI convergence" width="80%">

The capital policy stays near the 45-degree line, so capital moves slowly. High TFP raises next-period capital at each current capital level. Hours rise in high TFP states and fall slightly with capital.

<img src="figures/policy-functions.png" alt="Capital and labor policy functions" width="80%">

Output jumps when TFP changes. Consumption moves less because capital buffers resources. Investment absorbs most of the gap between output and consumption.

<img src="figures/simulation.png" alt="Simulated output, consumption, investment, and TFP" width="80%">

Consumption is smoother than output. Investment moves with output and is about four times as volatile. Hours are *procyclical*. Capital is persistent because it accumulates past investment.

<img src="figures/comovements.png" alt="HP-filtered cyclical comovements" width="80%">

The table gives standard HP-filtered moments from the simulated economy. Investment is the most volatile flow. Consumption is smoother than output. Capital has high autocorrelation because it is a stock.

### Business-cycle moments

Business-cycle moments, HP-filtered (lambda=1600), 5000-quarter simulation.

| Variable | Std Dev (%) | Corr with Y | Autocorr(1) |
|:---|---:|---:|---:|
| Output (Y) | 4.55 | 1.00 | 0.71 |
| Consumption (C) | 1.54 | 0.48 | 0.74 |
| Investment (I) | 18.75 | 0.96 | 0.69 |
| Hours (L) | 2.75 | 0.94 | 0.70 |
| Capital (K) | 1.32 | 0.07 | 0.95 |

## Takeaway

*Time to Build* showed that technology shocks alone, propagated through an optimizing household, reproduce the major comovements of the business cycle. The quantitative surprise was how well the fit held with so few free parameters and no nominal frictions. The paper founded the RBC program and its calibration methodology, which went on to anchor DSGE models at central banks worldwide.

## See also

- [Aiyagari saving and capital-market clearing](../aiyagari/README.md)
- [Optimal growth model](../optimal-growth/README.md)
- [Consumption-savings under income risk](../consumption-savings/README.md)

## References

- Kydland, F. and Prescott, E. (1982). "Time to Build and Aggregate Fluctuations." *Econometrica*, 50(6), 1345-1370.
- King, R., Plosser, C., and Rebelo, S. (1988). "Production, Growth and Business Cycles: I. The Basic Neoclassical Model." *Journal of Monetary Economics*, 21(2-3), 195-232.
- Cooley, T. and Prescott, E. (1995). "Economic Growth and Business Cycles." In Cooley (ed.), *Frontiers of Business Cycle Research*, Princeton University Press.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 12.
