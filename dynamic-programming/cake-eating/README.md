# Finite-Resource Cake Eating

> Optimal consumption of a finite, non-renewable resource over an infinite horizon.

## Overview

The cake-eating problem is a finite-resource allocation problem. An agent starts with wealth, or cake, and chooses how much to consume today versus how much to carry forward. There is no production, income, or uncertainty. Saving one more unit only preserves that unit for tomorrow, so the model isolates the shadow value of remaining wealth.

This stripped-down environment is useful because log utility gives an exact benchmark: the agent consumes a constant share of wealth each period. The same Bellman-equation logic carries into [optimal growth](../optimal-growth/), [consumption-savings](../consumption-savings/), and the later heterogeneous-agent savings tutorials, where resources also move through time but the state space is larger and the benchmark is no longer closed form.

## Equations

Let $W_t$ be cake or wealth at the start of period $t$. The agent chooses
consumption $c_t \in [0,W_t]$, and unconsumed cake becomes next period's state:

$$W_{t+1}=W_t-c_t.$$

Lifetime utility is

$$\sum_{t=0}^{\infty} \beta^t u(c_t), \qquad \beta \in (0,1).$$

With CRRA preferences,

$$u(c) = \frac{c^{1-\sigma}}{1-\sigma}, \qquad
u(c)=\log c \text{ when } \sigma=1.$$

The value function $V(W)$ solves

$$V(W) = \max_{0 \le c \le W} \bigl[ u(c) + \beta V(W-c) \bigr].$$

The consumption policy is $c^{\ast}(W)$ and the next-wealth policy is
$g(W)=W-c^{\ast}(W)$. In the log-utility case, the closed-form solution is

$$V(W) = \frac{\ln((1-\beta) W)}{1-\beta} + \frac{\beta \ln \beta}{(1-\beta)^2}$$

and

$$c^{\ast}(W) = (1-\beta) W, \qquad g(W)=\beta W.$$

The marginal value of cake is the shadow value of an extra unit of wealth:
$V'(W)=1/((1-\beta)W)$ under log utility.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\beta$  | 0.9 | Discount factor |
| $\sigma$ | 1.0 | CRRA coefficient; $1$ gives log utility |
| Wealth grid | 500 points | Uniform grid for $W$ |
| Consumption grid | 300 points | Feasible choices inside each Bellman update |
| $W \in$  | [0.01, 1.0] | Cake size range |
| Tolerance | 1e-06 | Sup-norm convergence criterion |
| Simulation periods | 30 | Depletion-path horizon |

## Solution Method

The numerical problem approximates $V(W)$ on a grid and searches over feasible consumption choices. The continuation value $V_n(W-c)$ is interpolated because next period's wealth usually does not land exactly on the grid.

```text
Algorithm: cake-eating value function iteration
Input: wealth grid W, discount factor beta, utility u, tolerance epsilon
Output: value function V and consumption policy c*(W)
Initialize V_0(W) = u(W)
repeat for n = 0, 1, 2, ...:
    for each wealth state W_i:
        build feasible choices c in [0, W_i]
        W_next = W_i - c
        continuation = interpolate V_n at W_next
        choose c that maximizes u(c) + beta * continuation
        record V_{n+1}(W_i) and c*(W_i)
    error = max_i |V_{n+1}(W_i) - V_n(W_i)|
until error < epsilon
```

The Bellman operator is a contraction, so this fixed-point iteration converges to the unique value function. Here it converged in **68 iterations** with sup-norm error **4.23e-07**. The closed-form log solution is not used to solve the model; it is used afterward as ground truth.

## Results

The computed value function can be compared with the log-utility ground truth. Concavity means extra cake has high value when the stock is low and lower value when the stock is already large. The largest value-function deviation outside the bottom decile of the grid is **2.52e-02**.

The numerical and analytical value functions nearly overlap. The remaining gap is a grid and interpolation error, not an economic disagreement.

<img src="figures/value-function.png" alt="Value function: numerical VFI vs analytical solution" width="80%">

The policy function is the economic object of interest. Under log utility, the agent consumes a constant share $(1-\beta)$ of current wealth. With $\beta=0.9$, that share is **10.0%**. The largest policy deviation outside the bottom decile is **3.24e-04**.

The numerical policy tracks the analytical straight line. Small departures come from the finite consumption grid and interpolation of the continuation value.

<img src="figures/policy-function.png" alt="Consumption policy: numerical vs analytical" width="80%">

The simulated policy traces the resource path over time. The analytical path is $W_t=\beta^t W_0$ and $c_t=(1-\beta)W_t$. The largest numerical depletion-path deviation over the simulation is **1.25e-03**.

Both wealth and consumption shrink geometrically. The analytical path makes the numerical error visible as a diagnostic of grid resolution rather than a separate economic effect.

<img src="figures/simulation.png" alt="Simulation: cake depletion and consumption paths starting from W=1" width="80%">

The table reports pointwise errors at selected wealth states. The cake-eating benchmark allows the numerical approximation to be audited directly before moving to models without closed forms.

**Numerical vs analytical solution at selected grid points**

|     W |   V(W) numerical |   V(W) analytical |   V error |   c* numerical |   c* analytical |   c* error |
|------:|-----------------:|------------------:|----------:|---------------:|----------------:|-----------:|
| 0.109 |         -54.6794 |          -54.6542 |  -0.0252  |         0.011  |          0.0109 |   3.54e-05 |
| 0.236 |         -46.9527 |          -46.9402 |  -0.0124  |         0.0237 |          0.0236 |   7.66e-05 |
| 0.363 |         -42.646  |          -42.6378 |  -0.00825 |         0.0364 |          0.0363 |   0.000118 |
| 0.49  |         -39.6455 |          -39.6393 |  -0.0062  |         0.0492 |          0.049  |   0.000159 |
| 0.617 |         -37.3406 |          -37.3356 |  -0.00495 |         0.0619 |          0.0617 |   0.0002   |
| 0.744 |         -35.4687 |          -35.4645 |  -0.00415 |         0.0746 |          0.0744 |   0.000241 |
| 0.871 |         -33.8925 |          -33.8889 |  -0.00351 |         0.0874 |          0.0871 |   0.000283 |
| 1     |         -32.5114 |          -32.5083 |  -0.00313 |         0.1003 |          0.1    |   0.000324 |

## Takeaway

Cake eating reduces dynamic programming to a resource-allocation problem: the state is remaining wealth, the control is current consumption, and the policy trades off current utility against the shadow value of wealth tomorrow. With log utility, the agent consumes the constant share $(1-\beta)$ and carries forward $\beta W$. The small numerical gaps shown above are grid and interpolation diagnostics. In optimal-growth and consumption-savings models, the same Bellman logic remains, but production, income risk, and borrowing constraints remove this closed-form safety check.

## References

- Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 4.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.
