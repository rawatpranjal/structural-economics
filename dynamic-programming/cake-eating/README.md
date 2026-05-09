# Finite-Resource Cake Eating

## Overview

A household owns a fixed cake and chooses consumption each period. The cake does not grow, and there is no income or uncertainty. Consuming more today leaves less cake for every future period.

The state is remaining cake $W_t$. The control is consumption $c_t$. The policy rule maps each stock into consumption. The value function prices the remaining stock.

Value function iteration solves the Bellman equation on a grid. Log utility gives a closed-form Euler rule. That rule lets us check the computed value and policy directly.

## Equations

Let $W_t$ be remaining cake at the start of period $t$.
The household chooses $c_t \in [0, W_t]$ and leaves next-period cake:

$$W_{t+1} = W_t - c_t, \qquad W_0 \text{ given}.$$

Preferences use discount factor $\beta \in (0,1)$ and CRRA flow utility:

$$\sum_{t=0}^{\infty} \beta^t u(c_t),
\qquad u(c)=\frac{c^{1-\sigma}}{1-\sigma},
\qquad u(c)=\log c \text{ when } \sigma=1.$$

The value function solves a one-state Bellman equation:

$$V(W) = \max_{0 \le c \le W} \{\, u(c) + \beta\, V(W-c) \,\}.$$

The first-order condition and envelope condition give the Euler equation:

$$u'(c_t) = \beta\, u'(c_{t+1}).$$

This says marginal utility rises as the cake stock falls.
In the log case, consumption falls at rate $\beta$.

Guessing a constant consumption share gives the closed-form policy:

$$c^{\ast}(W) = (1-\beta)\, W,
\qquad g(W) = W - c^{\ast}(W) = \beta\, W,$$

The matching value function is:

$$V(W) = \frac{\ln((1-\beta) W)}{1-\beta}
+ \frac{\beta \ln \beta}{(1-\beta)^2},
\qquad V'(W) = \frac{1}{(1-\beta)\,W}.$$

This closed form is the target for the numerical check.

## Model Setup

| Symbol | Value | Role |
|--------|-------|------|
| $\beta$ | 0.9 | Discount factor; closed-form saving rate is $\beta$ |
| $\sigma$ | 1.0 | CRRA curvature; $\sigma=1$ gives the log closed form |
| $W_0$ | 1.0 | Initial cake endowment |
| $W$ | $[0.01,\, 1.0]$ | Wealth grid for $V$ and $c^{\ast}$ |
| $N_W$ | 500 | Uniform grid points for the state $W$ |
| $N_c$ | 300 | Inner grid for the consumption choice at each state |
| Tolerance $\varepsilon$ | 1e-06 | Sup-norm convergence threshold for the Bellman operator |
| $T_{sim}$ | 30 | Periods simulated for the depletion path |

## Solution Method

Define the Bellman operator

$$(TV)(W) = \max_{0 \le c \le W} \{\, u(c) + \beta\, V(W-c) \,\}.$$

The computation applies this operator repeatedly. At each grid point, it searches over feasible consumption. The next stock $W-c$ is usually off grid. The continuation value is therefore interpolated.

```text
Algorithm: Cake-eating VFI
Input : wealth grid, choice grid size N_c, tolerance epsilon
Output: value V*(W_i), consumption policy c*(W_i)
  initialise V_0(W_i) = u(W_i)                     # guess: eat everything
  for n = 0, 1, 2, ... :
      for each state W_i :
          c_grid <- N_c points uniform on (0, W_i)
          W'     <- W_i - c_grid                   # next-period wealth
          V_cont <- interp(V_n, W')                # off-grid continuation
          obj    <- u(c_grid) + beta * V_cont
          V_{n+1}(W_i) <- max(obj)
          c*(W_i)      <- argmax(obj)
      err <- max_i | V_{n+1}(W_i) - V_n(W_i) |
      stop when err < epsilon
```

The iteration converges in **68 steps**. The final sup-norm residual is **4.23e-07**. The closed form is then computed on the same wealth grid.

## Results

Concavity is the main shape restriction on the value function. The numerical value curve lies on the closed-form curve except near the lower boundary. Outside the bottom decile, the largest sup-norm gap is **2.52e-02**. The lower-boundary gap comes from the log singularity.

<img src="figures/value-function.png" alt="Numerical value function plotted against the closed-form benchmark" width="80%">

Under log utility, the household consumes **10%** of remaining cake. The numerical policy follows the closed-form line. The dotted line marks immediate exhaustion. Above the bottom decile, the largest policy gap is **3.24e-04**.

<img src="figures/policy-function.png" alt="Consumption policy plotted against the closed-form $c=(1-\beta)W$ rule" width="80%">

Starting from $W_0=1$, the policy produces geometric depletion. Wealth follows $W_t = \beta^t W_0$. Consumption follows $c_t = (1-\beta)\beta^t W_0$. The numerical path stays within **1.25e-03** of the closed form.

<img src="figures/simulation.png" alt="Wealth and consumption paths starting from $W_0=1$, numerical against closed form" width="80%">

The table checks value and policy at eight wealth states. The residuals are smooth and small away from the lower boundary.

**Numerical vs closed-form solution at selected wealth states**

|     W |   V numerical |   V closed form |   V error |   c* numerical |   c* closed form |   c* error |
|------:|--------------:|----------------:|----------:|---------------:|-----------------:|-----------:|
| 0.109 |      -54.6794 |        -54.6542 |  -0.0252  |         0.011  |           0.0109 |   3.54e-05 |
| 0.236 |      -46.9527 |        -46.9402 |  -0.0124  |         0.0237 |           0.0236 |   7.66e-05 |
| 0.363 |      -42.646  |        -42.6378 |  -0.00825 |         0.0364 |           0.0363 |   0.000118 |
| 0.49  |      -39.6455 |        -39.6393 |  -0.0062  |         0.0492 |           0.049  |   0.000159 |
| 0.617 |      -37.3406 |        -37.3356 |  -0.00495 |         0.0619 |           0.0617 |   0.0002   |
| 0.744 |      -35.4687 |        -35.4645 |  -0.00415 |         0.0746 |           0.0744 |   0.000241 |
| 0.871 |      -33.8925 |        -33.8889 |  -0.00351 |         0.0874 |           0.0871 |   0.000283 |
| 1     |      -32.5114 |        -32.5083 |  -0.00313 |         0.1003 |           0.1    |   0.000324 |

## Takeaway

Cake eating isolates Bellman logic in a one-state resource problem. The computed policy should consume a constant share of remaining cake. In this log case, the share is $1-\beta$. The closed form makes value function iteration easy to inspect. The remaining errors are interpolation and choice-grid error.

## References

- Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 4.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.
