# Continuous-Time Cake Eating and Shadow Prices

> Optimal depletion of a finite resource, solved with Pontryagin's maximum principle.

## Overview

Cake eating is the stripped-down resource allocation problem: a planner starts with a fixed stock $W_0$ and must decide how quickly to turn it into consumption. In discrete time this appears as a Bellman equation, as in [Finite-Resource Cake Eating](../../dynamic-programming/cake-eating/). Here the same economics is written in continuous time, so the central object is the path $c(t)$ rather than a sequence of grid choices.

Consuming more now lowers the stock available later, while CRRA utility rewards smoothing. Pontryagin's maximum principle turns that tradeoff into a shadow price for the remaining cake. Because the cake itself does not enter utility except through feasible consumption, the present-value shadow price is constant and consumption declines smoothly rather than dropping to zero in finite time.

## Equations

For $\sigma \neq 1$, flow utility is
$$u(c)=\frac{c^{1-\sigma}}{1-\sigma}, \qquad u'(c)=c^{-\sigma},$$
with the log case obtained as $\sigma \to 1$. The continuous-time problem is
$$\max_{\{c(t)\}_{t\geq 0}} \int_0^\infty e^{-\rho t} u(c(t)) \, dt$$

subject to
$$\dot{W}(t)=-c(t), \qquad W(0)=W_0, \qquad c(t)\geq 0, \qquad W(t)\geq 0.$$

The present-value Hamiltonian is
$$\mathcal{H}(c,W,\lambda,t)=e^{-\rho t}u(c)-\lambda c.$$

The first-order and costate equations are
$$e^{-\rho t}c(t)^{-\sigma}=\lambda(t), \qquad
\dot{\lambda}(t)=-\frac{\partial \mathcal{H}}{\partial W}=0.$$

Since $\lambda(t)$ is constant in present value, differentiating the first-order
condition gives the Euler equation
$$\frac{\dot c(t)}{c(t)}=-\frac{\rho}{\sigma}.$$

The no-waste condition $\int_0^\infty c(t)\,dt=W_0$ pins down the initial
consumption rate:
$$c(t)=\frac{\rho}{\sigma}W_0 e^{-\rho t/\sigma}, \qquad
W(t)=W_0 e^{-\rho t/\sigma}.$$

The current-value shadow price is
$$\mu(t)=e^{\rho t}\lambda=c(t)^{-\sigma},$$
so $\dot{\mu}(t)/\mu(t)=\rho$. Scarcity grows in current-value terms even though
the present-value costate is flat.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\rho$    | 0.05 | Continuous discount rate |
| $\sigma$  | 2.0 | Relative risk aversion; IES $=1/\sigma$ |
| $W_0$     | 1.0 | Initial cake size |
| $\beta = e^{-\rho}$ | 0.9512 | Period-one discount factor used for the discrete reference path |
| $T$       | 80.0 | Plotting and ODE-check horizon |
| Evaluation points | 500 | Time points for plotting exact and numerical paths |

## Solution Method

Pontryagin's principle attaches a price to relaxing the stock constraint by one unit. The planner chooses $c(t)$ so that discounted marginal utility equals that price at every instant. Since $W$ has no direct payoff, the present-value price does not drift; all movement in consumption comes from discounting and curvature.

```text
Inputs: rho, sigma, W0, evaluation grid {t_m}
1. Write the present-value Hamiltonian H = exp(-rho t) u(c) - lambda c.
2. Use the FOC exp(-rho t) c(t)^(-sigma) = lambda(t).
3. Use the costate equation lambda_dot(t) = -H_W = 0.
4. Differentiate the FOC to obtain c_dot(t) / c(t) = -rho / sigma.
5. Use integral_0^infinity c(t) dt = W0 to set c(0) = (rho / sigma) W0.
6. Evaluate c(t), W(t), and mu(t) = c(t)^(-sigma) on {t_m}.
Output: exact consumption, stock, and shadow-price paths.
```

The ODE integration is a numerical check, not the source of the solution. It integrates $\dot{W}=-c$ and $\dot{c}=-(\rho/\sigma)c$ from the analytical initial condition and compares the resulting path with the exact one. The period-one discrete path is a nearby benchmark for the dynamic programming version of the same allocation problem.

**Verification:** Max absolute error in $W(t)$: 2.64e-11, in $c(t)$: 6.61e-13.

## Results

Consumption starts at $(\rho/\sigma)W_0$ and then falls at the constant proportional rate $\rho/\sigma$. The black markers show that direct ODE integration recovers the exact path. The stepped line is the period-one discrete allocation, which connects this continuous-time problem to the finite-resource dynamic-programming tutorial.

<img src="figures/consumption-path.png" alt="Exact continuous consumption path with ODE and discrete-time comparisons" width="80%">

The stock is depleted only asymptotically. That is not a numerical artifact: with CRRA utility and an infinite horizon, the planner keeps a tail of future consumption alive because marginal utility becomes large as consumption approaches zero.

<img src="figures/cake-remaining.png" alt="Exact cake stock with ODE and discrete-time comparisons" width="80%">

The present-value costate is flat because the resource stock has no direct payoff term. In current-value units the shadow price rises at rate $\rho$: one unit of cake left for a later date is scarce in utility terms even though its discounted value is equalized along the optimum.

<img src="figures/shadow-price.png" alt="Present-value and current-value shadow prices for cake" width="80%">

The exact solution is a benchmark for the numerical ODE path. The small errors below are solver error, not approximation error from a grid over choices.

**Selected Checks Against the Exact Continuous-Time Path**

|   t |   c(t) exact |   c(t) RK45 |   c error |   W(t) exact |   W(t) RK45 |   W error |
|----:|-------------:|------------:|----------:|-------------:|------------:|----------:|
|   0 |     0.025    |    0.025    |   0       |     1        |    1        |   0       |
|   5 |     0.021991 |    0.021991 |   1.8e-13 |     0.879628 |    0.879628 |   7.2e-12 |
|  10 |     0.019421 |    0.019421 |   1.1e-13 |     0.776852 |    0.776852 |   4.3e-12 |
|  20 |     0.015148 |    0.015148 |   2.5e-13 |     0.605923 |    0.605923 |   1e-11   |
|  30 |     0.011768 |    0.011768 |   2e-13   |     0.470713 |    0.470713 |   7.8e-12 |
|  50 |     0.007159 |    0.007159 |   3.3e-14 |     0.286361 |    0.286361 |   1.3e-12 |

## Takeaway

The costate is the intertemporal price of the remaining resource, not just formal machinery. In this problem the present-value price is constant, so optimality requires a declining consumption path that keeps discounted marginal utility equal across dates.

Higher impatience raises the depletion rate, while higher risk aversion slows it through the smoothing motive. The continuous-time formulation also clarifies what the discrete cake-eating Bellman problem is approximating: a smooth shadow-price condition, not merely a search over feasible consumption grid points.

## References

- Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 7.
- Kamien, M. and Schwartz, N. (2012). *Dynamic Optimization*. Dover, 2nd edition.
- Chiang, A. (1992). *Elements of Dynamic Optimization*. Waveland Press.
