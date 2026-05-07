# Fixed-Resource Consumption and Pontryagin Shadow Prices

> A finite-resource consumption problem in continuous time, solved by costate equations.

## Overview

A planner begins with one fixed stock $W_0$, perhaps an inherited reserve of a consumption good, and has to decide how quickly to draw it down. Eating more today gives utility now, but it also leaves less for every future date. With CRRA utility, the smoothing motive pushes the planner away from spending the whole stock early.

In discrete time this allocation can be written as a Bellman equation, as in [Finite-Resource Cake Eating](../../dynamic-programming/cake-eating/). Here the same economic tradeoff is written in continuous time. Instead of searching over a grid of next-period stocks, we use Pontryagin's maximum principle to price the remaining resource and recover the consumption path $c(t)$ directly. The costate variable is the shadow value of one more unit left for the future.

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
consumption rate and therefore the full path:
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
| $W_0$     | 1.0 | Initial resource stock |
| $\beta = e^{-\rho}$ | 0.9512 | Period-one discount factor for the discrete reference path |
| $T$       | 80.0 | Plotting horizon; the economic problem has an infinite horizon |
| Evaluation points | 500 | Time points for exact paths and ODE checks |

## Solution Method

Pontryagin's principle attaches a price to one extra unit of the resource stock. At each instant, the planner equates discounted marginal utility from current consumption with that shadow price. Since $W$ has no direct payoff, the present-value price does not drift. Discounting and preference curvature then determine how quickly consumption falls.

```text
Inputs: rho, sigma, W0, evaluation grid {t_m}
1. Write the present-value Hamiltonian H = exp(-rho t) u(c) - lambda c.
2. Use the FOC exp(-rho t) c(t)^(-sigma) = lambda(t).
3. Use the costate equation lambda_dot(t) = -H_W = 0.
4. Differentiate the FOC to obtain c_dot(t) / c(t) = -rho / sigma.
5. Use integral_0^infinity c(t) dt = W0 to set c(0) = (rho / sigma) W0.
6. Evaluate c(t), W(t), and mu(t) = c(t)^(-sigma) on {t_m}.
Output: consumption, remaining stock, and shadow-price paths.
```

The numerical work evaluates those analytical paths and checks them by ODE integration. The ODE check integrates $\dot{W}=-c$ and $\dot{c}=-(\rho/\sigma)c$ from the implied initial consumption rate, then compares the result with the closed-form path. The period-one discrete path gives a nearby benchmark for the dynamic-programming version of the same allocation problem.

**Verification:** Max absolute error in $W(t)$: 2.64e-11, in $c(t)$: 6.61e-13.

## Results

Consumption starts at $(\rho/\sigma)W_0$ and falls at the constant proportional rate $\rho/\sigma$. The ODE markers sit on the closed-form path, so the costate calculation and the differential equation give the same allocation. The stepped line is the period-one discrete allocation from the related finite-resource dynamic-programming problem.

<img src="figures/consumption-path.png" alt="Continuous consumption path with ODE and discrete-time comparisons" width="80%">

The stock is depleted only asymptotically. With CRRA utility and an infinite horizon, the planner keeps a tail of future consumption alive because marginal utility becomes large as consumption approaches zero.

<img src="figures/cake-remaining.png" alt="Exact cake stock with ODE and discrete-time comparisons" width="80%">

The present-value costate is flat because the resource stock has no direct payoff term. In current-value units the shadow price rises at rate $\rho$: one unit left for a later date is scarce in utility terms even though its discounted value is equalized along the optimum.

<img src="figures/shadow-price.png" alt="Present-value and current-value shadow prices for the resource stock" width="80%">

The exact solution is a benchmark for the numerical ODE path. The small errors below come from ODE solver tolerances rather than a grid over choices.

**Selected Checks Against the Continuous-Time Path**

|   t |   c(t) exact |   c(t) RK45 |   c error |   W(t) exact |   W(t) RK45 |   W error |
|----:|-------------:|------------:|----------:|-------------:|------------:|----------:|
|   0 |     0.025    |    0.025    |   0       |     1        |    1        |   0       |
|   5 |     0.021991 |    0.021991 |   1.8e-13 |     0.879628 |    0.879628 |   7.2e-12 |
|  10 |     0.019421 |    0.019421 |   1.1e-13 |     0.776852 |    0.776852 |   4.3e-12 |
|  20 |     0.015148 |    0.015148 |   2.5e-13 |     0.605923 |    0.605923 |   1e-11   |
|  30 |     0.011768 |    0.011768 |   2e-13   |     0.470713 |    0.470713 |   7.8e-12 |
|  50 |     0.007159 |    0.007159 |   3.3e-14 |     0.286361 |    0.286361 |   1.3e-12 |

## Takeaway

The costate is the intertemporal price of the remaining resource. In this problem the present-value price is constant, so optimality requires a declining consumption path that keeps discounted marginal utility equal across dates.

Higher impatience raises the depletion rate, while higher risk aversion slows it through the smoothing motive. The continuous-time formulation also clarifies the object behind the discrete cake-eating Bellman problem: a smooth shadow-price condition linking current consumption to all future scarcity.

## References

- Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 7.
- Kamien, M. and Schwartz, N. (2012). *Dynamic Optimization*. Dover, 2nd edition.
- Chiang, A. (1992). *Elements of Dynamic Optimization*. Waveland Press.
