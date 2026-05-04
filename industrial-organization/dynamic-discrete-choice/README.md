# Dynamic Discrete Choice

> Rust-style bus engine replacement with full-solution and CCP estimation.

## Overview

Dynamic discrete choice models are used when agents make repeated discrete decisions and today's action changes tomorrow's state. The canonical example is Rust's bus engine replacement problem. A bus operator observes mileage and chooses whether to replace the engine. In this normalization, replacement gives up the current keep payoff but resets future mileage and reduces future maintenance costs.

This tutorial solves the model, simulates panel data, and estimates the payoff parameters two ways: full-solution maximum likelihood and a Hotz-Miller conditional-choice-probability estimator.

## Equations

State $x$ is mileage. Action $a=1$ replaces the engine, while $a=0$ keeps it.
Replacement utility is normalized to zero:

$$u(x,1) = 0,$$

and the flow payoff from keeping the engine is:

$$u(x,0) = \theta_0 + \theta_1 x, \qquad \theta_1 < 0.$$

With Type-I extreme value shocks, the conditional value functions satisfy:

$$v_a(x) = u(x,a) + \beta \sum_{x'} F_a(x' \mid x)
\left[\log\left(\exp(v_1(x')) + \exp(v_0(x'))\right) + \gamma\right].$$

The replacement probability is:

$$P(a=1 \mid x) =
\frac{\exp(v_1(x))}{\exp(v_1(x))+\exp(v_0(x))}.$$

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\beta$ | 0.9 | Discount factor |
| $\theta_0$ | 2.00 | Keep-engine payoff intercept |
| $\theta_1$ | -0.15 | Mileage cost slope |
| Mileage states | 61 | Grid from 0 to 15 |
| Buses | 1500 | Simulated panel units |
| Periods | 35 | Observations per bus |

## Solution Method

**Full-solution ML** solves the value function for every candidate parameter vector and evaluates the likelihood of observed replacement decisions.

**CCP estimation** first estimates the policy rule nonparametrically with a logit in mileage and mileage squared. The Hotz-Miller inversion then recovers ex ante values implied by those estimated CCPs, avoiding a nested value-function solve inside the structural objective.

## Results

The keep option becomes less attractive as mileage rises. The replacement probability is therefore low at fresh-engine states and high at worn-engine states.

![Value functions and replacement probabilities](figures/value-and-ccp.png)
*Value functions and replacement probabilities*

Mileage drifts upward when the operator keeps the old engine and resets after replacement. The scattered points mark replacement decisions.

![Mileage histories for six simulated buses](figures/simulated-histories.png)
*Mileage histories for six simulated buses*

Both estimators recover the main economic pattern: replacement becomes more likely as mileage increases. Differences are largest in high-mileage states that are rarely observed because buses are usually replaced before reaching them.

![Policy rules implied by the true and estimated parameters](figures/estimated-policies.png)
*Policy rules implied by the true and estimated parameters*

The full-solution estimator nests a value-function solve inside the likelihood. The CCP estimator uses an estimated policy rule to reduce that computational burden.

**Structural parameter estimates**

| Parameter   |   True |   Full-solution ML |       CCP |
|:------------|-------:|-------------------:|----------:|
| theta_0     |   2    |           2.01812  |  2.01767  |
| theta_1     |  -0.15 |          -0.153463 | -0.153336 |

The simulated panel gives the estimators repeated choices at many mileage states.

**Simulation and solver diagnostics**

| Moment          |      Value |
|:----------------|-----------:|
| Repair rate     |   0.253181 |
| Average mileage |   2.21011  |
| VFI iterations  | 228        |
| Full ML success |   1        |
| CCP success     |   1        |

## Economic Takeaway

The hard part of dynamic discrete choice is the feedback between choices and future states. Full-solution likelihood is conceptually direct but expensive because each parameter guess requires solving the dynamic program. CCP methods trade some first-stage smoothing for speed by using observed choice probabilities to infer continuation values.

## Reproduce

```bash
python run.py
```

## References

- Rust, J. (1987). Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher. Econometrica.
- Hotz, V. J. and Miller, R. A. (1993). Conditional choice probabilities and the estimation of dynamic models. Review of Economic Studies.
- Aguirregabiria, V. and Mira, P. (2010). Dynamic discrete choice structural models: A survey. Journal of Econometrics.
