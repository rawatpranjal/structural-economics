# Deep Learning for Optimal Growth

> A tiny JAX neural net learns the Brock-Mirman saving rule from Euler residuals, then faces a closed-form audit.

## Overview

A planner allocates output between consumption today and capital tomorrow. The log Cobb-Douglas version is useful because the exact saving rule is known. That makes it a clean place to study a neural policy solver without hiding the economics.

Recent deep-learning macro methods turn dynamic models into empirical-risk minimization problems over simulated states. The loss can use lifetime rewards, Euler residuals, or Bellman residuals. This first example uses Euler residuals. The neural net proposes a feasible saving share, gradients update its weights, and the closed-form Brock-Mirman policy audits the result point by point.

## Equations

Capital fully depreciates each period. With state $k_t$, output is

$$
y_t = A k_t^{\alpha}, \qquad 0<\alpha<1,
$$

and feasibility is

$$
c_t + k_{t+1} = A k_t^{\alpha},
\qquad c_t>0, k_{t+1}>0.
$$

The planner maximizes

$$
\sum_{t=0}^{\infty}\beta^t \log c_t,
\qquad 0<\beta<1.
$$

The Euler equation is

$$
\frac{1}{c_t}
= \beta \frac{\alpha A k_{t+1}^{\alpha-1}}{c_{t+1}}.
$$

The code evaluates it as the log residual

$$
r(k;\theta)
= \log\left[
\beta \alpha A k'(k;\theta)^{\alpha-1}
\frac{c(k;\theta)}{c(k'(k;\theta);\theta)}
\right].
$$

Zero means the Euler equation holds.

The neural policy first chooses a saving share,

$$
s(k;\theta)
= s_{\min} + (s_{\max}-s_{\min})
\sigma\left(N_{\theta}(\log(k/k_{ss}))\right),
$$

then imposes feasibility by construction:

$$
k'(k;\theta)=s(k;\theta)A k^{\alpha},
\qquad
c(k;\theta)=(1-s(k;\theta))A k^{\alpha}.
$$

For this special case, the exact policy is

$$
k'(k)=\alpha\beta A k^{\alpha},
\qquad
c(k)=(1-\alpha\beta)A k^{\alpha},
$$

and the steady state is

$$
k_{ss}=(\alpha\beta A)^{1/(1-\alpha)}.
$$

## Model Setup

| Symbol | Value | Role |
|--------|-------|------|
| $\alpha$ | 0.36 | Capital share in $A k^{\alpha}$ |
| $\beta$ | 0.95 | Discount factor |
| $A$ | 2.0 | Total factor productivity |
| $k_{ss}$ | 0.5524 | Closed-form steady-state capital |
| $c_{ss}$ | 1.0629 | Closed-form steady-state consumption |
| Training states | [0.138, 1.381] | Uniform random draws around $k_{ss}$ |
| Neural net | 1-16-16-1 tanh MLP | Maps $\log(k/k_{ss})$ to a saving share |
| Saving-share bounds | [0.02, 0.95] | Keep $c$ and $k'$ feasible |
| Batch size | 256 | States per gradient step |
| Gradient steps | 6000 | Adam updates using JAX autodiff |

## Solution Method

The broader deep-learning recipe is to choose a parameterized policy, simulate or draw states, build an economic residual at those states, and minimize the average squared residual. In richer DSGE models the same template can use lifetime rewards or Bellman residuals. Here the Euler equation is enough because the deterministic growth model has one state and one policy.

```text
Algorithm: simulated-state Euler-residual training
Input: primitives alpha, beta, A; training interval K; neural weights theta
Output: feasible neural policy k'(k; theta)
Compute the exact steady state k_ss and draw minibatches k_i from K
For each k_i, map log(k_i/k_ss) through the MLP and sigmoid saving-share transform
Construct c_i, k'_i, and c(k'_i; theta) so feasibility holds by construction
Evaluate the log Euler residual r(k_i; theta)
Minimize mean_i r(k_i; theta)^2 with Adam and JAX autodiff
Audit the trained policy on an out-of-sample capital grid against the exact rule
```

The audit is not part of the training loss. It exists because this Brock-Mirman case has the closed-form rule $k'=\alpha\beta A k^\alpha$. The comparison shows what the residual-trained neural policy learned.

## Results

The trained policy is almost the exact constant-saving rule. The x-axis uses capital relative to the steady state, so the main economic object is the slope of the saving rule away from $k_{ss}$.

The exact and neural policy functions lie nearly on top of each other.

<img src="figures/policy-comparison.png" alt="Neural and closed-form capital policy" width="80%">

The residual plot is the numerical check. A small log Euler residual means the neural policy makes marginal utility today match discounted marginal utility tomorrow. The policy-error panel measures the same approximation directly against the closed-form rule, which is available only in this teaching example.

Residuals stay small across the grid, and the remaining policy error is centered near zero.

<img src="figures/euler-residuals.png" alt="Euler residuals and policy errors over the audit grid" width="80%">

Starting below the steady state gives a transition path. The exact and neural paths converge to the same steady state because the learned saving share reproduces the Brock-Mirman policy.

The neural policy tracks the closed-form transition path from the same initial capital.

<img src="figures/simulated-path.png" alt="Capital transition under neural and closed-form policies" width="80%">

The table reports the out-of-sample audit on the plotted grid. Policy errors are in capital units. The Euler residual is the maximum absolute log residual, so values near zero mean the intertemporal first-order condition is nearly satisfied.

**Neural Policy Audit**

|   final_training_loss |   mean_policy_error |   max_policy_error |   max_log_euler_residual |   final_path_error |   gradient_steps |
|----------------------:|--------------------:|-------------------:|-------------------------:|-------------------:|-----------------:|
|           2.31559e-08 |         4.43835e-05 |        0.000162423 |              0.000282804 |        0.000116169 |             6000 |

The learned saving share is nearly flat, as theory predicts. Across the audit grid its mean is 0.3420; the exact saving share is $\alpha\beta=0.3420$.

## Takeaway

Deep learning is not needed to solve this Brock-Mirman model. Its value here is pedagogical: the same residual-minimization machinery used in larger nonlinear macro models can be inspected in a case where the answer is known. Feasibility comes from the policy parameterization, training comes from Euler residuals on simulated states, and credibility comes from an out-of-sample closed-form audit.

## References

- Brock, W. A., and Mirman, L. J. (1972). *Optimal Economic Growth and Uncertainty: The Discounted Case*. Journal of Economic Theory.
- Maliar, L., Maliar, S., and Winant, P. (2022). *Deep Learning for Solving Dynamic Economic Models*. Lecture slides.
