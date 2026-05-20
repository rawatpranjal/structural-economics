# Buffer-Stock Saving with IID Income by Envelope-Equation Iteration

## Overview

A CRRA household faces IID labor income and cannot borrow. Wealth is a buffer against low income. The policy says how assets shape consumption and saving.

The object is the marginal continuation value $W_a(a)$. It measures the value of one more dollar before next period's income draw. EEI updates this curve directly with the envelope condition.

The computational need is to update this curve without solving for the whole value function. The Euler equation then recovers consumption at each asset-income state.

## Equations

The household enters with assets $a$ and IID income $y_j$.
Income has probabilities $\pi_j$ over $\{y_1,\dots,y_{n_y}\}$.
With gross return $R = 1+r$, the Bellman equation is

$$
V(a,y_j) = \max_{a' \geq \underline a}\,
\{\,u(R a + y_j - a') + \beta\,W(a')\,\},
\qquad
W(a') = \sum_{\ell=1}^{n_y}\pi_\ell\,V(a',y_\ell),
$$

The policy is $g(a,y_j)$.
Consumption is $c(a,y_j) = R a + y_j - g(a,y_j)$.

Preferences are CRRA:

$$
u(c) = \frac{c^{1-\gamma}-1}{1-\gamma},
\qquad
u'(c) = c^{-\gamma},
\qquad
(u')^{-1}(\mu) = \mu^{-1/\gamma}.
$$

At an interior optimum, the Euler equation uses only $W_a(a')$:

$$
u'(c(a,y_j)) = \beta\,W_a(g(a,y_j)).
$$

The envelope condition updates that object from the policy:

$$
W_a(a) = \sum_{\ell=1}^{n_y}\pi_\ell\,V_a(a,y_\ell) =
R\,\sum_{\ell=1}^{n_y}\pi_\ell\,u'(c(a,y_\ell)).
$$

These two equations close the system without using the value level.

The borrowing limit binds when the household wants $a' < \underline a$.
Then $g(a,y_j) = \underline a$ and the Euler inequality is

$$
u'(R a + y_j - \underline a) \geq \beta\,W_a(\underline a),
$$

This case produces high MPCs near zero assets.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| CRRA $\gamma$ | 2.0 | Curvature; sets the precautionary motive and the slope of $W_a$ |
| Discount factor $\beta$ | 0.95 | Annual time preference |
| Net rate $r$ | 0.03 | Exogenous risk-free return |
| Patience-return product $\beta R$ | 0.9785 | $<1$ rules out an unbounded asset target |
| Income mean $\mu_y$ | 1.0 | Normalisation |
| Income s.d. $\sigma_y$ | 0.2 | Width of the IID labor-income shock |
| Income states $n_y$ | 5 | Width-fitted equal-spaced normal grid |
| Borrowing limit $\underline a$ | 0.0 | Hard zero; binds with positive mass |
| Upper grid bound $\bar a$ | 50.0 | Wide enough to contain the simulated tail |
| EEI asset grid | 50 pts | Power-spaced; denser at $\underline a$ |
| Reference asset grid | 600 pts | Audit grid for the EEI policy |
| Convergence tolerance | 1e-06 | Sup-norm on the consumption iterates |
| Simulation | 50,000 households, 500 periods | Forward-iterated cross section |

## Solution Method

EEI starts from a consumption policy.
The envelope step computes $W_a(a_i)$ by averaging marginal utilities across income states.
The Euler step solves for current consumption at each $(a_i,y_j)$.

The Euler step solves a scalar root at each state.
It finds $c \in (0,\,Ra + y_j - \underline a)$ such that

$$
u'(c) = \beta\,W_a(R a + y_j - c).
$$

The borrowing check comes first.
If the household wants to borrow, the solver sets $a'=\underline a$.
Otherwise bisection solves the interior Euler equation.

This update carries only one curve across iterations.
The policy still depends on assets and income after the Euler step.

```text
Algorithm: EEI for IID-income buffer-stock saving
Inputs    asset grid {a_i}, income chain ({y_j}, {pi_j}),
          primitives (beta, R, gamma), borrowing limit a_min, tolerance eps
Output    consumption policy c(a, y), saving policy g(a, y),
          marginal continuation value W_a(a)

Initialise c_0(a_i, y_j) = (R - 1) a_i + y_j        # consume current resources
repeat n = 0, 1, 2, ...
    # 1. Envelope step: collapse the policy into W_a on the exogenous grid
    W_{a,n}(a_i) = R * sum_l pi_l * u'(c_n(a_i, y_l))

    # 2. Euler step at each (a_i, y_j)
    for each i, j:
        cash = R a_i + y_j
        if u'(cash - a_min) >= beta * W_{a,n}(a_min):
            g_{n+1}(a_i, y_j) = a_min                # constraint binds
            c_{n+1}(a_i, y_j) = cash - a_min
        else:
            # Solve u'(c) - beta * W_{a,n}(cash - c) = 0 by bisection on c.
            c_star = bisect(lambda c: u'(c) - beta * W_{a,n}(cash - c),
                            lo=eps, hi=cash - a_min - eps)
            g_{n+1}(a_i, y_j) = cash - c_star
            c_{n+1}(a_i, y_j) = c_star

    err = max_{i,j} |c_{n+1}(a_i, y_j) - c_n(a_i, y_j)|
until err < eps
```

The run keeps a 50-point EEI grid.
It also solves EGP and grid VFI on that grid.
A 600-point EGP solve checks the policy on $a \leq 20$.

EEI converged in **149 iterations**.
The maximum consumption gap against the fine-grid policy is 1.09e-02.
The same gap for next assets is 1.09e-02.

## Results

The consumption policy is increasing and concave in assets. Income shifts it because IID income enters cash on hand. Near the borrowing limit, consumption tracks available cash. The fine-grid EGP reference stays within 1.09e-02 on $a \leq 20$.

<img src="figures/consumption-policy.png" alt="EEI consumption policy with fine-grid reference" width="80%">

$W_a(a)$ is steep near zero assets. One more dollar is most valuable when the buffer is empty. The curve flattens as wealth rises. The envelope condition averages the state-specific marginal utilities.

<img src="figures/value-derivative.png" alt="Marginal continuation value with state-specific decomposition" width="80%">

The simulated asset distribution is right-skewed. Mean assets are 0.4124. 3.1% of households sit at the borrowing limit. IID income keeps the asset scale modest. The borrowing-limit mass raises the average MPC to 0.2197.

<img src="figures/wealth-distribution.png" alt="Simulated terminal wealth distribution under the EEI policy" width="80%">

EEI and EGP converge at nearly the same rate. Both update policies through the Euler equation and track a consumption-level sup-norm error. Grid VFI updates the value level and tracks a value-level error, which is intrinsically larger-scaled, so VFI needs more iterations to cross the same absolute tolerance. The iteration counts are read against different error metrics, so this is a fixed-point comparison and not a clean iteration race or a timing claim.

<img src="figures/convergence-comparison.png" alt="Convergence paths for EEI, EGP, and grid VFI on the same asset grid" width="80%">

The table reports the main economic moments and policy checks. The fine-grid rows show interpolation error, not a new model.

**Solution and Simulation Summary**

| Statistic                                | Value    |
|:-----------------------------------------|:---------|
| EEI iterations                           | 149      |
| Same-grid EGP iterations                 | 151      |
| Same-grid VFI iterations                 | 203      |
| Fine-grid reference points               | 600      |
| Fine-grid reference iterations           | 140      |
| Max consumption gap vs reference, a ≤ 20 | 1.09e-02 |
| Max next-asset gap vs reference, a ≤ 20  | 1.09e-02 |
| Mean assets                              | 0.4124   |
| Fraction at borrowing limit              | 3.1%     |
| Mean MPC, 0.10 transfer                  | 0.2197   |
| Perfect-foresight MPC limit              | 0.0413   |
| 10th percentile wealth                   | 0.074    |
| 50th percentile wealth                   | 0.374    |
| 90th percentile wealth                   | 0.799    |

## Takeaway

EEI is a fixed point for the same buffer-stock household. It iterates $W_a(a)$ instead of the value level. Low-wealth households consume more of a transfer. High-wealth households smooth toward the perfect-foresight MPC $\kappa^{\ast}\approx0.041$. Here $\kappa^{\ast} = R(\beta R)^{-1/\gamma}-1$ is the MPC in the perfect-foresight limit.

The computational lesson is simple. The envelope condition can be an update rule. EGP replaces the inner bisection with one analytic marginal-utility inverse per state, so each iteration does less work than the EEI Euler step. All three methods agree up to the fine-grid gap.

## References

- Arellano, C., Maliar, L., Maliar, S. and Tsyrennikov, V. (2016). Envelope Condition Method with an Application to Default Risk Models. *Journal of Economic Dynamics and Control*, 69, 436-459.
- Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), 312-320.
- Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.
- Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 18.
